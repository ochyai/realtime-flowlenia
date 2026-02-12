#!/usr/bin/env python3
"""CoreML-optimized Flow Lenia for macOS.

Performance optimizations over realtime_flowlenia_gpu.py:
  1. Mixed-precision (float16) — ~1.5-2x MPS throughput
  2. CoreML rendering on Apple Neural Engine — GPU freed for simulation
  3. torch.compile graph fusion (when available)
  4. Pre-allocated buffers to reduce allocation overhead

Requires: pip install coremltools  (optional — falls back to MPS rendering)

Controls:
  SPACE  = pause/resume    r = reset    q/ESC = quit
  s = screenshot    f = fullscreen    c = cycle colormap
  +/- = sim steps/frame    m = toggle modulation
  g = toggle glow          t = toggle thin-film
  d = toggle fluid distort b = benchmark mode
  1-5 = switch pattern     mouse click = perturbation
"""

import sys
import os
import time
import math

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from dataclasses import dataclass

# ── Device ──
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

USE_FP16 = DEVICE.type in ("mps", "cuda")

# ── Optional CoreML ──
try:
    import coremltools as ct
    HAS_COREML = True
except ImportError:
    HAS_COREML = False


# ── Config ──

@dataclass
class Cfg:
    X: int = 512
    Y: int = 512
    C: int = 3
    k: int = 9
    dt: float = 0.2
    sigma: float = 0.65
    growth_inject: float = 0.02
    diffusion: float = 0.005
    mod_speed: float = 1.0
    mod_depth: float = 0.4


def conn_from_matrix(mat):
    C = mat.shape[0]
    c0, c1 = [], [[] for _ in range(C)]
    i = 0
    for s in range(C):
        for t in range(C):
            n = int(mat[s, t])
            if n:
                c0 += [s] * n
                c1[t] += list(range(i, i + n))
            i += n
    return c0, c1


# ══════════════════════════════════════════════════════════════════════
#  CoreML Rendering Module
# ══════════════════════════════════════════════════════════════════════

class ToneMapGlowModule(torch.nn.Module):
    """Tone mapping + glow bloom — designed for CoreML ANE conversion.

    Runs on Apple Neural Engine, freeing the GPU for simulation.
    """

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (1, 3, H, W)
        out = img.clamp(0.0, 1.0) * 2.5
        out = out / (1.0 + out)
        out = out * out * (3.0 - 2.0 * out)

        # Saturation boost
        lum = 0.299 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.114 * out[:, 2:3]
        out = lum + (out - lum) * 1.8
        out = out.clamp(0.0, 1.0)

        # Glow (two-layer bloom)
        bright = (out - 0.3).clamp(min=0.0) * 1.5
        s1 = F.avg_pool2d(bright, 4, stride=4)
        g1 = F.interpolate(s1, size=(img.shape[2], img.shape[3]),
                           mode='bilinear', align_corners=False)
        s2 = F.avg_pool2d(s1, 4, stride=4)
        g2 = F.interpolate(s2, size=(img.shape[2], img.shape[3]),
                           mode='bilinear', align_corners=False)
        out = (out + g1 * 0.35 + g2 * 0.5).clamp(0.0, 1.0)
        return out


def build_coreml_render_model(X: int, Y: int):
    """Convert ToneMapGlow to a CoreML mlprogram for ANE."""
    if not HAS_COREML:
        return None
    try:
        module = ToneMapGlowModule().eval()
        example = torch.rand(1, 3, X, Y)
        traced = torch.jit.trace(module, example)
        model = ct.convert(
            traced,
            inputs=[ct.TensorType(name="img", shape=(1, 3, X, Y))],
            compute_units=ct.ComputeUnit.ALL,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
        )
        return model
    except Exception as e:
        print(f"  CoreML conversion failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
#  Engine — float16 optimized
# ══════════════════════════════════════════════════════════════════════

class FlowLeniaCoreML:
    """Flow Lenia engine with mixed-precision + CoreML optimizations."""

    def __init__(self, cfg: Cfg, seed: int = 0):
        self.cfg = cfg
        self.device = DEVICE
        self.t = 0.0
        D = DEVICE
        torch.manual_seed(seed)
        k, C = cfg.k, cfg.C

        # Parameters (float32 — converted to fp16 in step)
        self.R = torch.empty((), device=D).uniform_(8.0, 20.0)
        self.r = torch.empty(k, device=D).uniform_(0.2, 1.0)
        self.m0 = torch.empty(k, device=D).uniform_(0.05, 0.5)
        self.s0 = torch.empty(k, device=D).uniform_(0.001, 0.18)
        self.h0 = torch.empty(k, device=D).uniform_(0.01, 1.0)
        self.a = torch.empty(k, 3, device=D).uniform_(0.0, 1.0)
        self.b = torch.empty(k, 3, device=D).uniform_(0.001, 1.0)
        self.w = torch.empty(k, 3, device=D).uniform_(0.01, 0.5)

        M = np.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
        c0l, c1l = conn_from_matrix(M)
        self._c0 = torch.tensor(c0l, dtype=torch.long, device=D)
        c1m = torch.zeros(k, C, device=D)
        for c in range(C):
            for ki in c1l[c]:
                c1m[ki, c] = 1.0
        self._c1 = c1m

        # Half-precision copies of connection matrix
        self._c1_h = c1m.half() if USE_FP16 else c1m

        # Modulation harmonics
        rng = np.random.RandomState(seed + 777)
        sp = cfg.mod_speed
        nh = 6

        def mk(sh, lo, hi):
            return (
                torch.tensor(rng.uniform(lo, hi, sh) * sp, device=D, dtype=torch.float32),
                torch.tensor(rng.uniform(0, 2 * math.pi, sh), device=D, dtype=torch.float32),
            )

        self._mm = mk((k, nh), 0.003, 0.03)
        self._ms = mk((k, nh), 0.002, 0.02)
        self._mh = mk((k, nh), 0.004, 0.025)
        self._mdt = mk((1, nh), 0.001, 0.008)
        self._mflow = mk((1, nh), 0.002, 0.012)

        # Sobel weights (keep float32 + half copies)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32, device=D)
        ky = kx.t().contiguous()
        wc = torch.zeros(2 * C, 1, 3, 3, device=D)
        for c in range(C):
            wc[c, 0] = ky
            wc[C + c, 0] = kx
        self._sw_c = wc
        self._sw_c_h = wc.half() if USE_FP16 else wc
        w1 = torch.zeros(2, 1, 3, 3, device=D)
        w1[0, 0] = ky
        w1[1, 0] = kx
        self._sw_1 = w1
        self._sw_1_h = w1.half() if USE_FP16 else w1

        # Laplacian
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                           dtype=torch.float32, device=D)
        self._lw = lap.view(1, 1, 3, 3).expand(C, 1, 3, 3).clone()
        self._lw_h = self._lw.half() if USE_FP16 else self._lw

        # Position grids
        X, Y = cfg.X, cfg.Y
        gi, gj = torch.meshgrid(
            torch.arange(X, device=D, dtype=torch.float32),
            torch.arange(Y, device=D, dtype=torch.float32), indexing="ij")
        self._pi = gi.unsqueeze(-1)
        self._pj = gj.unsqueeze(-1)
        # Half-precision copies
        self._pi_h = self._pi.half() if USE_FP16 else self._pi
        self._pj_h = self._pj.half() if USE_FP16 else self._pj

        self.modulation_on = True

        # Try torch.compile
        self._compiled = False
        if hasattr(torch, "compile"):
            try:
                self._step_core = torch.compile(
                    self._step_core_impl, backend="aot_eager", fullgraph=False
                )
                self._compiled = True
            except Exception:
                self._step_core = self._step_core_impl
        else:
            self._step_core = self._step_core_impl

    def _mod(self, base, m, amp):
        if not self.modulation_on:
            return base
        f, p = m
        return base + torch.sin(f * self.t + p).mean(-1) * amp * self.cfg.mod_depth

    @torch.no_grad()
    def get_fK(self):
        X, Y, k = self.cfg.X, self.cfg.Y, self.cfg.k
        mid = X // 2
        c = np.mgrid[-mid:mid, -mid:mid].astype(np.float32)
        dist = torch.from_numpy(np.linalg.norm(c, axis=0)).to(DEVICE)
        K = torch.zeros(X, Y, k, device=DEVICE)
        for ki in range(k):
            D = dist / ((self.R.item() + 15) * self.r[ki].item())
            sig = 0.5 * (torch.tanh(-(D - 1) * 5) + 1)
            D3 = D.unsqueeze(-1)
            ker = (self.b[ki] * torch.exp(-((D3 - self.a[ki]) ** 2) / self.w[ki])).sum(-1)
            K[:, :, ki] = sig * ker
        K = K / K.sum(dim=(0, 1), keepdim=True).clamp(min=1e-10)
        return torch.fft.fft2(torch.fft.fftshift(K, dim=(0, 1)), dim=(0, 1))

    @torch.no_grad()
    def init_pattern(self, pat="dense_noise", seed=42):
        torch.manual_seed(seed)
        X, Y, C = self.cfg.X, self.cfg.Y, self.cfg.C
        if pat == "dense_noise":
            return torch.rand(X, Y, C, device=DEVICE) * 0.4
        elif pat == "random_center":
            A = torch.zeros(X, Y, C, device=DEVICE)
            sz = 50
            cx, cy = X // 2, Y // 2
            A[cx - sz // 2:cx + sz // 2, cy - sz // 2:cy + sz // 2] = \
                torch.rand(sz, sz, C, device=DEVICE)
            return A
        elif pat == "random_spots":
            A = torch.zeros(X, Y, C, device=DEVICE)
            for _ in range(8):
                cx = torch.randint(40, X - 40, (1,)).item()
                cy = torch.randint(40, Y - 40, (1,)).item()
                r = torch.randint(10, 30, (1,)).item()
                yy, xx = torch.meshgrid(
                    torch.arange(X, device=DEVICE),
                    torch.arange(Y, device=DEVICE), indexing="ij")
                mask = ((xx - cx) ** 2 + (yy - cy) ** 2).float() < r ** 2
                for c in range(C):
                    A[:, :, c] = torch.where(mask, torch.rand(1, device=DEVICE), A[:, :, c])
            return A
        elif pat == "circle":
            A = torch.zeros(X, Y, C, device=DEVICE)
            cx, cy = X // 2, Y // 2
            radius = min(X, Y) // 8
            yy, xx = torch.meshgrid(
                torch.arange(X, device=DEVICE),
                torch.arange(Y, device=DEVICE), indexing="ij")
            dist = ((xx - cx) ** 2 + (yy - cy) ** 2).float().sqrt()
            ring = torch.exp(-((dist - radius) ** 2) / (radius * 0.3) ** 2)
            for c in range(C):
                A[:, :, c] = ring * (0.5 + 0.5 * torch.rand(1, device=DEVICE))
            return A
        return torch.rand(X, Y, C, device=DEVICE) * 0.3

    def _step_core_impl(self, U_h, A_h, m_h, s_h, h_h, dt_mod_h, flow_scale_h,
                        sw_c, sw_1, lw, c1, pi, pj, X_f, Y_f, C, cfg_growth, cfg_diff):
        """Core computation in half precision. Separated for torch.compile."""
        # Growth
        G = (torch.exp(-((U_h - m_h) ** 2) / (2 * s_h ** 2)) * 2 - 1) * h_h
        Gc = torch.matmul(G, c1)

        # Sobel on Gc
        inp_gc = F.pad(Gc.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="circular")
        o_gc = F.conv2d(inp_gc.repeat(1, 2, 1, 1), sw_c, groups=2 * C).squeeze(0)
        nG = torch.stack([o_gc[:C].permute(1, 2, 0), o_gc[C:].permute(1, 2, 0)], dim=2)

        # Sobel on sum(A)
        A_sum = A_h.sum(-1, keepdim=True)
        inp_a = F.pad(A_sum.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="circular")
        o_a = F.conv2d(inp_a.repeat(1, 2, 1, 1), sw_1, groups=2).squeeze(0)
        nA_sob = o_a.permute(1, 2, 0).unsqueeze(-1)

        # Flow field
        al = (A_h[:, :, None, :] / C).square().clamp(0.0, 1.0)
        Ff = (nG * (1 - al) - nA_sob * al) * flow_scale_h

        # grid_sample advection
        fi, fj = Ff[:, :, 0, :], Ff[:, :, 1, :]
        si = (pi - dt_mod_h * fi) % X_f
        sj = (pj - dt_mod_h * fj) % Y_f

        pad = 2
        A_in = A_h.permute(2, 0, 1).unsqueeze(1)
        A_pad = F.pad(A_in, (pad, pad, pad, pad), mode="circular")
        XP = X_f + 2 * pad
        YP = Y_f + 2 * pad

        ni = 2.0 * (si + pad) / (XP - 1) - 1.0
        nj = 2.0 * (sj + pad) / (YP - 1) - 1.0
        grids = torch.stack([nj.permute(2, 0, 1), ni.permute(2, 0, 1)], dim=-1)

        nw = F.grid_sample(A_pad, grids, mode='bilinear',
                           padding_mode='zeros', align_corners=True)
        nw = nw.squeeze(1).permute(1, 2, 0)

        # Growth injection
        nw = nw + dt_mod_h * cfg_growth * Gc

        # Diffusion
        if cfg_diff > 0:
            lap_inp = F.pad(nw.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="circular")
            lap_out = F.conv2d(lap_inp, lw, groups=C).squeeze(0).permute(1, 2, 0)
            nw = nw + dt_mod_h * cfg_diff * lap_out

        return nw

    @torch.no_grad()
    def step(self, A, fK):
        cfg = self.cfg
        X, Y, C = cfg.X, cfg.Y, cfg.C
        self.t += 1.0

        # Modulated params (float32 — cheap, few elements)
        m = self._mod(self.m0, self._mm, 0.15).view(1, 1, -1)
        s = self._mod(self.s0, self._ms, 0.05).clamp(min=0.001).view(1, 1, -1)
        h = self._mod(self.h0, self._mh, 0.3).clamp(min=0.001).view(1, 1, -1)

        dt_mod = cfg.dt
        if self.modulation_on:
            f, p = self._mdt
            dt_mod *= 1.0 + 0.15 * torch.sin(f * self.t + p).mean().item() * cfg.mod_depth
        flow_scale = 1.0
        if self.modulation_on:
            f, p = self._mflow
            flow_scale = 1.0 + 0.3 * torch.sin(f * self.t + p).mean().item() * cfg.mod_depth

        # ── FFT in float32 (precision-sensitive) ──
        fA = torch.fft.fft2(A, dim=(0, 1))
        U = torch.fft.ifft2(fK * fA[:, :, self._c0], dim=(0, 1)).real

        # ── Growth + Flow + Advection in float16 ──
        if USE_FP16:
            U_h = U.half()
            A_h = A.half()
            m_h, s_h, h_h = m.half(), s.half(), h.half()
            dt_h = torch.tensor(dt_mod, device=DEVICE, dtype=torch.float16)
            fs_h = torch.tensor(flow_scale, device=DEVICE, dtype=torch.float16)
            sw_c, sw_1, lw = self._sw_c_h, self._sw_1_h, self._lw_h
            c1, pi, pj = self._c1_h, self._pi_h, self._pj_h
        else:
            U_h, A_h = U, A
            m_h, s_h, h_h = m, s, h
            dt_h = torch.tensor(dt_mod, device=DEVICE)
            fs_h = torch.tensor(flow_scale, device=DEVICE)
            sw_c, sw_1, lw = self._sw_c, self._sw_1, self._lw
            c1, pi, pj = self._c1, self._pi, self._pj

        nw = self._step_core(
            U_h, A_h, m_h, s_h, h_h, dt_h, fs_h,
            sw_c, sw_1, lw, c1, pi, pj,
            float(X), float(Y), C, cfg.growth_inject, cfg.diffusion,
        )

        # Soft mass damping
        nw_f = nw.float()
        target = 0.20
        cur = nw_f.mean()
        if cur > 1e-8:
            ratio = target / cur
            nw_f = nw_f * (1.0 + 0.05 * (ratio - 1.0))

        return nw_f.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  GPU Shader Effects (same as gpu version)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def gpu_noise(pos, t, scale=1.0, speed=0.1):
    p = pos * scale
    v = (torch.sin(p[..., 0] * 1.7 + t * speed * 0.8 + torch.sin(p[..., 1] * 2.3) * 0.5) *
         torch.cos(p[..., 1] * 2.1 + t * speed * 1.1 + torch.sin(p[..., 0] * 1.9) * 0.5))
    v = v + torch.sin(p[..., 0] * 3.3 + p[..., 1] * 1.1 + t * speed * 1.3) * 0.5
    v = v + torch.cos(p[..., 0] * 0.7 - p[..., 1] * 2.7 + t * speed * 0.6) * 0.3
    return v


@torch.no_grad()
def gpu_fbm2(pos, t, scale=1.0, speed=0.1):
    v = gpu_noise(pos, t, scale, speed) * 0.5
    v = v + gpu_noise(pos, t + 17.3, scale * 2.0, speed) * 0.25
    return v


@torch.no_grad()
def thin_film_interference(thickness, angle):
    wavelengths = torch.tensor([650.0, 510.0, 475.0], device=thickness.device)
    opt_path = 2.0 * 1.33 * thickness.unsqueeze(-1) * torch.cos(angle).unsqueeze(-1)
    phase = 2.0 * math.pi * opt_path / wavelengths.view(1, 1, 3)
    intensity = 0.5 + 0.5 * torch.cos(phase + torch.tensor([0.0, 2.094, 4.188],
                                                             device=thickness.device))
    intensity = intensity.pow(0.7)
    lum = intensity.mean(-1, keepdim=True)
    intensity = lum + (intensity - lum) * 1.8
    return intensity.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Renderer — with CoreML ANE path
# ══════════════════════════════════════════════════════════════════════

class LiquidRendererCoreML:
    def __init__(self, X, Y, device):
        self.device = device
        self.X, self.Y = X, Y

        gi = torch.linspace(0, 1, X, device=device)
        gj = torch.linspace(0, 1, Y, device=device)
        pi, pj = torch.meshgrid(gi, gj, indexing="ij")
        self.pos = torch.stack([pi, pj], dim=-1)

        base_i = torch.linspace(-1, 1, X, device=device)
        base_j = torch.linspace(-1, 1, Y, device=device)
        bi, bj = torch.meshgrid(base_i, base_j, indexing="ij")
        self._base_grid = torch.stack([bj, bi], dim=-1).unsqueeze(0)

        self.glow_on = False
        self.thin_film_on = False
        self.distort_on = True

        # CoreML rendering model
        self._coreml_model = None
        self._coreml_active = False

    def enable_coreml(self, model):
        if model is not None:
            self._coreml_model = model
            self._coreml_active = True

    def _to_lum(self, A):
        return 0.299 * A[:, :, 0] + 0.587 * A[:, :, 1] + 0.114 * A[:, :, 2]

    def _tonemap(self, lum):
        lum = lum * 2.5
        lum = lum / (1.0 + lum)
        lum = lum * lum * (3.0 - 2.0 * lum)
        return lum

    def _apply_distort(self, img_nchw, t):
        breath = 0.015 + 0.01 * math.sin(t * 0.005) + 0.005 * math.sin(t * 0.013)
        dx = gpu_fbm2(self.pos, t * 0.02, scale=1.5, speed=0.15) * breath
        dy = gpu_fbm2(self.pos + 50.0, t * 0.02, scale=1.5, speed=0.12) * breath
        dx = dx + gpu_noise(self.pos, t * 0.04, scale=3.0, speed=0.3) * breath * 0.3
        dy = dy + gpu_noise(self.pos + 80.0, t * 0.04, scale=3.0, speed=0.25) * breath * 0.3
        disp = torch.stack([dy * 2, dx * 2], dim=-1).unsqueeze(0)
        grid = self._base_grid + disp
        return F.grid_sample(img_nchw, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)

    def _apply_glow_mps(self, img_nchw):
        bright = (img_nchw - 0.3).clamp(min=0.0) * 1.5
        s1 = F.avg_pool2d(bright, 4, stride=4)
        g1 = F.interpolate(s1, size=img_nchw.shape[2:], mode='bilinear', align_corners=False)
        s2 = F.avg_pool2d(s1, 4, stride=4)
        g2 = F.interpolate(s2, size=img_nchw.shape[2:], mode='bilinear', align_corners=False)
        return (img_nchw + g1 * 0.35 + g2 * 0.5).clamp(0.0, 1.0)

    def _render_coreml(self, img_nchw):
        """Render tone mapping + glow on ANE via CoreML."""
        # MPS → numpy (zero-copy on Apple Silicon unified memory)
        inp = img_nchw.cpu().numpy().astype(np.float32)
        result = self._coreml_model.predict({"img": inp})
        # Find the output key
        out_key = list(result.keys())[0]
        rendered = np.array(result[out_key])
        return torch.from_numpy(rendered).to(self.device)

    @torch.no_grad()
    def render_mono(self, A, t):
        lum = self._tonemap(self._to_lum(A.clamp(0.0, 1.0)))
        img = lum.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1).clone()

        if self.distort_on:
            img = self._apply_distort(img, t)
        if self.glow_on:
            if self._coreml_active:
                img = self._render_coreml(img)
            else:
                img = self._apply_glow_mps(img)

        return np.ascontiguousarray(
            (img.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())

    @torch.no_grad()
    def render_mono_warm(self, A, t):
        lum = self._tonemap(self._to_lum(A.clamp(0.0, 1.0)))
        r = lum.pow(0.75)
        g = lum.pow(1.1) * 0.8
        b = lum.pow(1.8) * 0.4
        img = torch.stack([r, g, b], dim=0).unsqueeze(0)

        if self.distort_on:
            img = self._apply_distort(img, t)
        if self.glow_on:
            if self._coreml_active:
                img = self._render_coreml(img)
            else:
                img = self._apply_glow_mps(img)

        return np.ascontiguousarray(
            (img.squeeze(0)[[2, 1, 0]].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())

    @torch.no_grad()
    def render_liquid(self, A, t):
        out = A.clamp(0.0, 1.0)

        if self.thin_film_on:
            lum = self._to_lum(out)
            flow1 = gpu_fbm2(self.pos, t * 0.015, scale=2.0, speed=0.08)
            flow2 = gpu_fbm2(self.pos + 40.0, t * 0.02, scale=3.5, speed=0.12)
            thickness = 200.0 + 350.0 * (flow1 * 0.5 + 0.5) + 120.0 * flow2 + lum * 150.0
            angle = 0.2 + 0.35 * (flow2 * 0.5 + 0.5)
            film = thin_film_interference(thickness, angle)
            activity = lum.unsqueeze(-1).clamp(0.0, 1.0)
            out = out * 0.6 + film * activity * 0.4

        if self._coreml_active and self.glow_on:
            # Use CoreML for tone mapping + glow (ANE)
            img = out.permute(2, 0, 1).unsqueeze(0)
            if self.distort_on:
                img = self._apply_distort(img, t)
            img = self._render_coreml(img)
            return np.ascontiguousarray(
                (img.squeeze(0)[[2, 1, 0]].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())

        # MPS fallback: tone map
        out = out * 2.5
        out = out / (1.0 + out)
        out = out * out * (3.0 - 2.0 * out)
        lum2 = (0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]).unsqueeze(-1)
        out = lum2 + (out - lum2) * 1.8
        out = out.clamp(0.0, 1.0)

        img = out.permute(2, 0, 1).unsqueeze(0)
        if self.distort_on:
            img = self._apply_distort(img, t)
        if self.glow_on:
            img = self._apply_glow_mps(img)

        return np.ascontiguousarray(
            (img.squeeze(0)[[2, 1, 0]].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())

    @torch.no_grad()
    def render_direct(self, A):
        return np.ascontiguousarray(
            (A[:, :, [2, 1, 0]].clamp(0, 1) * 255).byte().cpu().numpy())


# ── Perturbation ──

def add_perturbation(A, x, y, device):
    H, W, C = A.shape
    radius = np.random.randint(8, 25)
    yy, xx = np.mgrid[max(0, y - radius):min(H, y + radius),
                       max(0, x - radius):min(W, x + radius)]
    if yy.size == 0 or xx.size == 0:
        return A
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2).astype(np.float32)
    mask = dist < radius
    blob = np.random.rand(yy.shape[0], yy.shape[1], C).astype(np.float32)
    blob *= mask[:, :, None]
    blob_t = torch.from_numpy(blob).to(device)
    A = A.clone()
    y_s, x_s = max(0, y - radius), max(0, x - radius)
    A[y_s:y_s + blob_t.shape[0], x_s:x_s + blob_t.shape[1]] += blob_t
    return A.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Benchmark
# ══════════════════════════════════════════════════════════════════════

def run_benchmark(fl, fK, A, n_steps=50):
    """Benchmark simulation throughput."""
    # Warm-up
    for _ in range(5):
        A = fl.step(A, fK)
    if DEVICE.type == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        A = fl.step(A, fK)
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    fps = n_steps / elapsed
    ms = elapsed / n_steps * 1000
    return fps, ms, A


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

PATTERNS = ["dense_noise", "random_center", "random_spots", "circle", "noise"]
PATTERN_KEYS = {ord(str(i + 1)): i for i in range(5)}
WINDOW_NAME = "Flow Lenia (CoreML)"
DISPLAY_SIZE = 1024


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["click"] = (x, y)


def main():
    cfg = Cfg()
    seed = 42

    print(f"Flow Lenia — CoreML Optimized")
    print(f"  Device: {DEVICE}")
    print(f"  Precision: {'float16 + float32 (mixed)' if USE_FP16 else 'float32'}")
    print(f"  CoreML: {'available' if HAS_COREML else 'not installed (pip install coremltools)'}")

    fl = FlowLeniaCoreML(cfg, seed=seed)
    print(f"  torch.compile: {'enabled' if fl._compiled else 'disabled'}")
    fK = fl.get_fK()
    A = fl.init_pattern("dense_noise", seed=seed)

    renderer = LiquidRendererCoreML(cfg.X, cfg.Y, DEVICE)

    # Build CoreML rendering model
    if HAS_COREML:
        print("  Building CoreML render model ...")
        coreml_model = build_coreml_render_model(cfg.X, cfg.Y)
        if coreml_model:
            renderer.enable_coreml(coreml_model)
            print("  CoreML ANE rendering: enabled")

    CMAP_NAMES = ["mono", "mono_warm", "liquid", "direct"]

    print(f"\n  Sim: {cfg.X}x{cfg.Y} C={cfg.C} k={cfg.k}")
    print(f"  Display: {DISPLAY_SIZE}x{DISPLAY_SIZE}")
    print()
    print("  SPACE=pause  r=reset  q/ESC=quit  s=screenshot  f=fullscreen")
    print("  c=colormap  +/-=steps  m=modulation  g=glow  t=thin-film  d=distort")
    print("  b=benchmark  1-5=pattern  mouse=perturbation")

    # Benchmark
    print("\nBenchmark ...")
    bench_fps, bench_ms, A = run_benchmark(fl, fK, A, n_steps=50)
    print(f"  {bench_fps:.1f} FPS ({bench_ms:.1f} ms/step)")

    # Re-init after benchmark
    A = fl.init_pattern("dense_noise", seed=seed)
    print("\nWarm-up ...")
    for _ in range(5):
        A = fl.step(A, fK)
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    print("Ready.\n")

    paused = False
    fullscreen = False
    steps_per_frame = 2
    current_pattern = "dense_noise"
    screenshot_counter = 0
    click_param = {"click": None}
    colormap_mode = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE, DISPLAY_SIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, click_param)

    frame_times = []
    fps = 0.0
    step_count = 0

    try:
        while True:
            t0 = time.perf_counter()

            if click_param["click"] is not None:
                cx, cy = click_param["click"]
                rect = cv2.getWindowImageRect(WINDOW_NAME)
                if rect[2] > 0 and rect[3] > 0:
                    sx = int(cx * cfg.X / rect[2])
                    sy = int(cy * cfg.Y / rect[3])
                else:
                    sx, sy = cx, cy
                A = add_perturbation(A, max(0, min(cfg.X - 1, sx)),
                                     max(0, min(cfg.Y - 1, sy)), DEVICE)
                click_param["click"] = None

            if not paused:
                for _ in range(steps_per_frame):
                    A = fl.step(A, fK)
                step_count += steps_per_frame

            if colormap_mode == 0:
                display = renderer.render_mono(A, fl.t)
            elif colormap_mode == 1:
                display = renderer.render_mono_warm(A, fl.t)
            elif colormap_mode == 2:
                display = renderer.render_liquid(A, fl.t)
            else:
                display = renderer.render_direct(A)

            dt_frame = time.perf_counter() - t0
            frame_times.append(dt_frame)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

            status = "PAUSED" if paused else f"step {step_count}"
            mod_str = "MOD" if fl.modulation_on else "---"
            fx = ""
            if colormap_mode <= 2:
                fx += "G" if renderer.glow_on else ""
                fx += "D" if renderer.distort_on else ""
                fx += "T" if renderer.thin_film_on else ""
            coreml_str = " [ANE]" if renderer._coreml_active else ""
            prec_str = "fp16" if USE_FP16 else "fp32"
            info = f"FPS:{fps:.0f} x{steps_per_frame} | {status} | {CMAP_NAMES[colormap_mode]} {mod_str} {fx} | {prec_str}{coreml_str}"
            cv2.putText(display, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                seed = int(time.time()) % 100000
                fl = FlowLeniaCoreML(cfg, seed=seed)
                fK = fl.get_fK()
                A = fl.init_pattern(current_pattern, seed=seed)
                for _ in range(5):
                    A = fl.step(A, fK)
                step_count = 0
                print(f"Reset seed={seed}")
            elif key == ord('s'):
                fname = f"flowlenia_{screenshot_counter:04d}.png"
                cv2.imwrite(fname, display)
                screenshot_counter += 1
                print(f"Saved: {fname}")
            elif key == ord('f'):
                fullscreen = not fullscreen
                prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)
            elif key == ord('c'):
                colormap_mode = (colormap_mode + 1) % len(CMAP_NAMES)
                print(f"Colormap: {CMAP_NAMES[colormap_mode]}")
            elif key == ord('m'):
                fl.modulation_on = not fl.modulation_on
                print(f"Modulation: {'ON' if fl.modulation_on else 'OFF'}")
            elif key == ord('g'):
                renderer.glow_on = not renderer.glow_on
                print(f"Glow: {'ON' if renderer.glow_on else 'OFF'}")
            elif key == ord('t'):
                renderer.thin_film_on = not renderer.thin_film_on
                print(f"Thin-film: {'ON' if renderer.thin_film_on else 'OFF'}")
            elif key == ord('d'):
                renderer.distort_on = not renderer.distort_on
                print(f"Distort: {'ON' if renderer.distort_on else 'OFF'}")
            elif key == ord('b'):
                print("Running benchmark ...")
                bfps, bms, _ = run_benchmark(fl, fK, A.clone(), n_steps=50)
                print(f"  {bfps:.1f} FPS ({bms:.1f} ms/step)")
            elif key == ord('+') or key == ord('='):
                steps_per_frame = min(steps_per_frame + 1, 10)
                print(f"Steps/frame: {steps_per_frame}")
            elif key == ord('-') or key == ord('_'):
                steps_per_frame = max(steps_per_frame - 1, 1)
                print(f"Steps/frame: {steps_per_frame}")
            elif key in PATTERN_KEYS:
                idx = PATTERN_KEYS[key]
                current_pattern = PATTERNS[idx]
                A = fl.init_pattern(current_pattern, seed=int(time.time()) % 100000)
                step_count = 0
                print(f"Pattern: {current_pattern}")

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
