#!/usr/bin/env python3
"""CoreML-optimized Flow Lenia for macOS.

Performance optimizations:
  1. Simulation at 512x512 with display upscale to 1024
  2. Mixed-precision — fp16 growth math + grid_sample data, fp32 coordinates
  3. CoreML rendering on Apple Neural Engine (optional)
  4. torch.compile graph fusion

Features:
  - Auto-evolving parameters with smooth interpolation
  - Pattern crossfade transitions
  - Liquid shader effects (distortion, thin-film, glow)

Controls:
  SPACE  = pause/resume    r = reset    q/ESC = quit
  s = screenshot    f = fullscreen    c = cycle colormap
  +/- = sim steps/frame    m = toggle modulation
  g = toggle glow          t = toggle thin-film
  d = toggle fluid distort b = benchmark mode
  e = toggle auto-evolution
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
    X: int = 512          # simulation resolution
    Y: int = 512
    C: int = 3
    k: int = 9
    dt: float = 0.2
    sigma: float = 0.65
    growth_inject: float = 0.02
    diffusion: float = 0.005
    mod_speed: float = 1.0
    mod_depth: float = 0.4
    # Evolution
    evolve_interval: float = 20.0   # seconds between parameter transitions
    evolve_duration: float = 5.0    # seconds for smooth transition
    crossfade_duration: float = 3.0 # seconds for pattern crossfade


DISPLAY_SIZE = 1024


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
    """Tone mapping + glow bloom — designed for CoreML ANE conversion."""

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        out = img.clamp(0.0, 1.0) * 2.5
        out = out / (1.0 + out)
        out = out * out * (3.0 - 2.0 * out)

        lum = 0.299 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.114 * out[:, 2:3]
        out = lum + (out - lum) * 1.8
        out = out.clamp(0.0, 1.0)

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
#  Parameter Evolution
# ══════════════════════════════════════════════════════════════════════

class ParamEvolver:
    """Smoothly evolves Lenia parameters over time."""

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.evolving = True
        self._last_evolve = time.perf_counter()
        self._transition_start = None
        self._target_params = None
        self._source_params = None

    def _snapshot_params(self, fl):
        return {
            'R': fl.R.clone(),
            'r': fl.r.clone(),
            'm0': fl.m0.clone(),
            's0': fl.s0.clone(),
            'h0': fl.h0.clone(),
            'a': fl.a.clone(),
            'b': fl.b.clone(),
            'w': fl.w.clone(),
        }

    def _generate_target(self, fl):
        D = fl.device
        k = fl.cfg.k
        return {
            'R': torch.empty((), device=D).uniform_(8.0, 20.0),
            'r': torch.empty(k, device=D).uniform_(0.2, 1.0),
            'm0': torch.empty(k, device=D).uniform_(0.05, 0.5),
            's0': torch.empty(k, device=D).uniform_(0.001, 0.18),
            'h0': torch.empty(k, device=D).uniform_(0.01, 1.0),
            'a': torch.empty(k, 3, device=D).uniform_(0.0, 1.0),
            'b': torch.empty(k, 3, device=D).uniform_(0.001, 1.0),
            'w': torch.empty(k, 3, device=D).uniform_(0.01, 0.5),
        }

    def _apply_lerp(self, fl, alpha):
        src, tgt = self._source_params, self._target_params
        # Smooth ease in/out
        t = alpha * alpha * (3.0 - 2.0 * alpha)
        for key in src:
            val = src[key] * (1 - t) + tgt[key] * t
            setattr(fl, key, val)

    def update(self, fl):
        if not self.evolving:
            return False

        now = time.perf_counter()

        if self._transition_start is not None:
            elapsed = now - self._transition_start
            duration = self.cfg.evolve_duration
            if elapsed < duration:
                alpha = elapsed / duration
                self._apply_lerp(fl, alpha)
                return True
            else:
                # Transition complete
                self._apply_lerp(fl, 1.0)
                self._transition_start = None
                self._last_evolve = now
                return True

        if now - self._last_evolve >= self.cfg.evolve_interval:
            self._source_params = self._snapshot_params(fl)
            self._target_params = self._generate_target(fl)
            self._transition_start = now
            return True

        return False

    @property
    def transitioning(self):
        return self._transition_start is not None


# ══════════════════════════════════════════════════════════════════════
#  Engine — mixed-precision optimized
# ══════════════════════════════════════════════════════════════════════

class FlowLeniaCoreML:
    """Flow Lenia engine with surgical mixed-precision.

    Float16: growth function (exp, matmul), grid_sample data.
    Float32: position grids, grid_sample coords, conv weights, FFT.
    """

    def __init__(self, cfg: Cfg, seed: int = 0):
        self.cfg = cfg
        self.device = DEVICE
        self.t = 0.0
        D = DEVICE
        torch.manual_seed(seed)
        k, C = cfg.k, cfg.C

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

        # Sobel weights (float32)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32, device=D)
        ky = kx.t().contiguous()
        wc = torch.zeros(2 * C, 1, 3, 3, device=D)
        for c in range(C):
            wc[c, 0] = ky
            wc[C + c, 0] = kx
        self._sw_c = wc
        w1 = torch.zeros(2, 1, 3, 3, device=D)
        w1[0, 0] = ky
        w1[1, 0] = kx
        self._sw_1 = w1

        # Laplacian (float32)
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                           dtype=torch.float32, device=D)
        self._lw = lap.view(1, 1, 3, 3).expand(C, 1, 3, 3).clone()

        # Position grids (float32 — critical for smooth grid_sample)
        X, Y = cfg.X, cfg.Y
        gi, gj = torch.meshgrid(
            torch.arange(X, device=D, dtype=torch.float32),
            torch.arange(Y, device=D, dtype=torch.float32), indexing="ij")
        self._pi = gi.unsqueeze(-1)
        self._pj = gj.unsqueeze(-1)

        self.modulation_on = True
        self._fK_dirty = True  # recompute kernel when params change

        # torch.compile
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
        self._fK_dirty = False
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

    def _step_core_impl(self, U, A, A_h, m_h, s_h, h_h, c1_h,
                        sw_c, sw_1, lw, pi, pj,
                        dt_mod, flow_scale, X_f, Y_f, C,
                        cfg_growth, cfg_diff):
        """Core step: fp16 growth math + fp16 grid_sample data, fp32 coords."""
        # ── Growth in fp16 ──
        if USE_FP16:
            U_h = U.half()
            G = (torch.exp(-((U_h - m_h) ** 2) / (2 * s_h ** 2)) * 2 - 1) * h_h
            Gc = torch.matmul(G, c1_h).float()
        else:
            G = (torch.exp(-((U - m_h) ** 2) / (2 * s_h ** 2)) * 2 - 1) * h_h
            Gc = torch.matmul(G, c1_h)

        # ── Sobel on Gc (float32) ──
        inp_gc = F.pad(Gc.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="circular")
        o_gc = F.conv2d(inp_gc.repeat(1, 2, 1, 1), sw_c, groups=2 * C).squeeze(0)
        nG = torch.stack([o_gc[:C].permute(1, 2, 0), o_gc[C:].permute(1, 2, 0)], dim=2)

        # ── Sobel on sum(A) (float32) ──
        A_sum = A.sum(-1, keepdim=True)
        inp_a = F.pad(A_sum.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="circular")
        o_a = F.conv2d(inp_a.repeat(1, 2, 1, 1), sw_1, groups=2).squeeze(0)
        nA_sob = o_a.permute(1, 2, 0).unsqueeze(-1)

        # ── Flow field (float32) ──
        al = (A[:, :, None, :] / C).square().clamp(0.0, 1.0)
        Ff = (nG * (1 - al) - nA_sob * al) * flow_scale

        # ── grid_sample advection: fp32 coords, fp16 data ──
        fi, fj = Ff[:, :, 0, :], Ff[:, :, 1, :]
        si = (pi - dt_mod * fi) % X_f
        sj = (pj - dt_mod * fj) % Y_f

        pad = 2
        # Use fp16 data for grid_sample (coords stay fp32)
        A_in = A_h.permute(2, 0, 1).unsqueeze(1) if USE_FP16 else A.permute(2, 0, 1).unsqueeze(1)
        A_pad = F.pad(A_in, (pad, pad, pad, pad), mode="circular")
        XP = X_f + 2 * pad
        YP = Y_f + 2 * pad

        ni = 2.0 * (si + pad) / (XP - 1) - 1.0
        nj = 2.0 * (sj + pad) / (YP - 1) - 1.0
        grids = torch.stack([nj.permute(2, 0, 1), ni.permute(2, 0, 1)], dim=-1)

        nw = F.grid_sample(A_pad, grids, mode='bilinear',
                           padding_mode='zeros', align_corners=True)
        nw = nw.squeeze(1).permute(1, 2, 0).float()

        # ── Growth injection ──
        nw = nw + dt_mod * cfg_growth * Gc

        # ── Diffusion ──
        if cfg_diff > 0:
            lap_inp = F.pad(nw.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="circular")
            lap_out = F.conv2d(lap_inp, lw, groups=C).squeeze(0).permute(1, 2, 0)
            nw = nw + dt_mod * cfg_diff * lap_out

        return nw

    @torch.no_grad()
    def step(self, A, fK):
        cfg = self.cfg
        X, Y, C = cfg.X, cfg.Y, cfg.C
        self.t += 1.0

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

        # FFT in float32
        fA = torch.fft.fft2(A, dim=(0, 1))
        U = torch.fft.ifft2(fK * fA[:, :, self._c0], dim=(0, 1)).real

        m_h = m.half() if USE_FP16 else m
        s_h = s.half() if USE_FP16 else s
        h_h = h.half() if USE_FP16 else h
        A_h = A.half() if USE_FP16 else A

        nw = self._step_core(
            U, A, A_h, m_h, s_h, h_h, self._c1_h,
            self._sw_c, self._sw_1, self._lw,
            self._pi, self._pj,
            dt_mod, flow_scale, float(X), float(Y), C,
            cfg.growth_inject, cfg.diffusion,
        )

        target = 0.20
        cur = nw.mean()
        if cur > 1e-8:
            ratio = target / cur
            nw = nw * (1.0 + 0.05 * (ratio - 1.0))

        return nw.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  GPU Shader Effects
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
#  Renderer — upscales sim to display resolution
# ══════════════════════════════════════════════════════════════════════

class LiquidRendererCoreML:
    def __init__(self, sim_x, sim_y, display_size, device):
        self.device = device
        self.sim_x, self.sim_y = sim_x, sim_y
        self.display_size = display_size

        # Shader grids at display resolution
        gi = torch.linspace(0, 1, display_size, device=device)
        gj = torch.linspace(0, 1, display_size, device=device)
        pi, pj = torch.meshgrid(gi, gj, indexing="ij")
        self.pos = torch.stack([pi, pj], dim=-1)

        base_i = torch.linspace(-1, 1, display_size, device=device)
        base_j = torch.linspace(-1, 1, display_size, device=device)
        bi, bj = torch.meshgrid(base_i, base_j, indexing="ij")
        self._base_grid = torch.stack([bj, bi], dim=-1).unsqueeze(0)

        self.glow_on = False
        self.thin_film_on = False
        self.distort_on = True

        self._coreml_model = None
        self._coreml_active = False

    def enable_coreml(self, model):
        if model is not None:
            self._coreml_model = model
            self._coreml_active = True

    def _upscale(self, A):
        """Upscale (X,Y,C) sim state to (display,display,C) with bilinear."""
        img = A.permute(2, 0, 1).unsqueeze(0)
        up = F.interpolate(img, size=(self.display_size, self.display_size),
                           mode='bilinear', align_corners=False)
        return up

    def _to_lum_nchw(self, img):
        return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

    def _tonemap_nchw(self, img):
        img = img * 2.5
        img = img / (1.0 + img)
        img = img * img * (3.0 - 2.0 * img)
        return img

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

    def _apply_glow(self, img_nchw):
        bright = (img_nchw - 0.3).clamp(min=0.0) * 1.5
        s1 = F.avg_pool2d(bright, 4, stride=4)
        g1 = F.interpolate(s1, size=img_nchw.shape[2:], mode='bilinear', align_corners=False)
        s2 = F.avg_pool2d(s1, 4, stride=4)
        g2 = F.interpolate(s2, size=img_nchw.shape[2:], mode='bilinear', align_corners=False)
        return (img_nchw + g1 * 0.35 + g2 * 0.5).clamp(0.0, 1.0)

    def _render_coreml(self, img_nchw):
        inp = img_nchw.cpu().numpy().astype(np.float32)
        result = self._coreml_model.predict({"img": inp})
        out_key = list(result.keys())[0]
        rendered = np.array(result[out_key])
        return torch.from_numpy(rendered).to(self.device)

    def _to_numpy_bgr(self, img_nchw):
        return np.ascontiguousarray(
            (img_nchw.squeeze(0)[[2, 1, 0]].permute(1, 2, 0).clamp(0, 1) * 255)
            .byte().cpu().numpy())

    @torch.no_grad()
    def render_mono(self, A, t):
        img = self._upscale(A.clamp(0.0, 1.0))
        lum = self._to_lum_nchw(img)
        lum = self._tonemap_nchw(lum)
        img = lum.expand(-1, 3, -1, -1).clone()

        if self.distort_on:
            img = self._apply_distort(img, t)
        if self.glow_on:
            if self._coreml_active:
                img = self._render_coreml(img)
            else:
                img = self._apply_glow(img)

        return np.ascontiguousarray(
            (img.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())

    @torch.no_grad()
    def render_mono_warm(self, A, t):
        img = self._upscale(A.clamp(0.0, 1.0))
        lum = self._to_lum_nchw(img)
        lum = self._tonemap_nchw(lum)
        r = lum.pow(0.75)
        g = lum.pow(1.1) * 0.8
        b = lum.pow(1.8) * 0.4
        img = torch.cat([r, g, b], dim=1)

        if self.distort_on:
            img = self._apply_distort(img, t)
        if self.glow_on:
            if self._coreml_active:
                img = self._render_coreml(img)
            else:
                img = self._apply_glow(img)

        return self._to_numpy_bgr(img)

    @torch.no_grad()
    def render_liquid(self, A, t):
        img = self._upscale(A.clamp(0.0, 1.0))

        if self.thin_film_on:
            lum = self._to_lum_nchw(img).squeeze(0).squeeze(0)
            flow1 = gpu_fbm2(self.pos, t * 0.015, scale=2.0, speed=0.08)
            flow2 = gpu_fbm2(self.pos + 40.0, t * 0.02, scale=3.5, speed=0.12)
            thickness = 200.0 + 350.0 * (flow1 * 0.5 + 0.5) + 120.0 * flow2 + lum * 150.0
            angle = 0.2 + 0.35 * (flow2 * 0.5 + 0.5)
            film = thin_film_interference(thickness, angle)
            activity = lum.unsqueeze(-1).clamp(0.0, 1.0)
            out_hwc = img.squeeze(0).permute(1, 2, 0)
            out_hwc = out_hwc * 0.6 + film * activity * 0.4
            img = out_hwc.permute(2, 0, 1).unsqueeze(0)

        img = self._tonemap_nchw(img)
        lum2 = self._to_lum_nchw(img)
        img = lum2 + (img - lum2) * 1.8
        img = img.clamp(0.0, 1.0)

        if self.distort_on:
            img = self._apply_distort(img, t)

        if self.glow_on:
            if self._coreml_active:
                img = self._render_coreml(img)
            else:
                img = self._apply_glow(img)

        return self._to_numpy_bgr(img)

    @torch.no_grad()
    def render_direct(self, A):
        img = self._upscale(A.clamp(0.0, 1.0))
        return self._to_numpy_bgr(img)


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


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["click"] = (x, y)


def main():
    cfg = Cfg()
    seed = 42

    print(f"Flow Lenia — CoreML Optimized")
    print(f"  Device: {DEVICE}")
    print(f"  Precision: {'mixed (fp16 growth/data + fp32 coords)' if USE_FP16 else 'float32'}")
    print(f"  CoreML: {'available' if HAS_COREML else 'not installed (pip install coremltools)'}")

    fl = FlowLeniaCoreML(cfg, seed=seed)
    print(f"  torch.compile: {'enabled' if fl._compiled else 'disabled'}")
    fK = fl.get_fK()
    A = fl.init_pattern("dense_noise", seed=seed)

    renderer = LiquidRendererCoreML(cfg.X, cfg.Y, DISPLAY_SIZE, DEVICE)

    if HAS_COREML:
        print("  Building CoreML render model ...")
        coreml_model = build_coreml_render_model(DISPLAY_SIZE, DISPLAY_SIZE)
        if coreml_model:
            renderer.enable_coreml(coreml_model)
            print("  CoreML ANE rendering: enabled")

    evolver = ParamEvolver(cfg)

    CMAP_NAMES = ["mono", "mono_warm", "liquid", "direct"]

    print(f"\n  Sim: {cfg.X}x{cfg.Y} C={cfg.C} k={cfg.k}")
    print(f"  Display: {DISPLAY_SIZE}x{DISPLAY_SIZE}")
    print(f"  Auto-evolution: every {cfg.evolve_interval:.0f}s (transition {cfg.evolve_duration:.0f}s)")
    print()
    print("  SPACE=pause  r=reset  q/ESC=quit  s=screenshot  f=fullscreen")
    print("  c=colormap  +/-=steps  m=modulation  g=glow  t=thin-film  d=distort")
    print("  e=auto-evolution  b=benchmark  1-5=pattern  mouse=perturbation")

    print("\nBenchmark (sim only) ...")
    bench_fps, bench_ms, A = run_benchmark(fl, fK, A, n_steps=50)
    print(f"  {bench_fps:.1f} FPS ({bench_ms:.1f} ms/step)")

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
    fK_refresh_counter = 0

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

            # Parameter evolution
            if not paused and evolver.evolving:
                changed = evolver.update(fl)
                if changed:
                    fK_refresh_counter += 1
                    # Refresh kernel every 10 frames during transition (not every frame)
                    if fK_refresh_counter >= 10:
                        fK = fl.get_fK()
                        fK_refresh_counter = 0

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
            evo_str = "EVO" if evolver.evolving else "---"
            trans_str = "*" if evolver.transitioning else ""
            fx = ""
            if colormap_mode <= 2:
                fx += "G" if renderer.glow_on else ""
                fx += "D" if renderer.distort_on else ""
                fx += "T" if renderer.thin_film_on else ""
            coreml_str = " [ANE]" if renderer._coreml_active else ""
            prec_str = "fp16+fp32" if USE_FP16 else "fp32"
            info = (f"FPS:{fps:.0f} x{steps_per_frame} | {status} | "
                    f"{CMAP_NAMES[colormap_mode]} {mod_str} {evo_str}{trans_str} {fx} | "
                    f"{prec_str}{coreml_str}")
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
                evolver = ParamEvolver(cfg)
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
            elif key == ord('e'):
                evolver.evolving = not evolver.evolving
                print(f"Auto-evolution: {'ON' if evolver.evolving else 'OFF'}")
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
