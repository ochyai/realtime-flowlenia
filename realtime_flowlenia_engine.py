#!/usr/bin/env python3
"""
Real-time Flow Lenia Engine -- proper Reintegration Tracking (PyTorch MPS).

Matches the original JAX Flow Lenia algorithm:
  - FFT-based kernel convolution
  - Bell-curve growth function
  - Sobel-based flow field (growth gradient + concentration gradient)
  - Reintegration Tracking transport (neighborhood scan with area overlap)

Resolution: 1024x1024, 3 channels (RGB), 9 kernels
"""

import time
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Config / State
# ---------------------------------------------------------------------------

@dataclass
class FlowLeniaConfig:
    X: int = 1024
    Y: int = 1024
    C: int = 3
    k: int = 9
    dt: float = 0.2
    dd: int = 5
    sigma: float = 0.65
    border: str = "torus"
    growth_inject: float = 0.0  # optional hybrid Lenia+Flow


@dataclass
class FlowLeniaState:
    A: torch.Tensor   # (X, Y, C)
    fK: torch.Tensor  # (X, Y, k) complex64


# ---------------------------------------------------------------------------
# Utility: conn_from_matrix
# ---------------------------------------------------------------------------

def conn_from_matrix(mat: np.ndarray) -> Tuple[List[int], List[List[int]]]:
    C = mat.shape[0]
    c0: List[int] = []
    c1: List[List[int]] = [[] for _ in range(C)]
    i = 0
    for s in range(C):
        for t in range(C):
            n = int(mat[s, t])
            if n:
                c0 = c0 + [s] * n
                c1[t] = c1[t] + list(range(i, i + n))
            i += n
    return c0, c1


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class RealtimeFlowLenia:
    """Flow Lenia with proper Reintegration Tracking for MPS."""

    def __init__(self, cfg: FlowLeniaConfig, seed: int = 0, preset: str = "default"):
        self.cfg = cfg
        self.device = DEVICE

        torch.manual_seed(seed)

        k = cfg.k
        C = cfg.C

        # --- Parameters (matching original JAX ranges) ---
        self.R = torch.empty((), device=self.device).uniform_(2.0, 25.0)
        self.r = torch.empty(k, device=self.device).uniform_(0.2, 1.0)
        self.m = torch.empty(k, device=self.device).uniform_(0.05, 0.50)
        self.s = torch.empty(k, device=self.device).uniform_(0.001, 0.18)
        self.h = torch.empty(k, device=self.device).uniform_(0.01, 1.0)
        self.a = torch.empty(k, 3, device=self.device).uniform_(0.0, 1.0)
        self.b = torch.empty(k, 3, device=self.device).uniform_(0.001, 1.0)
        self.w = torch.empty(k, 3, device=self.device).uniform_(0.01, 0.50)

        # --- Connection matrix (standard 3-channel cycling) ---
        M = np.array([[2, 1, 0],
                       [0, 2, 1],
                       [1, 0, 2]])
        c0_list, c1_list = conn_from_matrix(M)

        # --- Pre-cache constant tensors ---
        self._c0 = torch.tensor(c0_list, dtype=torch.long, device=self.device)

        c1_mat = torch.zeros(k, C, device=self.device)
        for c in range(C):
            for ki in c1_list[c]:
                c1_mat[ki, c] = 1.0
        self._c1_mat = c1_mat  # (k, C)

        # Pre-reshape growth params: (1, 1, k)
        self._m = self.m.view(1, 1, k)
        self._s = self.s.view(1, 1, k)
        self._h = self.h.view(1, 1, k)

        # --- Sobel kernels ---
        # kx detects horizontal gradient (d/dj), ky detects vertical gradient (d/di)
        kx = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=self.device)
        ky = kx.t().contiguous()

        # Weights for C channels: first C outputs = d/di (ky), next C = d/dj (kx)
        w_c = torch.zeros(2 * C, 1, 3, 3, device=self.device)
        for c in range(C):
            w_c[c, 0] = ky       # d/di (vertical gradient) → dim2=0
            w_c[C + c, 0] = kx   # d/dj (horizontal gradient) → dim2=1
        self._sobel_weight_C = w_c

        # Weights for 1 channel (density gradient)
        w_1 = torch.zeros(2, 1, 3, 3, device=self.device)
        w_1[0, 0] = ky   # d/di
        w_1[1, 0] = kx   # d/dj
        self._sobel_weight_1 = w_1

        # --- Position grid for reintegration tracking ---
        X, Y = cfg.X, cfg.Y
        ix = torch.arange(X, device=self.device, dtype=torch.float32)
        iy = torch.arange(Y, device=self.device, dtype=torch.float32)
        gi, gj = torch.meshgrid(ix, iy, indexing="ij")
        self._pos = torch.stack([gi, gj], dim=-1) + 0.5  # (X, Y, 2)

        # Torus wrap helper
        self._grid_size = torch.tensor([float(X), float(Y)],
                                        device=self.device).view(1, 1, 2, 1)

    # -----------------------------------------------------------------
    # Sobel (vectorized grouped conv2d)
    # -----------------------------------------------------------------

    def _sobel_C(self, x: torch.Tensor) -> torch.Tensor:
        """Sobel gradient for C-channel field. x: (X,Y,C) -> (X,Y,2,C)
        dim2: 0=d/di (row), 1=d/dj (col)"""
        C = x.shape[2]
        inp = x.permute(2, 0, 1).unsqueeze(0)  # (1, C, X, Y)
        if self.cfg.border == "torus":
            inp = F.pad(inp, (1, 1, 1, 1), mode="circular")
        else:
            inp = F.pad(inp, (1, 1, 1, 1), mode="replicate")
        inp2 = inp.repeat(1, 2, 1, 1)
        out = F.conv2d(inp2, self._sobel_weight_C, groups=2 * C).squeeze(0)
        di = out[:C].permute(1, 2, 0)   # d/di (vertical)
        dj = out[C:].permute(1, 2, 0)   # d/dj (horizontal)
        return torch.stack([di, dj], dim=2)  # (X, Y, 2, C)

    def _sobel_1(self, x: torch.Tensor) -> torch.Tensor:
        """Sobel gradient for 1-channel field. x: (X,Y,1) -> (X,Y,2,1)"""
        inp = x.permute(2, 0, 1).unsqueeze(0)
        if self.cfg.border == "torus":
            inp = F.pad(inp, (1, 1, 1, 1), mode="circular")
        else:
            inp = F.pad(inp, (1, 1, 1, 1), mode="replicate")
        inp2 = inp.repeat(1, 2, 1, 1)
        out = F.conv2d(inp2, self._sobel_weight_1, groups=2).squeeze(0)
        return out.permute(1, 2, 0).unsqueeze(-1)  # (X, Y, 2, 1)

    # -----------------------------------------------------------------
    # Kernel FFT
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _get_kernels_fft(self) -> torch.Tensor:
        X, Y, k = self.cfg.X, self.cfg.Y, self.cfg.k
        mid = X // 2
        coords = np.mgrid[-mid:mid, -mid:mid].astype(np.float32)
        dist = torch.from_numpy(np.linalg.norm(coords, axis=0)).to(self.device)

        K = torch.zeros(X, Y, k, device=self.device)
        for ki in range(k):
            D = dist / ((self.R.item() + 15.0) * self.r[ki].item())
            # sigmoid(-(D-1)*10) = 0.5*(tanh(-(D-1)*5)+1)
            sig = 0.5 * (torch.tanh(-(D - 1.0) * 5.0) + 1.0)
            D3 = D.unsqueeze(-1)
            ker = (self.b[ki] * torch.exp(-((D3 - self.a[ki]) ** 2) / self.w[ki])).sum(-1)
            K[:, :, ki] = sig * ker

        K = K / K.sum(dim=(0, 1), keepdim=True).clamp(min=1e-10)
        K = torch.fft.fftshift(K, dim=(0, 1))
        return torch.fft.fft2(K, dim=(0, 1))

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    @torch.no_grad()
    def initialize(self, seed: int = 0) -> FlowLeniaState:
        torch.manual_seed(seed)
        A = torch.zeros(self.cfg.X, self.cfg.Y, self.cfg.C, device=self.device)
        fK = self._get_kernels_fft()
        return FlowLeniaState(A=A, fK=fK)

    @torch.no_grad()
    def init_pattern(self, state: FlowLeniaState, pattern_type: str = "random_center",
                     seed: int = 42) -> FlowLeniaState:
        torch.manual_seed(seed)
        X, Y, C = self.cfg.X, self.cfg.Y, self.cfg.C

        if pattern_type == "dense_noise":
            # Uniform noise (matches original: noise * 0.4)
            A = torch.rand(X, Y, C, device=self.device) * 0.4
        elif pattern_type == "random_center":
            A = torch.zeros(X, Y, C, device=self.device)
            sz = 40
            cx, cy = X // 2, Y // 2
            A[cx - sz // 2: cx + sz // 2, cy - sz // 2: cy + sz // 2] = \
                torch.rand(sz, sz, C, device=self.device)
        elif pattern_type == "random_spots":
            A = torch.zeros(X, Y, C, device=self.device)
            for _ in range(5):
                cx = torch.randint(50, X - 50, (1,)).item()
                cy = torch.randint(50, Y - 50, (1,)).item()
                r = torch.randint(10, 30, (1,)).item()
                yy, xx = torch.meshgrid(
                    torch.arange(Y, device=self.device),
                    torch.arange(X, device=self.device), indexing="ij")
                mask = ((xx - cx) ** 2 + (yy - cy) ** 2).float() < r ** 2
                for c in range(C):
                    A[:, :, c] = torch.where(mask, torch.rand(1, device=self.device), A[:, :, c])
        elif pattern_type == "circle":
            A = torch.zeros(X, Y, C, device=self.device)
            cx, cy = X // 2, Y // 2
            radius = min(X, Y) // 8
            yy, xx = torch.meshgrid(
                torch.arange(Y, device=self.device),
                torch.arange(X, device=self.device), indexing="ij")
            dist = ((xx - cx) ** 2 + (yy - cy) ** 2).float().sqrt()
            ring = torch.exp(-((dist - radius) ** 2) / (radius * 0.3) ** 2)
            for c in range(C):
                A[:, :, c] = ring * (0.5 + 0.5 * torch.rand(1, device=self.device))
        elif pattern_type == "noise":
            A = torch.rand(X, Y, C, device=self.device) * 0.3
        else:
            raise ValueError(f"Unknown pattern_type: {pattern_type}")

        return FlowLeniaState(A=A, fK=state.fK)

    # -----------------------------------------------------------------
    # Simulation step — Reintegration Tracking
    # -----------------------------------------------------------------

    @torch.no_grad()
    def step(self, state: FlowLeniaState) -> FlowLeniaState:
        A = state.A    # (X, Y, C)
        fK = state.fK  # (X, Y, k) complex
        cfg = self.cfg

        # ---- 1. FFT convolution ----
        fA = torch.fft.fft2(A, dim=(0, 1))
        fAk = fA[:, :, self._c0]  # (X, Y, k)
        U = torch.fft.ifft2(fK * fAk, dim=(0, 1)).real

        # ---- 2. Growth function ----
        G = (torch.exp(-((U - self._m) ** 2) / (2.0 * self._s ** 2)) * 2.0 - 1.0) * self._h
        G_channel = torch.matmul(G, self._c1_mat)  # (X, Y, C)

        # ---- 3. Optional hybrid: direct growth injection ----
        if cfg.growth_inject > 0.0:
            A = (A + cfg.dt * cfg.growth_inject * G_channel).clamp(0.0, 1.0)

        # ---- 4. Flow field ----
        nabla_G = self._sobel_C(G_channel)              # (X, Y, 2, C)
        A_sum = A.sum(dim=-1, keepdim=True)              # (X, Y, 1)
        nabla_A = self._sobel_1(A_sum)                   # (X, Y, 2, 1)

        alpha = (A[:, :, None, :] / cfg.C).square().clamp(0.0, 1.0)
        F_flow = nabla_G * (1.0 - alpha) - nabla_A * alpha  # (X, Y, 2, C)

        # ---- 5. Compute target positions (mu) ----
        dd = cfg.dd
        sigma = cfg.sigma
        ma = dd - sigma

        mu = self._pos[..., None] + (cfg.dt * F_flow).clamp(-ma, ma)  # (X, Y, 2, C)

        if cfg.border == "wall":
            mu[:, :, 0, :] = mu[:, :, 0, :].clamp(sigma, cfg.X - sigma)
            mu[:, :, 1, :] = mu[:, :, 1, :].clamp(sigma, cfg.Y - sigma)

        # ---- 6. Reintegration Tracking (optimized with padding+slicing) ----
        X, Y = cfg.X, cfg.Y
        two_sigma = min(1.0, 2.0 * sigma)
        inv_four_sigma_sq = 1.0 / (4.0 * sigma ** 2)

        # Split pos into row/col for (X,Y,C)-shaped ops (avoids (X,Y,2,C) tensors)
        pos_i = self._pos[:, :, 0:1]  # (X, Y, 1)
        pos_j = self._pos[:, :, 1:2]  # (X, Y, 1)

        # Circular pad A and mu components to avoid roll copies
        # A: (X,Y,C) → (C,X,Y) → pad → slice as views
        A_t = A.permute(2, 0, 1).unsqueeze(0)           # (1, C, X, Y)
        A_pad = F.pad(A_t, (dd, dd, dd, dd), mode="circular").squeeze(0)  # (C, X+2dd, Y+2dd)
        A_pad = A_pad.permute(1, 2, 0)  # (X+2dd, Y+2dd, C)

        # mu components: split into i,j to work with (X,Y,C) tensors
        mu_i = mu[:, :, 0, :]  # (X, Y, C)
        mu_j = mu[:, :, 1, :]  # (X, Y, C)

        mu_i_t = mu_i.permute(2, 0, 1).unsqueeze(0)
        mu_i_pad = F.pad(mu_i_t, (dd, dd, dd, dd), mode="circular").squeeze(0).permute(1, 2, 0)

        mu_j_t = mu_j.permute(2, 0, 1).unsqueeze(0)
        mu_j_pad = F.pad(mu_j_t, (dd, dd, dd, dd), mode="circular").squeeze(0).permute(1, 2, 0)

        nA = torch.zeros_like(A)
        fX, fY = float(X), float(Y)

        for dx in range(-dd, dd + 1):
            ri = dd + dx
            for dy in range(-dd, dd + 1):
                rj = dd + dy
                # Slice from padded tensors (view — zero-copy)
                Ar = A_pad[ri:ri + X, rj:rj + Y]          # (X, Y, C)
                mur_i = mu_i_pad[ri:ri + X, rj:rj + Y]    # (X, Y, C)
                mur_j = mu_j_pad[ri:ri + X, rj:rj + Y]    # (X, Y, C)

                # Distance per axis
                d_i = (pos_i - mur_i).abs()  # (X, Y, C)
                d_j = (pos_j - mur_j).abs()  # (X, Y, C)

                if cfg.border == "torus":
                    d_i = torch.minimum(d_i, fX - d_i)
                    d_j = torch.minimum(d_j, fY - d_j)

                # Area overlap = product of 1D overlaps
                sz_i = (0.5 - d_i + sigma).clamp(0.0, two_sigma)
                sz_j = (0.5 - d_j + sigma).clamp(0.0, two_sigma)
                area = sz_i * sz_j * inv_four_sigma_sq  # (X, Y, C)

                nA = nA + Ar * area

        return FlowLeniaState(A=nA, fK=fK)

    # -----------------------------------------------------------------
    # Multi-step convenience
    # -----------------------------------------------------------------

    @torch.no_grad()
    def rollout(self, state: FlowLeniaState, steps: int = 100) -> FlowLeniaState:
        for _ in range(steps):
            state = self.step(state)
        return state


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    cfg = FlowLeniaConfig(X=1024, Y=1024, C=3, k=9, dt=0.2, border="torus")
    engine = RealtimeFlowLenia(cfg, seed=42)

    print("Initializing state ...")
    state = engine.initialize(seed=0)
    state = engine.init_pattern(state, "dense_noise", seed=123)

    print("Warm-up (5 steps) ...")
    for _ in range(5):
        state = engine.step(state)
    if DEVICE.type == "mps":
        torch.mps.synchronize()

    n_steps = 30
    print(f"Benchmarking {n_steps} steps at {cfg.X}x{cfg.Y} ...")
    t0 = time.perf_counter()
    for _ in range(n_steps):
        state = engine.step(state)
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    fps = n_steps / elapsed
    ms_per_step = elapsed / n_steps * 1000
    print(f"  {n_steps} steps in {elapsed:.3f}s")
    print(f"  {fps:.1f} FPS  ({ms_per_step:.1f} ms/step)")
    print(f"  State A range: [{state.A.min().item():.4f}, {state.A.max().item():.4f}]")
    print(f"  State A mean:  {state.A.mean().item():.6f}")
    print(f"  State A std:   {state.A.std().item():.6f}")
