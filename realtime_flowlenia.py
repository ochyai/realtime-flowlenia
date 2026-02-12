#!/usr/bin/env python3
"""Real-time Flow Lenia - Interactive Viewer (GPU-optimized)

Controls:
  SPACE  = pause / resume
  r      = reset with new random parameters and pattern
  q/ESC  = quit
  s      = save screenshot (PNG)
  1-5    = switch initial pattern
  f      = toggle fullscreen
  c      = cycle colormap
  +/-    = more/fewer sim steps per frame
  mouse  = click to add perturbation
"""

import sys
import numpy as np
import cv2
import torch
import time
from realtime_flowlenia_engine import RealtimeFlowLenia, FlowLeniaConfig, FlowLeniaState

# Pattern names mapped to keys 1-5
PATTERNS = ["dense_noise", "random_center", "random_spots", "circle", "noise"]
PATTERN_KEYS = {ord('1'): 0, ord('2'): 1, ord('3'): 2, ord('4'): 3, ord('5'): 4}

# Colormap modes
COLORMAP_NAMES = ["boosted_rgb", "direct_rgb", "hot", "viridis"]

WINDOW_NAME = "Flow Lenia"

# Display resolution (sim runs at full res, display downscaled for speed)
DISPLAY_SIZE = 1024


# ---------------------------------------------------------------------------
# GPU-side colour processing (avoids slow CPU numpy ops)
# ---------------------------------------------------------------------------

@torch.no_grad()
def gpu_boost_rgb(A: torch.Tensor) -> torch.Tensor:
    """Organic contrast + saturation boost on GPU.
    Emphasises dark background with vivid coloured organisms.
    A: (X, Y, 3) float32 in [0,1]  →  (X, Y, 3) float32 in [0,1]
    """
    # Mild S-curve contrast (keeps darks dark, boosts mids)
    out = A.clamp(0.0, 1.0)
    # Smooth S-curve: 3x^2 - 2x^3  (Hermite interpolation)
    out = out * out * (3.0 - 2.0 * out)

    # Per-channel normalisation (gentle — don't blow out)
    for c in range(3):
        ch = out[:, :, c]
        hi = ch.quantile(0.995)
        if hi > 0.05:
            out[:, :, c] = (ch / hi).clamp(0.0, 1.0)

    # Saturation boost: push away from luminance
    lum = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
    lum = lum.unsqueeze(-1)
    out = lum + (out - lum) * 2.2  # strong saturation for vivid RGB separation
    return out.clamp(0.0, 1.0)


def to_bgr_u8(frame_gpu: torch.Tensor, mode: str) -> np.ndarray:
    """Convert GPU tensor to BGR uint8 for OpenCV display."""
    if mode == "boosted_rgb":
        frame_gpu = gpu_boost_rgb(frame_gpu)
        # RGB -> BGR, scale to 255, transfer to CPU
        bgr = (frame_gpu[:, :, [2, 1, 0]] * 255).to(torch.uint8).cpu().numpy()
        return bgr

    if mode == "direct_rgb":
        bgr = (frame_gpu[:, :, [2, 1, 0]] * 255).to(torch.uint8).cpu().numpy()
        return bgr

    # Grayscale colormaps (hot, viridis) — compute luminance on GPU
    gray = (0.299 * frame_gpu[:, :, 0]
            + 0.587 * frame_gpu[:, :, 1]
            + 0.114 * frame_gpu[:, :, 2])
    gray_u8 = (gray.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

    if mode == "hot":
        return cv2.applyColorMap(gray_u8, cv2.COLORMAP_HOT)
    elif mode == "viridis":
        return cv2.applyColorMap(gray_u8, cv2.COLORMAP_VIRIDIS)
    else:
        bgr = (frame_gpu[:, :, [2, 1, 0]] * 255).to(torch.uint8).cpu().numpy()
        return bgr


def add_perturbation(state, x, y, device):
    """Add a random blob perturbation at pixel (x, y)."""
    H, W, C = state.A.shape
    radius = np.random.randint(10, 30)
    yy, xx = np.mgrid[max(0, y - radius):min(H, y + radius),
                       max(0, x - radius):min(W, x + radius)]
    if yy.size == 0 or xx.size == 0:
        return state
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2).astype(np.float32)
    mask = dist < radius
    blob = np.random.rand(yy.shape[0], yy.shape[1], C).astype(np.float32)
    blob *= mask[:, :, None]
    blob_t = torch.from_numpy(blob).to(device)
    A = state.A.clone()
    y_start = max(0, y - radius)
    x_start = max(0, x - radius)
    A[y_start:y_start + blob_t.shape[0],
      x_start:x_start + blob_t.shape[1]] += blob_t
    A = A.clamp(0.0, 1.0)
    return FlowLeniaState(A=A, fK=state.fK)


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["click"] = (x, y)


def main():
    cfg = FlowLeniaConfig(X=1024, Y=1024, C=3, k=9, dt=0.2, dd=3, border="torus")
    seed = 42
    fl = RealtimeFlowLenia(cfg, seed=seed)
    state = fl.initialize(seed)
    state = fl.init_pattern(state, "dense_noise", seed=seed)

    device = fl.device
    print(f"Flow Lenia Real-time Viewer (GPU-optimized)")
    print(f"  Device:     {device}")
    print(f"  Resolution: {cfg.X}x{cfg.Y}")
    print(f"  Channels:   {cfg.C}, Kernels: {cfg.k}")
    print(f"  dt={cfg.dt}, dd={cfg.dd}, sigma={cfg.sigma}")
    print()
    print("Controls:")
    print("  SPACE = pause/resume   r = reset   q/ESC = quit")
    print("  s = screenshot   1-5 = pattern   f = fullscreen   c = colormap")
    print("  +/- = sim steps per frame   Mouse click = perturbation")

    # Warm-up
    print("\nWarm-up (5 steps) ...")
    for _ in range(5):
        state = fl.step(state)
    if device.type == "mps":
        torch.mps.synchronize()
    print("Ready.\n")

    # State variables
    paused = False
    fullscreen = False
    colormap_idx = 0
    current_pattern = "dense_noise"
    screenshot_counter = 0
    click_param = {"click": None}
    steps_per_frame = 1  # reintegration tracking is heavier, 1 step per frame

    # Create window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE, DISPLAY_SIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, click_param)

    # FPS tracking
    frame_times = []
    fps = 0.0
    step_count = 0

    try:
        while True:
            t_start = time.perf_counter()

            # Handle mouse click
            if click_param["click"] is not None:
                cx, cy = click_param["click"]
                win_rect = cv2.getWindowImageRect(WINDOW_NAME)
                if win_rect[2] > 0 and win_rect[3] > 0:
                    sx = int(cx * cfg.X / win_rect[2]) if win_rect[2] != cfg.X else cx
                    sy = int(cy * cfg.Y / win_rect[3]) if win_rect[3] != cfg.Y else cy
                else:
                    sx, sy = cx, cy
                sx = max(0, min(cfg.X - 1, sx))
                sy = max(0, min(cfg.Y - 1, sy))
                state = add_perturbation(state, sx, sy, device)
                click_param["click"] = None

            # Simulation: multiple steps per display frame
            if not paused:
                for _ in range(steps_per_frame):
                    state = fl.step(state)
                step_count += steps_per_frame

            # Render — colour processing on GPU, then transfer
            display = to_bgr_u8(state.A, COLORMAP_NAMES[colormap_idx])

            # FPS overlay
            t_end = time.perf_counter()
            dt_frame = t_end - t_start
            frame_times.append(dt_frame)
            if len(frame_times) > 30:
                frame_times.pop(0)
            if frame_times:
                fps = 1.0 / (sum(frame_times) / len(frame_times))

            sim_fps = steps_per_frame * fps
            status = "PAUSED" if paused else f"step {step_count}"
            info = f"FPS:{fps:.0f} simFPS:{sim_fps:.0f} x{steps_per_frame} | {status} | {COLORMAP_NAMES[colormap_idx]}"
            cv2.putText(display, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display)

            # Key handling
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break

            elif key == ord(' '):
                paused = not paused
                print("Paused." if paused else "Resumed.")

            elif key == ord('r'):
                seed = int(time.time()) % 100000
                fl = RealtimeFlowLenia(cfg, seed=seed)
                state = fl.initialize(seed)
                state = fl.init_pattern(state, current_pattern, seed=seed)
                for _ in range(5):
                    state = fl.step(state)
                step_count = 0
                print(f"Reset seed={seed}, pattern={current_pattern}")

            elif key == ord('s'):
                fname = f"flowlenia_screenshot_{screenshot_counter:04d}.png"
                cv2.imwrite(fname, display)
                screenshot_counter += 1
                print(f"Saved: {fname}")

            elif key == ord('f'):
                fullscreen = not fullscreen
                prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)

            elif key == ord('c'):
                colormap_idx = (colormap_idx + 1) % len(COLORMAP_NAMES)
                print(f"Colormap: {COLORMAP_NAMES[colormap_idx]}")

            elif key == ord('+') or key == ord('='):
                steps_per_frame = min(steps_per_frame + 1, 10)
                print(f"Steps/frame: {steps_per_frame}")

            elif key == ord('-') or key == ord('_'):
                steps_per_frame = max(steps_per_frame - 1, 1)
                print(f"Steps/frame: {steps_per_frame}")

            elif key in PATTERN_KEYS:
                idx = PATTERN_KEYS[key]
                current_pattern = PATTERNS[idx]
                state = fl.init_pattern(state, current_pattern, seed=int(time.time()) % 100000)
                step_count = 0
                print(f"Pattern: {current_pattern}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
