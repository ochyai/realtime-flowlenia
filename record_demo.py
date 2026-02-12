#!/usr/bin/env python3
"""Record a demo GIF of Flow Lenia for the README.

Usage:
  python record_demo.py                    # defaults: 150 frames, 480px, 15fps
  python record_demo.py --frames 200 --size 384 --fps 12
  python record_demo.py --output demo.webp  # WebP output (smaller)
"""

import argparse
import os
import sys
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np
import torch

from realtime_flowlenia_gpu import FlowLeniaGPU, LiquidRenderer, Cfg, DEVICE


def record(n_frames=150, size=480, fps=15, steps_per_frame=3, output="demo.gif"):
    cfg = Cfg()
    fl = FlowLeniaGPU(cfg, seed=42)
    fK = fl.get_fK()
    A = fl.init_pattern("dense_noise", seed=42)
    renderer = LiquidRenderer(cfg.X, cfg.Y, DEVICE)

    print(f"Device: {DEVICE}")
    print(f"Recording {n_frames} frames at {size}x{size}, {fps} FPS")
    print(f"Steps per frame: {steps_per_frame}")

    # Warm-up
    print("Warm-up ...")
    for _ in range(30):
        A = fl.step(A, fK)
    if DEVICE.type == "mps":
        torch.mps.synchronize()

    tmpdir = tempfile.mkdtemp(prefix="flowlenia_")
    frame_paths = []
    t0 = time.perf_counter()

    for i in range(n_frames):
        for _ in range(steps_per_frame):
            A = fl.step(A, fK)

        progress = i / n_frames

        # Phase transitions for visual variety
        if progress < 0.30:
            frame = renderer.render_mono_warm(A, fl.t)
        elif progress < 0.35:
            # Transition: enable liquid effects
            renderer.thin_film_on = True
            renderer.distort_on = True
            frame = renderer.render_liquid(A, fl.t)
        elif progress < 0.65:
            frame = renderer.render_liquid(A, fl.t)
        elif progress < 0.70:
            renderer.glow_on = True
            frame = renderer.render_liquid(A, fl.t)
        else:
            frame = renderer.render_liquid(A, fl.t)

        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        path = os.path.join(tmpdir, f"frame_{i:04d}.png")
        cv2.imwrite(path, frame)
        frame_paths.append(path)

        if (i + 1) % 30 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i+1}/{n_frames} frames ({elapsed:.1f}s)")

    elapsed = time.perf_counter() - t0
    print(f"Captured {n_frames} frames in {elapsed:.1f}s")

    # Build GIF/WebP
    ext = os.path.splitext(output)[1].lower()
    has_ffmpeg = shutil.which("ffmpeg") is not None

    if has_ffmpeg:
        print(f"Creating {output} with ffmpeg (palette-optimized) ...")
        pattern = os.path.join(tmpdir, "frame_%04d.png")
        if ext == ".webp":
            cmd = [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", pattern,
                "-vf", f"scale={size}:{size}:flags=lanczos",
                "-loop", "0", "-quality", "80",
                output,
            ]
        else:
            palette = os.path.join(tmpdir, "palette.png")
            cmd_palette = [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", pattern,
                "-vf", f"fps={fps},scale={size}:-1:flags=lanczos,palettegen=max_colors=128",
                palette,
            ]
            subprocess.run(cmd_palette, capture_output=True)
            cmd = [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", pattern,
                "-i", palette,
                "-lavfi", f"fps={fps},scale={size}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                "-loop", "0",
                output,
            ]
        subprocess.run(cmd, capture_output=True)
    else:
        print(f"Creating {output} with Pillow ...")
        from PIL import Image
        frames = []
        for p in frame_paths:
            img = cv2.imread(p)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img_rgb))
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)

    fsize = os.path.getsize(output) / (1024 * 1024)
    print(f"Saved: {output} ({fsize:.1f} MB, {n_frames} frames, {n_frames/fps:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Record Flow Lenia demo GIF")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=str, default="demo.gif")
    args = parser.parse_args()
    record(args.frames, args.size, args.fps, args.steps, args.output)


if __name__ == "__main__":
    main()
