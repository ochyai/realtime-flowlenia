# Real-time Flow Lenia

GPU-accelerated real-time [Flow Lenia](https://arxiv.org/abs/2212.07906) simulation with liquid shader effects.

![Demo](demo.gif)
<!-- Replace demo.gif with an actual recording -->

## Features

- **GPU acceleration** -- PyTorch MPS (Apple Silicon) / CUDA / CPU fallback
- **Liquid shader rendering** -- fluid distortion, thin-film interference, glow bloom
- **Time-varying parameter modulation** -- multi-harmonic breathing keeps patterns alive
- **Multiple render modes** -- monochrome, warm amber, full liquid shader, direct RGB
- **Interactive controls** -- pause, reset, pattern switching, mouse perturbation
- **Real-time performance** -- 512x512 at 30+ FPS on MPS, higher on CUDA

## Requirements

- Python 3.9+
- PyTorch 2.0+
- OpenCV
- NumPy

## Installation

```bash
git clone https://github.com/ochyai/realtime-flowlenia.git
cd realtime-flowlenia
pip install -r requirements.txt
```

## Usage

**GPU version (recommended):**

```bash
python realtime_flowlenia_gpu.py
```

**CPU fallback version** (uses the engine with reintegration tracking):

```bash
python realtime_flowlenia.py
```

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `r` | Reset with new random parameters |
| `q` / `ESC` | Quit |
| `s` | Save screenshot (PNG) |
| `f` | Toggle fullscreen |
| `c` | Cycle colormap / render mode |
| `+` / `-` | Increase / decrease sim steps per frame |
| `m` | Toggle time-varying modulation |
| `g` | Toggle glow effect |
| `t` | Toggle thin-film interference |
| `d` | Toggle fluid distortion |
| `1`-`5` | Switch initial pattern |
| Mouse click | Add perturbation |

## Architecture

- `realtime_flowlenia_gpu.py` -- Self-contained GPU implementation with `grid_sample` advection and liquid shader effects (main entry point)
- `realtime_flowlenia_engine.py` -- Engine with proper reintegration tracking transport
- `realtime_flowlenia.py` -- CPU-friendly viewer using the engine

## Citation

This implementation is based on Flow Lenia:

```bibtex
@article{plantec2023flowlenia,
  title={Flow Lenia: Mass conservation for the study of virtual creatures in continuous cellular automata},
  author={Plantec, Erwan and Hamon, Gautier and Etcheverry, Mayalen and Oudeyer, Pierre-Yves and Moulin-Frier, Cl{\'e}ment and Chan, Bert Wang-Chak},
  journal={arXiv preprint arXiv:2212.07906},
  year={2023}
}
```

## License

MIT
