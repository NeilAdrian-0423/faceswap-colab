# 🎭 Face Swap for Colab (photos & videos)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neiladrian-0423/faceswap-colab/blob/main/faceswap_colab.ipynb)

One-image face swap with a **drag-and-drop UI**. Upload a **source face** and a **target photo or video**, click **Swap**, and download the result — no manual file paths, no editing cells.

![UI](https://img.shields.io/badge/UI-Gradio-orange) ![GPU](https://img.shields.io/badge/runtime-T4%20GPU-blue) ![python](https://img.shields.io/badge/python-3.13-green)

## Quick start

1. Click **Open in Colab** above.
2. `Runtime ▸ Change runtime type ▸ T4 GPU`.
3. Run **cell 1** (setup — once, ~2–3 min).
4. Run **cell 2** — a UI appears inline (and prints a public share link). Upload → tick options → **Swap** → **Download**.

## Features

- **Photos *and* videos** — auto-detects the target type; videos keep their original audio.
- **Drag-and-drop UI** (Gradio) — no path routing.
- **Enhance** toggle — optional GFPGAN pass for sharper faces.
- **Swap all faces** toggle — swap every face in frame, or just the largest.
- **GPU-accelerated** via `onnxruntime-gpu`.

## How it works

| Stage | Model |
|-------|-------|
| Face detection + landmarks + embedding | insightface `buffalo_l` |
| Face swap | `inswapper_128.onnx` |
| Optional enhancement | `GFPGANv1.4.onnx` |
| Audio re-mux | `ffmpeg` |

## Why this exists

The popular `dream80/roop_colab` notebook is **broken on current Colab**: it pins Python-3.10-era `numpy==1.24.3` and `torch==2.0.1+cu118`, which don't build/exist on Colab's **Python 3.13** runtime.

This notebook fixes that:

- Uses the **pure-python `insightface` 2.1 wheel** (installs on 3.13 with no compiler).
- Includes the **Tesla-T4 + cuDNN-9 workaround**. On T4, `onnxruntime-gpu` throws
  `CUDNN_FE ... HEURISTIC_QUERY_FAILED` on conv layers and silently falls back to (slow) CPU.
  Passing the CUDA provider option `cudnn_conv_algo_search='DEFAULT'` restores real GPU inference:

  ```python
  providers = [('CUDAExecutionProvider',
                {'cudnn_conv_algo_search': 'DEFAULT', 'cudnn_conv_use_max_workspace': '1'}),
               'CPUExecutionProvider']
  ```

## ⚠️ Responsible use

For consensual, creative use only. Swap faces that are **your own** or that you have **permission** to use. Do not create misleading, defamatory, or non-consensual content. You are responsible for what you make.

## Credits

Built on [insightface](https://github.com/deepinsight/insightface); model weights mirrored by the [ReActor](https://huggingface.co/datasets/Gourieff/ReActor) project. Face-swap concept popularized by [roop](https://github.com/s0md3v/roop).
