<h1 align="center">q1-FaceSwap</h1>

<p align="center">
  <strong>High-performance face swapping pipeline with swarm engine scheduling and TensorRT acceleration.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey" alt="Platform Windows | Linux"/>
  <img src="https://img.shields.io/badge/license-Proprietary-red" alt="License"/>
</p>

---

## Overview

q1-FaceSwap is a production-grade face swapping system that combines ONNX Runtime inference with an intelligent **swarm engine** to maximize GPU utilization. It automatically scales worker threads across pipeline stages (detect → swap → restore → parse) based on real-time queue pressure, keeping your GPU saturated at all times.

Built for **NVIDIA CUDA GPUs** with automatic fallback to CPU for users without dedicated hardware.

### Key Features

- **Swarm Engine** — Adaptive worker pool that dynamically shifts compute resources to the hottest pipeline stage, eliminating bottlenecks
- **TensorRT Acceleration** — Automatic TensorRT EP integration for maximum inference throughput
- **Multi-stage Pipeline** — Face detection → swapping → face restoration (GFPGAN/GPEN/CodeFormer) → face parsing
- **GPU Utilization Targeting** — Configurable GPU load target (default 95%) with live tuner
- **Rich Progress UI** — Real-time terminal dashboard with per-stage queue depths, FPS, and GPU metrics
- **Web UI / Desktop GUI** — React + Tauri frontend with live preview and full pipeline control
- **Video & Image Support** — Process single images or full video files with frame-level parallelism
- **Graceful Degradation** — Full CPU fallback when CUDA is unavailable

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA (optional, CPU works too)
- Windows or Linux

### Installation

```bash
# Clone the repository
git clone https://github.com/Pakaho-q1/q1-FACESWAP.git
cd q1-FACESWAP

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux
.venv\Scripts\activate     # Windows

# Install q1-FaceSwap
pip install -e .

# Download model assets
python scripts/download_models.py
```

### Download Pre-built GUI (Optional)

Grab the latest `gui.exe` from the [Releases](https://github.com/Pakaho-q1/q1-FACESWAP/releases) page and place it at:

```
q1-FACESWAP/dist/gui.exe
```

Then launch with:

```bash
python faceswap.py gui
```

### Basic Usage

```bash
# Run face swap on a single image
python faceswap.py run --face-model-name alice --input-path ./input/photo.jpg

# Run on a video
python faceswap.py run --face-model-name alice --format 2 --input-path ./input/video.mp4

# Launch Web UI in browser
python faceswap.py webui

# Launch Desktop GUI (requires gui.exe)
python faceswap.py gui

# Display help
python faceswap.py --help
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Swarm Engine                             │
│  ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐      │
│  │ Detect  │──▶│  Swap   │──▶│ Restore  │──▶│  Parse   │      │
│  │ Workers │   │ Workers │   │ Workers  │   │ Workers  │      │
│  └─────────┘   └─────────┘   └──────────┘   └──────────┘      │
│         ▲            ▲              ▲             ▲            │
│         └────────────┴──────────────┴─────────────┘            │
│                    Hot Stage Balancer                          │
│         Dynamically allocates workers to the                   │
│         most congested stage in real-time                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ONNX Runtime Backend                        │
│  TensorRT EP  │  CUDA EP  │  CPU EP  (auto-selected)           │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

| Stage | Component | Function |
|-------|-----------|----------|
| **Detect** | InsightFace buffalo_l | Face detection & landmark extraction |
| **Swap** | Inswapper_128 | Face swapping with source embedding |
| **Restore** | GFPGAN / GPEN / CodeFormer | Face restoration & enhancement |
| **Parse** | CelebAMask-HQ SegFormer | Face segmentation mask generation |

---

## Performance Tuning

The swarm engine automatically manages worker allocation, but you can configure its behavior:

```bash
# Target 95% GPU utilization (default)
python faceswap.py run --gpu-target-util 95

# Increase parallel workers per stage
python faceswap.py run --workers-per-stage 8

# Set provider preference (trt / cuda / cpu)
python faceswap.py run --provider-all trt
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKERS_PER_STAGE` | `8` | Max concurrent workers per pipeline stage |
| `WORKER_QUEUE_SIZE` | `64` | Inter-stage queue capacity |
| `GPU_TARGET_UTIL` | `95` | Target GPU utilization percentage |
| `PROVIDER_ALL` | `trt` | Default execution provider |

---

## Project Structure

```
q1-FACESWAP/
├── core/                     # Python backend
│   ├── processors/           # Pipeline stage processors
│   ├── swarm_engine.py       # Adaptive worker scheduler
│   ├── model_manager.py      # ONNX model lifecycle
│   ├── provider_policy.py    # GPU/CPU provider resolution
│   └── web_server.py         # REST API + static file server
├── webui/                    # React + Tauri desktop frontend
│   ├── src/                  # TypeScript/React application
│   └── src-tauri/            # Tauri (Rust) native shell
├── scripts/
│   ├── download_models.py    # Model asset downloader
│   ├── bootstrap.py          # Environment setup & diagnostics
│   └── build_gui.py          # Tauri desktop build script
├── assets/models/            # Downloaded ONNX models
├── dist/                     # Pre-built GUI binary location
├── input/                    # Input images/videos
└── output/                   # Processed output files
```

---

## Model Credits

q1-FaceSwap bundles and distributes the following third-party models. We are grateful to their creators:

| Model | Source | License |
|-------|--------|---------|
| **InsightFace** (buffalo_l, inswapper_128) | [InsightFace](https://github.com/deepinsight/insightface) | MIT |
| **GFPGAN** v1.4 | [GFPGAN](https://github.com/TencentARC/GFPGAN) | Apache 2.0 |
| **GPEN** BFR-512 / BFR-1024 | [GPEN](https://github.com/yangxy/GPEN) | Apache 2.0 |
| **CodeFormer** | [CodeFormer](https://github.com/sczhou/CodeFormer) | MIT |
| **CelebAMask-HQ SegFormer** | [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) | MIT |
| **FFmpeg** | [FFmpeg](https://ffmpeg.org/) | LGPL/GPL |

---

## Requirements

- Python 3.10+
- 4 GB+ VRAM recommended for GPU processing
- 8 GB+ system RAM
- Windows 10+ or Linux

---

## License

Proprietary. All rights reserved.

---

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for face detection and recognition models
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) for cross-platform inference engine
- [Tauri](https://tauri.app/) for the desktop GUI framework
- All model authors listed in [Model Credits](#model-credits)
