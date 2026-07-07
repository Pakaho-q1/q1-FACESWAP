"""
bootstrap.py
============
Bootstrap and verify the python environment for q1-FaceSwap.
This script checks python version, installs dependencies (compatible with venv,
conda, docker, etc.), resolves library conflicts, and runs GPU diagnostics.
"""
from __future__ import annotations

import os
import sys
import subprocess


def print_step(msg: str) -> None:
    print(f"\n{'-'*60}\n>>> {msg}\n{'-'*60}")


def print_ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def print_warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def print_err(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def run_command(command: list[str]) -> bool:
    try:
        subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> None:
    # -- Resolve repository root ----------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir).lower() == "scripts":
        repo_root = os.path.dirname(script_dir)
    else:
        repo_root = script_dir

    print_step("STEP 1: Python Runtime Environment Verification")
    py_ver = sys.version_info
    py_ver_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    
    # Check virtual environment or conda environment active
    in_venv = sys.prefix != sys.base_prefix or "CONDA_PREFIX" in os.environ
    if in_venv:
        print_ok(f"Running inside active environment: {sys.prefix}")
    else:
        print_warn("Not running in a virtual environment. We highly recommend using venv or conda.")

    if py_ver.major == 3 and py_ver.minor >= 10:
        print_ok(f"Python {py_ver_str} is supported.")
    else:
        print_err(f"Python {py_ver_str} is not supported. Please use Python 3.10 or higher.")
        sys.exit(1)

    print_step("STEP 2: Install Package Dependencies")
    requirements_path = os.path.join(repo_root, "requirements.txt")
    if os.path.exists(requirements_path):
        print("[INFO] Installing dependencies via pip (this may take a moment)...")
        # sys.executable ensures we use pip bound to the current venv/conda/docker python interpreter
        if run_command([sys.executable, "-m", "pip", "install", "-r", requirements_path]):
            print_ok("All package dependencies installed successfully.")
        else:
            print_err("Failed to install package dependencies. Please check your internet connection and permissions.")
            sys.exit(1)
    else:
        print_err(f"Dependency manifest not found at: {requirements_path}")
        sys.exit(1)

    print_step("STEP 3: GPU Acceleration — ONNX Runtime Auto-Detect")
    print("[INFO] Checking for NVIDIA CUDA-capable GPU...")
    has_cuda = False
    try:
        from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlShutdown
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        nvmlShutdown()
        has_cuda = device_count > 0
        if has_cuda:
            print_ok(f"Detected {device_count} NVIDIA GPU(s).")
        else:
            print_ok("No NVIDIA GPU detected.")
    except Exception:
        print_ok("nvidia-ml-py not available; cannot detect GPU.")

    if has_cuda:
        print("[INFO] CUDA GPU found. Upgrading to onnxruntime-gpu for hardware acceleration...")
        # onnxruntime-gpu replaces onnxruntime (CPU) — they conflict
        if run_command([sys.executable, "-m", "pip", "install", "onnxruntime-gpu", "numpy<2.0.0"]):
            print_ok("onnxruntime-gpu installed successfully.")
        else:
            print_warn("onnxruntime-gpu install failed; falling back to CPU onnxruntime.")
    else:
        print_ok("Using CPU onnxruntime (safe default).")
        print("[HINT] To enable GPU later: pip install onnxruntime-gpu")
        # Remove stale onnxruntime-gpu if previously installed on a system that no longer has CUDA
        if run_command([sys.executable, "-m", "pip", "show", "onnxruntime-gpu"]):
            print("[INFO] Removing stale onnxruntime-gpu (no CUDA detected)...")
            run_command([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime-gpu"])

    print_step("STEP 4: GPU Acceleration Diagnostics")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"Available execution providers: {providers}")

        if "CUDAExecutionProvider" in providers:
            print_ok("ONNX Runtime successfully initialized CUDA (NVIDIA GPU).")
        else:
            print_warn("ONNX Runtime cannot access CUDA. GPU acceleration might be disabled.")

        if "TensorrtExecutionProvider" in providers:
            print_ok("ONNX Runtime successfully initialized TensorRT.")
        else:
            print_warn("TensorRT execution provider is not loaded.")
    except Exception as exc:
        print_err(f"Error loading ONNX Runtime: {exc}")

    print_step("STEP 5: Validate Critical Path Assets")
    # Resolve assets located in default locations inside workspace
    models_dir = os.path.join(repo_root, "assets", "models")
    
    # We will search the project directory and verify if standard models are present
    critical_paths = {
        "models directory": os.path.isdir(models_dir),
        "Inswapper Model (inswapper_128.onnx)": os.path.isfile(os.path.join(models_dir, "inswapper_128.onnx")),
        "Face Restore Model (GFPGANv1.4.onnx)": os.path.isfile(os.path.join(models_dir, "GFPGANv1.4.onnx")),
        "Face Parser Model (Segformer_CelebAMask-HQ.onnx)": os.path.isfile(os.path.join(models_dir, "Segformer_CelebAMask-HQ.onnx")),
        "FFmpeg Binary": os.path.isfile(os.path.join(models_dir, "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")),
    }

    all_assets_present = True
    for name, exists in critical_paths.items():
        if exists:
            print_ok(f"Verified: {name}")
        else:
            print_warn(f"Missing: {name}")
            all_assets_present = False

    print_step("Diagnostic Summary")
    if all_assets_present and ("CUDAExecutionProvider" in locals().get("providers", [])):
        print("[SUCCESS] Environment is fully configured and ready to run q1-FaceSwap.")
    else:
        print("[NOTICE] Diagnostics completed with warnings. Some features or assets are missing.")


if __name__ == "__main__":
    main()
