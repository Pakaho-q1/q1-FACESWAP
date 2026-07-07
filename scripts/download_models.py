"""
download_models.py
==================
One-command model downloader for q1-FaceSwap.

Downloads all required ONNX models and ffmpeg from HuggingFace.
Works standalone — no config or prior setup needed.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --models-dir ./assets/models
    python scripts/download_models.py --force
"""
from __future__ import annotations

import argparse
import os
import sys


def resolve_repo_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir).lower() == "scripts":
        return os.path.dirname(script_dir)
    return script_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download q1-FaceSwap model assets")
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Target directory for models (default: <repo_root>/assets/models)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download models even if they already exist",
    )
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    models_dir = args.models_dir or os.path.join(repo_root, "assets", "models")

    # Ensure the q1-faceswap package is importable
    sys.path.insert(0, repo_root)
    try:
        from core.config import _default_model_manifest, _sync_models_from_manifest
    except ImportError as exc:
        print(f"[FAIL] Cannot import core.config. Make sure q1-faceswap is installed.\n  {exc}")
        sys.exit(1)

    manifest = _default_model_manifest()
    filenames = {m["filename"] for m in manifest["models"]}

    print(f"Downloading {len(filenames)} model(s) to: {models_dir}")
    print(f"Source: huggingface.co/Pakaho-q1/onnx-models\n")

    os.makedirs(models_dir, exist_ok=True)
    _sync_models_from_manifest(models_dir, manifest, preload_models=args.force, required_filenames=filenames)

    print(f"\n[DONE] All models downloaded to {models_dir}")
    print("You can now run: python faceswap.py ... or q1-faceswap ...")


if __name__ == "__main__":
    main()
