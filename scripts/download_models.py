"""
download_models.py
==================
One-command model downloader for q1-FaceSwap.

Downloads all required ONNX models and ffmpeg from HuggingFace.
Parallel downloads (4 concurrent) with auto-fallback concurrency.
Uses huggingface_hub if available, fallback to urllib.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --models-dir ./assets/models
    python scripts/download_models.py --force
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


REPO_ID = "Pakaho-q1/onnx-models"


def _download_url(url: str, dest: str, progress, task_id: int) -> None:
    tmp = dest + ".tmp"
    req = urllib.request.Request(url, headers={"User-Agent": "q1-FaceSwap/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        if total:
            progress.update(task_id, total=total)
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 128)
                if not chunk:
                    break
                f.write(chunk)
                progress.update(task_id, advance=len(chunk))
    os.replace(tmp, dest)


def _download_one(entry: dict, models_dir: str, progress, task_id: int) -> bool:
    filename = entry["filename"]
    url = entry["url"]
    dest = os.path.join(models_dir, filename)

    # Try huggingface_hub first
    try:
        from huggingface_hub import hf_hub_download

        progress.update(task_id, description=f"[cyan]{filename}[/]")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False,
            resume=True,
            quiet=True,
        )
        if os.path.isfile(dest):
            return True
    except Exception:
        pass

    # Fallback to urllib
    progress.update(task_id, description=f"[cyan]{filename}[/] (http)")
    try:
        _download_url(url, dest, progress, task_id)
        return True
    except Exception:
        return False


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
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent downloads (default: 4, auto-reduces on failure)",
    )
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    models_dir = args.models_dir or os.path.join(repo_root, "assets", "models")

    sys.path.insert(0, repo_root)
    try:
        from core.config import _default_model_manifest
    except ImportError as exc:
        print(f"[FAIL] Cannot import core.config. Make sure q1-faceswap is installed.\n  {exc}")
        sys.exit(1)

    manifest = _default_model_manifest()
    entries = manifest["models"]

    os.makedirs(models_dir, exist_ok=True)

    to_download = []
    for entry in entries:
        path = os.path.join(models_dir, entry["filename"])
        if os.path.isfile(path):
            if args.force:
                os.remove(path)
                print(f"  Removed existing: {entry['filename']}")
            else:
                continue
        to_download.append(entry)

    if not to_download:
        print(f"All models present in {models_dir}. Use --force to re-download.")
        return

    from rich.console import Console
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    console = Console()
    console.print(f"\nDownloading [cyan]{len(to_download)}[/] model(s) to: [yellow]{models_dir}[/]")
    has_hf = False
    try:
        import huggingface_hub  # noqa: F401
        has_hf = True
    except ImportError:
        pass
    via = "huggingface_hub" if has_hf else "urllib (no huggingface_hub)"
    console.print(f"Source: [blue]huggingface.co/{REPO_ID}[/] | via: [yellow]{via}[/]\n")

    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    concurrency = max(1, args.workers)
    remaining = list(to_download)
    task_ids: dict[str, int] = {}

    with progress:
        while remaining and concurrency > 0:
            batch, remaining = remaining[:concurrency], remaining[concurrency:]
            failed_batch = []

            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {}
                for entry in batch:
                    tid = progress.add_task(entry["filename"])
                    task_ids[entry["filename"]] = tid
                    futures[pool.submit(_download_one, entry, models_dir, progress, tid)] = entry

                for fut in as_completed(futures):
                    entry = futures[fut]
                    tid = task_ids[entry["filename"]]
                    ok = fut.result()
                    if ok:
                        progress.update(tid, description=f"[green]{entry['filename']}[/]")
                    else:
                        progress.update(tid, description=f"[red]{entry['filename']}[/]")
                        failed_batch.append(entry)

            if failed_batch:
                concurrency -= 1
                remaining = failed_batch + remaining
                console.print(
                    f"  [yellow]Retrying {len(failed_batch)} failed with {max(1, concurrency)} worker(s)...[/]"
                )

    if remaining:
        console.print(f"\n[red bold]Failed to download {len(remaining)} model(s).[/]")
        for entry in remaining:
            console.print(f"  [red]{entry['filename']}[/]")
        sys.exit(1)

    console.print(f"\n[green bold]Done![/] Models saved to: [yellow]{models_dir}[/]")
    console.print("Run: [bold]python faceswap.py ...[/] or [bold]python faceswap.py gui[/]")


if __name__ == "__main__":
    main()
