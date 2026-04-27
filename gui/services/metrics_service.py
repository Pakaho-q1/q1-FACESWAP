from __future__ import annotations

import os
from typing import Any


def build_output_name(filename: str, suffix: str) -> str:
    if not suffix:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}{suffix}{ext}"


def scan_selected_job_status(
    selected_format: str,
    input_path_value: str,
    output_path_value: str,
    suffix: str,
    failed_counts: dict[str, int],
    controller_running: bool,
) -> dict[str, int]:
    selected_fmt = str(selected_format or "image").lower()
    in_dir = os.path.expanduser(str(input_path_value or "").strip())
    out_dir = os.path.expanduser(str(output_path_value or "").strip())
    image_ext = {".png", ".jpg", ".jpeg"}
    video_ext = {".mp4", ".mkv", ".avi", ".mov"}

    total = 0
    done = 0
    input_files: list[str] = []

    if os.path.isdir(in_dir):
        try:
            with os.scandir(in_dir) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    _, ext = os.path.splitext(entry.name)
                    ext_l = ext.lower()
                    if selected_fmt == "video" and ext_l in video_ext:
                        input_files.append(entry.name)
                    elif selected_fmt != "video" and ext_l in image_ext:
                        input_files.append(entry.name)
        except OSError:
            input_files = []

    total = len(input_files)
    if total > 0 and os.path.isdir(out_dir):
        expected_names = {build_output_name(name, suffix) for name in input_files}
        try:
            with os.scandir(out_dir) as out_entries:
                for entry in out_entries:
                    if entry.is_file() and entry.name in expected_names:
                        done += 1
        except OSError:
            done = 0

    failed = failed_counts["video"] if selected_fmt == "video" else failed_counts["image"]
    planned = max(0, total - done)
    running = 1 if controller_running and planned > 0 else 0
    return {
        "total": total,
        "done": done,
        "planned": planned,
        "running": running,
        "failed": max(0, failed),
    }


def format_eta(seconds: float) -> str:
    if seconds < 0 or not seconds or seconds > 864000:
        return "--:--"
    sec = int(seconds)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def compute_throughput(
    progress_units_done: int,
    progress_units_total: int,
    elapsed_seconds: float,
    latest_write_fps: float,
    fallback_remaining: int,
) -> dict[str, Any]:
    elapsed = max(0.0, float(elapsed_seconds))
    done = max(0, int(progress_units_done))
    total = max(0, int(progress_units_total))

    items_per_min = 0.0 if elapsed <= 0 else ((done * 60.0) / elapsed)
    if latest_write_fps > 0:
        write_fps = float(latest_write_fps)
    elif elapsed > 0 and done > 0:
        write_fps = done / elapsed
    else:
        write_fps = 0.0

    if total > 0:
        remaining = max(0, total - done)
    else:
        remaining = max(0, int(fallback_remaining))

    rate_items_per_sec = 0.0 if elapsed <= 0 else (done / elapsed)
    eta_seconds = (remaining / rate_items_per_sec) if rate_items_per_sec > 0 else -1.0

    return {
        "items_per_min": items_per_min,
        "write_fps": write_fps,
        "eta_seconds": eta_seconds,
    }


def merge_pipeline_metrics(
    planned_counts: dict[str, int],
    completed_counts: dict[str, int],
    progress_units_done: int,
    progress_units_total: int,
    metrics: dict[str, Any],
) -> tuple[dict[str, int], dict[str, int], int, int]:
    if not metrics:
        return planned_counts, completed_counts, progress_units_done, progress_units_total

    if "planned_images" in metrics:
        planned_counts["image"] = max(planned_counts["image"], int(metrics.get("planned_images", 0)))
    if "planned_videos" in metrics:
        planned_counts["video"] = max(planned_counts["video"], int(metrics.get("planned_videos", 0)))
    if "written_images" in metrics:
        completed_counts["image"] = max(completed_counts["image"], int(metrics.get("written_images", 0)))
    if "written_videos" in metrics:
        completed_counts["video"] = max(completed_counts["video"], int(metrics.get("written_videos", 0)))
    if "planned_images" in metrics:
        progress_units_total = max(progress_units_total, int(metrics.get("planned_images", 0)))
    if "planned_videos" in metrics:
        progress_units_total = max(progress_units_total, int(metrics.get("planned_videos", 0)))
    if "written_images" in metrics:
        progress_units_done = max(progress_units_done, int(metrics.get("written_images", 0)))
    if "written_videos" in metrics:
        progress_units_done = max(progress_units_done, int(metrics.get("written_videos", 0)))

    return planned_counts, completed_counts, progress_units_done, progress_units_total
