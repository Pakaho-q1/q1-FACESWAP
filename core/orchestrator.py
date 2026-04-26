from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
from core.errors import PipelineError
from core.pipeline_state import PipelineState
from core.preview_utils import encode_preview_data_url
from core.types import RuntimeContext


@dataclass(frozen=True)
class JobPlan:
    kind: str
    input_path: str
    output_path: str
    output_suffix: str
    models_dir: str
    temp_audio_dir: str
    ffmpeg_cmd: str
    skip_existing: bool
    max_frames: int
    face_name: str
    enable_swapper: bool
    enable_restore: bool
    enable_parser: bool
    workers_per_stage: int


@dataclass(frozen=True)
class ImageWorkItem:
    filename: str
    output_name: str


@dataclass(frozen=True)
class VideoWorkItem:
    filename: str
    size_bytes: int
    output_name: str


@dataclass(frozen=True)
class VideoOutputPaths:
    output_name: str
    final_video: str
    temp_video: str
    recovery_video: str
    temp_audio: str


def build_output_name(filename: str, suffix: str) -> str:
    if not suffix:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}{suffix}{ext}"


def commit_image_output(output_dir: str, image_name: str, image_bgr) -> bool:
    out_path = os.path.join(output_dir, image_name)
    return bool(cv2.imwrite(out_path, image_bgr))


def build_video_output_paths(
    output_dir: str,
    temp_audio_dir: str,
    output_name: str,
    index: int,
) -> VideoOutputPaths:
    os.makedirs(temp_audio_dir, exist_ok=True)
    base_name, ext = os.path.splitext(output_name)
    return VideoOutputPaths(
        output_name=output_name,
        final_video=os.path.join(output_dir, output_name),
        temp_video=os.path.join(output_dir, f"{base_name}-temp{ext}"),
        recovery_video=os.path.join(output_dir, f"{base_name}-recovery{ext}"),
        temp_audio=os.path.join(temp_audio_dir, f"temp_audio_{index}.aac"),
    )

def finalize_video_output(paths: VideoOutputPaths) -> None:
    temp_exists = os.path.exists(paths.temp_video)
    temp_has_content = temp_exists and os.path.getsize(paths.temp_video) > 0

    # Use replace so reruns can overwrite existing outputs safely on Windows.
    if temp_has_content:
        os.replace(paths.temp_video, paths.final_video)
    elif temp_exists:
        # Cleanup zero-byte temp artifacts.
        os.remove(paths.temp_video)

    # If final output is present, stale recovery file is no longer needed.
    if os.path.exists(paths.final_video) and os.path.exists(paths.recovery_video):
        os.remove(paths.recovery_video)


def stage_recovery_file(paths: VideoOutputPaths) -> bool:
    temp_exists = os.path.exists(paths.temp_video)
    recovery_exists = os.path.exists(paths.recovery_video)
    final_exists = os.path.exists(paths.final_video)

    # Stale recovery after successful finalize from older run.
    if recovery_exists and final_exists and not temp_exists:
        try:
            os.remove(paths.recovery_video)
        except OSError:
            pass
        return False

    if not temp_exists:
        return False

    # Atomic replace prevents "cannot rename because recovery exists" loops.
    os.replace(paths.temp_video, paths.recovery_video)
    return True


def build_plan(ctx: RuntimeContext) -> JobPlan:
    run_cfg = ctx.config
    kind = "image" if run_cfg.format_is_image else "video"
    return JobPlan(
        kind=kind,
        input_path=run_cfg.input_path,
        output_path=run_cfg.output_dir,
        output_suffix=run_cfg.output_suffix,
        models_dir=run_cfg.models_dir,
        temp_audio_dir=run_cfg.temp_audio_dir,
        ffmpeg_cmd=run_cfg.ffmpeg_cmd,
        skip_existing=run_cfg.skip_existing,
        max_frames=run_cfg.max_frames,
        face_name=run_cfg.face_name,
        enable_swapper=run_cfg.enable_swapper,
        enable_restore=run_cfg.enable_restore,
        enable_parser=run_cfg.enable_parser,
        workers_per_stage=run_cfg.workers_per_stage,
    )


def discover_image_work(plan: JobPlan) -> List[ImageWorkItem]:
    image_files = [
        f
        for f in os.listdir(plan.input_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if plan.skip_existing:
        existing_outputs = set(os.listdir(plan.output_path))
        image_files = [
            f
            for f in image_files
            if build_output_name(f, plan.output_suffix) not in existing_outputs
        ]
    if plan.max_frames > 0:
        image_files = image_files[: plan.max_frames]
    return [
        ImageWorkItem(
            filename=f,
            output_name=build_output_name(f, plan.output_suffix),
        )
        for f in image_files
    ]


def discover_video_work(plan: JobPlan) -> List[VideoWorkItem]:
    video_files = [
        f
        for f in os.listdir(plan.input_path)
        if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))
    ]
    ordered: List[Tuple[str, int]] = sorted(
        [(f, os.path.getsize(os.path.join(plan.input_path, f))) for f in video_files],
        key=lambda x: x[1],
    )
    if plan.skip_existing:
        existing_outputs = set(os.listdir(plan.output_path))
        ordered = [
            (f, size)
            for f, size in ordered
            if build_output_name(f, plan.output_suffix) not in existing_outputs
        ]
    return [
        VideoWorkItem(
            filename=f,
            size_bytes=size,
            output_name=build_output_name(f, plan.output_suffix),
        )
        for f, size in ordered
    ]


def _job_state_path(plan: JobPlan) -> str:
    return os.path.join(plan.output_path, f".job_state_{plan.kind}.json")


def run_job(
    ctx: RuntimeContext,
    process_images_fn: Callable,
    process_videos_fn: Callable,
    get_gpu_utilization: Callable,
    runtime_ui=None,
    pipeline_state: Optional[PipelineState] = None,
):
    if pipeline_state is None:
        raise PipelineError("pipeline_state is required for run_job")
    state = pipeline_state
    plan = build_plan(ctx)

    if plan.kind == "image":
        legacy_state_path = _job_state_path(plan)
        if os.path.exists(legacy_state_path):
            try:
                os.remove(legacy_state_path)
            except OSError:
                pass

        image_items = discover_image_work(plan)
        output_name_by_id: Dict[str, str] = {item.filename: item.output_name for item in image_items}
        pending_items = list(image_items)

        def on_item_start(item_id: str):
            ctx.emit_event("item_started", {"kind": "image", "item_id": item_id})

        def on_item_result(item_id: str, result_image):
            nonlocal image_preview_last_ts
            if result_image is None:
                ctx.emit_event("item_failed", {"kind": "image", "item_id": item_id, "reason": "empty_result_image"})
            else:
                out_name = output_name_by_id.get(item_id, build_output_name(item_id, plan.output_suffix))
                if commit_image_output(plan.output_path, out_name, result_image):
                    state.metrics.increment("written_images", 1)
                    ctx.emit_event("item_completed", {"kind": "image", "item_id": item_id})
                    now_ts = time.perf_counter()
                    if (
                        state.preview_enabled
                        and state.preview_callback is not None
                        and (now_ts - image_preview_last_ts) >= state.preview_interval_s
                    ):
                        preview_data_url = encode_preview_data_url(result_image, max_width=640, jpeg_quality=75)
                        if preview_data_url is not None:
                            ctx.emit_event(
                                "preview",
                                {
                                    "kind": "image",
                                    "item_id": item_id,
                                    "output_name": out_name,
                                    "data_url": preview_data_url,
                                },
                            )
                            image_preview_last_ts = now_ts
                else:
                    ctx.emit_event("item_failed", {"kind": "image", "item_id": item_id, "reason": "image_commit_failed"})

        state.metrics.increment("planned_images", len(pending_items))
        image_preview_last_ts = 0.0
        process_images_fn(
            get_gpu_utilization,
            plan.workers_per_stage,
            runtime_ui,
            pending_images=[item.filename for item in pending_items],
            input_path=plan.input_path,
            output_dir=plan.output_path,
            output_suffix=plan.output_suffix,
            pipeline_state=state,
            on_item_start=on_item_start,
            on_item_result=on_item_result,
        )
    else:
        # Video mode is now filesystem-truth based (no .job_state_video).
        legacy_state_path = _job_state_path(plan)
        if os.path.exists(legacy_state_path):
            try:
                os.remove(legacy_state_path)
            except OSError:
                pass

        video_items = discover_video_work(plan)
        output_name_by_id: Dict[str, str] = {item.filename: item.output_name for item in video_items}
        pending_items = list(video_items)

        def on_item_start(item_id: str):
            ctx.emit_event("item_started", {"kind": "video", "item_id": item_id})

        def on_video_finalize(item_id: str, paths):
            expected_name = output_name_by_id.get(item_id, paths.output_name)
            if paths.output_name != expected_name:
                ctx.emit_event("item_failed", {"kind": "video", "item_id": item_id, "reason": "output_name_mismatch"})
                return False, "output_name_mismatch"

            finalize_video_output(paths)
            if os.path.exists(paths.final_video):
                state.metrics.increment("written_videos", 1)
                ctx.emit_event("item_completed", {"kind": "video", "item_id": item_id})
                return True, ""

            ctx.emit_event("item_failed", {"kind": "video", "item_id": item_id, "reason": "final_video_missing"})
            return False, "final_video_missing"

        def on_item_done(item_id: str, success: bool, error: str = ""):
            if success:
                # Completion is decided only after finalize callback.
                return
            else:
                ctx.emit_event("item_failed", {"kind": "video", "item_id": item_id, "reason": error})

        state.metrics.increment("planned_videos", len(pending_items))
        process_videos_fn(
            get_gpu_utilization,
            plan.workers_per_stage,
            runtime_ui,
            video_list=[(item.filename, item.size_bytes) for item in pending_items],
            input_path=plan.input_path,
            output_dir=plan.output_path,
            output_suffix=plan.output_suffix,
            temp_audio_dir=plan.temp_audio_dir,
            max_frames=plan.max_frames,
            ffmpeg_cmd=plan.ffmpeg_cmd,
            pipeline_state=state,
            on_item_start=on_item_start,
            on_item_done=on_item_done,
            on_video_finalize=on_video_finalize,
        )
    return state.metrics.snapshot()


def resume_job(
    ctx: RuntimeContext,
    process_images_fn: Callable,
    process_videos_fn: Callable,
    get_gpu_utilization: Callable,
    runtime_ui=None,
    pipeline_state: Optional[PipelineState] = None,
):
    # Current pipeline auto-recovers by design (especially video temp/recovery path).
    # Resume delegates to run_job until checkpointed state is introduced in later phases.
    return run_job(
        ctx=ctx,
        process_images_fn=process_images_fn,
        process_videos_fn=process_videos_fn,
        get_gpu_utilization=get_gpu_utilization,
        runtime_ui=runtime_ui,
        pipeline_state=pipeline_state,
    )
