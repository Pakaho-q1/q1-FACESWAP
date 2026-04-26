import logging
import os
import queue
import threading
import time

import cv2
import numpy as np

from core.ffmpeg_runner import (
    close_wait_process,
    extract_audio_copy,
    open_decoder,
    open_encoder,
    terminate_process,
)
from core.errors import PipelineError
from core.orchestrator import (
    build_output_name,
    build_video_output_paths,
    finalize_video_output,
    stage_recovery_file,
)
from core.pipeline_state import PipelineState
from core.preview_utils import encode_preview_data_url
from core.ui_log import ui_print


logger = logging.getLogger(__name__)


def _state(pipeline_state: PipelineState | None) -> PipelineState:
    if pipeline_state is None:
        raise PipelineError("pipeline_state is required for process_videos/frame_writer")
    return pipeline_state


def _backlog_pressure(state: PipelineState) -> float:
    queues = state.queues
    components = [
        queues.detect.qsize() / max(1, queues.detect.maxsize),
        queues.swap.qsize() / max(1, queues.swap.maxsize),
    ]
    if state.config.enable_restore:
        components.append(queues.restore.qsize() / max(1, queues.restore.maxsize))
    if state.config.enable_parser:
        components.append(queues.parse.qsize() / max(1, queues.parse.maxsize))
    return sum(components) / float(len(components))


def _short_video_label(filename, max_chars=15):
    stem, ext = os.path.splitext(filename)
    if len(stem) <= max_chars:
        return filename
    return f"{stem[:max_chars]}... {ext}"


def frame_writer(process_out, start_frame_id, ingest_done_event, ingest_state, ui=None, pipeline_state=None):
    """Read completed frames and stream them to FFmpeg in order."""
    state = _state(pipeline_state)
    queues = state.queues
    next_frame_to_write = start_frame_id
    buffer = {}
    last_preview_ts = 0.0
    fps_window_start_ts = None
    fps_window_items = 0
    last_write_fps = 0.0

    while True:
        if state.abort_event.is_set() and ingest_done_event.is_set():
            break
        try:
            task = queues.out.get(timeout=1.0)
        except queue.Empty:
            if ingest_done_event.is_set() and next_frame_to_write >= ingest_state["end_frame_id"]:
                break
            continue

        try:
            if task is None:
                if ingest_done_event.is_set() and next_frame_to_write >= ingest_state["end_frame_id"]:
                    break
                continue

            frame_id, frame = task
            buffer[frame_id] = frame

            while next_frame_to_write in buffer:
                out_frame = buffer.pop(next_frame_to_write)
                if process_out.poll() is None and out_frame is not None:
                    try:
                        process_out.stdin.write(out_frame.tobytes())
                    except (OSError, IOError):
                        logger.exception("video_encode_write_failed")
                now_ts = time.perf_counter()
                preview_callback = state.preview_callback
                if (
                    state.preview_enabled
                    and out_frame is not None
                    and preview_callback is not None
                    and (now_ts - last_preview_ts) >= state.preview_interval_s
                ):
                    preview_data_url = encode_preview_data_url(out_frame, max_width=640, jpeg_quality=70)
                    if preview_data_url is not None:
                        preview_callback(
                            {
                                "kind": "video",
                                "item_id": ingest_state.get("item_id", ""),
                                "frame_id": int(next_frame_to_write),
                                "data_url": preview_data_url,
                            }
                        )
                        last_preview_ts = now_ts
                next_frame_to_write += 1
                if ui is not None:
                    ui.advance_progress(1)
                now_ts_fps = time.perf_counter()
                if fps_window_start_ts is None:
                    fps_window_start_ts = now_ts_fps
                fps_window_items += 1
                elapsed_fps = now_ts_fps - fps_window_start_ts
                if elapsed_fps >= 0.5:
                    last_write_fps = float(fps_window_items) / max(1e-6, elapsed_fps)
                    fps_window_start_ts = now_ts_fps
                    fps_window_items = 0
                if state.write_fps_callback is not None:
                    state.write_fps_callback(
                        {
                            "label": "video",
                            "write_fps": float(last_write_fps),
                            "written": int(next_frame_to_write - start_frame_id),
                            "total": int(ingest_state.get("end_frame_id", 0)),
                            "item_id": ingest_state.get("item_id", ""),
                        }
                    )
                if state.progress_callback is not None:
                    total = int(ingest_state.get("end_frame_id", 0))
                    state.progress_callback("video", next_frame_to_write, total)
        finally:
            queues.out.task_done()

        if ingest_done_event.is_set() and next_frame_to_write >= ingest_state["end_frame_id"]:
            break


def process_videos(
    get_gpu_utilization,
    workers_per_stage=4,
    ui=None,
    video_list=None,
    input_path=None,
    output_dir=None,
    output_suffix=None,
    temp_audio_dir=None,
    max_frames=None,
    ffmpeg_cmd=None,
    pipeline_state: PipelineState | None = None,
    on_item_start=None,
    on_item_done=None,
    on_video_finalize=None,
):
    """Extract frames, process them, and encode output videos."""
    state = _state(pipeline_state)
    queues = state.queues

    if video_list is None:
        raise ValueError("video_list is required. Build work items in orchestrator before calling process_videos().")
    if not input_path:
        raise ValueError("input_path is required.")
    if not output_dir:
        raise ValueError("output_dir is required.")
    if output_suffix is None:
        raise ValueError("output_suffix is required.")
    if not temp_audio_dir:
        raise ValueError("temp_audio_dir is required.")
    if max_frames is None:
        raise ValueError("max_frames is required.")
    if not ffmpeg_cmd:
        raise ValueError("ffmpeg_cmd is required.")

    if not video_list:
        ui_print("?? There are no remaining videos!", "There are no remaining videos!")
        for _ in range(workers_per_stage):
            queues.detect.put(None)
        return

    for idx, (filename, _size) in enumerate(video_list, 1):
        if state.abort_event.is_set():
            logger.warning("video_processing_aborted_before_item", extra={"index": idx})
            break
        if on_item_start is not None:
            on_item_start(filename)
        input_video = os.path.join(input_path, filename)
        output_name = build_output_name(filename, output_suffix)
        paths = build_video_output_paths(output_dir, temp_audio_dir, output_name, idx)

        process_in = None
        process_out = None
        writer_thread = None
        ingest_done_event = None
        ingest_state = None

        if stage_recovery_file(paths):
            ui_print(
                f"\n?? Found stuck file {filename} -> Prepare to restore completed frames!",
                f"\nFound stuck file {filename} -> Prepare to restore completed frames!",
            )
        else:
            ui_print(f"\n?? Rendering: {filename}", f"\nRendering: {filename}")

        item_success = False
        try:
            capture = cv2.VideoCapture(input_video)
            orig_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()

            if orig_width <= 0 or orig_height <= 0 or fps <= 0 or total_frames <= 0:
                raise ValueError(f"Invalid video metadata: {filename}")

            max_video_frames = total_frames if max_frames <= 0 else min(total_frames, max_frames)

            width = orig_width - (orig_width % 2)
            height = orig_height - (orig_height % 2)

            has_audio = extract_audio_copy(ffmpeg_cmd, input_video, paths.temp_audio)
            process_out = open_encoder(
                ffmpeg_cmd=ffmpeg_cmd,
                width=width,
                height=height,
                fps=fps,
                temp_video=paths.temp_video,
                temp_audio=paths.temp_audio if has_audio else None,
            )

            if ui is not None:
                short_name = _short_video_label(output_name, max_chars=15)
                ui.set_progress(total=max_video_frames, description=f"Rendering {short_name}")
            recovered_frames = 0

            if os.path.exists(paths.recovery_video):
                if ui is not None:
                    short_name = _short_video_label(output_name, max_chars=15)
                    ui.set_progress_description(f"Recovering {short_name}")
                recovery_capture = cv2.VideoCapture(paths.recovery_video)
                while recovered_frames < max_video_frames:
                    if state.abort_event.is_set():
                        logger.warning("video_recovery_aborted", extra={"file": filename})
                        break
                    ret, frame = recovery_capture.read()
                    if not ret:
                        break

                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height))

                    if process_out.poll() is None and process_out.stdin is not None:
                        try:
                            process_out.stdin.write(frame.tobytes())
                            recovered_frames += 1
                            if ui is not None:
                                ui.advance_progress(1)
                            if state.progress_callback is not None:
                                state.progress_callback("video", recovered_frames, max_video_frames)
                        except (OSError, IOError):
                            logger.exception("video_recovery_write_failed", extra={"file": filename})
                            break
                recovery_capture.release()
                if ui is not None:
                    short_name = _short_video_label(output_name, max_chars=15)
                    ui.set_progress_description(f"Rendering {short_name}")
                ui_print(
                    f"? Recovery successful {recovered_frames} frames! Continuing AI pipeline...",
                    f"Recovery successful {recovered_frames} frames! Continuing AI pipeline...",
                )

            process_in = open_decoder(ffmpeg_cmd=ffmpeg_cmd, input_video=input_video, fps=fps)
            frame_size = orig_width * orig_height * 3

            if recovered_frames > 0 and process_in.stdout is not None:
                for _ in range(recovered_frames):
                    process_in.stdout.read(frame_size)

            ingest_done_event = threading.Event()
            ingest_state = {"end_frame_id": max_video_frames, "item_id": filename}

            writer_thread = threading.Thread(
                target=frame_writer,
                args=(process_out, recovered_frames, ingest_done_event, ingest_state, ui, state),
                name="video-writer",
            )
            writer_thread.start()

            frame_id = recovered_frames
            is_paused = False
            while frame_id < max_video_frames:
                if state.abort_event.is_set():
                    logger.warning("video_processing_aborted", extra={"file": filename, "frame_id": frame_id})
                    break
                try:
                    gpu_util = get_gpu_utilization()
                except Exception:
                    logger.exception("gpu_utilization_probe_failed")
                    gpu_util = 0

                backlog_pressure = _backlog_pressure(state)

                if not is_paused and gpu_util >= 99:
                    is_paused = True
                elif is_paused and gpu_util < 96 and backlog_pressure < 0.88:
                    is_paused = False

                if backlog_pressure >= 0.92:
                    is_paused = True

                if is_paused:
                    time.sleep(0.03)
                    continue

                if process_in.stdout is None:
                    break
                in_bytes = process_in.stdout.read(frame_size)
                if not in_bytes or len(in_bytes) != frame_size:
                    break

                frame = np.frombuffer(in_bytes, np.uint8).reshape((orig_height, orig_width, 3))
                if orig_width != width or orig_height != height:
                    frame = cv2.resize(frame, (width, height))

                # Avoid hard block when queue is full so stop signal can interrupt feed loop.
                while True:
                    if state.abort_event.is_set():
                        break
                    try:
                        queues.detect.put((frame_id, frame), timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if state.abort_event.is_set():
                    break
                frame_id += 1

            ingest_state["end_frame_id"] = frame_id
            ingest_done_event.set()
            writer_thread.join()
            writer_thread = None

            close_wait_process(process_in)
            close_wait_process(process_out)
            if ui is not None:
                ui.finish_progress()

            if state.abort_event.is_set():
                # User-stop path: keep temp/recovery files for resumable processing.
                logger.info(
                    "video_stop_requested_keep_temp",
                    extra={"file": filename, "temp_video": paths.temp_video, "recovery_video": paths.recovery_video},
                )
            else:
                if on_video_finalize is not None:
                    item_success, finalize_error = on_video_finalize(filename, paths)
                    if not item_success and on_item_done is not None:
                        on_item_done(filename, False, finalize_error)
                else:
                    finalize_video_output(paths)
                    item_success = os.path.exists(paths.final_video)
                if item_success and on_item_done is not None:
                    on_item_done(filename, True, "")
                elif not item_success and on_video_finalize is None and on_item_done is not None:
                    on_item_done(filename, False, "final_video_missing")

        except Exception as exc:
            logger.exception("video_processing_failed", extra={"file": filename, "error": str(exc)})
            if on_item_done is not None:
                on_item_done(filename, False, str(exc))
            state.request_abort()
            if ingest_done_event is not None and ingest_state is not None:
                ingest_state["end_frame_id"] = 0
                ingest_done_event.set()
            if writer_thread is not None and writer_thread.is_alive():
                writer_thread.join(timeout=5.0)
            terminate_process(process_in)
            terminate_process(process_out)
            break
        finally:
            if os.path.exists(paths.temp_audio):
                try:
                    os.remove(paths.temp_audio)
                except OSError:
                    logger.warning("temp_audio_cleanup_failed", extra={"path": paths.temp_audio})
        if state.abort_event.is_set():
            break

    for _ in range(workers_per_stage):
        queues.detect.put(None)
