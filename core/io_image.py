from __future__ import annotations

import logging
import os
import queue
import threading
import time

import cv2

from core.orchestrator import build_output_name
from core.errors import PipelineError
from core.pipeline_state import PipelineState
from core.ui_log import ui_print


logger = logging.getLogger(__name__)


def _state(pipeline_state: PipelineState | None) -> PipelineState:
    if pipeline_state is None:
        raise PipelineError("pipeline_state is required for process_images/image_writer")
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


def image_writer(
    total_images: int,
    output_dir: str,
    output_suffix: str,
    ui=None,
    pipeline_state: PipelineState | None = None,
    on_item_result=None,
):
    """Wait for processed images and save them in order of completion.

    Exits early if ``pipeline_state.abort_event`` is set so that a crashed
    pipeline never leaves this thread blocked forever.
    """
    state = _state(pipeline_state)
    queues = state.queues
    written = 0
    fps_window_start_ts = None
    fps_window_items = 0
    last_write_fps = 0.0

    while written < total_images:
        # Abort check: pipeline signalled a fatal error.
        if state.abort_event.is_set():
            logger.warning("image_writer_aborted", extra={"written": written, "total": total_images})
            break

        try:
            task = queues.out.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            if task is None:
                # Unexpected sentinel — count it so the loop can exit cleanly.
                written += 1
                continue

            img_file, res_img = task
            if on_item_result is not None:
                on_item_result(img_file, res_img)
            elif res_img is not None:
                out_name = build_output_name(img_file, output_suffix)
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, res_img)

            if ui is not None:
                ui.advance_progress(1)
            written += 1
            now_ts = time.perf_counter()
            if fps_window_start_ts is None:
                fps_window_start_ts = now_ts
            fps_window_items += 1
            elapsed = now_ts - fps_window_start_ts
            if elapsed >= 0.5:
                last_write_fps = float(fps_window_items) / max(1e-6, elapsed)
                fps_window_start_ts = now_ts
                fps_window_items = 0
            if state.write_fps_callback is not None:
                state.write_fps_callback(
                    {
                        "label": "image",
                        "write_fps": float(last_write_fps),
                        "written": int(written),
                        "total": int(total_images),
                    }
                )
            if state.progress_callback is not None:
                state.progress_callback("image", written, total_images)
        finally:
            queues.out.task_done()


def process_images(
    get_gpu_utilization,
    workers_per_stage=4,
    ui=None,
    pending_images=None,
    input_path=None,
    output_dir=None,
    output_suffix=None,
    pipeline_state: PipelineState | None = None,
    on_item_start=None,
    on_item_result=None,
):
    """Feed planned image work into the processing pipeline."""
    state = _state(pipeline_state)
    queues = state.queues

    if pending_images is None:
        raise ValueError(
            "pending_images is required. Build work items in orchestrator before calling process_images()."
        )
    if not input_path:
        raise ValueError("input_path is required.")
    if not output_dir:
        raise ValueError("output_dir is required.")
    if output_suffix is None:
        raise ValueError("output_suffix is required.")

    if not pending_images:
        ui_print("No more leftover photos to complete!", "No more leftover photos to complete!")
        for _ in range(workers_per_stage):
            queues.detect.put(None)
        return

    ui_print(
        f"Start Image Pipeline: {len(pending_images)} Images",
        f"Start Image Pipeline: {len(pending_images)} Images",
    )
    if ui is not None:
        ui.set_progress(total=len(pending_images), description="Processing Images")

    writer_thread = threading.Thread(
        target=image_writer,
        args=(len(pending_images), output_dir, output_suffix, ui, pipeline_state, on_item_result),
        name="image-writer",
    )
    writer_thread.start()

    is_paused = False

    try:
        for img_file in pending_images:
            if on_item_start is not None:
                on_item_start(img_file)

            while True:
                if state.abort_event.is_set():
                    return

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
                break

            img_path = os.path.join(input_path, img_file)
            frame = cv2.imread(img_path)

            if frame is not None:
                # Avoid hard block when queue is full so stop signal can interrupt feed loop.
                while True:
                    if state.abort_event.is_set():
                        return
                    try:
                        queues.detect.put((img_file, frame), timeout=0.2)
                        break
                    except queue.Full:
                        continue
            else:
                logger.warning("image_read_failed", extra={"path": img_path})
                queues.out.put((img_file, None))
    except Exception:
        logger.exception("process_images_feed_failed")
        state.request_abort()
        raise
    finally:
        writer_thread.join()
        if ui is not None:
            ui.finish_progress()
        for _ in range(workers_per_stage):
            queues.detect.put(None)
