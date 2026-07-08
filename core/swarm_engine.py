from __future__ import annotations

import logging
import threading
import time
from queue import Empty

from core.errors import PipelineError
from core.pipeline_state import PipelineState
from core.processors.detect_proc import run_detect
from core.processors.parse_proc import run_parse
from core.processors.restore_proc import run_restore
from core.processors.swap_proc import run_swap


logger = logging.getLogger(__name__)

# Maximum seconds a worker will wait for a concurrency slot before re-checking
# pipeline health.  Prevents permanent deadlock if a slot is never released.
_SLOT_WAIT_TIMEOUT_S = 5.0


def _state(pipeline_state: PipelineState | None) -> PipelineState:
    if pipeline_state is None:
        raise PipelineError("pipeline_state is required for swarm_engine operations")
    return pipeline_state


def _ordered_stages(pipeline_state: PipelineState):
    return pipeline_state.ordered_stages()


def _upstream_stage(stage_name, pipeline_state: PipelineState):
    if stage_name == "swap":
        return "detect"
    if stage_name == "restore":
        return "swap"
    if stage_name == "parse":
        return "restore" if pipeline_state.config.enable_restore else "swap"
    return None


def _next_hottest_stage(sizes, current_stage):
    ranked = sorted(sizes.items(), key=lambda item: item[1], reverse=True)
    for stage_name, _ in ranked:
        if stage_name != current_stage:
            return stage_name
    return current_stage


def _queue_capacities(pipeline_state: PipelineState):
    queues = pipeline_state.queues
    caps = {"detect": queues.detect.maxsize, "swap": queues.swap.maxsize}
    if pipeline_state.config.enable_restore:
        caps["restore"] = queues.restore.maxsize
    if pipeline_state.config.enable_parser:
        caps["parse"] = queues.parse.maxsize
    return caps


def _choose_hot_stage(sizes, pipeline_state: PipelineState):
    downstream_priority = {"detect": 0, "swap": 1, "restore": 2, "parse": 3}
    caps = _queue_capacities(pipeline_state)
    ranked = []
    for stage_name, size in sizes.items():
        cap = max(1, caps.get(stage_name, 1))
        pressure = float(size) / float(cap)
        ranked.append((pressure, downstream_priority.get(stage_name, 0), size, stage_name))
    ranked.sort(reverse=True)
    return ranked[0][3]


def _stage_depth(stage_name):
    depth = {"detect": 0, "swap": 1, "restore": 2, "parse": 3}
    return depth.get(stage_name, 0)


def _deepest_congested_stage(sizes, pipeline_state: PipelineState, min_pressure=0.85):
    caps = _queue_capacities(pipeline_state)
    selected = None
    for stage_name, size in sizes.items():
        cap = max(1, caps.get(stage_name, 1))
        pressure = float(size) / float(cap)
        if pressure >= min_pressure:
            if selected is None or _stage_depth(stage_name) > _stage_depth(selected):
                selected = stage_name
    return selected


def _push_tuner_status(
    now_ts, last_ui_ts, gpu_util, mode_name, hot_stage, sizes, permits, ui,
    pipeline_state: PipelineState,
):
    if now_ts - last_ui_ts < 0.4:
        return last_ui_ts

    if ui is not None:
        ui.update_tuner(
            gpu_util=gpu_util,
            mode_name=mode_name,
            hot_stage=hot_stage,
            sizes=sizes,
            permits=permits,
            ordered_stages=_ordered_stages(pipeline_state),
        )
    if pipeline_state.tuner_callback is not None:
        pipeline_state.tuner_callback(
            {
                "gpu_util": int(gpu_util),
                "mode_name": mode_name,
                "hot_stage": hot_stage,
                "sizes": dict(sizes),
                "permits": dict(permits),
                "ordered_stages": list(_ordered_stages(pipeline_state)),
            }
        )

    return now_ts


# ---------------------------------------------------------------------------
# Slot management — with deadlock protection
# ---------------------------------------------------------------------------

def wait_for_slot(stage: str, pipeline_state: PipelineState) -> bool:
    """Acquire a concurrency slot for *stage*.

    Returns True if a slot was acquired, False if the abort_event fired
    while waiting (caller should treat this as a shutdown signal).

    A bounded ``_SLOT_WAIT_TIMEOUT_S`` timeout on each ``wait()`` call
    prevents permanent deadlock when ``release_slot`` is never called due
    to an unexpected exception elsewhere in the pipeline.
    """
    with pipeline_state.slot_cond:
        while True:
            if pipeline_state.abort_event.is_set():
                return False
            if pipeline_state.stage_active[stage] < pipeline_state.stage_concurrency[stage]:
                pipeline_state.stage_active[stage] += 1
                return True
            # Timed wait: re-evaluate conditions every few seconds.
            pipeline_state.slot_cond.wait(timeout=_SLOT_WAIT_TIMEOUT_S)


def release_slot(stage: str, pipeline_state: PipelineState) -> None:
    with pipeline_state.slot_cond:
        pipeline_state.stage_active[stage] = max(0, pipeline_state.stage_active[stage] - 1)
        pipeline_state.slot_cond.notify_all()


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def worker_detect(pipeline_state: PipelineState | None = None):
    state = _state(pipeline_state)
    queues = state.queues
    model_manager = state.model_manager
    if model_manager is None:
        raise PipelineError("state.model_manager is required for worker_detect")

    while True:
        acquired = wait_for_slot("detect", state)
        if not acquired:
            break

        got_task = False
        try:
            task = queues.detect.get(timeout=1.0)
            got_task = True

            if task is None:
                queues.swap.put(None)
                break

            frame_id, frame = task
            try:
                faces = run_detect(frame, model_manager)
            except Exception:
                logger.exception("detect_failed", extra={"frame_id": frame_id})
                queues.out.put((frame_id, frame))
                continue

            if not faces:
                queues.out.put((frame_id, frame))
            else:
                queues.swap.put((frame_id, frame, faces))
        except Empty:
            pass
        finally:
            if got_task:
                queues.detect.task_done()
            release_slot("detect", state)


def worker_swap(pipeline_state: PipelineState | None = None):
    state = _state(pipeline_state)
    queues = state.queues
    model_manager = state.model_manager
    proc_cfg = state.proc_cfg          # ← immutable, no global import needed
    if model_manager is None:
        raise PipelineError("state.model_manager is required for worker_swap")

    while True:
        acquired = wait_for_slot("swap", state)
        if not acquired:
            break

        got_task = False
        try:
            task = queues.swap.get(timeout=1.0)
            got_task = True

            if task is None:
                if state.config.enable_restore:
                    queues.restore.put(None)
                elif state.config.enable_parser:
                    queues.parse.put(None)
                else:
                    queues.out.put(None)
                break

            frame_id, orig_frame, faces = task
            try:
                res_frame, swapped_faces_data = run_swap(orig_frame, faces, model_manager, proc_cfg)
            except Exception:
                logger.exception("swap_failed", extra={"frame_id": frame_id})
                queues.out.put((frame_id, orig_frame))
                continue

            if state.config.enable_restore:
                queues.restore.put((frame_id, orig_frame, swapped_faces_data))
            elif state.config.enable_parser:
                queues.parse.put((frame_id, orig_frame, swapped_faces_data))
            else:
                queues.out.put((frame_id, res_frame))
        except Empty:
            pass
        finally:
            if got_task:
                queues.swap.task_done()
            release_slot("swap", state)


def worker_restore(pipeline_state: PipelineState | None = None):
    state = _state(pipeline_state)
    queues = state.queues
    model_manager = state.model_manager
    proc_cfg = state.proc_cfg
    if model_manager is None:
        raise PipelineError("state.model_manager is required for worker_restore")

    while True:
        acquired = wait_for_slot("restore", state)
        if not acquired:
            break

        got_task = False
        try:
            task = queues.restore.get(timeout=1.0)
            got_task = True

            if task is None:
                if state.config.enable_parser:
                    queues.parse.put(None)
                else:
                    queues.out.put(None)
                break

            frame_id, orig_frame, swapped_faces_data = task
            try:
                res_frame, restored_faces_data = run_restore(
                    orig_frame, swapped_faces_data, model_manager, proc_cfg
                )
            except Exception:
                logger.exception("restore_failed", extra={"frame_id": frame_id})
                queues.out.put((frame_id, orig_frame))
                continue

            if state.config.enable_parser:
                queues.parse.put((frame_id, orig_frame, restored_faces_data))
            else:
                queues.out.put((frame_id, res_frame))
        except Empty:
            pass
        finally:
            if got_task:
                queues.restore.task_done()
            release_slot("restore", state)


def worker_parse(pipeline_state: PipelineState | None = None):
    state = _state(pipeline_state)
    queues = state.queues
    model_manager = state.model_manager
    proc_cfg = state.proc_cfg
    if model_manager is None:
        raise PipelineError("state.model_manager is required for worker_parse")

    while True:
        acquired = wait_for_slot("parse", state)
        if not acquired:
            break

        got_task = False
        try:
            task = queues.parse.get(timeout=1.0)
            got_task = True

            if task is None:
                queues.out.put(None)
                break

            frame_id, orig_frame, previous_faces_data = task
            try:
                res_frame = run_parse(orig_frame, previous_faces_data, model_manager, proc_cfg)
            except Exception:
                logger.exception("parse_failed", extra={"frame_id": frame_id})
                queues.out.put((frame_id, orig_frame))
                continue

            queues.out.put((frame_id, res_frame))
        except Empty:
            pass
        finally:
            if got_task:
                queues.parse.task_done()
            release_slot("parse", state)


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------

def swarm_tuner(
    stop_event,
    get_gpu_utilization,
    max_workers,
    ui=None,
    pipeline_state: PipelineState | None = None,
):
    state = _state(pipeline_state)
    queues = state.queues
    high_watermark = int(state.config.high_watermark)
    low_watermark = int(state.config.low_watermark)
    switch_cooldown_s = float(state.config.switch_cooldown_s)
    target_util = state.config.gpu_target_util

    mode = "normal"
    drain_stage = None
    last_switch_ts = 0.0
    last_ui_ts = 0.0

    while not stop_event.is_set():
        sizes = {"detect": queues.detect.qsize(), "swap": queues.swap.qsize()}
        if state.config.enable_restore:
            sizes["restore"] = queues.restore.qsize()
        if state.config.enable_parser:
            sizes["parse"] = queues.parse.qsize()

        hottest_stage = _choose_hot_stage(sizes, state)
        now_ts = time.time()

        try:
            gpu_util = get_gpu_utilization()
        except Exception:
            logger.exception("gpu_utilization_probe_failed")
            gpu_util = 0

        if mode == "normal":
            if sizes[hottest_stage] > high_watermark and (now_ts - last_switch_ts) >= switch_cooldown_s:
                mode = "drain"
                drain_stage = hottest_stage
                last_switch_ts = now_ts
        else:
            congested_deep_stage = _deepest_congested_stage(sizes, state, min_pressure=0.90)
            if (
                congested_deep_stage is not None
                and congested_deep_stage != drain_stage
                and _stage_depth(congested_deep_stage) > _stage_depth(drain_stage)
                and (now_ts - last_switch_ts) >= switch_cooldown_s
            ):
                drain_stage = congested_deep_stage
                last_switch_ts = now_ts

            hottest_now = _choose_hot_stage(sizes, state)
            if (
                hottest_now != drain_stage
                and _stage_depth(hottest_now) >= _stage_depth(drain_stage)
                and sizes.get(hottest_now, 0) >= sizes.get(drain_stage, 0) + 6
                and (now_ts - last_switch_ts) >= switch_cooldown_s
            ):
                drain_stage = hottest_now
                last_switch_ts = now_ts

            current_size = sizes.get(drain_stage, 0)
            if current_size <= low_watermark and (now_ts - last_switch_ts) >= switch_cooldown_s:
                candidate_stage = _next_hottest_stage(sizes, drain_stage)
                if sizes.get(candidate_stage, 0) > low_watermark:
                    drain_stage = candidate_stage
                    last_switch_ts = now_ts
                else:
                    mode = "normal"
                    drain_stage = None
                    last_switch_ts = now_ts

        with state.slot_cond:
            if mode == "drain" and drain_stage in state.stage_concurrency:
                if drain_stage == "detect" and queues.swap.qsize() >= int(queues.swap.maxsize * 0.80):
                    drain_stage = "swap"
                    last_switch_ts = now_ts

                for stage in state.stage_concurrency:
                    state.stage_concurrency[stage] = 2
                state.stage_concurrency[drain_stage] = max_workers
                feeder = _upstream_stage(drain_stage, state)
                if feeder in state.stage_concurrency:
                    state.stage_concurrency[feeder] = 1
            else:
                if state.config.tuner_mode == "stable":
                    low_threshold = max(70, target_util - 15)
                    high_threshold = max(80, target_util - 5)
                    if gpu_util < low_threshold:
                        base_quota = max(1, (max_workers // 2))
                    elif gpu_util < high_threshold:
                        base_quota = max(1, (max_workers // 2) + 1)
                    else:
                        base_quota = max(1, (max_workers // 3) + 1)
                elif state.config.tuner_mode == "max_util":
                    low_threshold = max(70, target_util - 10)
                    high_threshold = max(80, target_util - 2)
                    if gpu_util < low_threshold:
                        base_quota = max(2, max_workers - 1)
                    elif gpu_util < high_threshold:
                        base_quota = max(2, max_workers - 2)
                    else:
                        base_quota = max(1, (max_workers // 2) + 1)
                else:
                    low_threshold = max(70, target_util - 8)
                    high_threshold = max(80, target_util + 1)
                    if gpu_util < low_threshold:
                        base_quota = max(2, max_workers - 1)
                    elif gpu_util < high_threshold:
                        base_quota = max(2, max_workers - 2)
                    else:
                        base_quota = max(1, (max_workers // 2) + 1)
                for stage in state.stage_concurrency:
                    state.stage_concurrency[stage] = base_quota
                state.stage_concurrency[hottest_stage] = max_workers

                if queues.detect.qsize() > int(queues.detect.maxsize * 0.85):
                    state.stage_concurrency["detect"] = 1
                    state.stage_concurrency["swap"] = max_workers
                if queues.swap.qsize() > int(queues.swap.maxsize * 0.85):
                    state.stage_concurrency["detect"] = 1
                    state.stage_concurrency["swap"] = max_workers
                    if state.config.enable_restore:
                        state.stage_concurrency["restore"] = max_workers
                if state.config.enable_restore and queues.restore.qsize() > int(queues.restore.maxsize * 0.85):
                    state.stage_concurrency["detect"] = 1
                    state.stage_concurrency["swap"] = 2
                    state.stage_concurrency["restore"] = max_workers
                    if state.config.enable_parser:
                        state.stage_concurrency["parse"] = max_workers

            state.slot_cond.notify_all()

        hot_for_ui = drain_stage if mode == "drain" and drain_stage is not None else hottest_stage
        last_ui_ts = _push_tuner_status(
            now_ts, last_ui_ts, int(gpu_util), mode, hot_for_ui,
            sizes, state.stage_concurrency, ui, state,
        )
        time.sleep(0.08)


# ---------------------------------------------------------------------------
# Worker launcher
# ---------------------------------------------------------------------------

def start_swarm_workers(workers_per_stage=3, pipeline_state: PipelineState | None = None):
    state = _state(pipeline_state)
    threads = []
    for _ in range(workers_per_stage):
        threads.append(
            threading.Thread(target=worker_detect, kwargs={"pipeline_state": state}, name="detect-worker")
        )
        threads.append(
            threading.Thread(target=worker_swap, kwargs={"pipeline_state": state}, name="swap-worker")
        )
        if state.config.enable_restore:
            threads.append(
                threading.Thread(target=worker_restore, kwargs={"pipeline_state": state}, name="restore-worker")
            )
        if state.config.enable_parser:
            threads.append(
                threading.Thread(target=worker_parse, kwargs={"pipeline_state": state}, name="parse-worker")
            )

    for thread in threads:
        thread.start()

    return threads
