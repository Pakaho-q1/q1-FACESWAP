from __future__ import annotations

from dataclasses import replace
import logging
import threading
import warnings
import json
import os
from typing import Any, Callable, Dict, Optional

from core.errors import ConfigError, FaceSwapError, PipelineError
from core.pipeline_state import create_pipeline_state
from core.types import RunConfig, RuntimeContext, build_run_config_from_cfg, validate_run_config
from core.ui_log import ui_print


logger = logging.getLogger(__name__)
_GUI_EVENT_PREFIX = "__Q1_GUI__"


def _emit_gui_line(payload: Dict[str, Any]) -> None:
    try:
        print(f"{_GUI_EVENT_PREFIX}{json.dumps(payload, ensure_ascii=False)}", flush=True)
    except Exception:
        # Telemetry channel must never break pipeline execution.
        pass


def _attach_gui_telemetry_hooks(runtime_ctx: RuntimeContext) -> None:
    if os.environ.get("Q1_GUI_EVENTS", "").strip() != "1":
        return

    previous_on_event = runtime_ctx.hooks.on_event
    previous_on_progress = runtime_ctx.hooks.on_progress

    def _on_event(name: str, payload: Dict[str, Any]) -> None:
        _emit_gui_line({"type": "event", "name": str(name), "payload": dict(payload or {})})
        if previous_on_event is not None:
            previous_on_event(name, payload or {})

    def _on_progress(label: str, completed: int, total: int) -> None:
        _emit_gui_line(
            {
                "type": "progress",
                "label": str(label),
                "completed": int(completed),
                "total": int(total),
            }
        )
        if previous_on_progress is not None:
            previous_on_progress(label, int(completed), int(total))

    runtime_ctx.hooks.on_event = _on_event
    runtime_ctx.hooks.on_progress = _on_progress


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def build_gpu_probe():
    try:
        import pynvml

        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        def get_gpu_utilization():
            try:
                return pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
            except Exception:
                logger.exception("gpu_utilization_read_failed")
                return 0

        ui_print("? GPU verification system available", "[OK] GPU verification system available")
        return get_gpu_utilization
    except Exception:
        def get_gpu_utilization():
            return 0

        ui_print(
            "?? Could not find pynvml or NVIDIA GPU, only queue counting is used.",
            "[WARN] Could not find pynvml or NVIDIA GPU, only queue counting is used.",
        )
        return get_gpu_utilization


def run_pipeline(
    cfg_module=None,
    runtime_ctx=None,
    get_gpu_utilization: Optional[Callable[[], int]] = None,
    runtime_ui=None,
    external_stop_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    warnings.filterwarnings("ignore", category=FutureWarning)

    if runtime_ctx is None:
        if cfg_module is None:
            import core.config as cfg_module
        if hasattr(cfg_module, "ensure_cli_initialized"):
            cfg_module.ensure_cli_initialized()
        runtime_ctx = RuntimeContext(config=build_run_config_from_cfg(cfg_module))
    elif cfg_module is None:
        import core.config as cfg_module

    validate_run_config(runtime_ctx.config)
    _attach_gui_telemetry_hooks(runtime_ctx)

    logging.basicConfig(
        level=getattr(logging, cfg_module.LOG_LEVEL, logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if getattr(cfg_module, "PRINT_EFFECTIVE_CONFIG", False):
        effective = {}
        if hasattr(cfg_module, "get_effective_config"):
            effective = cfg_module.get_effective_config()
        ui_print(
            "[EFFECTIVE_CONFIG]\n" + json.dumps(effective, indent=2, ensure_ascii=False),
            "[EFFECTIVE_CONFIG]\n" + json.dumps(effective, indent=2, ensure_ascii=False),
        )

    if cfg_module.DRY_RUN:
        ui_print("[DRY_RUN] Configuration validated successfully. Exiting before inference.")
        return {"dry_run": True}

    # Import heavy runtime modules only when inference is requested.
    from core.io_image import process_images
    from core.io_video import process_videos
    from core.model_manager import init_models
    from core.orchestrator import run_job
    from core.runtime_ui import RuntimeUI
    from core.swarm_engine import start_swarm_workers, swarm_tuner
    workers_per_stage = runtime_ctx.config.workers_per_stage
    pipeline_state = create_pipeline_state(runtime_ctx.config)
    pipeline_state.progress_callback = runtime_ctx.emit_progress
    pipeline_state.write_fps_callback = lambda payload: runtime_ctx.emit_event("write_fps", payload)
    pipeline_state.tuner_callback = lambda payload: runtime_ctx.emit_event("tuner_status", payload)
    pipeline_state.preview_callback = lambda payload: runtime_ctx.emit_event("preview", payload)
    pipeline_state.preview_enabled = _env_bool("Q1_GUI_PREVIEW_ENABLED", True)
    preview_fps = max(0.5, min(30.0, _env_float("Q1_GUI_PREVIEW_FPS", 2.5)))
    pipeline_state.preview_interval_s = 1.0 / preview_fps
    manager = init_models(runtime_ctx)
    pipeline_state.model_manager = manager

    owns_ui = runtime_ui is None
    if runtime_ui is None:
        runtime_ui = RuntimeUI()
        runtime_ui.start()

    if get_gpu_utilization is None:
        get_gpu_utilization = build_gpu_probe()

    stop_tuner = threading.Event()
    cancel_thread = None
    if external_stop_event is not None:
        def _watch_external_stop() -> None:
            external_stop_event.wait()
            if external_stop_event.is_set():
                pipeline_state.request_abort()
                stop_tuner.set()

        cancel_thread = threading.Thread(
            target=_watch_external_stop,
            name="swarm-external-stop",
            daemon=True,
        )
        cancel_thread.start()

    tuner_thread = threading.Thread(
        target=swarm_tuner,
        args=(stop_tuner, get_gpu_utilization, workers_per_stage, runtime_ui, pipeline_state),
        name="swarm-tuner",
    )
    tuner_thread.start()

    ui_print(
        f"\n?? Factory opening Swarm {workers_per_stage}x{workers_per_stage}...",
        f"\nFactory opening Swarm {workers_per_stage}x{workers_per_stage}...",
    )
    ui_print(
        f"?? Queue Config: worker={runtime_ctx.config.worker_queue_size} | out={runtime_ctx.config.out_queue_size}",
        f"Queue Config: worker={runtime_ctx.config.worker_queue_size} | out={runtime_ctx.config.out_queue_size}",
    )
    ui_print(
        f"?? Tuner: mode={runtime_ctx.config.tuner_mode} | target_gpu={runtime_ctx.config.gpu_target_util}%",
        f"Tuner: mode={runtime_ctx.config.tuner_mode} | target_gpu={runtime_ctx.config.gpu_target_util}%",
    )

    worker_threads = start_swarm_workers(workers_per_stage, pipeline_state=pipeline_state)
    try:
        runtime_ctx.emit_event("pipeline_start", {"workers_per_stage": workers_per_stage})
        result_metrics = run_job(
            ctx=runtime_ctx,
            process_images_fn=process_images,
            process_videos_fn=process_videos,
            get_gpu_utilization=get_gpu_utilization,
            runtime_ui=runtime_ui,
            pipeline_state=pipeline_state,
        )
    except FaceSwapError:
        pipeline_state.request_abort()
        raise
    except Exception as exc:
        pipeline_state.request_abort()
        raise PipelineError(str(exc)) from exc
    finally:
        pipeline_state.request_abort()
        stop_tuner.set()
        for thread in worker_threads:
            thread.join(timeout=10.0)
            if thread.is_alive():
                logger.warning("worker_join_timeout", extra={"thread": thread.name})
        tuner_thread.join(timeout=10.0)
        if tuner_thread.is_alive():
            logger.warning("tuner_join_timeout", extra={"thread": tuner_thread.name})
        if manager is not None:
            try:
                manager.close()
            except Exception:
                logger.exception("model_manager_close_failed")
        runtime_ctx.models.manager = None
        runtime_ctx.models.app = None
        runtime_ctx.models.swapper = None
        runtime_ctx.models.restore_session = None
        runtime_ctx.models.parser_session = None
        if owns_ui:
            runtime_ui.stop()
        if cancel_thread is not None and cancel_thread.is_alive():
            cancel_thread.join(timeout=1.0)

    runtime_ctx.emit_event("pipeline_complete", {"metrics": dict(result_metrics)})
    ui_print("\n?? All work completed. Pipeline complete.", "\nAll work completed. Pipeline complete.")
    return result_metrics


def run_image_job(
    config: RunConfig,
    get_gpu_utilization: Optional[Callable[[], int]] = None,
    runtime_ui=None,
) -> Dict[str, Any]:
    image_cfg = replace(config, format_is_image=True)
    return run_pipeline(
        runtime_ctx=RuntimeContext(config=image_cfg),
        get_gpu_utilization=get_gpu_utilization,
        runtime_ui=runtime_ui,
    )


def run_video_job(
    config: RunConfig,
    get_gpu_utilization: Optional[Callable[[], int]] = None,
    runtime_ui=None,
) -> Dict[str, Any]:
    video_cfg = replace(config, format_is_image=False)
    return run_pipeline(
        runtime_ctx=RuntimeContext(config=video_cfg),
        get_gpu_utilization=get_gpu_utilization,
        runtime_ui=runtime_ui,
    )


def resume_pipeline_job(
    config: RunConfig,
    get_gpu_utilization: Optional[Callable[[], int]] = None,
    runtime_ui=None,
) -> Dict[str, Any]:
    # Current orchestrator recovery is integrated in run_job, so resume maps to run.
    try:
        return run_pipeline(
            runtime_ctx=RuntimeContext(config=config),
            get_gpu_utilization=get_gpu_utilization,
            runtime_ui=runtime_ui,
        )
    except ConfigError:
        raise
