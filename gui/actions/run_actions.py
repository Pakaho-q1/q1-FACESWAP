from __future__ import annotations

import time
from typing import Any, Callable


class StoreRef:
    def __init__(self, store: Any, key: str) -> None:
        self._store = store
        self._key = key

    def set(self, value: Any) -> None:
        setattr(self._store, self._key, value)

    def get(self) -> Any:
        return getattr(self._store, self._key)


def run_pipeline_action(
    *,
    run_handler: Callable[[dict[str, Any]], None],
    collect_values: Callable[[], dict[str, Any]],
    validate_form_inline: Callable[[], bool],
    ui: Any,
    set_health: Callable[[], dict[str, Any]],
    open_tensorrt_dialog: Callable[[], bool],
    set_queue_preview: Callable[[], None],
    save_project_settings: Callable[[str, dict[str, Any]], Any],
    project_root: str,
    controller: Any,
    store: Any,
    preview_engine: Any,
    preview_reset_js: Callable[[], None],
    throughput_fps: Any,
    throughput_items_min: Any,
    throughput_eta: Any,
    update_job_queue_status: Callable[[], None],
    progress_bar: Any,
    progress_label: Any,
    progress_count: Any,
    progress_pct: Any,
    run_btn: Any,
    stop_btn: Any,
    append_log: Callable[[str], None],
) -> None:
    action_ctx = {
        "collect_values": collect_values,
        "validate_form_inline": validate_form_inline,
        "ui": ui,
        "set_health": set_health,
        "open_tensorrt_dialog": open_tensorrt_dialog,
        "set_queue_preview": set_queue_preview,
        "save_project_settings": save_project_settings,
        "project_root": project_root,
        "controller": controller,
        "store": store,
        "run_started_at_ref": StoreRef(store, "run_started_at"),
        "processing_started_at_ref": StoreRef(store, "processing_started_at"),
        "last_progress_done_ref": StoreRef(store, "last_progress_done"),
        "last_progress_ts_ref": StoreRef(store, "last_progress_ts"),
        "progress_units_done_ref": StoreRef(store, "progress_units_done"),
        "progress_units_total_ref": StoreRef(store, "progress_units_total"),
        "progress_units_label_ref": StoreRef(store, "progress_units_label"),
        "last_pipeline_metrics_ref": StoreRef(store, "last_pipeline_metrics"),
        "latest_outqueue_write_fps_ref": StoreRef(store, "latest_outqueue_write_fps"),
        "last_preview_render_ts_ref": StoreRef(store, "last_preview_render_ts"),
        "last_preview_signature_ref": StoreRef(store, "last_preview_signature"),
        "preview_active_layer_ref": StoreRef(store, "preview_active_layer"),
        "preview_engine": preview_engine,
        "preview_reset_js": preview_reset_js,
        "throughput_fps": throughput_fps,
        "throughput_items_min": throughput_items_min,
        "throughput_eta": throughput_eta,
        "update_job_queue_status": update_job_queue_status,
        "progress_bar": progress_bar,
        "progress_label": progress_label,
        "progress_count": progress_count,
        "progress_pct": progress_pct,
        "run_btn": run_btn,
        "stop_btn": stop_btn,
        "append_log": append_log,
    }
    run_handler(action_ctx)


def stop_pipeline_action(
    *,
    request_stop_handler: Callable[[dict[str, Any]], None],
    controller: Any,
    append_log: Callable[[str], None],
    stop_btn: Any,
    ui: Any,
) -> None:
    request_stop_handler(
        {
            "controller": controller,
            "append_log": append_log,
            "stop_btn": stop_btn,
            "ui": ui,
        }
    )


def apply_preview_fps_limit(preview_fps_limit_control: Any, preview_engine: Any) -> None:
    try:
        value = max(0.5, min(30.0, float(preview_fps_limit_control.value or 2.5)))
    except Exception:  # noqa: BLE001
        value = 2.5
    preview_engine.set_base_fps(value)
