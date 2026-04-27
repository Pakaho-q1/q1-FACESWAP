from __future__ import annotations

import time
from typing import Any

try:
    from gui.config.validator import validate_run_config
except ImportError:
    from config.validator import validate_run_config  # type: ignore

from .pipeline_actions import start_pipeline, stop_pipeline


def run_handler(ctx: dict[str, Any]) -> None:
    values = ctx["collect_values"]()
    if not ctx["validate_form_inline"]():
        ctx["ui"].notify("Please fix validation errors before run", color="warning")
        return
    config_errors, config_warnings = validate_run_config(values)
    if config_errors:
        ctx["ui"].notify(
            "Invalid run config: " + "; ".join(config_errors[:3]),
            color="negative",
            multi_line=True,
        )
        for err in config_errors:
            ctx["append_log"](f"config_validation_error: {err}")
        return
    for warn in config_warnings:
        ctx["append_log"](f"config_validation_warning: {warn}")
    ctx["set_health"]()
    if (
        str(values.get("provider_all", "")).lower() == "trt"
        and ctx["open_tensorrt_dialog"]()
    ):
        ctx["ui"].notify("TensorRT is required for provider=trt", color="warning")
        return
    ctx["set_queue_preview"]()
    ctx["save_project_settings"](ctx["project_root"], values)
    if not start_pipeline(ctx["controller"], values):
        ctx["ui"].notify("Pipeline is already running", color="warning")
        return
    # reset counters/state in store
    store = ctx.get("store")
    if store is not None:
        store.reset_for_run()

    # initialize timing/preview/metrics UI pieces
    ctx["run_started_at_ref"].set(time.perf_counter())
    ctx["processing_started_at_ref"].set(0.0)
    ctx["last_progress_done_ref"].set(0)
    ctx["last_progress_ts_ref"].set(0.0)
    ctx["progress_units_done_ref"].set(0)
    ctx["progress_units_total_ref"].set(0)
    ctx["progress_units_label_ref"].set("work")
    ctx["last_pipeline_metrics_ref"].set({})
    ctx["latest_outqueue_write_fps_ref"].set(0.0)
    ctx["last_preview_render_ts_ref"].set(0.0)
    ctx["last_preview_signature_ref"].set("")
    ctx["preview_active_layer_ref"].set("a")
    ctx["preview_engine"].reset()
    ctx["preview_reset_js"]()
    ctx["throughput_items_min"].set_text("0.0")
    ctx["throughput_fps"].set_text("[0.0/fps]")
    ctx["throughput_eta"].set_text("[--.--]")
    ctx["update_job_queue_status"]()
    ctx["progress_bar"].set_value(0.0)
    ctx["progress_label"].set_text("Progress: running")
    ctx["progress_count"].set_text("0/0")
    ctx["progress_pct"].set_text("0%")
    ctx["run_btn"].disable()
    ctx["stop_btn"].enable()
    ctx["append_log"]("Start pipeline")


def request_stop_handler(ctx: dict[str, Any]) -> None:
    # request stop via pipeline action
    if stop_pipeline(ctx["controller"]):
        ctx["append_log"]("Stop requested by user")
        ctx["stop_btn"].disable()
        ctx["ui"].notify("Stop requested", color="warning")


def clear_handler(ctx: dict[str, Any]) -> None:
    # clear logs, preview, and reset UI
    ctx["controller"].state.clear_logs()
    clear_fn = getattr(ctx["log_view"], "clear", None)
    if callable(clear_fn):
        clear_fn()
    ctx["progress_label"].set_text("Progress: idle")
    ctx["progress_bar"].set_value(0.0)
    ctx["progress_count"].set_text("0/0")
    ctx["progress_pct"].set_text("0%")
    ctx["preview_meta"].set_text("Preview: waiting...")
    ctx["preview_engine"].reset()
    ctx["preview_active_layer_ref"].set("a")
    ctx["preview_reset_js"]()
    ctx["last_preview_render_ts_ref"].set(0.0)
    ctx["last_preview_signature_ref"].set("")
    ctx["clear_errors"]()


def poll_handler(ctx: dict[str, Any]) -> None:
    # handle controller events and update UI using provided ctx
    normalize_controller_event = ctx["normalize_controller_event"]
    latest_tuner_payload: dict[str, Any] | None = None
    for raw_kind, raw_payload in ctx["controller"].poll_events():
        event = normalize_controller_event(raw_kind, raw_payload)
        if event.kind == "progress":
            done = int(event.payload.get("completed", 0))
            total = int(event.payload.get("total", 0))
            label = str(event.payload.get("label", "work"))
            ctx["progress_units_done_ref"].set(max(0, done))
            ctx["progress_units_total_ref"].set(max(0, total))
            ctx["progress_units_label_ref"].set(label)
            pct = 0.0 if total <= 0 else min(1.0, float(done) / float(total))
            ctx["progress_bar"].set_value(pct)
            progress_pct_value = int(round(pct * 100.0))
            ctx["progress_last_percent_ref"].set(progress_pct_value)
            _ = label
            ctx["progress_label"].set_text("Progress:running")
            ctx["progress_count"].set_text(f"{done}/{total}")
            ctx["progress_pct"].set_text(f"{progress_pct_value}%")
            ctx["update_throughput"](done)
        elif event.kind == "event":
            name = event.name
            data = event.payload
            if name == "preview":
                if ctx["preview_paused_ref"].get() or not bool(
                    ctx["preview_enabled"].value
                ):
                    continue
                data_url = str(data.get("data_url", ""))
                if data_url:
                    ctx["enqueue_preview_payload"](dict(data))
            elif name == "write_fps":
                ctx["latest_outqueue_write_fps_ref"].set(
                    float(data.get("write_fps", 0.0) or 0.0)
                )
                payload_written = int(
                    data.get("written", ctx["progress_units_done_ref"].get())
                    or ctx["progress_units_done_ref"].get()
                )
                payload_total = int(
                    data.get("total", ctx["progress_units_total_ref"].get())
                    or ctx["progress_units_total_ref"].get()
                )
                ctx["progress_units_done_ref"].set(
                    max(ctx["progress_units_done_ref"].get(), payload_written)
                )
                ctx["progress_units_total_ref"].set(
                    max(ctx["progress_units_total_ref"].get(), payload_total)
                )
                ctx["update_throughput"]()
            elif name == "item_started":
                if ctx["processing_started_at_ref"].get() <= 0:
                    ctx["processing_started_at_ref"].set(time.perf_counter())
                ctx["progress_label"].set_text("Progress:running")
                ctx["update_job_queue_status"]()
            elif name == "item_completed":
                kind_name = str(data.get("kind", "work")).lower()
                if kind_name in ctx["store"].completed_counts:
                    ctx["store"].completed_counts[kind_name] += 1
                ctx["progress_label"].set_text("Progress:running")
                ctx["update_job_queue_status"]()
                ctx["update_throughput"]()
            elif name == "item_failed":
                kind_name = str(data.get("kind", "work")).lower()
                reason = str(data.get("reason", "unknown"))
                item_id = str(data.get("item_id", ""))
                if kind_name in ctx["store"].failed_counts:
                    ctx["store"].failed_counts[kind_name] += 1
                ctx["add_error_row"]("event", f"{kind_name} failed: {item_id}", reason)
                ctx["progress_label"].set_text("Progress:running")
                ctx["update_job_queue_status"]()
                ctx["update_throughput"]()
            elif name == "pipeline_complete":
                metrics = dict(data.get("metrics", {}))
                ctx["last_pipeline_metrics_ref"].set(dict(metrics))
                ctx["apply_pipeline_metrics"](metrics)
                ctx["throughput_eta"].set_text("[00.00]")
                ctx["progress_label"].set_text("Progress:completed")
                ctx["append_log"](f"Pipeline metrics: {metrics}")
                ctx["set_gallery"](force_scan=True)
            elif name == "tuner_status":
                latest_tuner_payload = dict(data)
            elif name == "controller_metrics":
                set_metrics = ctx.get("set_controller_metrics")
                if callable(set_metrics):
                    set_metrics(dict(data))
        elif event.kind == "log":
            ctx["append_log"](str(event.payload.get("message", "")))
        elif event.kind == "done":
            ctx["append_log"](f"Done. Metrics: {event.payload}")
            ctx["apply_pipeline_metrics"](ctx["last_pipeline_metrics_ref"].get())
            ctx["throughput_eta"].set_text("[00.00]")
            ctx["progress_label"].set_text("Progress: completed")
            ctx["progress_bar"].set_value(1.0)
            done_total = ctx["progress_units_total_ref"].get()
            ctx["progress_count"].set_text(f"{done_total}/{done_total}")
            ctx["progress_pct"].set_text("100%")
            ctx["controller"].finish()
            ctx["run_btn"].enable()
            ctx["stop_btn"].disable()
            ctx["set_gallery"](force_scan=True)
            ctx["update_job_queue_status"]()
            ctx["validate_form_inline"]()
        elif event.kind == "stopped":
            ctx["append_log"]("Stopped by user.")
            ctx["progress_label"].set_text("Progress: stopped by user")
            ctx["controller"].finish()
            ctx["run_btn"].enable()
            ctx["stop_btn"].disable()
            ctx["update_job_queue_status"]()
            ctx["validate_form_inline"]()
        elif event.kind == "error":
            err_payload = event.payload.get("value", event.payload)
            ctx["append_log"](f"Error: {err_payload}")
            ctx["add_error_row"](
                "controller", "Unhandled pipeline error", str(err_payload)
            )
            ctx["progress_label"].set_text("Progress: failed")
            ctx["controller"].finish()
            ctx["run_btn"].enable()
            ctx["stop_btn"].disable()
            ctx["update_job_queue_status"]()
            ctx["validate_form_inline"]()

    if latest_tuner_payload is not None:
        ctx["tuner_tick_ref"].set(ctx["tuner_tick_ref"].get() + 1)
        ctx["x_hist"].append(str(ctx["tuner_tick_ref"].get()))
        gpu_util = int(latest_tuner_payload.get("gpu_util", 0))
        ctx["preview_engine"].set_gpu_util(gpu_util)
        ctx["gpu_hist"].append(gpu_util)
        sizes = dict(latest_tuner_payload.get("sizes", {}))
        permits = dict(latest_tuner_payload.get("permits", {}))
        ctx["q_detect_hist"].append(int(sizes.get("detect", 0)))
        ctx["q_swap_hist"].append(int(sizes.get("swap", 0)))
        ctx["q_restore_hist"].append(int(sizes.get("restore", 0)))
        ctx["q_parse_hist"].append(int(sizes.get("parse", 0)))
        ctx["p_detect_hist"].append(int(permits.get("detect", 0)))
        ctx["p_swap_hist"].append(int(permits.get("swap", 0)))
        ctx["p_restore_hist"].append(int(permits.get("restore", 0)))
        ctx["p_parse_hist"].append(int(permits.get("parse", 0)))
        ctx["tuner_gpu"].set_text(f"{gpu_util}%")
        ctx["tuner_mode_live"].set_text(str(latest_tuner_payload.get("mode_name", "normal")))
        ctx["tuner_hot"].set_text(str(latest_tuner_payload.get("hot_stage", "-")))
        ctx["q_detect_label"].set_text(str(sizes.get("detect", 0)))
        ctx["q_swap_label"].set_text(str(sizes.get("swap", 0)))
        ctx["q_restore_label"].set_text(str(sizes.get("restore", 0)))
        ctx["q_parse_label"].set_text(str(sizes.get("parse", 0)))
        ctx["p_detect_label"].set_text(str(permits.get("detect", 0)))
        ctx["p_swap_label"].set_text(str(permits.get("swap", 0)))
        ctx["p_restore_label"].set_text(str(permits.get("restore", 0)))
        ctx["p_parse_label"].set_text(str(permits.get("parse", 0)))
        ctx["gpu_chart"].options["xAxis"]["data"] = list(ctx["x_hist"])
        ctx["gpu_chart"].options["series"][0]["data"] = list(ctx["gpu_hist"])
        ctx["gpu_chart"].update()
        ctx["queue_chart"].options["xAxis"]["data"] = list(ctx["x_hist"])
        ctx["queue_chart"].options["series"][0]["data"] = list(ctx["q_detect_hist"])
        ctx["queue_chart"].options["series"][1]["data"] = list(ctx["q_swap_hist"])
        ctx["queue_chart"].options["series"][2]["data"] = list(ctx["q_restore_hist"])
        ctx["queue_chart"].options["series"][3]["data"] = list(ctx["q_parse_hist"])
        ctx["queue_chart"].update()
        ctx["permit_chart"].options["xAxis"]["data"] = list(ctx["x_hist"])
        ctx["permit_chart"].options["series"][0]["data"] = list(ctx["p_detect_hist"])
        ctx["permit_chart"].options["series"][1]["data"] = list(ctx["p_swap_hist"])
        ctx["permit_chart"].options["series"][2]["data"] = list(ctx["p_restore_hist"])
        ctx["permit_chart"].options["series"][3]["data"] = list(ctx["p_parse_hist"])
        ctx["permit_chart"].update()
