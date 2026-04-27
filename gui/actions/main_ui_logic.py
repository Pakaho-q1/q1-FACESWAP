from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time
from typing import Any

from nicegui import app, ui

try:
    from gui.actions.dialog_actions import close_then, open_dialog
    from gui.actions.gallery_actions import (
        open_gallery_preview as open_gallery_preview_action,
        open_output_in_explorer as open_output_in_explorer_action,
        render_gallery,
        render_gallery_rows,
    )
    from gui.actions.health_actions import (
        run_setup_wizard_checks_action,
        set_health_action,
    )
    from gui.actions.preview_actions import (
        flush_preview_render,
        sync_preview_card,
        toggle_preview_pause,
    )
    from gui.actions.run_actions import (
        apply_preview_fps_limit,
        run_pipeline_action,
        stop_pipeline_action,
    )
    from gui.actions.ui_handlers import poll_handler, request_stop_handler, run_handler
    from gui.runtime.preview_bridge import preview_reset_script, preview_swap_script
    from gui.services.metrics_service import (
        compute_throughput,
        format_eta,
        merge_pipeline_metrics,
        scan_selected_job_status,
    )
    from gui.services.settings_service import AutosaveCoordinator
except ImportError:
    from actions.dialog_actions import close_then, open_dialog  # type: ignore
    from actions.gallery_actions import (  # type: ignore
        open_gallery_preview as open_gallery_preview_action,
        open_output_in_explorer as open_output_in_explorer_action,
        render_gallery,
        render_gallery_rows,
    )
    from actions.health_actions import run_setup_wizard_checks_action, set_health_action  # type: ignore
    from actions.preview_actions import flush_preview_render, sync_preview_card, toggle_preview_pause  # type: ignore
    from actions.run_actions import apply_preview_fps_limit, run_pipeline_action, stop_pipeline_action  # type: ignore
    from actions.ui_handlers import poll_handler, request_stop_handler, run_handler  # type: ignore
    from runtime.preview_bridge import preview_reset_script, preview_swap_script  # type: ignore
    from services.metrics_service import compute_throughput, format_eta, merge_pipeline_metrics, scan_selected_job_status  # type: ignore
    from services.settings_service import AutosaveCoordinator  # type: ignore


class _LocalRef:
    def __init__(self, box: dict[str, Any], key: str) -> None:
        self._box = box
        self._key = key

    def set(self, value: Any) -> None:
        self._box[self._key] = value

    def get(self) -> Any:
        return self._box[self._key]


class _StoreRef:
    def __init__(self, store: Any, key: str) -> None:
        self._store = store
        self._key = key

    def set(self, value: Any) -> None:
        setattr(self._store, self._key, value)

    def get(self) -> Any:
        return getattr(self._store, self._key)


def wire_main_ui_logic(ctx: dict[str, Any]) -> None:
    controller = ctx["controller"]
    store = ctx["store"]
    defaults = ctx["defaults"]
    project_root = ctx["project_root"]
    face_names = ctx["face_names"]
    preview_engine = ctx["preview_engine"]
    preview_dom_id = ctx["preview_dom_id"]

    # UI controls and labels
    face_select = ctx["face_select"]
    face_build_name = ctx["face_build_name"]
    face_build_input_path = ctx["face_build_input_path"]
    face_build_upload = ctx["face_build_upload"]
    face_build_upload_status = ctx["face_build_upload_status"]
    face_build_clear_upload_btn = ctx["face_build_clear_upload_btn"]
    face_build_provider = ctx["face_build_provider"]
    face_build_min_images = ctx["face_build_min_images"]
    face_build_status = ctx["face_build_status"]
    face_build_open_btn = ctx["face_build_open_btn"]
    build_face_btn = ctx["build_face_btn"]
    refresh_faces_btn = ctx["refresh_faces_btn"]
    clear_face_build_btn = ctx["clear_face_build_btn"]
    format_select = ctx["format_select"]
    input_path = ctx["input_path"]
    output_path = ctx["output_path"]
    provider_all = ctx["provider_all"]
    tuner_mode = ctx["tuner_mode"]
    workers_per_stage = ctx["workers_per_stage"]
    worker_queue_size = ctx["worker_queue_size"]
    out_queue_size = ctx["out_queue_size"]
    gpu_target_util = ctx["gpu_target_util"]
    high_watermark = ctx["high_watermark"]
    low_watermark = ctx["low_watermark"]
    switch_cooldown_s = ctx["switch_cooldown_s"]
    max_frames = ctx["max_frames"]
    max_retries = ctx["max_retries"]
    parser_mask_blur = ctx["parser_mask_blur"]
    swapper_blend = ctx["swapper_blend"]
    restore_weight = ctx["restore_weight"]
    restore_blend = ctx["restore_blend"]
    restore_choice = ctx["restore_choice"]
    parser_choice = ctx["parser_choice"]
    use_swaper = ctx["use_swaper"]
    use_restore = ctx["use_restore"]
    use_parser = ctx["use_parser"]
    preserve_swap_eyes = ctx["preserve_swap_eyes"]
    dry_run = ctx["dry_run"]
    preview_enabled = ctx["preview_enabled"]
    preview_fps_limit = ctx["preview_fps_limit"]
    swapper_settings_panel = ctx["swapper_settings_panel"]
    restore_settings_panel = ctx["restore_settings_panel"]
    parser_settings_panel = ctx["parser_settings_panel"]

    validate_face_label = ctx["validate_face_label"]
    validate_input_label = ctx["validate_input_label"]
    validate_output_label = ctx["validate_output_label"]

    queue_image = ctx["queue_image"]
    queue_video = ctx["queue_video"]
    queue_selected = ctx["queue_selected"]
    queue_planned_status = ctx["queue_planned_status"]
    queue_running_status = ctx["queue_running_status"]
    queue_done_status = ctx["queue_done_status"]
    queue_failed_status = ctx["queue_failed_status"]
    queue_hint = ctx["queue_hint"]

    throughput_fps = ctx["throughput_fps"]
    throughput_items_min = ctx["throughput_items_min"]
    throughput_eta = ctx["throughput_eta"]

    tuner_gpu = ctx["tuner_gpu"]
    tuner_mode_live = ctx["tuner_mode_live"]
    tuner_hot = ctx["tuner_hot"]
    q_detect_label = ctx["q_detect_label"]
    q_swap_label = ctx["q_swap_label"]
    q_restore_label = ctx["q_restore_label"]
    q_parse_label = ctx["q_parse_label"]
    p_detect_label = ctx["p_detect_label"]
    p_swap_label = ctx["p_swap_label"]
    p_restore_label = ctx["p_restore_label"]
    p_parse_label = ctx["p_parse_label"]
    q_swap_card = ctx["q_swap_card"]
    q_restore_card = ctx["q_restore_card"]
    q_parse_card = ctx["q_parse_card"]
    p_swap_card = ctx["p_swap_card"]
    p_restore_card = ctx["p_restore_card"]
    p_parse_card = ctx["p_parse_card"]
    gpu_chart = ctx["gpu_chart"]
    queue_chart = ctx["queue_chart"]
    permit_chart = ctx["permit_chart"]

    progress_label = ctx["progress_label"]
    progress_bar = ctx["progress_bar"]
    progress_count = ctx["progress_count"]
    progress_pct = ctx["progress_pct"]
    preview_expansion = ctx["preview_expansion"]
    preview_meta = ctx["preview_meta"]

    log_view = ctx["log_view"]
    gallery_status = ctx["gallery_status"]
    gallery_tabs = ctx["gallery_tabs"]
    gallery_items = ctx["gallery_items"]

    run_btn = ctx["run_btn"]
    stop_btn = ctx["stop_btn"]
    pause_preview_btn = ctx["pause_preview_btn"]

    stop_confirm_dialog = ctx["stop_confirm_dialog"]
    stop_confirm_yes_btn = ctx["stop_confirm_yes_btn"]
    init_confirm_dialog = ctx["init_confirm_dialog"]
    init_confirm_yes_btn = ctx["init_confirm_yes_btn"]
    error_dialog = ctx["error_dialog"]
    clear_error_btn = ctx["clear_error_btn"]
    error_count_label = ctx["error_count_label"]
    error_list = ctx["error_list"]
    error_count_badge = ctx["error_count_badge"]

    model_status_dialog = ctx["model_status_dialog"]
    refresh_model_dialog_btn = ctx["refresh_model_dialog_btn"]
    download_all_models_btn = ctx["download_all_models_btn"]
    open_download_center_btn = ctx["open_download_center_btn"]
    model_status_summary = ctx["model_status_summary"]
    model_status_list = ctx["model_status_list"]
    download_center_dialog = ctx["download_center_dialog"]
    pause_resume_downloads_btn = ctx["pause_resume_downloads_btn"]
    clear_finished_downloads_btn = ctx["clear_finished_downloads_btn"]
    download_center_summary = ctx["download_center_summary"]
    download_center_list = ctx["download_center_list"]
    tensorrt_dialog = ctx["tensorrt_dialog"]
    trt_missing_label = ctx["trt_missing_label"]
    trt_target_label = ctx["trt_target_label"]

    gallery_preview_title = ctx["gallery_preview_title"]
    gallery_popup_image = ctx["gallery_popup_image"]
    gallery_popup_video = ctx["gallery_popup_video"]
    gallery_preview_dialog = ctx["gallery_preview_dialog"]

    setup_wizard_btn = ctx["setup_wizard_btn"]
    init_btn = ctx["init_btn"]
    wizard_run_checks_btn = ctx["wizard_run_checks_btn"]
    setup_wizard_dialog = ctx["setup_wizard_dialog"]
    face_build_dialog = ctx["face_build_dialog"]
    wizard_project_status = ctx["wizard_project_status"]
    wizard_model_status = ctx["wizard_model_status"]
    wizard_trt_status = ctx["wizard_trt_status"]

    refresh_health_btn = ctx["refresh_health_btn"]
    refresh_queue_btn = ctx["refresh_queue_btn"]
    refresh_gallery_btn = ctx["refresh_gallery_btn"]
    health_models_card = ctx["health_models_card"]
    health_tensorrt_card = ctx["health_tensorrt_card"]
    health_models = ctx["health_models"]
    health_ffmpeg = ctx["health_ffmpeg"]
    health_tensorrt = ctx["health_tensorrt"]
    health_disk = ctx["health_disk"]
    health_writable = ctx["health_writable"]
    health_missing = ctx["health_missing"]

    open_error_btn = ctx["open_error_btn"]
    clear_btn = ctx["clear_btn"]
    download_service = ctx["download_service"]
    face_model_service = ctx["face_model_service"]

    # series/histories
    x_hist = ctx["x_hist"]
    gpu_hist = ctx["gpu_hist"]
    q_detect_hist = ctx["q_detect_hist"]
    q_swap_hist = ctx["q_swap_hist"]
    q_restore_hist = ctx["q_restore_hist"]
    q_parse_hist = ctx["q_parse_hist"]
    p_detect_hist = ctx["p_detect_hist"]
    p_swap_hist = ctx["p_swap_hist"]
    p_restore_hist = ctx["p_restore_hist"]
    p_parse_hist = ctx["p_parse_hist"]
    error_rows = ctx["error_rows"]

    planned_counts = store.planned_counts
    completed_counts = store.completed_counts
    failed_counts = store.failed_counts
    latest_health_report: dict[str, Any] = {}
    local_box: dict[str, Any] = {
        "tuner_tick": int(ctx.get("tuner_tick", 0) or 0),
        "notified_face_build_done": set(),
        "face_build_uploaded_files": [],
        "gallery_page": 1,
        "gallery_page_size": 50,
        "gallery_total_pages": 0,
        "gallery_total_rows": 0,
        "gallery_signature": "",
        "runtime_last_ui_refresh_ts": 0.0,
        "queue_hint_base": "Preview checks input path only.",
    }

    def preview_reset_js() -> None:
        ui.run_javascript(preview_reset_script(preview_dom_id))

    def sync_stage_visibility() -> None:
        swap_enabled = bool(use_swaper.value)
        restore_enabled = bool(use_restore.value)
        parser_enabled = bool(use_parser.value)
        for element in (swapper_settings_panel, q_swap_card, p_swap_card):
            element.set_visibility(swap_enabled)
        for element in (restore_settings_panel, q_restore_card, p_restore_card):
            element.set_visibility(restore_enabled)
        for element in (parser_settings_panel, q_parse_card, p_parse_card):
            element.set_visibility(parser_enabled)
        preserve_swap_eyes.set_visibility(parser_enabled)

    def preview_swap_js(data_url: str) -> None:
        ui.run_javascript(preview_swap_script(preview_dom_id, data_url))

    def enqueue_preview_payload(payload: dict[str, Any]) -> None:
        preview_engine.ingest(payload)

    def open_output_in_explorer(path_value: str) -> None:
        ok, msg = open_output_in_explorer_action(path_value)
        if not ok and msg:
            ui.notify(msg, color="warning")

    def download_all_missing_models() -> None:
        queued = download_service.download_all_missing_models(latest_health_report)
        ui.notify(
            (
                f"Queued downloads: {queued}"
                if queued > 0
                else "No missing models to download"
            ),
            color="positive" if queued > 0 else "info",
        )

    def toggle_pause_all_downloads() -> None:
        download_service.toggle_pause_all_downloads(pause_resume_downloads_btn)

    def clear_finished_downloads() -> None:
        download_service.clear_finished_downloads()

    def render_download_center() -> None:
        download_service.render_download_center(
            download_center_list, download_center_summary
        )

    def open_tensorrt_dialog() -> bool:
        return download_service.open_tensorrt_dialog(
            tensorrt_dialog, trt_missing_label, trt_target_label
        )

    def render_model_status_dialog() -> None:
        download_service.render_model_status_dialog(
            model_status_list=model_status_list,
            model_status_summary=model_status_summary,
            tensorrt_dialog=tensorrt_dialog,
            trt_missing_label=trt_missing_label,
            trt_target_label=trt_target_label,
        )

    def open_model_status_dialog() -> None:
        set_health()
        render_model_status_dialog()
        open_dialog(model_status_dialog)

    def open_gallery_preview(row: dict[str, str]) -> None:
        ok, msg = open_gallery_preview_action(
            row=row,
            output_path_value=str(output_path.value or ""),
            register_media_root=ctx["register_media_root"],
            to_media_url=ctx["to_media_url"],
            gallery_preview_title=gallery_preview_title,
            gallery_popup_video=gallery_popup_video,
            gallery_popup_image=gallery_popup_image,
            gallery_preview_dialog=gallery_preview_dialog,
        )
        if not ok and msg:
            ui.notify(msg, color="warning")

    def set_health() -> dict[str, Any]:
        nonlocal latest_health_report
        latest_health_report = set_health_action(
            project_root=project_root,
            output_path_value=str(output_path.value or ""),
            collect_runtime_health=ctx["collect_runtime_health"],
            apply_health_report=ctx["apply_health_report"],
            health_models=health_models,
            health_ffmpeg=health_ffmpeg,
            health_tensorrt=health_tensorrt,
            health_disk=health_disk,
            health_writable=health_writable,
            health_missing=health_missing,
        )
        return latest_health_report

    status_cache: dict[str, Any] = {
        "key": None,
        "value": {"total": 0, "done": 0, "planned": 0, "running": 0, "failed": 0},
        "ts": 0.0,
        "dirty": True,
        "scan_inflight": False,
        "scan_requested": False,
        "scan_seq": 0,
    }
    status_cache_lock = threading.Lock()

    def _status_cache_key() -> tuple[Any, ...]:
        return (
            str(format_select.value or "image").lower(),
            str(input_path.value or "").strip(),
            str(output_path.value or "").strip(),
            str(defaults.get("output_suffix", "") or ""),
            bool(controller.state.running),
            int(failed_counts.get("image", 0)),
            int(failed_counts.get("video", 0)),
        )

    def _scan_status_worker(
        scan_seq: int, cache_key: tuple[Any, ...], payload: dict[str, Any]
    ) -> None:
        value = scan_selected_job_status(
            selected_format=str(payload.get("selected_format", "image")),
            input_path_value=str(payload.get("input_path_value", "")),
            output_path_value=str(payload.get("output_path_value", "")),
            suffix=str(payload.get("suffix", "")),
            failed_counts=dict(payload.get("failed_counts", {})),
            controller_running=bool(payload.get("controller_running", False)),
        )
        with status_cache_lock:
            latest_seq = int(status_cache.get("scan_seq", 0))
            if scan_seq < latest_seq:
                return
            status_cache["key"] = cache_key
            status_cache["value"] = dict(value)
            status_cache["ts"] = time.monotonic()
            status_cache["dirty"] = bool(status_cache.get("scan_requested", False))
            status_cache["scan_requested"] = False
            status_cache["scan_inflight"] = False

    def _trigger_status_scan(force: bool = False) -> None:
        key = _status_cache_key()
        now = time.monotonic()
        refresh_s = 2.5 if bool(controller.state.running) else 5.0
        payload = {
            "selected_format": str(format_select.value or "image"),
            "input_path_value": str(input_path.value or ""),
            "output_path_value": str(output_path.value or ""),
            "suffix": str(defaults.get("output_suffix", "") or ""),
            "failed_counts": dict(failed_counts),
            "controller_running": bool(controller.state.running),
        }

        start_worker = False
        scan_seq = 0
        with status_cache_lock:
            if force:
                status_cache["dirty"] = True
            key_match = status_cache.get("key") == key
            age = now - float(status_cache.get("ts", 0.0))
            fresh_enough = (
                key_match
                and not bool(status_cache.get("dirty", True))
                and age < refresh_s
            )
            if fresh_enough:
                return
            if bool(status_cache.get("scan_inflight", False)):
                status_cache["scan_requested"] = True
                return
            status_cache["scan_inflight"] = True
            status_cache["scan_requested"] = False
            status_cache["scan_seq"] = int(status_cache.get("scan_seq", 0)) + 1
            scan_seq = int(status_cache["scan_seq"])
            start_worker = True

        if start_worker:
            worker = threading.Thread(
                target=_scan_status_worker,
                args=(scan_seq, key, payload),
                daemon=True,
            )
            worker.start()

    def scan_selected_status(force: bool = False) -> dict[str, int]:
        _trigger_status_scan(force=force)
        with status_cache_lock:
            return dict(status_cache["value"])

    def update_job_queue_status(force: bool = False) -> None:
        status = scan_selected_status(force=force)
        queue_planned_status.set_text(str(status["planned"]))
        queue_running_status.set_text(str(status["running"]))
        queue_done_status.set_text(str(status["done"]))
        queue_failed_status.set_text(str(status["failed"]))

    def set_queue_preview() -> None:
        stats = ctx["preview_job_queue"](str(input_path.value or ""))
        planned_counts["image"] = int(stats.get("image", 0))
        planned_counts["video"] = int(stats.get("video", 0))
        queue_image.set_text(str(planned_counts["image"]))
        queue_video.set_text(str(planned_counts["video"]))
        selected_fmt = str(format_select.value or "image").lower()
        selected_count = (
            planned_counts["video"]
            if selected_fmt == "video"
            else planned_counts["image"]
        )
        queue_selected.set_text(str(selected_count))
        update_job_queue_status(force=True)
        status = scan_selected_status(force=False)
        local_box["queue_hint_base"] = (
            f"Input scan: images={planned_counts['image']} videos={planned_counts['video']} "
            f"(format={selected_fmt}) | done={status['done']} pending={status['planned']}"
        )
        queue_hint.set_text(local_box["queue_hint_base"])

    def set_controller_metrics(metrics: dict[str, Any]) -> None:
        queue_depth = int(metrics.get("queue_depth", 0) or 0)
        dropped_logs = int(metrics.get("dropped_logs", 0) or 0)
        dropped_events = int(metrics.get("dropped_events", 0) or 0)
        prefix = str(local_box.get("queue_hint_base", ""))
        queue_hint.set_text(
            f"{prefix} | ctl_q={queue_depth} drop(log/event)={dropped_logs}/{dropped_events}"
        )

    def refresh_face_models() -> None:
        latest_names = list(ctx["list_face_names"](project_root))
        face_names.clear()
        face_names.extend(latest_names)
        selected = str(face_select.value or "").strip()
        set_options_fn = getattr(face_select, "set_options", None)
        if callable(set_options_fn):
            set_options_fn(face_names)
        else:
            setattr(face_select, "options", face_names)
            update_fn = getattr(face_select, "update", None)
            if callable(update_fn):
                update_fn()
        if selected and selected in face_names:
            face_select.value = selected
        elif face_names:
            face_select.value = face_names[0]
        else:
            face_select.value = None
        validate_form_inline()

    def _update_face_upload_status() -> None:
        files = list(local_box["face_build_uploaded_files"])
        face_build_upload_status.set_text(f"Uploaded: {len(files)} files")

    def _sanitize_uploaded_filename(raw_name: str, index: int) -> str:
        name = Path(str(raw_name or "").strip()).name
        if not name:
            name = f"upload_{index}.png"
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
        if not safe_name:
            safe_name = f"upload_{index}.png"
        return safe_name

    async def _on_face_upload(event: Any) -> None:
        upload_file = getattr(event, "file", None)
        if upload_file is None:
            return
        try:
            raw_data = await upload_file.read()
        except Exception:
            ui.notify("Failed reading uploaded file", color="negative")
            return
        if not isinstance(raw_data, (bytes, bytearray)):
            ui.notify("Uploaded payload is invalid", color="negative")
            return
        filename = _sanitize_uploaded_filename(
            getattr(upload_file, "name", ""),
            len(local_box["face_build_uploaded_files"]) + 1,
        )
        local_box["face_build_uploaded_files"].append(
            {"name": filename, "data": bytes(raw_data)}
        )
        _update_face_upload_status()

    def _clear_uploaded_face_files() -> None:
        local_box["face_build_uploaded_files"] = []
        _update_face_upload_status()
        ui.notify("Cleared uploaded face images", color="info")

    def _prepare_face_build_input_dir() -> str:
        uploaded_files = list(local_box["face_build_uploaded_files"])
        if not uploaded_files:
            return str(face_build_input_path.value or "").strip()
        upload_root = Path(project_root) / "assets" / "runtime" / "face_build_uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="face_build_", dir=str(upload_root)))
        for idx, item in enumerate(uploaded_files, start=1):
            file_name = _sanitize_uploaded_filename(str(item.get("name", "")), idx)
            payload = bytes(item.get("data", b""))
            (temp_dir / file_name).write_bytes(payload)
        return str(temp_dir)

    def start_face_model_build() -> None:
        name = str(face_build_name.value or "").strip()
        src = _prepare_face_build_input_dir()
        if not name:
            ui.notify("Face model name is required", color="warning")
            return
        if not src:
            ui.notify("Face images path is required", color="warning")
            return
        started = face_model_service.start_build(
            face_name=name,
            input_dir=src,
            provider=str(face_build_provider.value or "trt"),
            min_accepted_images=int(face_build_min_images.value or 1),
            overwrite=True,
        )
        if not started:
            ui.notify(
                "Face model build is already running for this name", color="warning"
            )
            return
        face_build_status.set_text(f"[{name}] queued")
        if local_box["face_build_uploaded_files"]:
            ui.notify(
                "Face model build queued (using uploaded files)", color="positive"
            )
        else:
            ui.notify("Face model build queued", color="positive")

    def clear_face_build_state() -> None:
        face_model_service.clear_finished()
        face_build_status.set_text("Face model builder is idle")

    def tick_face_model_builds() -> None:
        snapshot = face_model_service.snapshot_build_state()
        if not snapshot:
            return
        ordered = sorted(snapshot.items(), key=lambda item: item[0].casefold())
        active_row = ordered[0]
        for item in ordered:
            if str(item[1].get("status", "")) in {"queued", "running"}:
                active_row = item
                break
        face_name, state_row = active_row
        status = str(state_row.get("status", ""))
        progress = float(state_row.get("progress", 0.0) or 0.0)
        detail = str(state_row.get("detail", "") or "")
        face_build_status.set_text(
            f"[{face_name}] {status} {int(progress * 100.0):d}% {detail}".strip()
        )

        notified = local_box["notified_face_build_done"]
        completed_faces = []
        for finished_name, finished_row in ordered:
            finished_status = str(finished_row.get("status", ""))
            finished_detail = str(finished_row.get("detail", "") or "")
            if finished_status == "done" and finished_name not in notified:
                completed_faces.append(finished_name)
                notified.add(finished_name)
                ui.notify(
                    f"Face model '{finished_name}' built successfully", color="positive"
                )
            elif finished_status == "error" and finished_name not in notified:
                notified.add(finished_name)
                ui.notify(
                    f"Face model '{finished_name}' build failed: {finished_detail}",
                    color="negative",
                )
        if completed_faces:
            refresh_face_models()

    def update_throughput(done_now: int | None = None) -> None:
        now = time.perf_counter()
        if done_now is not None:
            if done_now > 0 and store.processing_started_at <= 0:
                store.processing_started_at = now
            if store.last_progress_ts > 0 and done_now >= store.last_progress_done:
                dt = max(1e-6, now - store.last_progress_ts)
                _ = (done_now - store.last_progress_done) / dt
            store.last_progress_done = done_now
            store.last_progress_ts = now

        elapsed = max(
            0.0,
            now
            - (
                store.processing_started_at
                if store.processing_started_at > 0
                else store.run_started_at
            ),
        )
        fallback_remaining = scan_selected_status(force=False)["planned"]
        t = compute_throughput(
            progress_units_done=store.progress_units_done,
            progress_units_total=store.progress_units_total,
            elapsed_seconds=elapsed,
            latest_write_fps=store.latest_outqueue_write_fps,
            fallback_remaining=fallback_remaining,
        )
        throughput_items_min.set_text(f"{float(t['items_per_min']):.1f}")
        throughput_fps.set_text(f"[{float(t['write_fps']):.1f}/fps]")
        throughput_eta.set_text(
            f"[{format_eta(float(t['eta_seconds'])).replace(':', '.')}]"
        )

    def apply_pipeline_metrics(metrics: dict[str, Any]) -> None:
        _, _, progress_units_done, progress_units_total = merge_pipeline_metrics(
            planned_counts=planned_counts,
            completed_counts=completed_counts,
            progress_units_done=store.progress_units_done,
            progress_units_total=store.progress_units_total,
            metrics=metrics,
        )
        store.progress_units_done = progress_units_done
        store.progress_units_total = progress_units_total
        update_job_queue_status(force=True)
        update_throughput()

    def validate_form_inline() -> bool:
        return ctx["validate_form"](
            face_name=str(face_select.value or ""),
            face_names=face_names,
            input_path_value=str(input_path.value or ""),
            output_path_value=str(output_path.value or ""),
            validate_face_label=validate_face_label,
            validate_input_label=validate_input_label,
            validate_output_label=validate_output_label,
            run_btn=run_btn,
            controller_running=bool(controller.state.running),
        )

    def _render_gallery_tab_buttons(total_pages: int) -> None:
        clear_tabs = getattr(gallery_tabs, "clear", None)
        if callable(clear_tabs):
            clear_tabs()
        if total_pages <= 1:
            return

        current_page = int(local_box["gallery_page"])
        start = max(1, current_page - 4)
        end = min(total_pages, start + 8)
        if end - start < 8:
            start = max(1, end - 8)

        with gallery_tabs:
            for page_no in range(start, end + 1):
                is_active = page_no == current_page
                btn_color = "primary" if is_active else "secondary"
                btn_props = "dense" if is_active else "dense flat"
                ui.button(
                    f"Tab {page_no}",
                    on_click=lambda _e=None, p=page_no: _open_gallery_page(p),
                    color=btn_color,
                ).props(btn_props).classes("text-xs")

    def _render_gallery_page(page_payload: dict[str, Any]) -> None:
        rows = list(page_payload.get("rows", []))
        total = int(page_payload.get("total", 0) or 0)
        page = int(page_payload.get("page", 1) or 1)
        total_pages = int(page_payload.get("total_pages", 0) or 0)
        stats = dict(page_payload.get("stats", {}))

        local_box["gallery_page"] = page
        local_box["gallery_total_pages"] = total_pages
        local_box["gallery_total_rows"] = total

        first_name = str(rows[0]["name"]) if rows else ""
        last_name = str(rows[-1]["name"]) if rows else ""
        signature = f"{total}:{page}:{total_pages}:{first_name}:{last_name}:{int(bool(stats.get('scan_inflight', False)))}"
        if signature == str(local_box.get("gallery_signature", "")):
            return
        local_box["gallery_signature"] = signature

        render_gallery_rows(
            rows=rows,
            gallery_items=gallery_items,
            on_view=open_gallery_preview,
            on_open=open_output_in_explorer,
        )
        _render_gallery_tab_buttons(total_pages)
        if total <= 0:
            gallery_status.set_text("No output files found")
            return
        scan_state = "scanning" if bool(stats.get("scan_inflight", False)) else "ready"
        duration_ms = float(stats.get("last_scan_duration_ms", 0.0) or 0.0)
        queue_depth = int(stats.get("queue_depth", 0) or 0)
        gallery_status.set_text(
            f"{total} outputs | tab {page}/{total_pages} | showing {len(rows)} items | "
            f"scan={scan_state} {duration_ms:.0f}ms q={queue_depth}"
        )

    def _open_gallery_page(page_no: int) -> None:
        local_box["gallery_page"] = max(1, int(page_no))
        set_gallery(force_scan=False)

    def set_gallery(force_scan: bool = False) -> None:
        output_path_value = str(output_path.value or "")
        ctx["register_media_root"](output_path_value)
        page_payload = ctx["list_output_gallery_page"](
            project_root=project_root,
            output_path=output_path_value,
            page=int(local_box["gallery_page"]),
            page_size=int(local_box["gallery_page_size"]),
            force_scan=bool(force_scan),
        )
        _render_gallery_page(page_payload)

    def add_error_row(source: str, message: str, detail: str = "") -> None:
        row = {"source": source, "message": message.strip(), "detail": detail.strip()}
        error_rows.appendleft(row)
        error_count_label.set_text(f"{len(error_rows)} critical entries")
        error_count_badge.set_text(f"Errors: {len(error_rows)}")
        clear_items = getattr(error_list, "clear", None)
        if callable(clear_items):
            clear_items()
        for idx, entry in enumerate(error_rows):
            with error_list:
                with ui.row().classes("w-full items-center gap-2"):
                    ui.badge(entry["source"]).props("outline").classes("text-rose-700")
                    ui.label(entry["message"][:180]).classes(
                        "text-xs text-rose-900 grow"
                    )
                    if entry["detail"]:
                        ui.button(
                            "Details",
                            on_click=lambda _e=None, d=entry["detail"]: ui.notify(
                                d, multi_line=True, timeout=8000
                            ),
                            color="negative",
                        ).props("flat dense")
            if idx >= 39:
                break

    def clear_errors() -> None:
        error_rows.clear()
        error_count_label.set_text("0 critical entries")
        error_count_badge.set_text("Errors: 0")
        clear_items = getattr(error_list, "clear", None)
        if callable(clear_items):
            clear_items()

    def do_initialize_structure() -> None:
        ok, reason = ctx["validate_project_path"](project_root)
        if not ok:
            ui.notify(reason, color="negative")
            return
        free_gb = ctx["disk_space_gb"](project_root)
        ctx["ensure_workspace"](project_root)
        set_health()
        if free_gb >= 0 and free_gb < 10.0:
            ui.notify(
                f"Project structure initialized, but low disk space: {free_gb:.1f} GB free",
                color="warning",
            )
        else:
            ui.notify("Project structure initialized successfully", color="positive")

    def append_log(message: str) -> None:
        stamped = store.append_log(message)
        log_view.push(stamped)
        lower = message.lower()
        if (
            "traceback" in lower
            or "error" in lower
            or "failed" in lower
            or "exception" in lower
            or "critical" in lower
        ):
            add_error_row("log", message)

    def collect_values() -> dict[str, Any]:
        selected_face = str(face_select.value or "").strip()
        return {
            "project_path": project_root,
            "face_name": selected_face,
            "face_model_name": selected_face,
            "format": format_select.value or "image",
            "input_path": input_path.value or "",
            "output_path": output_path.value or "",
            "provider_all": provider_all.value or "trt",
            "tuner_mode": tuner_mode.value or "auto",
            "workers_per_stage": int(workers_per_stage.value or 8),
            "worker_queue_size": int(worker_queue_size.value or 64),
            "out_queue_size": int(out_queue_size.value or 128),
            "gpu_target_util": int(gpu_target_util.value or 95),
            "high_watermark": int(high_watermark.value or 12),
            "low_watermark": int(low_watermark.value or 4),
            "switch_cooldown_s": float(switch_cooldown_s.value or 0.35),
            "max_frames": int(max_frames.value or 0),
            "max_retries": int(max_retries.value or 2),
            "parser_mask_blur": int(parser_mask_blur.value or 21),
            "swapper_blend": float(swapper_blend.value or 0.7),
            "restore_weight": float(restore_weight.value or 0.7),
            "restore_blend": float(restore_blend.value or 0.7),
            "restore_choice": str(restore_choice.value or "1"),
            "parser_choice": str(parser_choice.value or "1"),
            "use_swaper": bool(use_swaper.value),
            "use_restore": bool(use_restore.value),
            "use_parser": bool(use_parser.value),
            "preserve_swap_eyes": bool(preserve_swap_eyes.value),
            "dry_run": bool(dry_run.value),
            "preview_enabled": bool(preview_enabled.value),
            "preview_fps_limit": float(preview_fps_limit.value or 2.5),
        }

    autosave_coordinator = AutosaveCoordinator(
        project_root=project_root,
        collect_values=collect_values,
        save_project_settings=ctx["save_project_settings"],
        default_delay_s=0.6,
    )

    def on_run() -> None:
        run_pipeline_action(
            run_handler=run_handler,
            collect_values=collect_values,
            validate_form_inline=validate_form_inline,
            ui=ui,
            set_health=set_health,
            open_tensorrt_dialog=open_tensorrt_dialog,
            set_queue_preview=set_queue_preview,
            save_project_settings=ctx["save_project_settings"],
            project_root=project_root,
            controller=controller,
            store=store,
            preview_engine=preview_engine,
            preview_reset_js=preview_reset_js,
            throughput_fps=throughput_fps,
            throughput_items_min=throughput_items_min,
            throughput_eta=throughput_eta,
            update_job_queue_status=update_job_queue_status,
            progress_bar=progress_bar,
            progress_label=progress_label,
            progress_count=progress_count,
            progress_pct=progress_pct,
            run_btn=run_btn,
            stop_btn=stop_btn,
            append_log=append_log,
        )

    def on_stop() -> None:
        if not controller.state.running:
            return
        open_dialog(stop_confirm_dialog)

    def do_stop() -> None:
        stop_pipeline_action(
            request_stop_handler=request_stop_handler,
            controller=controller,
            append_log=append_log,
            stop_btn=stop_btn,
            ui=ui,
        )

    def on_clear() -> None:
        store.clear_logs()
        clear_fn = getattr(log_view, "clear", None)
        if callable(clear_fn):
            clear_fn()
        progress_label.set_text("Progress: idle")
        progress_bar.set_value(0.0)
        progress_count.set_text("0/0")
        progress_pct.set_text("0%")
        preview_meta.set_text("Preview: waiting...")
        preview_engine.reset()
        store.preview_active_layer = "a"
        preview_reset_js()
        store.last_preview_render_ts = 0.0
        store.last_preview_signature = ""
        clear_errors()

    def on_confirm_stop() -> None:
        close_then(stop_confirm_dialog, do_stop)

    def on_confirm_initialize() -> None:
        close_then(init_confirm_dialog, do_initialize_structure)

    def on_open_error_dialog() -> None:
        open_dialog(error_dialog)

    def on_open_download_center() -> None:
        render_download_center()
        open_dialog(download_center_dialog)

    def run_setup_wizard_checks() -> None:
        run_setup_wizard_checks_action(
            project_root=project_root,
            validate_project_path=ctx["validate_project_path"],
            check_tensorrt_status=ctx["check_tensorrt_status"],
            refresh_health=set_health,
            run_setup_checks=ctx["run_setup_checks"],
            wizard_project_status=wizard_project_status,
            wizard_model_status=wizard_model_status,
            wizard_trt_status=wizard_trt_status,
        )

    def on_open_setup_wizard() -> None:
        run_setup_wizard_checks()
        open_dialog(setup_wizard_dialog)

    def tick_model_downloads() -> None:
        active = download_service.has_active_downloads()
        snapshot = download_service.snapshot_download_state()
        if bool(getattr(model_status_dialog, "value", False)):
            render_model_status_dialog()
            if active or snapshot:
                set_health()
        if bool(getattr(download_center_dialog, "value", False)):
            render_download_center()

    def tick_runtime_cards() -> None:
        now = time.monotonic()
        if (now - float(local_box.get("runtime_last_ui_refresh_ts", 0.0))) < 0.5:
            return
        local_box["runtime_last_ui_refresh_ts"] = now
        update_job_queue_status()
        set_gallery(force_scan=False)
        if controller.state.running:
            update_throughput()

    def poll_events() -> None:
        poll_handler(
            {
                "normalize_controller_event": ctx["normalize_controller_event"],
                "controller": controller,
                "preview_enabled": preview_enabled,
                "enqueue_preview_payload": enqueue_preview_payload,
                "update_throughput": update_throughput,
                "progress_bar": progress_bar,
                "progress_label": progress_label,
                "progress_count": progress_count,
                "progress_pct": progress_pct,
                "progress_units_done_ref": _StoreRef(store, "progress_units_done"),
                "progress_units_total_ref": _StoreRef(store, "progress_units_total"),
                "progress_units_label_ref": _StoreRef(store, "progress_units_label"),
                "progress_last_percent_ref": _StoreRef(store, "progress_last_percent"),
                "preview_paused_ref": _StoreRef(store, "preview_paused"),
                "latest_outqueue_write_fps_ref": _StoreRef(
                    store, "latest_outqueue_write_fps"
                ),
                "processing_started_at_ref": _StoreRef(store, "processing_started_at"),
                "store": store,
                "update_job_queue_status": update_job_queue_status,
                "add_error_row": add_error_row,
                "last_pipeline_metrics_ref": _StoreRef(store, "last_pipeline_metrics"),
                "apply_pipeline_metrics": apply_pipeline_metrics,
                "throughput_eta": throughput_eta,
                "append_log": append_log,
                "set_gallery": set_gallery,
                "preview_engine": preview_engine,
                "tuner_tick_ref": _LocalRef(local_box, "tuner_tick"),
                "x_hist": x_hist,
                "gpu_hist": gpu_hist,
                "q_detect_hist": q_detect_hist,
                "q_swap_hist": q_swap_hist,
                "q_restore_hist": q_restore_hist,
                "q_parse_hist": q_parse_hist,
                "p_detect_hist": p_detect_hist,
                "p_swap_hist": p_swap_hist,
                "p_restore_hist": p_restore_hist,
                "p_parse_hist": p_parse_hist,
                "tuner_gpu": tuner_gpu,
                "tuner_mode_live": tuner_mode_live,
                "tuner_hot": tuner_hot,
                "q_detect_label": q_detect_label,
                "q_swap_label": q_swap_label,
                "q_restore_label": q_restore_label,
                "q_parse_label": q_parse_label,
                "p_detect_label": p_detect_label,
                "p_swap_label": p_swap_label,
                "p_restore_label": p_restore_label,
                "p_parse_label": p_parse_label,
                "gpu_chart": gpu_chart,
                "queue_chart": queue_chart,
                "permit_chart": permit_chart,
                "run_btn": run_btn,
                "stop_btn": stop_btn,
                "validate_form_inline": validate_form_inline,
                "set_controller_metrics": set_controller_metrics,
            }
        )

    run_btn.on_click(on_run)
    stop_btn.on_click(on_stop)
    stop_confirm_yes_btn.on_click(on_confirm_stop)
    pause_preview_btn.on_click(
        lambda: toggle_preview_pause(
            store=store,
            preview_enabled=preview_enabled,
            preview_expansion=preview_expansion,
            pause_preview_btn=pause_preview_btn,
        )
    )
    clear_btn.on_click(on_clear)
    open_error_btn.on_click(on_open_error_dialog)
    setup_wizard_btn.on_click(on_open_setup_wizard)
    init_btn.on_click(lambda: open_dialog(init_confirm_dialog))
    init_confirm_yes_btn.on_click(on_confirm_initialize)

    health_models_card.on("click", lambda _e: open_model_status_dialog())
    health_tensorrt_card.on("click", lambda _e: open_tensorrt_dialog())
    refresh_model_dialog_btn.on_click(render_model_status_dialog)
    download_all_models_btn.on_click(download_all_missing_models)
    open_download_center_btn.on_click(on_open_download_center)
    pause_resume_downloads_btn.on_click(toggle_pause_all_downloads)
    clear_finished_downloads_btn.on_click(clear_finished_downloads)

    wizard_run_checks_btn.on_click(run_setup_wizard_checks)
    refresh_health_btn.on_click(set_health)
    refresh_queue_btn.on_click(set_queue_preview)
    refresh_gallery_btn.on_click(lambda: set_gallery(force_scan=True))
    clear_error_btn.on_click(clear_errors)
    build_face_btn.on_click(start_face_model_build)
    face_build_open_btn.on_click(lambda: open_dialog(face_build_dialog))
    refresh_faces_btn.on_click(refresh_face_models)
    face_build_clear_upload_btn.on_click(_clear_uploaded_face_files)
    clear_face_build_btn.on_click(clear_face_build_state)
    face_build_upload.on_upload(_on_face_upload)

    input_path.on_value_change(
        lambda _e: (
            set_queue_preview(),
            validate_form_inline(),
            update_job_queue_status(),
        )
    )
    output_path.on_value_change(
        lambda _e: (set_health(), set_gallery(force_scan=True), validate_form_inline())
    )
    face_select.on_value_change(lambda _e: validate_form_inline())
    format_select.on_value_change(
        lambda _e: (set_queue_preview(), update_job_queue_status(), update_throughput())
    )
    provider_all.on_value_change(
        lambda _e: (
            set_health(),
            (
                open_tensorrt_dialog()
                if str(provider_all.value or "").lower() == "trt"
                else None
            ),
        )
    )

    preview_enabled.on_value_change(
        lambda _e: sync_preview_card(
            preview_enabled=preview_enabled,
            store=store,
            preview_expansion=preview_expansion,
            pause_preview_btn=pause_preview_btn,
        )
    )
    use_swaper.on_value_change(lambda _e: sync_stage_visibility())
    use_restore.on_value_change(lambda _e: sync_stage_visibility())
    use_parser.on_value_change(lambda _e: sync_stage_visibility())
    preview_fps_limit.on_value_change(
        lambda _e: apply_preview_fps_limit(preview_fps_limit, preview_engine)
    )

    for control in [
        face_select,
        format_select,
        input_path,
        output_path,
        provider_all,
        tuner_mode,
        workers_per_stage,
        worker_queue_size,
        out_queue_size,
        gpu_target_util,
        high_watermark,
        low_watermark,
        switch_cooldown_s,
        max_frames,
        max_retries,
        parser_mask_blur,
        swapper_blend,
        restore_weight,
        restore_blend,
        restore_choice,
        parser_choice,
        use_swaper,
        use_restore,
        use_parser,
        preserve_swap_eyes,
        dry_run,
        preview_enabled,
        preview_fps_limit,
    ]:
        autosave_coordinator.bind(control)

    autosave_coordinator.schedule(0.1)
    set_health()
    set_queue_preview()
    set_gallery(force_scan=True)
    validate_form_inline()
    update_job_queue_status()
    run_setup_wizard_checks()
    sync_stage_visibility()
    sync_preview_card(
        preview_enabled=preview_enabled,
        store=store,
        preview_expansion=preview_expansion,
        pause_preview_btn=pause_preview_btn,
    )
    _update_face_upload_status()

    ui.timer(0.2, poll_events)
    ui.timer(
        0.05,
        lambda: flush_preview_render(
            preview_engine=preview_engine,
            store=store,
            preview_enabled=preview_enabled,
            preview_meta=preview_meta,
            swap_preview=preview_swap_js,
        ),
    )
    ui.timer(0.6, tick_model_downloads)
    ui.timer(0.6, tick_face_model_builds)
    ui.timer(0.6, tick_runtime_cards)

    app.on_shutdown(lambda: preview_engine.shutdown())
    app.on_shutdown(lambda: autosave_coordinator.shutdown())
    shutdown_output_index_service = ctx.get("shutdown_output_index_service")
    if callable(shutdown_output_index_service):
        app.on_shutdown(lambda: shutdown_output_index_service())
