from __future__ import annotations

from collections import deque
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from nicegui import app, ui

# Ensure project root is importable when running as script: `python gui/main.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from gui.actions.dialog_actions import close_then, open_dialog
    from gui.actions.gallery_actions import (
        open_gallery_preview as open_gallery_preview_action,
        open_output_in_explorer as open_output_in_explorer_action,
        render_gallery,
    )
    from gui.actions.main_ui_logic import wire_main_ui_logic
    from gui.actions.ui_handlers import poll_handler, request_stop_handler, run_handler
    from gui.components.cards import metric_card
    from gui.components.dialogs import (
        build_error_dialog,
        build_gallery_preview_dialog,
        build_initialize_confirm_dialog,
        build_face_model_build_dialog,
        build_model_dialogs,
        build_setup_wizard_dialog,
        build_stop_confirm_dialog,
    )
    from gui.controller import PipelineController
    from gui.config.loader import load_gui_defaults
    from gui.project_bootstrap import (
        check_tensorrt_status,
        disk_space_gb,
        ensure_workspace,
        list_face_names,
        list_latest_outputs,
        load_project_settings,
        collect_runtime_health,
        preview_job_queue,
        read_model_manifest_status,
        save_project_settings,
        validate_project_path,
    )
    from gui.runtime.media import (
        register_media_root,
        register_media_route,
        to_media_url,
    )
    from gui.runtime.preview_bridge import (
        build_preview_container,
        preview_reset_script,
        preview_swap_script,
        register_preview_bridge_assets,
    )
    from gui.runtime.preview_engine import PreviewEngine
    from gui.services.download_service import ModelDownloadService
    from gui.services.face_model_service import FaceModelService
    from gui.services.health_validation_service import (
        apply_health_report,
        run_setup_checks,
        validate_form,
        validate_output_path,
    )
    from gui.services.metrics_service import (
        compute_throughput,
        format_eta,
        merge_pipeline_metrics,
        scan_selected_job_status,
    )
    from gui.services.settings_service import AutosaveCoordinator
    from gui.services.orchestrator import normalize_controller_event
    from gui.state import get_store, AppStore
    from gui.widgets import add_path_picker
except ImportError:
    from actions.dialog_actions import close_then, open_dialog  # type: ignore
    from actions.gallery_actions import (  # type: ignore
        open_gallery_preview as open_gallery_preview_action,
        open_output_in_explorer as open_output_in_explorer_action,
        render_gallery,
    )
    from actions.main_ui_logic import wire_main_ui_logic  # type: ignore
    from actions.ui_handlers import poll_handler, request_stop_handler, run_handler  # type: ignore
    from components.cards import metric_card  # type: ignore
    from components.dialogs import (  # type: ignore
        build_error_dialog,
        build_gallery_preview_dialog,
        build_initialize_confirm_dialog,
        build_face_model_build_dialog,
        build_model_dialogs,
        build_setup_wizard_dialog,
        build_stop_confirm_dialog,
    )
    from controller import PipelineController  # type: ignore
    from config.loader import load_gui_defaults  # type: ignore
    from project_bootstrap import (  # type: ignore
        check_tensorrt_status,
        disk_space_gb,
        ensure_workspace,
        list_face_names,
        list_latest_outputs,
        load_project_settings,
        collect_runtime_health,
        preview_job_queue,
        read_model_manifest_status,
        save_project_settings,
        validate_project_path,
    )
    from runtime.media import register_media_root, register_media_route, to_media_url  # type: ignore
    from runtime.preview_bridge import (  # type: ignore
        build_preview_container,
        preview_reset_script,
        preview_swap_script,
        register_preview_bridge_assets,
    )
    from runtime.preview_engine import PreviewEngine  # type: ignore
    from services.download_service import ModelDownloadService  # type: ignore
    from services.face_model_service import FaceModelService  # type: ignore
    from services.health_validation_service import (  # type: ignore
        apply_health_report,
        run_setup_checks,
        validate_form,
        validate_output_path,
    )
    from services.metrics_service import (  # type: ignore
        compute_throughput,
        format_eta,
        merge_pipeline_metrics,
        scan_selected_job_status,
    )
    from services.settings_service import AutosaveCoordinator  # type: ignore
    from services.orchestrator import normalize_controller_event  # type: ignore
    from state import get_store, AppStore  # type: ignore
    from widgets import add_path_picker  # type: ignore


MODEL_FALLBACK_URLS: dict[str, str] = {
    "1k3d68.onnx": "https://huggingface.co/Pakaho-q1/onnx-models/resolve/main/buffalo_l/1k3d68.onnx",
    "2d106det.onnx": "https://huggingface.co/Pakaho-q1/onnx-models/resolve/main/buffalo_l/2d106det.onnx",
    "det_10g.onnx": "https://huggingface.co/Pakaho-q1/onnx-models/resolve/main/buffalo_l/det_10g.onnx",
    "genderage.onnx": "https://huggingface.co/Pakaho-q1/onnx-models/resolve/main/buffalo_l/genderage.onnx",
    "w600k_r50.onnx": "https://huggingface.co/Pakaho-q1/onnx-models/resolve/main/buffalo_l/w600k_r50.onnx",
    "GFPGANv1.4.onnx": "https://huggingface.co/Pakaho-q1/onnx-models/resolve/main/GFPGANv1.4.onnx",
    "GPEN-BFR-1024.onnx": "https://huggingface.co/Pakaho-q1/onnx-models/resolve/main/GPEN-BFR-1024.onnx",
    "GPEN-BFR-512.onnx": "https://huggingface.co/Pakaho-q1/onnx-models/resolve/main/GPEN-BFR-512.onnx",
    "Segformer_CelebAMask-HQ.onnx": "https://huggingface.co/Pakaho-q1/onnx-models/resolve/main/Segformer_CelebAMask-HQ.onnx",
    "codeformer.onnx": "https://huggingface.co/Pakaho-q1/onnx-models/resolve/main/codeformer.onnx",
    "faceparser_resnet34.onnx": "https://huggingface.co/Pakaho-q1/onnx-models/resolve/main/faceparser_resnet34.onnx",
    "ffmpeg.exe": "https://huggingface.co/Pakaho-q1/onnx-models/resolve/main/ffmpeg.exe",
    "inswapper_128.onnx": "https://huggingface.co/Pakaho-q1/onnx-models/resolve/main/inswapper_128.onnx",
}


def build_main_ui(root: Any, project_root: str) -> None:
    register_media_route()
    controller = PipelineController(get_store())
    defaults = load_gui_defaults(project_root, load_project_settings)
    face_names = list_face_names(project_root)
    register_media_root(defaults.get("output_path", ""))
    store = get_store()

    chart_max_points = 60
    tuner_tick = 0
    gpu_hist: deque[int] = deque(maxlen=chart_max_points)
    q_detect_hist: deque[int] = deque(maxlen=chart_max_points)
    q_swap_hist: deque[int] = deque(maxlen=chart_max_points)
    q_restore_hist: deque[int] = deque(maxlen=chart_max_points)
    q_parse_hist: deque[int] = deque(maxlen=chart_max_points)
    p_detect_hist: deque[int] = deque(maxlen=chart_max_points)
    p_swap_hist: deque[int] = deque(maxlen=chart_max_points)
    p_restore_hist: deque[int] = deque(maxlen=chart_max_points)
    p_parse_hist: deque[int] = deque(maxlen=chart_max_points)
    x_hist: deque[str] = deque(maxlen=chart_max_points)
    error_rows: deque[dict[str, str]] = deque(maxlen=200)
    planned_counts = store.planned_counts
    completed_counts = store.completed_counts
    failed_counts = store.failed_counts
    progress_last_percent = store.progress_last_percent
    latest_health_report: dict[str, Any] = {}
    download_service = ModelDownloadService(
        project_root=project_root,
        model_fallback_urls=MODEL_FALLBACK_URLS,
        read_model_manifest_status=read_model_manifest_status,
        check_tensorrt_status=check_tensorrt_status,
    )
    face_model_service = FaceModelService(project_root=project_root)
    # runtime metrics live in store (SSOT)
    progress_units_done = store.progress_units_done
    progress_units_total = store.progress_units_total
    progress_units_label = store.progress_units_label
    last_pipeline_metrics: dict[str, Any] = store.last_pipeline_metrics
    latest_outqueue_write_fps = store.latest_outqueue_write_fps
    # Local copies (keep existing code using locals working while SSOT stays authoritative)
    run_started_at = store.run_started_at
    processing_started_at = store.processing_started_at
    last_progress_done = store.last_progress_done
    last_progress_ts = store.last_progress_ts
    last_preview_render_ts = store.last_preview_render_ts
    last_preview_signature = store.last_preview_signature
    preview_active_layer = store.preview_active_layer
    preview_engine = PreviewEngine(base_fps=float(defaults["preview_fps_limit"]))
    # preview_active_layer stored in store.preview_active_layer
    preview_dom_id = f"q1_preview_{int(time.time() * 1000)}"

    class _NoopText:
        def set_text(self, _value: str) -> None:
            return None

    class _SideCollapseController:
        def __init__(
            self,
            panel: Any,
            body: Any,
            header_lead: Any,
            title_label: Any,
            toggle_button: Any,
        ) -> None:
            self._panel = panel
            self._body = body
            self._header_lead = header_lead
            self._title_label = title_label
            self._toggle_button = toggle_button
            self.value = True
            self._apply()

        def set_value(self, value: bool) -> None:
            self.value = bool(value)
            self._apply()

        def _apply(self) -> None:
            if self.value:
                self._panel.classes(remove="is-collapsed")
                self._body.classes(remove="hidden")
                self._header_lead.classes(remove="hidden")
                self._title_label.classes(remove="hidden")
                self._toggle_button.set_text("</>")
                self._toggle_button.tooltip("Collapse preview")
            else:
                self._panel.classes(add="is-collapsed")
                self._body.classes(add="hidden")
                self._header_lead.classes(add="hidden")
                self._title_label.classes(add="hidden")
                self._toggle_button.set_text("</>")
                self._toggle_button.tooltip("Expand preview")

    with root:
        with ui.row().classes("app-header w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label("q1-FaceSwap").classes("text-3xl font-bold text-slate-800")
                ui.label(
                    "Latency-aware Multi-stage Orchestrator with Dynamic Worker Auto-scaling."
                ).classes("text-sm text-slate-500")
            with ui.row().classes("items-center gap-2"):
                ui.badge("Project Path").props("outline").classes("text-slate-700")
                ui.label(project_root).classes("text-xs text-slate-500")
                setup_wizard_btn = ui.button("Setup Wizard", color="primary").props(
                    "outline"
                )
                init_btn = ui.button("Initialize Project Structure", color="secondary")

        with ui.expansion(
            "Project Settings",
            icon="tune",
            value=True,
        ).classes("card-soft settings-panel w-full"):
            with ui.grid(columns=1).classes("w-full gap-3"):
                default_format = (
                    "image" if defaults["format"] not in {"video", "2"} else "video"
                )
                with ui.row().classes("items-end w-full gap-2"):
                    face_select = (
                        ui.select(
                            face_names,
                            value=(
                                defaults["face_name"]
                                if defaults["face_name"] in face_names
                                else None
                            ),
                            label="Face Model",
                        )
                        .props("clearable")
                        .classes("grow")
                    )
                    face_build_open_btn = ui.button(
                        "Build Face Model",
                        icon="construction",
                        color="secondary",
                    ).props("dense")
                format_select = ui.select(
                    {"image": "image", "video": "video"},
                    value=default_format,
                    label="Format",
                )
                with ui.row().classes("items-center w-full"):
                    input_path = (
                        ui.input("Input Path", value=defaults["input_path"])
                        .props("clearable")
                        .classes("grow")
                    )
                    add_path_picker(input_path, "Select Input Folder", pick_file=False)
                with ui.row().classes("items-center w-full"):
                    output_path = (
                        ui.input("Output Path", value=defaults["output_path"])
                        .props("clearable")
                        .classes("grow")
                    )
                    add_path_picker(
                        output_path, "Select Output Folder", pick_file=False
                    )
                provider_all = ui.select(
                    {"trt": "trt", "cuda": "cuda", "cpu": "cpu"},
                    value=defaults["provider_all"],
                    label="Provider",
                )
                tuner_mode = ui.select(
                    {"auto": "auto", "max_util": "max_util", "stable": "stable"},
                    value=defaults["tuner_mode"],
                    label="Tuner Mode",
                )
            with ui.column().classes("w-full gap-1 pt-2"):
                validate_face_label = ui.label("").classes("text-xs")
                validate_input_label = ui.label("").classes("text-xs")
                validate_output_label = ui.label("").classes("text-xs")

        with ui.expansion(
            "Processing Settings",
            icon="tune",
            value=True,
        ).classes("card-soft settings-panel w-full"):
            with ui.grid(columns=2).classes("w-full gap-3"):
                workers_per_stage = ui.number(
                    "Workers/Stage", value=defaults["workers_per_stage"], min=1, max=128
                ).props("dense outlined")
                worker_queue_size = ui.number(
                    "Worker Queue", value=defaults["worker_queue_size"], min=4, max=4096
                ).props("dense outlined")
                out_queue_size = ui.number(
                    "Out Queue", value=defaults["out_queue_size"], min=8, max=8192
                ).props("dense outlined")
                gpu_target_util = ui.number(
                    "GPU Target Util",
                    value=defaults["gpu_target_util"],
                    min=50,
                    max=100,
                ).props("dense outlined")
                high_watermark = ui.number(
                    "High Watermark", value=defaults["high_watermark"], min=1, max=4096
                ).props("dense outlined")
                low_watermark = ui.number(
                    "Low Watermark", value=defaults["low_watermark"], min=0, max=4096
                ).props("dense outlined")
                switch_cooldown_s = ui.number(
                    "Switch Cooldown (s)",
                    value=defaults["switch_cooldown_s"],
                    min=0.0,
                    max=60.0,
                ).props("dense outlined step=0.05")
                max_frames = ui.number(
                    "Max Frames", value=defaults["max_frames"], min=0, max=999999
                ).props("dense outlined")
                max_retries = ui.number(
                    "Max Retries", value=defaults["max_retries"], min=1, max=20
                ).props("dense outlined")
                preview_fps_limit = ui.number(
                    "Preview FPS",
                    value=defaults["preview_fps_limit"],
                    min=1.0,
                    max=30.0,
                ).props("dense outlined")

        # Stage controls
        with ui.expansion(
            "Stage Controls",
            icon="settings_input_component",
            value=True,
        ).classes("card-soft settings-panel w-full shadow-md text-primary"):

            with ui.grid(columns=3).classes(
                "settings-panel w-full gap-x-8 gap-y-2 items-center bg-white p-4 rounded-lg border border-slate-200"
            ):
                # Row 1
                use_swaper = ui.checkbox("Use Swapper", value=defaults["use_swaper"])
                use_restore = ui.checkbox("Use Restore", value=defaults["use_restore"])
                use_parser = ui.checkbox("Use Parser", value=defaults["use_parser"])

                # Row 2
                preserve_swap_eyes = ui.checkbox(
                    "Preserve Eyes", value=defaults["preserve_swap_eyes"]
                )
                preview_enabled = ui.checkbox(
                    "Preview", value=defaults["preview_enabled"]
                )
                dry_run = ui.checkbox("Dry Run", value=defaults["dry_run"]).classes(
                    "text-rose-600"
                )

            # Per-stage settings
            with ui.column().classes("w-full gap-4"):

                # --- Swapper Section ---
                with ui.column().classes("w-full gap-2") as swapper_settings_panel:
                    with ui.row().classes("w-full"):
                        # Stretch to fill the available width.
                        swapper_blend = (
                            ui.number(
                                "Swapper Blend",
                                value=defaults["swapper_blend"],
                                min=0.0,
                                max=1.0,
                            )
                            .props("dense outlined step=0.05")
                            .classes("w-full")
                        )

                # --- Restore Section ---
                with ui.column().classes("w-full gap-2") as restore_settings_panel:
                    # Use a 3-column grid to keep the controls balanced.
                    with ui.grid(columns=3).classes("w-full gap-3"):
                        restore_choice = (
                            ui.select(
                                {
                                    "1": "GFPGAN",
                                    "2": "GPEN-512",
                                    "3": "GPEN-1024",
                                    "4": "CodeFormer",
                                },
                                value=defaults["restore_choice"],
                                label="Restore Model",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )

                        restore_weight = (
                            ui.number(
                                "Restore Weight",
                                value=defaults["restore_weight"],
                                min=0.0,
                                max=1.0,
                            )
                            .props("dense outlined step=0.05")
                            .classes("w-full")
                        )

                        restore_blend = (
                            ui.number(
                                "Restore Blend",
                                value=defaults["restore_blend"],
                                min=0.0,
                                max=1.0,
                            )
                            .props("dense outlined step=0.05")
                            .classes("w-full")
                        )

                # --- Parser Section ---
                with ui.column().classes("w-full gap-2") as parser_settings_panel:
                    with ui.grid(columns=2).classes("w-full gap-3"):
                        parser_choice = (
                            ui.select(
                                {"1": "BiSeNet", "2": "SegFormer"},
                                value=defaults["parser_choice"],
                                label="Parser Model",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )

                        parser_mask_blur = (
                            ui.number(
                                "Parser Mask Blur",
                                value=defaults["parser_mask_blur"],
                                min=1,
                                max=255,
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )

        with ui.card().classes(
            "card-soft runtime-panel runtime-summary runtime-summary-card runtime-health w-full shadow-sm rounded-lg"
        ):
            with ui.row().classes(
                "w-full items-center justify-between q-px-sm bg-slate-50 border-b border-slate-100"
            ):
                with ui.row().classes("items-center gap-1"):
                    ui.icon("monitor_heart", color="primary").classes("text-sm")
                    ui.label("System Health").classes(
                        "text-sm font-black text-slate-500 uppercase tracking-tighter"
                    )

                refresh_health_btn = (
                    ui.button(icon="sync", color="secondary")
                    .props("flat dense size=sm")
                    .classes("text-slate-400")
                )

            # Main content container
            with ui.column().classes("w-full q-pa-none gap-1"):

                # Top row: models, FFmpeg, TensorRT
                with ui.grid(columns=3).classes("w-full q-pa-none gap-1"):
                    # 1. Models
                    with ui.row().classes(
                        "items-center no-wrap gap-1 q-px-xs border border-violet-100 rounded bg-violet-50/30"
                    ) as health_models_card:
                        ui.element("div").classes("w-1 h-3 bg-violet-400 rounded-full")
                        ui.label("MOD:").classes(
                            "text-[12px] font-bold text-violet-700"
                        )
                        health_models = ui.label("-").classes(
                            "text-[12px] font-black text-violet-900"
                        )

                    # 2. FFmpeg
                    with ui.row().classes(
                        "items-center no-wrap gap-1 q-px-xs border border-cyan-100 rounded bg-cyan-50/30"
                    ):
                        ui.element("div").classes("w-1 h-3 bg-cyan-400 rounded-full")
                        ui.label("FFM:").classes("text-[12px] font-bold text-cyan-700")
                        health_ffmpeg = ui.label("-").classes(
                            "text-[12px] font-black text-cyan-900"
                        )

                    # 3. TensorRT
                    with ui.row().classes(
                        "items-center no-wrap gap-1 q-px-xs border border-sky-100 rounded bg-sky-50/30"
                    ) as health_tensorrt_card:
                        ui.element("div").classes("w-1 h-3 bg-sky-400 rounded-full")
                        ui.label("TRT:").classes("text-[12px] font-bold text-sky-700")
                        health_tensorrt = ui.label("-").classes(
                            "text-[12px] font-black text-sky-900"
                        )

                # Bottom row: disk and writable output
                with ui.grid(columns=2).classes("w-full gap-1"):
                    # 4. Free Disk
                    with ui.row().classes(
                        "items-center no-wrap gap-1 q-px-xs border border-amber-100 rounded bg-amber-50/30"
                    ):
                        ui.element("div").classes("w-1 h-3 bg-amber-400 rounded-full")
                        ui.label("DSK:").classes("text-[12px] font-bold text-amber-700")
                        health_disk = ui.label("-").classes(
                            "text-[12px] font-black text-amber-900"
                        )

                    # 5. Writable Output
                    with ui.row().classes(
                        "items-center no-wrap gap-1 q-px-xs border border-emerald-100 rounded bg-emerald-50/30"
                    ):
                        ui.element("div").classes("w-1 h-3 bg-emerald-400 rounded-full")
                        ui.label("WRT:").classes(
                            "text-[12px] font-bold text-emerald-700"
                        )
                        health_writable = ui.label("-").classes(
                            "text-[12px] font-black text-emerald-900"
                        )

            # Error Label
            health_missing = ui.label("").classes(
                "w-full q-px-sm text-[12px] text-rose-600 italic"
            )

        with ui.card().classes(
            "card-soft runtime-panel runtime-summary runtime-summary-card runtime-queue w-full shadow-sm rounded-lg"
        ):
            with ui.row().classes(
                "w-full items-center justify-between q-px-sm bg-slate-50 border-b border-slate-100"
            ):
                with ui.row().classes("items-center gap-1"):
                    ui.icon("memory", color="primary").classes("text-sm")
                    ui.label("Job Queue").classes(
                        "text-sm font-black text-slate-500 uppercase tracking-tighter"
                    )
                refresh_queue_btn = ui.button(icon="sync", color="secondary").props(
                    "flat dense size=sm"
                )

            with ui.column().classes("w-full q-pa-none gap-1"):
                # Top row: media types
                with ui.grid(columns=3).classes("w-full gap-1"):
                    # Image Jobs
                    with ui.row().classes(
                        "items-center justify-between q-px-xs border border-blue-100 rounded bg-blue-50/30"
                    ):
                        ui.label("IMG:").classes("text-[12px] font-bold text-blue-700")
                        queue_image = ui.label("0").classes(
                            "text-[10px] font-black text-blue-900"
                        )
                    # Video Jobs
                    with ui.row().classes(
                        "items-center justify-between q-px-xs border border-rose-100 rounded bg-rose-50/30"
                    ):
                        ui.label("VID:").classes("text-[12px] font-bold text-rose-700")
                        queue_video = ui.label("0").classes(
                            "text-[10px] font-black text-rose-900"
                        )
                    # Selected
                    with ui.row().classes(
                        "items-center justify-between q-px-xs border border-slate-100 rounded bg-slate-50/30"
                    ):
                        ui.label("SEL:").classes("text-[12px] font-bold text-slate-700")
                        queue_selected = ui.label("0").classes(
                            "text-[10px] font-black text-slate-900"
                        )

                # Bottom row: job status
                with ui.grid(columns=4).classes("w-full gap-1"):
                    # Planned
                    with ui.column().classes(
                        "items-center gap-0 border border-slate-100 rounded q-pa-none bg-white"
                    ):
                        ui.label("PLN").classes(
                            "text-[7px] font-bold text-slate-400 uppercase"
                        )
                        queue_planned_status = ui.label("0").classes(
                            "text-[10px] font-bold text-slate-600"
                        )
                    # Running
                    with ui.column().classes(
                        "items-center gap-0 border border-blue-100 rounded q-pa-none bg-white"
                    ):
                        ui.label("RUN").classes(
                            "text-[7px] font-bold text-blue-400 uppercase"
                        )
                        queue_running_status = ui.label("0").classes(
                            "text-[10px] font-bold text-blue-600"
                        )
                    # Done
                    with ui.column().classes(
                        "items-center gap-0 border border-emerald-100 rounded q-pa-none bg-white"
                    ):
                        ui.label("DONE").classes(
                            "text-[7px] font-bold text-emerald-400 uppercase"
                        )
                        queue_done_status = ui.label("0").classes(
                            "text-[10px] font-bold text-emerald-600"
                        )
                    # Failed
                    with ui.column().classes(
                        "items-center gap-0 border border-rose-100 rounded q-pa-none bg-white"
                    ):
                        ui.label("FAIL").classes(
                            "text-[7px] font-bold text-rose-400 uppercase"
                        )
                        queue_failed_status = ui.label("0").classes(
                            "text-[10px] font-bold text-rose-600"
                        )

                queue_hint = ui.label("Preview checks input path only.").classes(
                    "text-[8px] text-slate-400 italic q-px-xs"
                )

        # --- Section: Tuner Live ---
        with ui.card().classes(
            "card-soft runtime-panel runtime-summary runtime-summary-card runtime-tuner w-full shadow-sm rounded-lg"
        ):
            with ui.row().classes(
                "w-full items-center q-px-sm bg-slate-50 border-b border-slate-100 gap-1"
            ):
                ui.icon("settings_input_component", color="primary").classes("text-sm")
                ui.label("Tuner Live").classes(
                    "text-sm font-black text-slate-500 uppercase tracking-tighter"
                )

                # Row 1: GPU, mode, hot stage
            with ui.grid(columns=3).classes("w-full gap-1 q-pa-xs"):
                # GPU Usage
                with ui.row().classes(
                    "items-center justify-between q-px-xs border border-indigo-100 rounded bg-indigo-50/30"
                ):
                    ui.label("GPU:").classes("text-[12px] font-bold text-indigo-700")
                    tuner_gpu = ui.label("0%").classes(
                        "text-[10px] font-black text-indigo-900"
                    )
                # Mode
                with ui.row().classes(
                    "items-center justify-between q-px-xs border border-emerald-100 rounded bg-emerald-50/30"
                ):
                    ui.label("MODE:").classes("text-[12px] font-bold text-emerald-700")
                    tuner_mode_live = ui.label("norm").classes(
                        "text-[10px] font-black text-emerald-900"
                    )
                # Hot Stage
                with ui.row().classes(
                    "items-center justify-between q-px-xs border border-amber-100 rounded bg-amber-50/30"
                ):
                    ui.label("HOT:").classes("text-[12px] font-bold text-amber-700")
                    tuner_hot = ui.label("-").classes(
                        "text-[10px] font-black text-amber-900"
                    )

                # Rows 2-3: queue and permit mini-badges
            with ui.column().classes("w-full q-pa-xs gap-1"):
                # Q Series
                with ui.grid(columns=4).classes("w-full gap-1"):
                    with ui.row().classes(
                        "items-center no-wrap gap-1 q-px-xs border border-blue-100 rounded bg-blue-50/20"
                    ):
                        ui.label("QD:").classes("text-[8px] font-bold text-blue-600")
                        q_detect_label = ui.label("0").classes(
                            "text-[12px] font-black text-blue-900"
                        )
                    with ui.row().classes(
                        "items-center no-wrap gap-1 q-px-xs border border-orange-100 rounded bg-orange-50/20"
                    ) as q_swap_card:
                        ui.label("QS:").classes("text-[8px] font-bold text-orange-600")
                        q_swap_label = ui.label("0").classes(
                            "text-[12px] font-black text-orange-900"
                        )
                    with ui.row().classes(
                        "items-center no-wrap gap-1 q-px-xs border border-green-100 rounded bg-green-50/20"
                    ) as q_restore_card:
                        ui.label("QR:").classes("text-[8px] font-bold text-green-600")
                        q_restore_label = ui.label("0").classes(
                            "text-[12px] font-black text-green-900"
                        )
                    with ui.row().classes(
                        "items-center no-wrap gap-1 q-px-xs border border-rose-100 rounded bg-rose-50/20"
                    ) as q_parse_card:
                        ui.label("QP:").classes("text-[8px] font-bold text-rose-600")
                        q_parse_label = ui.label("0").classes(
                            "text-[12px] font-black text-rose-900"
                        )

                # P Series
                with ui.grid(columns=4).classes("w-full gap-1"):
                    with ui.row().classes(
                        "items-center no-wrap gap-1 q-px-xs border border-blue-100 rounded bg-blue-50/20"
                    ):
                        ui.label("PD:").classes("text-[8px] font-bold text-blue-600")
                        p_detect_label = ui.label("0").classes(
                            "text-[12px] font-black text-blue-900"
                        )
                    with ui.row().classes(
                        "items-center no-wrap gap-1 q-px-xs border border-orange-100 rounded bg-orange-50/20"
                    ) as p_swap_card:
                        ui.label("PS:").classes("text-[8px] font-bold text-orange-600")
                        p_swap_label = ui.label("0").classes(
                            "text-[12px] font-black text-orange-900"
                        )
                    with ui.row().classes(
                        "items-center no-wrap gap-1 q-px-xs border border-green-100 rounded bg-green-50/20"
                    ) as p_restore_card:
                        ui.label("PR:").classes("text-[8px] font-bold text-green-600")
                        p_restore_label = ui.label("0").classes(
                            "text-[12px] font-black text-green-900"
                        )
                    with ui.row().classes(
                        "items-center no-wrap gap-1 q-px-xs border border-rose-100 rounded bg-rose-50/20"
                    ) as p_parse_card:
                        ui.label("PP:").classes("text-[8px] font-bold text-rose-600")
                        p_parse_label = ui.label("0").classes(
                            "text-[12px] font-black text-rose-900"
                        )

            # Row 4: performance charts
        with (
            ui.card()
            .tight()
            .classes(
                "card-soft runtime-panel runtime-charts w-full shadow-sm rounded-lg overflow-hidden border border-slate-200"
            )
        ):
            # Subtle header keeps the charts visually grouped.
            with ui.row().classes(
                "w-full items-center gap-2 q-px-sm q-py-xs bg-slate-50 border-b border-slate-100"
            ):
                ui.icon("show_chart", color="primary").classes("text-sm")
                ui.label("Performance Charts").classes(
                    "text-[sm] font-black text-slate-500 uppercase tracking-tighter"
                )

            # Three charts laid out in one row
            with ui.grid(columns=3).classes("w-full gap-0 divide-x divide-slate-100"):

                # --- Chart 1: GPU ---
                with ui.column().classes("w-full q-pa-xs"):
                    gpu_chart = ui.echart(
                        {
                            "animation": False,
                            "grid": {"left": 25, "right": 5, "top": 10, "bottom": 20},
                            "xAxis": {
                                "type": "category",
                                "data": [],
                                "show": False,
                            },  # Hide the X axis to save vertical space.
                            "yAxis": {
                                "type": "value",
                                "min": 0,
                                "max": 100,
                                "axisLabel": {"fontSize": 8},
                            },
                            "series": [
                                {
                                    "name": "GPU %",
                                    "type": "line",
                                    "smooth": True,
                                    "showSymbol": False,
                                    "areaStyle": {
                                        "opacity": 0.1
                                    },  # Use a light fill to ground the line.
                                    "lineStyle": {"width": 1.5, "color": "#6366f1"},
                                    "data": [],
                                }
                            ],
                        }
                    ).classes("w-full h-24")

                # --- Chart 2: Queue ---
                with ui.column().classes("w-full q-pa-xs"):
                    queue_chart = ui.echart(
                        {
                            "animation": False,
                            "legend": {
                                "top": 0,
                                "itemSize": 6,
                                "textStyle": {"fontSize": 7},
                                "icon": "circle",
                            },
                            "grid": {"left": 25, "right": 5, "top": 20, "bottom": 20},
                            "xAxis": {"type": "category", "data": [], "show": False},
                            "yAxis": {
                                "type": "value",
                                "min": 0,
                                "axisLabel": {"fontSize": 8},
                            },
                            "series": [
                                {
                                    "name": "det",
                                    "type": "line",
                                    "showSymbol": False,
                                    "data": [],
                                },
                                {
                                    "name": "swp",
                                    "type": "line",
                                    "showSymbol": False,
                                    "data": [],
                                },
                                {
                                    "name": "res",
                                    "type": "line",
                                    "showSymbol": False,
                                    "data": [],
                                },
                                {
                                    "name": "par",
                                    "type": "line",
                                    "showSymbol": False,
                                    "data": [],
                                },
                            ],
                        }
                    ).classes("w-full h-24")

                # --- Chart 3: Permit ---
                with ui.column().classes("w-full q-pa-xs"):
                    permit_chart = ui.echart(
                        {
                            "animation": False,
                            "legend": {
                                "top": 0,
                                "itemSize": 6,
                                "textStyle": {"fontSize": 7},
                                "icon": "circle",
                            },
                            "grid": {"left": 25, "right": 5, "top": 20, "bottom": 20},
                            "xAxis": {"type": "category", "data": [], "show": False},
                            "yAxis": {
                                "type": "value",
                                "min": 0,
                                "axisLabel": {"fontSize": 8},
                            },
                            "series": [
                                {
                                    "name": "det",
                                    "type": "line",
                                    "showSymbol": False,
                                    "data": [],
                                },
                                {
                                    "name": "swp",
                                    "type": "line",
                                    "showSymbol": False,
                                    "data": [],
                                },
                                {
                                    "name": "res",
                                    "type": "line",
                                    "showSymbol": False,
                                    "data": [],
                                },
                                {
                                    "name": "par",
                                    "type": "line",
                                    "showSymbol": False,
                                    "data": [],
                                },
                            ],
                        }
                    ).classes("w-full h-24")

        with ui.card().classes(
            "card-soft runtime-panel runtime-floating-controls action-strip w-full shadow-md rounded-xl overflow-hidden"
        ):

            # Action buttons
            with ui.row().classes("w-full items-center gap-2"):
                # Primary run action
                run_btn = ui.button(
                    "START PIPELINE", icon="play_arrow", color="primary"
                ).classes("grow font-semibold shadow-sm text-xs")
                # Secondary stop action
                stop_btn = ui.button("STOP", icon="stop", color="negative").classes(
                    "w-24 shadow-sm text-xs"
                )
                stop_btn.disable()

                # Compact preview pause toggle
                pause_preview_btn = (
                    ui.button(icon="pause_presentation", color="warning")
                    .props("flat round")
                    .tooltip("Pause Preview")
                    .classes("text-xs")
                )

                store.preview_paused = False
            # Single-line progress: status | bar | percent
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                progress_label = ui.label("Idle").classes(
                    "text-xs font-semibold text-slate-700 whitespace-nowrap shrink-0"
                )
                throughput_fps = ui.label("[0.0/fps]").classes(
                    "text-xs font-semibold text-indigo-700 whitespace-nowrap shrink-0"
                )
                throughput_eta = ui.label("[--.--]").classes(
                    "text-xs font-semibold text-amber-700 whitespace-nowrap shrink-0"
                )
                progress_bar = (
                    ui.linear_progress(value=0.0, color="primary")
                    .classes("grow h-2 rounded-full min-w-0")
                    .style("text-indent: -9999px; overflow: hidden;")
                )
                progress_count = ui.label("0/0").classes(
                    "text-xs font-semibold text-slate-500 whitespace-nowrap shrink-0"
                )
                progress_pct = ui.label("0%").classes(
                    "text-xs font-black text-primary whitespace-nowrap shrink-0"
                )
                throughput_items_min = ui.label("0.0").classes("hidden")

            # Preview visibility follows the preview controls.

        # --- 2. Runtime Console + Live Preview ---
        with ui.row().classes(
            "runtime-panel runtime-console-preview-row w-full no-wrap q-pa-none items-stretch gap-2"
        ):
            with ui.card().classes(
                "card-soft runtime-console w-full shadow-sm rounded-xl overflow-hidden q-pa-none"
            ):
                # Console Header
                with ui.row().classes(
                    "w-full items-center justify-between bg-slate-100 q-px-md border-b border-slate-200"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("terminal", color="slate-700").classes("text-lg")
                        ui.label("Runtime Console").classes(
                            "text-sm font-bold text-slate-700"
                        )

                    with ui.row().classes("items-center gap-2"):
                        error_count_badge = ui.badge(
                            "0 Errors", color="rose-500"
                        ).classes("font-bold")
                        open_error_btn = (
                            ui.button("Inspector", icon="bug_report", color="negative")
                            .props("flat dense")
                            .classes("text-xs font-bold")
                        )
                        clear_btn = (
                            ui.button("Clear", icon="delete_sweep", color="slate")
                            .props("flat dense")
                            .classes("text-xs font-bold")
                        )

                # Log viewer
                with ui.element("div").classes(
                    "runtime-console-body w-full bg-[#1e1e1e] p-3"
                ):
                    log_view = ui.log(max_lines=1200).classes(
                        "w-full h-full font-mono text-[11px] text-emerald-400 bg-transparent no-shadow leading-relaxed"
                    )

            with ui.card().classes(
                "card-soft runtime-preview runtime-preview-side shadow-sm rounded-xl overflow-hidden q-pa-none"
            ) as preview_panel:
                with ui.row().classes(
                    "runtime-preview-header w-full items-center justify-between bg-slate-100 q-px-sm border-b border-slate-200"
                ):
                    with ui.row().classes("runtime-preview-lead items-center gap-2") as preview_header_lead:
                        ui.icon("visibility", color="slate-700").classes("text-lg")
                        preview_title = ui.label("Live Preview").classes(
                            "runtime-preview-title text-sm font-bold text-slate-700"
                        )
                    preview_toggle_btn = (
                        ui.button("</>", color="secondary")
                        .props("flat dense")
                        .classes("runtime-preview-toggle text-slate-600")
                        .tooltip("Collapse preview")
                    )

                with ui.column().classes(
                    "runtime-preview-body w-full q-pt-none q-px-xs q-pb-sm"
                ) as preview_body:
                    build_preview_container(preview_dom_id)
                preview_meta = _NoopText()
                preview_expansion = _SideCollapseController(
                    preview_panel,
                    preview_body,
                    preview_header_lead,
                    preview_title,
                    preview_toggle_btn,
                )
                preview_expansion.set_value(bool(defaults["preview_enabled"]))
                preview_toggle_btn.on_click(
                    lambda: preview_expansion.set_value(
                        not bool(preview_expansion.value)
                    )
                )
        # Output gallery
        with ui.expansion("Output Gallery", icon="photo_library", value=False).classes(
            "w-full card-soft shadow-sm rounded-xl font-bold bg-white border border-slate-200"
        ):

            with ui.column().classes("w-full q-pa-sm gap-2 font-normal"):
                with ui.row().classes("w-full items-center justify-between"):
                    gallery_status = ui.label("No output yet").classes(
                        "text-xs text-slate-500 italic"
                    )
                    refresh_gallery_btn = (
                        ui.button("Refresh", icon="sync", color="secondary")
                        .props("flat dense")
                        .classes("text-xs")
                    )

                # Gallery items container
                gallery_items = ui.row().classes("w-full gap-2")

        stop_confirm_refs = build_stop_confirm_dialog()
        stop_confirm_dialog = stop_confirm_refs.dialog
        stop_confirm_yes_btn = stop_confirm_refs.confirm_btn

        init_confirm_refs = build_initialize_confirm_dialog()
        init_confirm_dialog = init_confirm_refs.dialog
        init_confirm_yes_btn = init_confirm_refs.confirm_btn

        error_dialog_refs = build_error_dialog()
        error_dialog = error_dialog_refs.dialog
        clear_error_btn = error_dialog_refs.clear_btn
        error_count_label = error_dialog_refs.count_label
        error_list = error_dialog_refs.error_list

        model_dialogs_refs = build_model_dialogs()
        model_status_dialog = model_dialogs_refs.model_status_dialog
        refresh_model_dialog_btn = model_dialogs_refs.refresh_model_dialog_btn
        download_all_models_btn = model_dialogs_refs.download_all_models_btn
        open_download_center_btn = model_dialogs_refs.open_download_center_btn
        model_status_summary = model_dialogs_refs.model_status_summary
        model_status_list = model_dialogs_refs.model_status_list
        download_center_dialog = model_dialogs_refs.download_center_dialog
        pause_resume_downloads_btn = model_dialogs_refs.pause_resume_downloads_btn
        clear_finished_downloads_btn = model_dialogs_refs.clear_finished_downloads_btn
        download_center_summary = model_dialogs_refs.download_center_summary
        download_center_list = model_dialogs_refs.download_center_list
        tensorrt_dialog = model_dialogs_refs.tensorrt_dialog
        trt_missing_label = model_dialogs_refs.trt_missing_label
        trt_target_label = model_dialogs_refs.trt_target_label

        gallery_preview_refs = build_gallery_preview_dialog()
        gallery_preview_dialog = gallery_preview_refs.dialog
        gallery_preview_title = gallery_preview_refs.title
        gallery_popup_image = gallery_preview_refs.image
        gallery_popup_video = gallery_preview_refs.video

        setup_wizard_refs = build_setup_wizard_dialog()
        setup_wizard_dialog = setup_wizard_refs.dialog
        wizard_run_checks_btn = setup_wizard_refs.run_checks_btn
        wizard_project_status = setup_wizard_refs.project_status
        wizard_model_status = setup_wizard_refs.model_status
        wizard_trt_status = setup_wizard_refs.trt_status

        face_model_build_refs = build_face_model_build_dialog(defaults["provider_all"])
        face_build_dialog = face_model_build_refs.dialog
        face_build_name = face_model_build_refs.name_input
        face_build_input_path = face_model_build_refs.input_path_input
        face_build_upload = face_model_build_refs.upload_input
        face_build_upload_status = face_model_build_refs.upload_status_label
        face_build_clear_upload_btn = face_model_build_refs.clear_upload_btn
        face_build_provider = face_model_build_refs.provider_select
        face_build_min_images = face_model_build_refs.min_images_input
        face_build_status = face_model_build_refs.status_label
        build_face_btn = face_model_build_refs.build_btn
        refresh_faces_btn = face_model_build_refs.refresh_faces_btn
        clear_face_build_btn = face_model_build_refs.clear_status_btn

        logic_ctx = dict(locals())
        logic_ctx.update(
            {
                "validate_form": validate_form,
                "preview_job_queue": preview_job_queue,
                "register_media_root": register_media_root,
                "to_media_url": to_media_url,
                "list_latest_outputs": list_latest_outputs,
                "collect_runtime_health": collect_runtime_health,
                "apply_health_report": apply_health_report,
                "save_project_settings": save_project_settings,
                "validate_project_path": validate_project_path,
                "disk_space_gb": disk_space_gb,
                "ensure_workspace": ensure_workspace,
                "check_tensorrt_status": check_tensorrt_status,
                "run_setup_checks": run_setup_checks,
                "normalize_controller_event": normalize_controller_event,
            }
        )
        wire_main_ui_logic(logic_ctx)


def register_ui_assets() -> None:
    ui.add_head_html(
        """
        <style>
          .app-shell {
            background: #f6f8fb;
            min-height: 100vh;
            display: block !important;
            padding: 12px !important;
          }
          .app-header {
            position: sticky;
            top: 0;
            z-index: 20;
            width: 100%;
            padding: 10px 12px;
            margin-bottom: 12px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(14px);
          }
          .workspace-shell {
            width: 100%;
            display: grid;
            grid-template-columns: minmax(360px, 420px) minmax(0, 1fr);
            gap: 10px;
            align-items: start;
          }
          .settings-rail,
          .runtime-workspace {
            min-width: 0;
          }
          .settings-rail { gap: 12px; }
          .settings-rail {
            display: flex;
            flex-direction: column;
          }
          .runtime-workspace {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
            align-items: stretch;
          }
          .settings-rail {
            position: sticky;
            top: 86px;
            max-height: calc(100vh - 100px);
            overflow-y: auto;
            padding-right: 4px;
          }
          .runtime-workspace {
            min-height: calc(100vh - 100px);
          }
          .settings-panel,
          .runtime-panel {
            width: 100%;
          }
          .action-strip {
            position: sticky;
            bottom: 0;
            z-index: 15;
            padding: 6px 8px;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.94);
            backdrop-filter: blur(12px);
          }
          .card-soft {
            max-width: none !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            border: 1px solid #e2e8f0;
            border-radius: 7px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
          }
          .runtime-workspace .q-card {
            min-height: 0;
          }
          .runtime-workspace .q-card .q-card {
            border-radius: 6px;
          }
          .runtime-workspace > .runtime-panel { grid-column: 1 / -1; order: 20; }
          .runtime-workspace > .runtime-summary {
            grid-column: auto;
            order: 1;
            height: 100%;
            min-height: 188px;
          }
          .runtime-summary-card .summary-header {
            min-height: 30px;
            padding: 2px 4px;
            margin-bottom: 4px;
          }
          .runtime-summary-card .summary-header .q-icon {
            font-size: 1rem !important;
          }
          .runtime-summary-card .summary-header .q-btn {
            font-size: 11px;
          }
          .runtime-summary-card .summary-header .q-btn .q-btn__content {
            gap: 4px;
          }
          .runtime-summary-card .q-card__section {
            padding: 4px 6px;
          }
          .runtime-workspace > .runtime-health { order: 1; }
          .runtime-workspace > .runtime-queue { order: 2; }
          .runtime-workspace > .runtime-tuner { order: 3; }
          .runtime-workspace > .runtime-console-preview-row {
            grid-column: 1 / -1;
            order: 0;
          }
          .runtime-console-preview-row > .runtime-console {
            flex: 1 1 0;
            min-width: 0;
            height: 352px;
            display: flex;
            flex-direction: column;
          }
          .runtime-console-preview-row > .runtime-preview-side {
            flex: 1 1 0;
            min-width: 0;
            max-width: none;
            height: 352px;
            display: flex;
            flex-direction: column;
            transition: none;
          }
          .runtime-console-preview-row .runtime-console-body,
          .runtime-console-preview-row .runtime-preview-body {
            flex: 1 1 auto;
            min-height: 0;
          }
          .runtime-console-preview-row .runtime-console-body .q-log {
            height: 100% !important;
          }
          .runtime-preview-header {
            position: relative;
            display: flex;
            align-items: center;
            width: 100%;
          }
          .runtime-preview-lead {
            flex: 1 1 auto;
            min-width: 0;
          }
          .runtime-preview-toggle {
            margin-left: auto;
            flex: 0 0 auto;
            align-self: center;
          }
          .runtime-console-preview-row > .runtime-preview-side.is-collapsed {
            flex: 0 0 52px;
            min-width: 52px;
            max-width: 52px;
          }
          .runtime-console-preview-row > .runtime-preview-side.is-collapsed .runtime-preview-header {
            justify-content: flex-end;
            padding-left: 0;
            padding-right: 0;
          }
          .runtime-console-preview-row > .runtime-preview-side.is-collapsed .runtime-preview-title,
          .runtime-console-preview-row > .runtime-preview-side.is-collapsed .runtime-preview-lead {
            display: none;
          }
          .runtime-workspace > .runtime-charts { grid-column: 1 / -1; order: 5; }
          .runtime-workspace > .runtime-floating-controls {
            grid-column: 1 / -1;
            order: 30;
            position: sticky;
            bottom: 10px;
            z-index: 30;
            width: min(880px, calc(100% - 16px));
            justify-self: center;
            border-radius: 14px !important;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.14);
            background: rgba(255, 255, 255, 0.96);
            backdrop-filter: blur(14px);
          }
          .log-card { border: 1px solid #334155; border-radius: 12px; background: #0b1220; }
          @media (max-width: 1100px) {
            .workspace-shell { grid-template-columns: 1fr; }
            .runtime-workspace { grid-template-columns: 1fr; }
            .runtime-workspace > .runtime-health,
            .runtime-workspace > .runtime-queue,
            .runtime-workspace > .runtime-tuner,
            .runtime-workspace > .runtime-charts,
            .runtime-workspace > .runtime-console-preview-row {
              grid-column: 1 / -1;
            }
            .runtime-console-preview-row {
              flex-direction: column;
            }
            .runtime-console-preview-row > .runtime-preview-side,
            .runtime-console-preview-row > .runtime-preview-side.is-collapsed {
              flex: 1 1 auto;
              min-width: 0;
              max-width: none;
            }
            .settings-rail {
              position: static;
              max-height: none;
              overflow: visible;
              padding-right: 0;
            }
          }
        </style>
        """
    )
    ui.add_body_html(
        """
        <script>
        (function() {
          function organizeQ1Layout() {
            const shell = document.querySelector('.app-shell');
            if (!shell || shell.dataset.q1LayoutReady === '1') return;
            const header = shell.querySelector('.app-header');
            const settings = Array.from(shell.querySelectorAll(':scope > .settings-panel'));
            const runtime = Array.from(shell.querySelectorAll(':scope > .runtime-panel'));
            if (!settings.length || !runtime.length) return;

            const workspace = document.createElement('div');
            workspace.className = 'workspace-shell';
            const settingsRail = document.createElement('div');
            settingsRail.className = 'settings-rail';
            const runtimeWorkspace = document.createElement('div');
            runtimeWorkspace.className = 'runtime-workspace';
            workspace.appendChild(settingsRail);
            workspace.appendChild(runtimeWorkspace);

            if (header && header.nextSibling) {
              shell.insertBefore(workspace, header.nextSibling);
            } else {
              shell.appendChild(workspace);
            }

            settings.forEach((node) => settingsRail.appendChild(node));
            runtime.forEach((node) => runtimeWorkspace.appendChild(node));
            shell.dataset.q1LayoutReady = '1';
          }

          const timer = window.setInterval(function() {
            organizeQ1Layout();
            const shell = document.querySelector('.app-shell');
            if (shell && shell.dataset.q1LayoutReady === '1') {
              window.clearInterval(timer);
            }
          }, 100);
          window.addEventListener('load', organizeQ1Layout);
        })();
        </script>
        """
    )
    register_preview_bridge_assets()
