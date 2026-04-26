from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nicegui import ui

try:
    from gui.widgets import add_path_picker
except ImportError:
    from widgets import add_path_picker  # type: ignore


@dataclass
class ConfirmDialogRefs:
    dialog: Any
    confirm_btn: Any


@dataclass
class ErrorDialogRefs:
    dialog: Any
    clear_btn: Any
    count_label: Any
    error_list: Any


@dataclass
class ModelDialogsRefs:
    model_status_dialog: Any
    refresh_model_dialog_btn: Any
    download_all_models_btn: Any
    open_download_center_btn: Any
    model_status_summary: Any
    model_status_list: Any
    download_center_dialog: Any
    pause_resume_downloads_btn: Any
    clear_finished_downloads_btn: Any
    download_center_summary: Any
    download_center_list: Any
    tensorrt_dialog: Any
    trt_missing_label: Any
    trt_target_label: Any


@dataclass
class GalleryPreviewDialogRefs:
    dialog: Any
    title: Any
    image: Any
    video: Any


@dataclass
class SetupWizardDialogRefs:
    dialog: Any
    run_checks_btn: Any
    project_status: Any
    model_status: Any
    trt_status: Any


@dataclass
class FaceModelBuildDialogRefs:
    dialog: Any
    name_input: Any
    input_path_input: Any
    upload_input: Any
    upload_status_label: Any
    clear_upload_btn: Any
    provider_select: Any
    min_images_input: Any
    status_label: Any
    build_btn: Any
    refresh_faces_btn: Any
    clear_status_btn: Any


def build_stop_confirm_dialog() -> ConfirmDialogRefs:
    with ui.dialog() as dialog, ui.card().classes("w-[420px]"):
        ui.label("Confirm Stop").classes("text-lg font-semibold")
        ui.label(
            "Stop pipeline now? Current item may pause and resume next run."
        ).classes("text-sm text-slate-600")
        with ui.row().classes("justify-end w-full gap-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            confirm_btn = ui.button("Stop Now", color="negative")
    return ConfirmDialogRefs(dialog=dialog, confirm_btn=confirm_btn)


def build_initialize_confirm_dialog() -> ConfirmDialogRefs:
    with ui.dialog() as dialog, ui.card().classes("w-[460px]"):
        ui.label("Confirm Initialize Structure").classes("text-lg font-semibold")
        ui.label(
            "Initialize/repair project folders and copy default assets/docs if missing?",
        ).classes("text-sm text-slate-600")
        with ui.row().classes("justify-end w-full gap-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            confirm_btn = ui.button("Initialize", color="primary")
    return ConfirmDialogRefs(dialog=dialog, confirm_btn=confirm_btn)


def build_error_dialog() -> ErrorDialogRefs:
    with ui.dialog() as dialog, ui.card().classes("w-[980px] max-w-[96vw]"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Error Inspector").classes("text-lg font-semibold")
            clear_btn = ui.button("Clear Errors", color="secondary").props(
                "flat dense"
            )
        count_label = ui.label("0 critical entries").classes("text-xs text-slate-600")
        with ui.scroll_area().classes(
            "w-full h-[62vh] border border-rose-100 rounded p-2 bg-rose-50"
        ):
            error_list = ui.column().classes("w-full gap-1")
    return ErrorDialogRefs(
        dialog=dialog,
        clear_btn=clear_btn,
        count_label=count_label,
        error_list=error_list,
    )


def build_model_dialogs() -> ModelDialogsRefs:
    with (
        ui.dialog() as model_status_dialog,
        ui.card().classes("w-[960px] max-w-[96vw]"),
    ):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Model Status").classes("text-lg font-semibold")
            with ui.row().classes("items-center gap-2"):
                refresh_model_dialog_btn = ui.button("Refresh", color="secondary").props(
                    "flat dense"
                )
                download_all_models_btn = ui.button(
                    "Download All Missing", color="primary"
                ).props("flat dense")
                open_download_center_btn = ui.button(
                    "Download Center", color="secondary"
                ).props("flat dense")
        model_status_summary = ui.label("-").classes("text-xs text-slate-600")
        with ui.scroll_area().classes("w-full h-[62vh] border border-slate-200 rounded p-2"):
            model_status_list = ui.column().classes("w-full gap-1")

    with (
        ui.dialog() as download_center_dialog,
        ui.card().classes("w-[980px] max-w-[96vw]"),
    ):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Download Center").classes("text-lg font-semibold")
            with ui.row().classes("items-center gap-2"):
                pause_resume_downloads_btn = ui.button("Pause All", color="warning").props(
                    "flat dense"
                )
                clear_finished_downloads_btn = ui.button(
                    "Clear Finished", color="secondary"
                ).props("flat dense")
        download_center_summary = ui.label("No downloads yet").classes(
            "text-xs text-slate-600"
        )
        with ui.scroll_area().classes("w-full h-[62vh] border border-slate-200 rounded p-2"):
            download_center_list = ui.column().classes("w-full gap-1")

    with (
        ui.dialog() as tensorrt_dialog,
        ui.card().classes("w-[760px] max-w-[96vw]"),
    ):
        ui.label("TensorRT Required").classes("text-lg font-semibold text-slate-800")
        ui.label(
            "Provider is set to TRT but TensorRT binaries are missing or incomplete.",
        ).classes("text-sm text-slate-600")
        trt_missing_label = ui.label("-").classes("text-xs text-rose-700")
        trt_target_label = ui.label("-").classes("text-xs text-slate-600")
        with ui.row().classes("justify-end w-full gap-2"):
            ui.button("Close", on_click=tensorrt_dialog.close).props("flat")
            ui.link(
                "Download TensorRT 10.x",
                "https://developer.nvidia.com/tensorrt/download/10x",
            ).props("target=_blank")

    return ModelDialogsRefs(
        model_status_dialog=model_status_dialog,
        refresh_model_dialog_btn=refresh_model_dialog_btn,
        download_all_models_btn=download_all_models_btn,
        open_download_center_btn=open_download_center_btn,
        model_status_summary=model_status_summary,
        model_status_list=model_status_list,
        download_center_dialog=download_center_dialog,
        pause_resume_downloads_btn=pause_resume_downloads_btn,
        clear_finished_downloads_btn=clear_finished_downloads_btn,
        download_center_summary=download_center_summary,
        download_center_list=download_center_list,
        tensorrt_dialog=tensorrt_dialog,
        trt_missing_label=trt_missing_label,
        trt_target_label=trt_target_label,
    )


def build_gallery_preview_dialog() -> GalleryPreviewDialogRefs:
    with (
        ui.dialog() as dialog,
        ui.card().classes("w-[1100px] max-w-[96vw]"),
    ):
        with ui.row().classes("w-full items-center justify-between"):
            title = ui.label("Output Preview").classes("text-lg font-semibold")
            ui.button("Close", on_click=dialog.close).props("flat")
        image = ui.image("").classes("w-full h-[72vh] object-contain bg-slate-100 rounded")
        video = ui.video("").classes("w-full h-[72vh] bg-slate-900 rounded")
        video.set_visibility(False)
    return GalleryPreviewDialogRefs(dialog=dialog, title=title, image=image, video=video)


def build_setup_wizard_dialog() -> SetupWizardDialogRefs:
    with (
        ui.dialog() as dialog,
        ui.card().classes("w-[840px] max-w-[96vw]"),
    ):
        ui.label("Setup Wizard").classes("text-lg font-semibold")
        ui.label("project path -> model check -> trt check").classes("text-xs text-slate-500")
        with ui.column().classes("w-full gap-2"):
            with ui.card().classes("w-full border border-slate-200"):
                ui.label("Step 1: Project Path").classes("text-sm font-semibold text-slate-700")
                project_status = ui.label("-").classes("text-xs text-slate-600")
            with ui.card().classes("w-full border border-slate-200"):
                ui.label("Step 2: Model Check").classes("text-sm font-semibold text-slate-700")
                model_status = ui.label("-").classes("text-xs text-slate-600")
            with ui.card().classes("w-full border border-slate-200"):
                ui.label("Step 3: TensorRT Check").classes("text-sm font-semibold text-slate-700")
                trt_status = ui.label("-").classes("text-xs text-slate-600")
        with ui.row().classes("w-full justify-end gap-2"):
            run_checks_btn = ui.button("Run Checks", color="primary")
            ui.button("Close", on_click=dialog.close).props("flat")
    return SetupWizardDialogRefs(
        dialog=dialog,
        run_checks_btn=run_checks_btn,
        project_status=project_status,
        model_status=model_status,
        trt_status=trt_status,
    )


def build_face_model_build_dialog(default_provider: str = "trt") -> FaceModelBuildDialogRefs:
    with ui.dialog() as dialog, ui.card().classes("w-[760px] max-w-[96vw]"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Build Face Model").classes("text-lg font-semibold")
            ui.button("Close", on_click=dialog.close).props("flat")
        with ui.column().classes("w-full gap-2"):
            name_input = ui.input("New Face Model Name", value="").props("clearable").classes("w-full")
            with ui.row().classes("items-center w-full"):
                input_path_input = (
                    ui.input("Face Images Path", value="")
                    .props("clearable")
                    .classes("grow")
                )
                add_path_picker(
                    input_path_input,
                    "Select Face Images Folder",
                    pick_file=False,
                )
            ui.label("Or Drag & Drop Face Images").classes("text-xs text-slate-500")
            upload_input = ui.upload(
                multiple=True,
                auto_upload=True,
                max_files=200,
            ).props("accept=.png,.jpg,.jpeg,.webp,.bmp").classes("w-full")
            upload_status_label = ui.label("Uploaded: 0 files").classes("text-xs text-slate-600")
            provider_select = ui.select(
                {"trt": "trt", "cuda": "cuda", "cpu": "cpu"},
                value=str(default_provider or "trt"),
                label="Build Provider",
            ).classes("w-full")
            min_images_input = ui.number(
                "Min Accepted Images",
                value=1,
                min=1,
                max=1000,
            ).props("dense outlined").classes("w-full")
            status_label = ui.label("Face model builder is idle").classes("text-xs text-slate-600")
            with ui.row().classes("w-full items-center justify-end gap-2"):
                clear_upload_btn = ui.button("Clear Uploaded Files", color="secondary").props("flat dense")
                clear_status_btn = ui.button("Clear Build Status", color="secondary").props("flat dense")
                refresh_faces_btn = ui.button("Refresh Faces", icon="refresh", color="primary").props("flat dense")
                build_btn = ui.button("Build Face Model", icon="construction", color="secondary").props("dense")
    return FaceModelBuildDialogRefs(
        dialog=dialog,
        name_input=name_input,
        input_path_input=input_path_input,
        upload_input=upload_input,
        upload_status_label=upload_status_label,
        clear_upload_btn=clear_upload_btn,
        provider_select=provider_select,
        min_images_input=min_images_input,
        status_label=status_label,
        build_btn=build_btn,
        refresh_faces_btn=refresh_faces_btn,
        clear_status_btn=clear_status_btn,
    )
