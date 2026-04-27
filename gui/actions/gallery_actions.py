from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from nicegui import ui


def open_output_in_explorer(path_value: str) -> tuple[bool, str]:
    candidate = Path(str(path_value or "").strip()).expanduser()
    if not candidate.exists():
        return False, f"Path not found: {candidate}"
    try:
        subprocess.Popen(["explorer", "/select,", str(candidate)], shell=False)
    except Exception as exc:  # noqa: BLE001
        return False, f"Cannot open file location: {exc}"
    return True, ""


def open_gallery_preview(
    row: dict[str, str],
    output_path_value: str,
    register_media_root: Callable[[str], None],
    to_media_url: Callable[[str], str],
    gallery_preview_title: Any,
    gallery_popup_video: Any,
    gallery_popup_image: Any,
    gallery_preview_dialog: Any,
) -> tuple[bool, str]:
    name = str(row.get("name", "output"))
    kind = str(row.get("kind", "image"))
    path_value = str(row.get("path", "")).strip()
    if not path_value:
        return False, "Invalid output path"
    register_media_root(output_path_value)
    path_uri = to_media_url(path_value)
    gallery_preview_title.set_text(name)
    if kind == "video":
        gallery_popup_video.set_source(path_uri)
        gallery_popup_video.set_visibility(True)
        gallery_popup_image.set_source("")
        gallery_popup_image.set_visibility(False)
    else:
        gallery_popup_image.set_source(path_uri)
        gallery_popup_image.set_visibility(True)
        gallery_popup_video.set_source("")
        gallery_popup_video.set_visibility(False)
    gallery_preview_dialog.open()
    return True, ""


def render_gallery(
    output_path_value: str,
    gallery_items: Any,
    gallery_status: Any,
    list_latest_outputs: Callable[[str, int], list[dict[str, str]]],
    on_view: Callable[[dict[str, str]], None],
    on_open: Callable[[str], None],
) -> None:
    clear_items = getattr(gallery_items, "clear", None)
    if callable(clear_items):
        clear_items()
    outputs = list_latest_outputs(str(output_path_value or ""), limit=24)
    if not outputs:
        gallery_status.set_text("No output files found")
        return

    gallery_status.set_text(f"{len(outputs)} recent outputs")
    for row in outputs:
        with gallery_items:
            with ui.card().classes("w-[230px] border border-slate-200 bg-white"):
                ui.label(f"[{row['kind']}]").classes("text-xs text-slate-500")
                ui.label(str(row["name"])).classes(
                    "text-sm font-medium text-slate-700 break-all"
                )
                with ui.row().classes("w-full gap-1 justify-between"):
                    ui.button(
                        "View",
                        on_click=lambda _e=None, r=row: on_view(r),
                        color="primary",
                    ).props("dense")
                    ui.button(
                        "Open",
                        on_click=lambda _e=None, p=row["path"]: on_open(p),
                        color="secondary",
                    ).props("dense flat")


def render_gallery_rows(
    rows: list[dict[str, str]],
    gallery_items: Any,
    on_view: Callable[[dict[str, str]], None],
    on_open: Callable[[str], None],
) -> None:
    clear_items = getattr(gallery_items, "clear", None)
    if callable(clear_items):
        clear_items()
    for row in rows:
        with gallery_items:
            with ui.card().classes("w-[230px] border border-slate-200 bg-white"):
                ui.label(f"[{row['kind']}]").classes("text-xs text-slate-500")
                ui.label(str(row["name"])).classes(
                    "text-sm font-medium text-slate-700 break-all"
                )
                with ui.row().classes("w-full gap-1 justify-between"):
                    ui.button(
                        "View",
                        on_click=lambda _e=None, r=row: on_view(r),
                        color="primary",
                    ).props("dense")
                    ui.button(
                        "Open",
                        on_click=lambda _e=None, p=row["path"]: on_open(p),
                        color="secondary",
                    ).props("dense flat")
