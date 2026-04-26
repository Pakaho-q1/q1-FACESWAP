from __future__ import annotations

import os
from pathlib import Path

from nicegui import ui


def add_path_picker(target_input, title: str, pick_file: bool = False) -> None:
    dialog = ui.dialog()
    state = {"cwd": str(Path(target_input.value or os.getcwd()).resolve())}

    with dialog, ui.card().classes("w-[700px]"):
        ui.label(title).classes("text-lg font-semibold")
        cwd_label = ui.label("")
        list_box = ui.column().classes("w-full max-h-80 overflow-auto border rounded p-2 gap-1")

        def refresh() -> None:
            try:
                root = Path(state["cwd"]).resolve()
            except Exception:
                root = Path(os.getcwd()).resolve()
                state["cwd"] = str(root)
            cwd_label.set_text(f"Current: {root}")
            list_box.clear()
            with list_box:
                try:
                    entries = list(root.iterdir())
                except Exception:
                    entries = []
                dirs = sorted([p for p in entries if p.is_dir()], key=lambda p: p.name.lower())
                files = sorted([p for p in entries if p.is_file()], key=lambda p: p.name.lower())
                for d in dirs:
                    ui.button(
                        f"[DIR] {d.name}",
                        on_click=lambda _, p=d: _enter_dir(p),
                    ).props("flat dense").classes("justify-start w-full")
                if pick_file:
                    for f in files:
                        ui.button(
                            f"[FILE] {f.name}",
                            on_click=lambda _, p=f: _select_file(p),
                        ).props("flat dense").classes("justify-start w-full")

        def _enter_dir(path: Path) -> None:
            state["cwd"] = str(path.resolve())
            refresh()

        def _go_up() -> None:
            state["cwd"] = str(Path(state["cwd"]).resolve().parent)
            refresh()

        def _select_current_dir() -> None:
            target_input.set_value(str(Path(state["cwd"]).resolve()))
            dialog.close()

        def _select_file(path: Path) -> None:
            target_input.set_value(str(path.resolve()))
            dialog.close()

        with ui.row().classes("gap-2"):
            ui.button("Up", on_click=lambda: _go_up())
            ui.button("Refresh", on_click=lambda: refresh())
            if not pick_file:
                ui.button("Use This Folder", on_click=lambda: _select_current_dir(), color="primary")
            ui.button("Close", on_click=lambda: dialog.close())

        refresh()

    ui.button("Browse", on_click=dialog.open).props("outline dense")
