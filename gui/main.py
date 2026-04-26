from __future__ import annotations

import sys
from pathlib import Path

from nicegui import ui

# Ensure project root is importable when running as script: `python gui/main.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from gui.main_ui import build_main_ui, register_ui_assets
    from gui.project_bootstrap import (
        detect_requires_project_path,
        disk_space_gb,
        ensure_workspace,
        load_last_project_root,
        validate_project_path,
    )
    from gui.state import reset_store
    from gui.widgets import add_path_picker
except ImportError:
    from main_ui import build_main_ui, register_ui_assets  # type: ignore
    from project_bootstrap import (  # type: ignore
        detect_requires_project_path,
        disk_space_gb,
        ensure_workspace,
        load_last_project_root,
        validate_project_path,
    )
    from state import reset_store  # type: ignore
    from widgets import add_path_picker  # type: ignore


def build_page() -> None:
    ui.dark_mode(False)
    ui.page_title("q1-FaceSwap")
    register_ui_assets()

    requires_project = detect_requires_project_path()
    saved_project = load_last_project_root()
    # Source mode: always use current repository root as default workspace.
    # Wheel/site-packages mode: use last saved project path if available.
    if requires_project:
        default_project = saved_project or ""
    else:
        default_project = str(PROJECT_ROOT)
    normalized_default = ensure_workspace(default_project) if default_project else ""

    with ui.column().classes("app-shell w-full p-4 gap-4") as root:
        if requires_project and not saved_project:
            ui.label("Initial Setup").classes("text-2xl font-bold text-slate-800")
            ui.label(
                "Detected site-packages runtime. Please choose a writable project path."
            ).classes("text-sm text-slate-600")
            with ui.card().classes("card-soft w-full max-w-3xl"):
                project_input = ui.input(
                    "Project Path", value=str(Path.home() / "Documents")
                ).classes("w-full")
                setup_status = ui.label("").classes("text-sm")
                with ui.row().classes("items-center w-full"):
                    add_path_picker(
                        project_input, "Select Project Root", pick_file=False
                    )
                    validate_btn = ui.button("Validate", color="secondary")
                    apply_btn = ui.button(
                        "Create Workspace and Continue", color="primary"
                    )

                def run_validation() -> tuple[bool, str]:
                    chosen = str(project_input.value or "").strip()
                    if not chosen:
                        setup_status.set_text("Project Path is required")
                        setup_status.classes("text-red-600")
                        return False, ""
                    ok, reason = validate_project_path(chosen)
                    if not ok:
                        setup_status.set_text(reason)
                        setup_status.classes("text-red-600")
                        return False, chosen
                    free_gb = disk_space_gb(chosen)
                    if free_gb >= 0 and free_gb < 10.0:
                        setup_status.set_text(
                            f"Path is writable. Low disk space: {free_gb:.1f} GB free"
                        )
                        setup_status.classes("text-amber-700")
                    elif free_gb >= 0:
                        setup_status.set_text(
                            f"Path is writable. Free space: {free_gb:.1f} GB"
                        )
                        setup_status.classes("text-emerald-700")
                    else:
                        setup_status.set_text("Path is writable.")
                        setup_status.classes("text-emerald-700")
                    return True, chosen

                def apply_project() -> None:
                    ok, chosen = run_validation()
                    if not ok:
                        ui.notify(
                            "Please choose a valid writable path", color="warning"
                        )
                        return
                    resolved = ensure_workspace(chosen)
                    reset_store()
                    root.clear()
                    build_main_ui(root, resolved)

                validate_btn.on_click(lambda: run_validation())
                apply_btn.on_click(apply_project)
        else:
            reset_store()
            build_main_ui(root, normalized_default)


build_page()


if __name__ in {"__main__", "__mp_main__"}:
    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    ui.run(title="q1-FaceSwap", reload=True)
