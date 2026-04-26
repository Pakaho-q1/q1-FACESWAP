from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable


def apply_health_report(
    project_root: str,
    output_path_value: str,
    collect_runtime_health: Callable[[str, str], dict[str, Any]],
    health_models: Any,
    health_ffmpeg: Any,
    health_tensorrt: Any,
    health_disk: Any,
    health_writable: Any,
    health_missing: Any,
) -> dict[str, Any]:
    report = collect_runtime_health(project_root, output_path_value)
    health_models.set_text(f"{report['models_found']}/{report['models_total']}")
    health_ffmpeg.set_text("OK" if report["ffmpeg_ok"] else "Missing")
    health_tensorrt.set_text("OK" if report["tensorrt_ok"] else "Missing")
    free_gb = float(report.get("free_disk_gb", -1.0))
    health_disk.set_text(f"{free_gb:.1f} GB" if free_gb >= 0 else "Unknown")
    health_writable.set_text("OK" if report["writable_output"] else "No Access")
    missing_models = list(report.get("missing_models", []))
    if missing_models:
        health_missing.set_text(f"Missing models: {', '.join(missing_models[:8])}")
    else:
        health_missing.set_text("")
    return report


def validate_output_path(path_value: str) -> tuple[bool, str]:
    p = Path(path_value).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix="q1fs_out_", dir=str(p))
        os.close(fd)
        Path(tmp_name).unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        return False, f"Output path is not writable: {exc}"
    return True, "Output path writable."


def validate_form(
    face_name: str,
    face_names: list[str],
    input_path_value: str,
    output_path_value: str,
    validate_face_label: Any,
    validate_input_label: Any,
    validate_output_label: Any,
    run_btn: Any,
    controller_running: bool,
) -> bool:
    selected_face = str(face_name or "").strip()
    if not selected_face:
        validate_face_label.set_text("Face model is required.")
        validate_face_label.classes("text-xs text-rose-700")
        face_ok = False
    elif selected_face not in face_names:
        validate_face_label.set_text(f"Face model not found in assets/faces: {selected_face}")
        validate_face_label.classes("text-xs text-rose-700")
        face_ok = False
    else:
        validate_face_label.set_text("Face model ready.")
        validate_face_label.classes("text-xs text-emerald-700")
        face_ok = True

    in_path = Path(str(input_path_value or "").strip()).expanduser()
    input_files = 0
    if not in_path.is_dir():
        validate_input_label.set_text("Input path does not exist or is not a directory.")
        validate_input_label.classes("text-xs text-rose-700")
        input_ok = False
    else:
        try:
            input_files = sum(1 for p in in_path.iterdir() if p.is_file())
        except OSError:
            input_files = 0
        if input_files <= 0:
            validate_input_label.set_text("Input folder is empty.")
            validate_input_label.classes("text-xs text-amber-700")
            input_ok = False
        else:
            validate_input_label.set_text(f"Input folder ready ({input_files} files).")
            validate_input_label.classes("text-xs text-emerald-700")
            input_ok = True

    out_ok, out_msg = validate_output_path(str(output_path_value or "").strip())
    validate_output_label.set_text(out_msg)
    validate_output_label.classes("text-xs text-emerald-700" if out_ok else "text-xs text-rose-700")

    valid = face_ok and input_ok and out_ok
    if controller_running:
        run_btn.disable()
    else:
        if valid:
            run_btn.enable()
        else:
            run_btn.disable()
    return valid


def run_setup_checks(
    project_root: str,
    validate_project_path: Callable[[str], tuple[bool, str]],
    check_tensorrt_status: Callable[[str], dict[str, Any]],
    refresh_health: Callable[[], dict[str, Any]],
    wizard_project_status: Any,
    wizard_model_status: Any,
    wizard_trt_status: Any,
) -> None:
    ok_path, reason = validate_project_path(project_root)
    if ok_path:
        wizard_project_status.set_text(f"OK: {project_root}")
        wizard_project_status.classes("text-xs text-emerald-700")
    else:
        wizard_project_status.set_text(f"ERROR: {reason}")
        wizard_project_status.classes("text-xs text-rose-700")

    report = refresh_health()
    models_found = int(report.get("models_found", 0))
    models_total = int(report.get("models_total", 0))
    if models_total > 0 and models_found == models_total:
        wizard_model_status.set_text(f"OK: {models_found}/{models_total} models available")
        wizard_model_status.classes("text-xs text-emerald-700")
    else:
        wizard_model_status.set_text(f"WARNING: {models_found}/{models_total} models available")
        wizard_model_status.classes("text-xs text-amber-700")

    trt_state = check_tensorrt_status(project_root)
    if bool(trt_state.get("ok", False)):
        wizard_trt_status.set_text("OK: TensorRT runtime complete")
        wizard_trt_status.classes("text-xs text-emerald-700")
    else:
        wizard_trt_status.set_text(
            f"WARNING: missing {', '.join(list(trt_state.get('missing', [])))} | target: {trt_state.get('bin', '')}"
        )
        wizard_trt_status.classes("text-xs text-amber-700")
