from __future__ import annotations

from typing import Any, Callable


def set_health_action(
    *,
    project_root: str,
    output_path_value: str,
    collect_runtime_health: Callable[[str, str], dict[str, Any]],
    apply_health_report: Callable[..., dict[str, Any]],
    health_models: Any,
    health_ffmpeg: Any,
    health_tensorrt: Any,
    health_disk: Any,
    health_writable: Any,
    health_missing: Any,
) -> dict[str, Any]:
    return apply_health_report(
        project_root=project_root,
        output_path_value=output_path_value,
        collect_runtime_health=collect_runtime_health,
        health_models=health_models,
        health_ffmpeg=health_ffmpeg,
        health_tensorrt=health_tensorrt,
        health_disk=health_disk,
        health_writable=health_writable,
        health_missing=health_missing,
    )


def run_setup_wizard_checks_action(
    *,
    project_root: str,
    validate_project_path: Callable[[str], tuple[bool, str]],
    check_tensorrt_status: Callable[[str], dict[str, Any]],
    refresh_health: Callable[[], dict[str, Any]],
    run_setup_checks: Callable[..., None],
    wizard_project_status: Any,
    wizard_model_status: Any,
    wizard_trt_status: Any,
) -> None:
    run_setup_checks(
        project_root=project_root,
        validate_project_path=validate_project_path,
        check_tensorrt_status=check_tensorrt_status,
        refresh_health=refresh_health,
        wizard_project_status=wizard_project_status,
        wizard_model_status=wizard_model_status,
        wizard_trt_status=wizard_trt_status,
    )
