from __future__ import annotations

from pathlib import Path
from typing import Any

_ALLOWED_FORMATS = {"image", "video"}
_ALLOWED_PROVIDER = {"cpu", "cuda", "trt"}
_ALLOWED_TUNER = {"auto", "max_util", "stable"}
_ALLOWED_FILE_SORTING = {
    "date_modified_newest",
    "date_modified_oldest",
    "date_created_newest",
    "date_created_oldest",
    "size_smallest_largest",
    "size_largest_smallest",
    "name_az",
    "name_za",
}


def validate_run_config(values: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Validate normalized run values before pipeline start.

    Returns (errors, warnings).
    """
    errors: list[str] = []
    warnings: list[str] = []

    face_name = str(values.get("face_name", "")).strip()
    if not face_name:
        errors.append("face_name is required")

    fmt = str(values.get("format", "")).strip().lower()
    if fmt not in _ALLOWED_FORMATS:
        errors.append("format must be image or video")

    provider = str(values.get("provider_all", "")).strip().lower()
    if provider not in _ALLOWED_PROVIDER:
        errors.append("provider_all must be one of: cpu, cuda, trt")

    tuner_mode = str(values.get("tuner_mode", "")).strip().lower()
    if tuner_mode not in _ALLOWED_TUNER:
        errors.append("tuner_mode must be one of: auto, max_util, stable")

    file_sorting = str(values.get("file_sorting", "")).strip().lower()
    if file_sorting not in _ALLOWED_FILE_SORTING:
        errors.append("file_sorting is invalid")

    def _as_int(key: str, minimum: int, maximum: int | None = None) -> int | None:
        raw = values.get(key)
        try:
            v = int(raw)
        except Exception:  # noqa: BLE001
            errors.append(f"{key} must be an integer")
            return None
        if v < minimum:
            errors.append(f"{key} must be >= {minimum}")
        if maximum is not None and v > maximum:
            errors.append(f"{key} must be <= {maximum}")
        return v

    def _as_float(key: str, minimum: float, maximum: float | None = None) -> float | None:
        raw = values.get(key)
        try:
            v = float(raw)
        except Exception:  # noqa: BLE001
            errors.append(f"{key} must be a number")
            return None
        if v < minimum:
            errors.append(f"{key} must be >= {minimum}")
        if maximum is not None and v > maximum:
            errors.append(f"{key} must be <= {maximum}")
        return v

    _as_int("workers_per_stage", 1, 128)
    _as_int("worker_queue_size", 4, 8192)
    _as_int("out_queue_size", 8, 16384)
    _as_int("gpu_target_util", 0, 100)
    _as_int("high_watermark", 1, 8192)
    low_watermark = _as_int("low_watermark", 0, 8192)
    high_watermark = _as_int("high_watermark", 1, 8192)
    if low_watermark is not None and high_watermark is not None and low_watermark > high_watermark:
        errors.append("low_watermark must be <= high_watermark")

    _as_float("switch_cooldown_s", 0.0, 120.0)
    _as_int("max_frames", 0, 10_000_000)
    _as_int("max_retries", 0, 100)
    _as_int("parser_mask_blur", 1, 255)
    _as_float("swapper_blend", 0.0, 1.0)
    _as_float("restore_weight", 0.0, 1.0)
    _as_float("restore_blend", 0.0, 1.0)
    _as_float("preview_fps_limit", 0.5, 60.0)

    input_path = Path(str(values.get("input_path", "")).strip())
    output_path = Path(str(values.get("output_path", "")).strip())
    if not str(input_path):
        errors.append("input_path is required")
    elif not input_path.exists() or not input_path.is_dir():
        errors.append("input_path must be an existing directory")

    if not str(output_path):
        errors.append("output_path is required")
    else:
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            probe = output_path / ".q1_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"output_path is not writable: {exc}")

    if provider == "trt":
        warnings.append("provider_all=trt requires TensorRT runtime to be installed")

    return errors, warnings
