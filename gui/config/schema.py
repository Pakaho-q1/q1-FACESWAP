from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def to_bool(raw: Any, fallback: bool) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    return fallback


def to_int(raw: Any, fallback: int) -> int:
    try:
        return int(str(raw).strip())
    except Exception:  # noqa: BLE001
        return fallback


def to_float(raw: Any, fallback: float) -> float:
    try:
        return float(str(raw).strip())
    except Exception:  # noqa: BLE001
        return fallback


@dataclass(frozen=True)
class GuiDefaults:
    project_path: str
    face_name: str
    format: str
    input_path: str
    output_path: str
    output_suffix: str
    provider_all: str
    tuner_mode: str
    workers_per_stage: int
    worker_queue_size: int
    out_queue_size: int
    gpu_target_util: int
    high_watermark: int
    low_watermark: int
    switch_cooldown_s: float
    max_frames: int
    max_retries: int
    parser_mask_blur: int
    swapper_blend: float
    restore_weight: float
    restore_blend: float
    restore_choice: str
    parser_choice: str
    use_swaper: bool
    use_restore: bool
    use_parser: bool
    preserve_swap_eyes: bool
    dry_run: bool
    preview_enabled: bool
    preview_fps_limit: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_path": self.project_path,
            "face_name": self.face_name,
            "format": self.format,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "output_suffix": self.output_suffix,
            "provider_all": self.provider_all,
            "tuner_mode": self.tuner_mode,
            "workers_per_stage": self.workers_per_stage,
            "worker_queue_size": self.worker_queue_size,
            "out_queue_size": self.out_queue_size,
            "gpu_target_util": self.gpu_target_util,
            "high_watermark": self.high_watermark,
            "low_watermark": self.low_watermark,
            "switch_cooldown_s": self.switch_cooldown_s,
            "max_frames": self.max_frames,
            "max_retries": self.max_retries,
            "parser_mask_blur": self.parser_mask_blur,
            "swapper_blend": self.swapper_blend,
            "restore_weight": self.restore_weight,
            "restore_blend": self.restore_blend,
            "restore_choice": self.restore_choice,
            "parser_choice": self.parser_choice,
            "use_swaper": self.use_swaper,
            "use_restore": self.use_restore,
            "use_parser": self.use_parser,
            "preserve_swap_eyes": self.preserve_swap_eyes,
            "dry_run": self.dry_run,
            "preview_enabled": self.preview_enabled,
            "preview_fps_limit": self.preview_fps_limit,
        }
