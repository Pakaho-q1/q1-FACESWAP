from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .schema import GuiDefaults, to_bool, to_float, to_int


def load_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.is_file():
        return data
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if value and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            data[key] = value
    return data


def load_gui_defaults(project_root: str, load_project_settings: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    project = Path(project_root)
    assets = project / "assets"
    default_input_path = str(project / "input")
    default_output_path = str(project / "output")
    defaults: dict[str, str] = {
        "face_model_name": "",
        "format": "image",
        "input_path": default_input_path,
        "output_path": default_output_path,
        "provider_all": "trt",
        "tuner_mode": "auto",
        "workers_per_stage": "8",
        "worker_queue_size": "64",
        "out_queue_size": "128",
        "gpu_target_util": "95",
        "high_watermark": "12",
        "low_watermark": "4",
        "switch_cooldown_s": "0.35",
        "max_frames": "0",
        "max_retries": "2",
        "parser_mask_blur": "21",
        "swaper_weigh": "0.70",
        "restore_weigh": "0.70",
        "restore_blend": "0.70",
        "restore_choice": "1",
        "parser_choice": "1",
        "use_swaper": "true",
        "use_restore": "true",
        "use_parser": "true",
        "preserve_swap_eyes": "true",
        "dry_run": "false",
        "preview_enabled": "true",
        "preview_fps_limit": "2.5",
    }
    env_defaults = load_env_file(assets / ".env")
    env_user = load_env_file(assets / ".env_user")
    settings = {str(k).lower(): v for k, v in load_project_settings(project_root).items()}
    overrides: dict[str, Any] = {}
    overrides.update(env_defaults)
    overrides.update(env_user)
    overrides.update(settings)
    merged = dict(defaults)
    merged.update(overrides)
    migration_notes: list[str] = []

    face_name_raw = str(overrides.get("face_name", "")).strip()
    if not face_name_raw:
        legacy_face_name = str(overrides.get("face_model_name", "")).strip()
        if legacy_face_name:
            face_name_raw = legacy_face_name
            migration_notes.append("Migrated FACE_MODEL_NAME -> face_name")
    if not face_name_raw:
        face_name_raw = str(defaults.get("face_model_name", "")).strip()

    input_path_raw = str(overrides.get("input_path", "")).strip()
    if not input_path_raw:
        legacy_input_path = str(overrides.get("input_dir", "")).strip()
        if legacy_input_path:
            input_path_raw = legacy_input_path
            migration_notes.append("Migrated INPUT_DIR -> input_path")
    if not input_path_raw:
        input_path_raw = default_input_path

    output_path_raw = str(overrides.get("output_path", "")).strip()
    if not output_path_raw:
        legacy_output_path = str(overrides.get("output_dir", "")).strip()
        if legacy_output_path:
            output_path_raw = legacy_output_path
            migration_notes.append("Migrated OUTPUT_DIR -> output_path")
    if not output_path_raw:
        output_path_raw = default_output_path

    worker_queue_raw = overrides.get("worker_queue_size")
    if worker_queue_raw is None:
        worker_queue_raw = overrides.get("worker-queue")
        if worker_queue_raw is not None:
            migration_notes.append("Migrated worker-queue -> worker_queue_size")
    if worker_queue_raw is None:
        worker_queue_raw = merged.get("worker_queue_size", "64")

    obj = GuiDefaults(
        project_path=project_root,
        face_name=face_name_raw,
        format=str(merged.get("format", "image")),
        input_path=input_path_raw,
        output_path=output_path_raw,
        output_suffix=str(merged.get("output_suffix", "")),
        provider_all=str(merged.get("provider_all", "trt")),
        tuner_mode=str(merged.get("tuner_mode", "auto")),
        workers_per_stage=to_int(merged.get("workers_per_stage", "8"), 8),
        worker_queue_size=to_int(worker_queue_raw, 64),
        out_queue_size=to_int(merged.get("out_queue_size", "128"), 128),
        gpu_target_util=to_int(merged.get("gpu_target_util", "95"), 95),
        high_watermark=to_int(merged.get("high_watermark", "12"), 12),
        low_watermark=to_int(merged.get("low_watermark", "4"), 4),
        switch_cooldown_s=to_float(merged.get("switch_cooldown_s", "0.35"), 0.35),
        max_frames=to_int(merged.get("max_frames", "0"), 0),
        max_retries=to_int(merged.get("max_retries", "2"), 2),
        parser_mask_blur=to_int(merged.get("parser_mask_blur", "21"), 21),
        swapper_blend=to_float(merged.get("swaper_weigh", "0.70"), 0.7),
        restore_weight=to_float(merged.get("restore_weigh", "0.70"), 0.7),
        restore_blend=to_float(merged.get("restore_blend", "0.70"), 0.7),
        restore_choice=str(merged.get("restore_choice", "1")),
        parser_choice=str(merged.get("parser_choice", "1")),
        use_swaper=to_bool(merged.get("use_swaper", "true"), True),
        use_restore=to_bool(merged.get("use_restore", "true"), True),
        use_parser=to_bool(merged.get("use_parser", "true"), True),
        preserve_swap_eyes=to_bool(merged.get("preserve_swap_eyes", "true"), True),
        dry_run=to_bool(merged.get("dry_run", "false"), False),
        preview_enabled=to_bool(merged.get("preview_enabled", "true"), True),
        preview_fps_limit=to_float(merged.get("preview_fps_limit", "2.5"), 2.5),
    )
    payload = obj.as_dict()
    payload["_migration_notes"] = migration_notes
    return payload
