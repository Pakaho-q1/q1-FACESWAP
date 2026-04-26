from __future__ import annotations
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable, Dict, Optional

from core.errors import ConfigError


@dataclass(frozen=True)
class ProviderPolicy:
    default: str
    detect: str
    swap: str
    restore: str
    parse: str

    def for_stage(self, stage: str) -> str:
        mapping = {
            "detect": self.detect,
            "swap": self.swap,
            "restore": self.restore,
            "parse": self.parse,
        }
        return mapping.get(stage, self.default)


@dataclass(frozen=True)
class RunConfig:
    # ── Identity ──────────────────────────────────────────────────────────────
    face_name: str
    format_is_image: bool
    input_path: str
    output_dir: str

    # ── Pipeline switches ─────────────────────────────────────────────────────
    enable_swapper: bool
    enable_restore: bool
    enable_parser: bool

    # ── Swapper ───────────────────────────────────────────────────────────────
    swapper_blend: float

    # ── Restore ───────────────────────────────────────────────────────────────
    restore_choice: str        # "1" GFPGAN | "2" GPEN512 | "3" GPEN1024 | "4" CodeFormer
    restore_size: int          # derived from restore_choice
    restore_model_name: str    # derived from restore_choice
    restore_weight: float
    restore_blend: float

    # ── Parser ────────────────────────────────────────────────────────────────
    parser_choice: str         # "1" BiSeNet | "2" SegFormer
    parser_type: str           # "bisenet" | "segformer"  (derived)
    parser_mask_blur: int
    preserve_swap_eyes: bool

    # ── Runtime ───────────────────────────────────────────────────────────────
    workers_per_stage: int
    worker_queue_size: int
    out_queue_size: int
    tuner_mode: str
    gpu_target_util: int
    high_watermark: int
    low_watermark: int
    switch_cooldown_s: float
    max_retries: int
    max_frames: int
    skip_existing: bool
    output_suffix: str

    # ── Paths ─────────────────────────────────────────────────────────────────
    models_dir: str
    assets_dir: str
    ffmpeg_cmd: str            # cross-platform: set by build_run_config_from_cfg
    insightface_root: str
    faces_dir: str
    temp_audio_dir: str
    tensorrt_dir: str
    source_face_path: str
    swapper_model: str
    restore_model_path: str
    parser_model: str
    trt_cache_dir: str
    trt_cache_detect_dir: str
    trt_cache_swap_dir: str
    trt_cache_restore_dir: str
    trt_cache_parser_dir: str

    # ── Providers ─────────────────────────────────────────────────────────────
    provider_policy: ProviderPolicy


# ---------------------------------------------------------------------------
# Dataclasses that make up RuntimeContext
# ---------------------------------------------------------------------------

@dataclass
class RunState:
    stop_tuner: Event = field(default_factory=Event)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunHooks:
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_progress: Optional[Callable[[str, int, int], None]] = None


@dataclass
class ModelRegistry:
    manager: Optional[Any] = None
    app: Optional[Any] = None
    swapper: Optional[Any] = None
    restore_session: Optional[Any] = None
    parser_session: Optional[Any] = None


@dataclass
class RuntimeContext:
    config: RunConfig
    state: RunState = field(default_factory=RunState)
    models: ModelRegistry = field(default_factory=ModelRegistry)
    hooks: RunHooks = field(default_factory=RunHooks)

    def emit_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self.hooks.on_event is not None:
            self.hooks.on_event(name, payload or {})

    def emit_progress(self, label: str, completed: int, total: int) -> None:
        if self.hooks.on_progress is not None:
            self.hooks.on_progress(label, int(completed), int(total))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_run_config(run_config: RunConfig) -> None:
    if not run_config.face_name:
        raise ConfigError("face_name is required")
    if not run_config.input_path:
        raise ConfigError("input_path is required")
    if not run_config.output_dir:
        raise ConfigError("output_dir is required")
    if run_config.workers_per_stage < 1 or run_config.workers_per_stage > 128:
        raise ConfigError("workers_per_stage must be in range [1, 128]")
    if run_config.worker_queue_size < 4 or run_config.worker_queue_size > 4096:
        raise ConfigError("worker_queue_size must be in range [4, 4096]")
    if run_config.out_queue_size < 8 or run_config.out_queue_size > 8192:
        raise ConfigError("out_queue_size must be in range [8, 8192]")
    if run_config.gpu_target_util < 50 or run_config.gpu_target_util > 100:
        raise ConfigError("gpu_target_util must be in range [50, 100]")
    if run_config.high_watermark < 1:
        raise ConfigError("high_watermark must be >= 1")
    if run_config.low_watermark < 0:
        raise ConfigError("low_watermark must be >= 0")
    if run_config.low_watermark >= run_config.high_watermark:
        raise ConfigError("low_watermark must be < high_watermark")
    if run_config.max_retries < 1 or run_config.max_retries > 20:
        raise ConfigError("max_retries must be in range [1, 20]")
    if not run_config.models_dir:
        raise ConfigError("models_dir is required")
    if not run_config.assets_dir:
        raise ConfigError("assets_dir is required")
    if not run_config.ffmpeg_cmd:
        raise ConfigError("ffmpeg_cmd is required")
    if not run_config.insightface_root:
        raise ConfigError("insightface_root is required")
    for name, value in [
        ("swapper_blend", run_config.swapper_blend),
        ("restore_weight", run_config.restore_weight),
        ("restore_blend", run_config.restore_blend),
    ]:
        if value < 0.0 or value > 1.0:
            raise ConfigError(f"{name} must be in range [0.0, 1.0]")
    if run_config.parser_mask_blur < 1 or run_config.parser_mask_blur > 255:
        raise ConfigError("parser_mask_blur must be in range [1, 255]")
    if run_config.parser_mask_blur % 2 == 0:
        raise ConfigError("parser_mask_blur must be an odd number")


# ---------------------------------------------------------------------------
# Factory: build RunConfig from the legacy cfg_module (CLI path)
# ---------------------------------------------------------------------------

def build_run_config_from_cfg(cfg_module) -> RunConfig:
    provider_policy = ProviderPolicy(
        default=cfg_module.PROVIDER_ALL,
        detect=cfg_module.PROVIDER_POLICY.get("detect", cfg_module.PROVIDER_ALL),
        swap=cfg_module.PROVIDER_POLICY.get("swap", cfg_module.PROVIDER_ALL),
        restore=cfg_module.PROVIDER_POLICY.get("restore", cfg_module.PROVIDER_ALL),
        parse=cfg_module.PROVIDER_POLICY.get("parse", cfg_module.PROVIDER_ALL),
    )

    models_dir = cfg_module.MODELS_DIR
    ffmpeg_cmd = cfg_module.FFMPEG_CMD

    return RunConfig(
        face_name=cfg_module.FACE_NAME,
        format_is_image=cfg_module.FORMAT_IS_IMAGE,
        input_path=cfg_module.INPUT_PATH,
        output_dir=cfg_module.OUTPUT_DIR,
        enable_swapper=cfg_module.ENABLE_SWAPPER,
        enable_restore=cfg_module.ENABLE_RESTORE,
        enable_parser=cfg_module.ENABLE_PARSER,
        swapper_blend=cfg_module.SWAPPER_BLEND,
        restore_choice=cfg_module.RESTORE_CHOICE,
        restore_size=cfg_module.RESTORE_SIZE,
        restore_model_name=cfg_module.RESTORE_MODEL_NAME,
        restore_weight=cfg_module.RESTORE_WEIGHT,
        restore_blend=cfg_module.RESTORE_BLEND,
        parser_choice=cfg_module.PARSER_CHOICE,
        parser_type=cfg_module.PARSER_TYPE,
        parser_mask_blur=cfg_module.PARSER_MASK_BLUR,
        preserve_swap_eyes=cfg_module.PRESERVE_SWAP_EYES,
        workers_per_stage=cfg_module.WORKERS_PER_STAGE,
        worker_queue_size=cfg_module.WORKER_QUEUE_SIZE,
        out_queue_size=cfg_module.OUT_QUEUE_SIZE,
        tuner_mode=cfg_module.TUNER_MODE,
        gpu_target_util=cfg_module.GPU_TARGET_UTIL,
        high_watermark=cfg_module.HIGH_WATERMARK,
        low_watermark=cfg_module.LOW_WATERMARK,
        switch_cooldown_s=cfg_module.SWITCH_COOLDOWN_S,
        max_retries=cfg_module.MAX_RETRIES,
        max_frames=cfg_module.MAX_FRAMES,
        skip_existing=cfg_module.SKIP_EXISTING,
        output_suffix=cfg_module.OUTPUT_SUFFIX,
        models_dir=models_dir,
        assets_dir=cfg_module.ASSETS_HOME,
        ffmpeg_cmd=ffmpeg_cmd,
        insightface_root=cfg_module.INSIGHTFACE_ROOT,
        faces_dir=cfg_module.FACES_DIR,
        temp_audio_dir=cfg_module.TEMP_AUDIO_DIR,
        tensorrt_dir=cfg_module.TENSORRT_DIR,
        source_face_path=cfg_module.SOURCE_FACE_PATH,
        swapper_model=cfg_module.SWAPPER_MODEL,
        restore_model_path=cfg_module.RESTORE_MODEL_PATH,
        parser_model=cfg_module.PARSER_MODEL,
        trt_cache_dir=cfg_module.TRT_CACHE_DIR,
        trt_cache_detect_dir=cfg_module.TRT_CACHE_DETECT_DIR,
        trt_cache_swap_dir=cfg_module.TRT_CACHE_SWAP_DIR,
        trt_cache_restore_dir=cfg_module.TRT_CACHE_RESTORE_DIR,
        trt_cache_parser_dir=cfg_module.TRT_CACHE_PARSER_DIR,
        provider_policy=provider_policy,
    )
