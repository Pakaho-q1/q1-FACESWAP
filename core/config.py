import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

from core.project_layout import (
    ensure_project_layout,
    is_site_packages_path,
    normalize_project_root,
)

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CORE_DIR)
SOURCE_ASSETS_DIR = os.path.join(BASE_DIR, "assets")
SOURCE_DOCS_DIR = os.path.join(SOURCE_ASSETS_DIR, "docs")
ENV_OFFICIAL_PATH = os.path.join(SOURCE_ASSETS_DIR, ".env")
ENV_USER_PATH = os.path.join(SOURCE_ASSETS_DIR, ".env_user")


def _parse_bool(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_format(value):
    text = str(value).strip().lower()
    if text in {"1", "image"}:
        return "image"
    if text in {"2", "video"}:
        return "video"
    raise argparse.ArgumentTypeError("FORMAT must be one of: image, video.")


def _load_env_file(path):
    data = {}
    if not os.path.isfile(path):
        return data

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            data[key] = value
    return data


def _first_defined(merged, keys, fallback=""):
    for key in keys:
        if key in merged and str(merged[key]).strip() != "":
            return str(merged[key]).strip()
    return fallback


def _get_env_str(merged, key, fallback=""):
    return _first_defined(merged, [key], fallback)


def _get_env_bool(merged, key, fallback):
    raw = _first_defined(merged, [key], "")
    if raw == "":
        return fallback
    return _parse_bool(raw)


def _get_env_int(merged, key, fallback):
    raw = _first_defined(merged, [key], "")
    if raw == "":
        return fallback
    try:
        return int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer for {key}: {raw}") from exc


def _get_env_float(merged, key, fallback):
    raw = _first_defined(merged, [key], "")
    if raw == "":
        return fallback
    try:
        return float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float for {key}: {raw}") from exc


def _get_env_choice(merged, key, fallback, allowed):
    raw = _first_defined(merged, [key], "")
    if raw == "":
        return fallback
    if raw not in allowed:
        raise argparse.ArgumentTypeError(
            f"Invalid value for {key}: {raw} (allowed: {','.join(allowed)})"
        )
    return raw


def _parse_provider(value, field_name):
    normalized = str(value).strip().lower()
    if normalized in {"cpu", "cuda", "trt", "auto"}:
        return normalized
    raise argparse.ArgumentTypeError(
        f"{field_name} must be one of: auto, cpu, cuda, trt."
    )


def _platform_ffmpeg_name() -> str:
    """Return the correct ffmpeg binary name for the current OS."""
    return "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"


def _default_user_home_root() -> str:
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        return os.path.join(local_app_data, "q1-faceswap")
    # Cross-platform fallback when LOCALAPPDATA is unavailable.
    return os.path.join(os.path.expanduser("~"), ".q1-faceswap")


def _resolve_default_project_root(base_dir: str) -> str:
    if is_site_packages_path(base_dir):
        return normalize_project_root(_default_user_home_root())
    return normalize_project_root(base_dir)


TENSORRT_DOWNLOAD_URL = "https://developer.nvidia.com/tensorrt/download/10x"


def _manifest_path(models_dir: str) -> str:
    return os.path.join(models_dir, "model_manifest.json")


def _default_model_manifest() -> dict:
    base = "https://huggingface.co/Pakaho-q1/onnx-models/resolve/main"
    return {
        "version": 1,
        "models": [
            {"filename": "ffmpeg.exe", "url": f"{base}/ffmpeg.exe", "sha256": ""},
            {"filename": "inswapper_128.onnx", "url": f"{base}/inswapper_128.onnx", "sha256": ""},
            {"filename": "faceparser_resnet34.onnx", "url": f"{base}/faceparser_resnet34.onnx", "sha256": ""},
            {"filename": "codeformer.onnx", "url": f"{base}/codeformer.onnx", "sha256": ""},
            {"filename": "Segformer_CelebAMask-HQ.onnx", "url": f"{base}/Segformer_CelebAMask-HQ.onnx", "sha256": ""},
            {"filename": "GPEN-BFR-512.onnx", "url": f"{base}/GPEN-BFR-512.onnx", "sha256": ""},
            {"filename": "GPEN-BFR-1024.onnx", "url": f"{base}/GPEN-BFR-1024.onnx", "sha256": ""},
            {"filename": "GFPGANv1.4.onnx", "url": f"{base}/GFPGANv1.4.onnx", "sha256": ""},
            {"filename": "1k3d68.onnx", "url": f"{base}/buffalo_l/1k3d68.onnx", "sha256": ""},
            {"filename": "2d106det.onnx", "url": f"{base}/buffalo_l/2d106det.onnx", "sha256": ""},
            {"filename": "det_10g.onnx", "url": f"{base}/buffalo_l/det_10g.onnx", "sha256": ""},
            {"filename": "genderage.onnx", "url": f"{base}/buffalo_l/genderage.onnx", "sha256": ""},
            {"filename": "w600k_r50.onnx", "url": f"{base}/buffalo_l/w600k_r50.onnx", "sha256": ""},
        ],
    }


def _ensure_model_manifest(models_dir: str) -> str:
    path = _manifest_path(models_dir)
    if os.path.isfile(path):
        return path
    os.makedirs(models_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_default_model_manifest(), f, indent=2, ensure_ascii=False)
    return path


def _load_model_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("model_manifest.json must contain a JSON object.")
    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise argparse.ArgumentTypeError("model_manifest.json must contain non-empty 'models' list.")
    normalized = []
    for idx, item in enumerate(models):
        if not isinstance(item, dict):
            raise argparse.ArgumentTypeError(f"model_manifest.json models[{idx}] must be an object.")
        filename = str(item.get("filename", "")).strip()
        url = str(item.get("url", "")).strip()
        sha256 = str(item.get("sha256", "")).strip().lower()
        if not filename:
            raise argparse.ArgumentTypeError(f"model_manifest.json models[{idx}] missing filename.")
        if not url:
            raise argparse.ArgumentTypeError(f"model_manifest.json models[{idx}] missing url.")
        if sha256 and len(sha256) != 64:
            raise argparse.ArgumentTypeError(
                f"model_manifest.json models[{idx}] sha256 must be 64 hex chars when provided."
            )
        normalized.append({"filename": filename, "url": url, "sha256": sha256})
    return {"version": int(data.get("version", 1)), "models": normalized}


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_file(url: str, destination: str) -> None:
    tmp_path = destination + ".tmp"
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp_path, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp_path, destination)


def _sync_models_from_manifest(
    models_dir: str,
    manifest: dict,
    preload_models: bool,
    required_filenames: set[str],
) -> None:
    manifest_entries = {entry["filename"]: entry for entry in manifest["models"]}

    missing_required = [name for name in sorted(required_filenames) if name not in manifest_entries]
    if missing_required:
        raise argparse.ArgumentTypeError(
            "model_manifest.json missing required models: " + ", ".join(missing_required)
        )

    for filename in required_filenames:
        entry = manifest_entries[filename]
        local_path = os.path.join(models_dir, filename)
        expected_sha = entry["sha256"]
        if os.path.isfile(local_path):
            if expected_sha and _sha256_file(local_path) != expected_sha:
                if not preload_models:
                    raise argparse.ArgumentTypeError(
                        f"Checksum mismatch for {filename}. Set --preload-models true to re-download."
                    )
                os.remove(local_path)
            else:
                continue
        if not preload_models:
            raise argparse.ArgumentTypeError(
                f"Required model missing: {local_path}. "
                "Set --preload-models true or place the file manually."
            )
        try:
            _download_file(entry["url"], local_path)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise argparse.ArgumentTypeError(
                f"Failed downloading {filename} from {entry['url']}: {exc}"
            ) from exc
        if expected_sha and _sha256_file(local_path) != expected_sha:
            raise argparse.ArgumentTypeError(
                f"Checksum mismatch after download for {filename}."
            )


CODE_DEFAULTS = {
    "PROJECT_PATH": "",
    "FACE_MODEL_NAME": "",
    "FORMAT": "image",
    "INPUT_PATH": "",
    "USE_RESTORE": "true",
    "RESTORE_CHOICE": "1",
    "USE_PARSER": "true",
    "PARSER_CHOICE": "1",
    "WORKERS_PER_STAGE": "8",
    "WORKER_QUEUE_SIZE": "64",
    "OUT_QUEUE_SIZE": "128",
    "TUNER_MODE": "auto",
    "GPU_TARGET_UTIL": "95",
    "HIGH_WATERMARK": "12",
    "LOW_WATERMARK": "4",
    "SWITCH_COOLDOWN_S": "0.35",
    "PRESERVE_SWAP_EYES": "true",
    "PARSER_MASK_BLUR": "21",
    "DRY_RUN": "false",
    "LOG_LEVEL": "warning",
    "OUTPUT_SUFFIX": "",
    "SKIP_EXISTING": "true",
    "MAX_FRAMES": "0",
    "MAX_RETRIES": "2",
    "PRINT_EFFECTIVE_CONFIG": "false",
    "OUTPUT_PATH": "",
    "FACE_MODEL_PATH": "",
    "PROVIDER_ALL": "trt",
    "PROVIDER_SWAPER": "auto",
    "PROVIDER_RESTORE": "auto",
    "PROVIDER_PARSER": "auto",
    "PROVIDER_DETECT": "auto",
    "USE_SWAPER": "true",
    "SWAPER_WEIGH": "0.70",
    "RESTORE_WEIGH": "0.70",
    "RESTORE_BLEND": "0.70",
    "MODEL_HOME": "",
    "ASSETS_HOME": "",
    "PRELOAD_MODELS": "false",
}

ENV_VALUES = dict(CODE_DEFAULTS)
ENV_VALUES.update(_load_env_file(ENV_OFFICIAL_PATH))
ENV_VALUES.update(_load_env_file(ENV_USER_PATH))

# Allow legacy env keys.
if _first_defined(ENV_VALUES, ["FACE_MODEL_NAME"], "") == "":
    legacy_face = _first_defined(ENV_VALUES, ["FACE_NAME"], "")
    if legacy_face:
        ENV_VALUES["FACE_MODEL_NAME"] = legacy_face
if _first_defined(ENV_VALUES, ["INPUT_PATH"], "") == "":
    legacy_input = _first_defined(ENV_VALUES, ["INPUT_DIR"], "")
    if legacy_input:
        ENV_VALUES["INPUT_PATH"] = legacy_input


parser = argparse.ArgumentParser(description="Unified Face Swap System (Image/Video)")
parser.add_argument(
    "--project-path", "--project_path", dest="PROJECT_PATH", type=str,
    default=_get_env_str(ENV_VALUES, "PROJECT_PATH", ""),
    help="Base directory where q1-FACESWAP workspace will be created/used.",
)
parser.add_argument(
    "--model-home", "--model_home", dest="MODEL_HOME", type=str,
    default=_get_env_str(ENV_VALUES, "MODEL_HOME", ""),
    help="Directory containing model binaries and assets (overrides default model path).",
)
parser.add_argument(
    "--assets-home", "--assets_home", dest="ASSETS_HOME", type=str,
    default=_get_env_str(ENV_VALUES, "ASSETS_HOME", ""),
    help="Directory for writable runtime assets (trt_cache/faces/temp_audio/TensorRT).",
)
parser.add_argument(
    "--preload-models", "--preload_models", dest="PRELOAD_MODELS", type=_parse_bool,
    default=_get_env_bool(ENV_VALUES, "PRELOAD_MODELS", False),
    help="Download missing required models from model_manifest.json before validation.",
)
parser.add_argument(
    "--face-name", "--face-model-name", dest="FACE_MODEL_NAME", type=str,
    default=_get_env_str(ENV_VALUES, "FACE_MODEL_NAME", ""),
    help="Face model name",
)
parser.add_argument(
    "--format", type=_parse_format,
    default=_parse_format(_first_defined(ENV_VALUES, ["FORMAT"], "image")),
    help="image or video (legacy: 1=image, 2=video).",
)
parser.add_argument(
    "--input-dir", "--input-path", dest="INPUT_PATH", type=str,
    default=_get_env_str(ENV_VALUES, "INPUT_PATH", ""),
    help="Input directory path",
)
parser.add_argument(
    "--use-restore", type=_parse_bool,
    default=_get_env_bool(ENV_VALUES, "USE_RESTORE", True),
    help="Enable restore stage (true/false).",
)
parser.add_argument(
    "--restore-choice", type=str, choices=["1", "2", "3", "4"],
    default=_get_env_choice(ENV_VALUES, "RESTORE_CHOICE", "1", {"1", "2", "3", "4"}),
    help="1:GFPGAN 2:GPEN512 3:GPEN1024 4:CodeFormer",
)
parser.add_argument(
    "--use-parser", type=_parse_bool,
    default=_get_env_bool(ENV_VALUES, "USE_PARSER", True),
    help="Enable parser stage (true/false).",
)
parser.add_argument(
    "--parser-choice", type=str, choices=["1", "2"],
    default=_get_env_choice(ENV_VALUES, "PARSER_CHOICE", "1", {"1", "2"}),
    help="1:BiSeNet 2:SegFormer",
)
parser.add_argument(
    "--workers-per-stage", type=int,
    default=_get_env_int(ENV_VALUES, "WORKERS_PER_STAGE", 8),
    help="Workers spawned per pipeline stage.",
)
parser.add_argument(
    "--worker-queue-size", type=int,
    default=_get_env_int(ENV_VALUES, "WORKER_QUEUE_SIZE", 64),
    help="Queue size for worker stages.",
)
parser.add_argument(
    "--out-queue-size", type=int,
    default=_get_env_int(ENV_VALUES, "OUT_QUEUE_SIZE", 128),
    help="Queue size for output stage.",
)
parser.add_argument(
    "--tuner-mode", type=str, choices=["auto", "max_util", "stable"],
    default=_get_env_choice(ENV_VALUES, "TUNER_MODE", "auto", {"auto", "max_util", "stable"}),
    help="Tuner strategy mode.",
)
parser.add_argument(
    "--gpu-target-util", type=int,
    default=_get_env_int(ENV_VALUES, "GPU_TARGET_UTIL", 95),
    help="Target GPU utilization percentage for tuner.",
)
parser.add_argument(
    "--high-watermark", type=int,
    default=_get_env_int(ENV_VALUES, "HIGH_WATERMARK", 12),
    help="Queue high watermark threshold for entering drain mode.",
)
parser.add_argument(
    "--low-watermark", type=int,
    default=_get_env_int(ENV_VALUES, "LOW_WATERMARK", 4),
    help="Queue low watermark threshold for leaving drain mode.",
)
parser.add_argument(
    "--switch-cooldown-s", type=float,
    default=_get_env_float(ENV_VALUES, "SWITCH_COOLDOWN_S", 0.35),
    help="Minimum seconds between tuner mode/stage switches.",
)
parser.add_argument(
    "--preserve-swap-eyes", type=_parse_bool,
    default=_get_env_bool(ENV_VALUES, "PRESERVE_SWAP_EYES", True),
    help="Force swapped-eye preservation in parser stage.",
)
parser.add_argument(
    "--parser-mask-blur", type=int,
    default=_get_env_int(ENV_VALUES, "PARSER_MASK_BLUR", 21),
    help="Parser mask blur kernel size (odd number).",
)
parser.add_argument(
    "--dry-run", type=_parse_bool,
    default=_get_env_bool(ENV_VALUES, "DRY_RUN", False),
    help="Validate configuration and exit without inference.",
)
parser.add_argument(
    "--log-level", type=str, choices=["warning", "info", "debug"],
    default=_get_env_choice(ENV_VALUES, "LOG_LEVEL", "warning", {"warning", "info", "debug"}),
    help="Application log level.",
)
parser.add_argument(
    "--output-suffix", type=str,
    default=_get_env_str(ENV_VALUES, "OUTPUT_SUFFIX", ""),
    help="Suffix appended to output filenames.",
)
parser.add_argument(
    "--skip-existing", type=_parse_bool,
    default=_get_env_bool(ENV_VALUES, "SKIP_EXISTING", True),
    help="Skip files that already exist in output.",
)
parser.add_argument(
    "--max-frames", type=int,
    default=_get_env_int(ENV_VALUES, "MAX_FRAMES", 0),
    help="Process only first N frames/images. 0 means no limit.",
)
parser.add_argument(
    "--max-retries", type=int,
    default=_get_env_int(ENV_VALUES, "MAX_RETRIES", 2),
    help="Maximum attempts per item including initial attempt.",
)
parser.add_argument(
    "--print-effective-config", type=_parse_bool,
    default=_get_env_bool(ENV_VALUES, "PRINT_EFFECTIVE_CONFIG", False),
    help="Print merged effective configuration at startup.",
)
parser.add_argument(
    "--output-path", type=str,
    default=_get_env_str(ENV_VALUES, "OUTPUT_PATH", ""),
    help="Override output directory path.",
)
parser.add_argument(
    "--face-model-path", type=str,
    default=_get_env_str(ENV_VALUES, "FACE_MODEL_PATH", ""),
    help="Override source face model (.safetensors) path.",
)
parser.add_argument(
    "--provider-all", type=str,
    default=_get_env_str(ENV_VALUES, "PROVIDER_ALL", "trt"),
    help="Default provider for all stages (cpu/cuda/trt).",
)
parser.add_argument(
    "--provider-swaper", type=str,
    default=_get_env_str(ENV_VALUES, "PROVIDER_SWAPER", "auto"),
    help="Provider override for swap stage.",
)
parser.add_argument(
    "--provider-restore", type=str,
    default=_get_env_str(ENV_VALUES, "PROVIDER_RESTORE", "auto"),
    help="Provider override for restore stage.",
)
parser.add_argument(
    "--provider-parser", type=str,
    default=_get_env_str(ENV_VALUES, "PROVIDER_PARSER", "auto"),
    help="Provider override for parser stage.",
)
parser.add_argument(
    "--provider-detect", type=str,
    default=_get_env_str(ENV_VALUES, "PROVIDER_DETECT", "auto"),
    help="Provider override for detect stage.",
)
parser.add_argument(
    "--use-swaper", type=_parse_bool,
    default=_get_env_bool(ENV_VALUES, "USE_SWAPER", True),
    help="Enable swap stage (true/false).",
)
parser.add_argument(
    "--swaper-weigh", type=float,
    default=_get_env_float(ENV_VALUES, "SWAPER_WEIGH", 0.70),
    help="Swap blend ratio in range [0,1].",
)
parser.add_argument(
    "--restore-weigh", type=float,
    default=_get_env_float(ENV_VALUES, "RESTORE_WEIGH", 0.70),
    help="Restore model fidelity/weight hint in range [0,1].",
)
parser.add_argument(
    "--restore-blend", type=float,
    default=_get_env_float(ENV_VALUES, "RESTORE_BLEND", 0.70),
    help="Restore blending ratio in range [0,1].",
)

_CLI_INITIALIZED = False


def _apply_parsed_args(args, validate_paths):
    global FACE_NAME, FORMAT_IS_IMAGE, INPUT_PATH
    global ENABLE_RESTORE, ENABLE_PARSER, RESTORE_CHOICE, PARSER_CHOICE
    global WORKERS_PER_STAGE, WORKER_QUEUE_SIZE, OUT_QUEUE_SIZE
    global TUNER_MODE, GPU_TARGET_UTIL, HIGH_WATERMARK, LOW_WATERMARK, SWITCH_COOLDOWN_S
    global PRESERVE_SWAP_EYES, PARSER_MASK_BLUR
    global DRY_RUN, LOG_LEVEL, OUTPUT_SUFFIX, SKIP_EXISTING, MAX_FRAMES, MAX_RETRIES
    global PRINT_EFFECTIVE_CONFIG, ENABLE_SWAPPER, SWAPPER_BLEND, RESTORE_WEIGHT, RESTORE_BLEND
    global PRELOAD_MODELS
    global PROVIDER_ALL, PROVIDER_POLICY
    global PROJECT_PATH, MODEL_HOME, ASSETS_HOME, MODELS_DIR, TRT_CACHE_DIR, TRT_CACHE_DETECT_DIR, TRT_CACHE_SWAP_DIR
    global TRT_CACHE_RESTORE_DIR, TRT_CACHE_PARSER_DIR, INSIGHTFACE_ROOT, FACES_DIR
    global TEMP_AUDIO_DIR, TENSORRT_HOME
    global SOURCE_FACE_PATH, SWAPPER_MODEL, PARSER_TYPE, PARSER_MODEL
    global RESTORE_MODEL_NAME, RESTORE_SIZE, RESTORE_MODEL_PATH
    global OUTPUT_DIR, FFMPEG_CMD, TENSORRT_DIR

    FACE_NAME = args.FACE_MODEL_NAME.strip()
    FORMAT_IS_IMAGE = args.format == "image"
    INPUT_PATH = args.INPUT_PATH.strip()

    ENABLE_RESTORE = bool(args.use_restore)
    ENABLE_PARSER = bool(args.use_parser)
    RESTORE_CHOICE = args.restore_choice
    PARSER_CHOICE = args.parser_choice

    WORKERS_PER_STAGE = args.workers_per_stage
    WORKER_QUEUE_SIZE = args.worker_queue_size
    OUT_QUEUE_SIZE = args.out_queue_size

    TUNER_MODE = args.tuner_mode
    GPU_TARGET_UTIL = args.gpu_target_util
    HIGH_WATERMARK = args.high_watermark
    LOW_WATERMARK = args.low_watermark
    SWITCH_COOLDOWN_S = args.switch_cooldown_s

    PRESERVE_SWAP_EYES = bool(args.preserve_swap_eyes)
    PARSER_MASK_BLUR = args.parser_mask_blur

    DRY_RUN = bool(args.dry_run)
    LOG_LEVEL = args.log_level.upper()

    OUTPUT_SUFFIX = args.output_suffix.strip()
    SKIP_EXISTING = bool(args.skip_existing)
    MAX_FRAMES = max(0, args.max_frames)
    MAX_RETRIES = args.max_retries
    PRINT_EFFECTIVE_CONFIG = bool(args.print_effective_config)
    PRELOAD_MODELS = bool(args.PRELOAD_MODELS)

    ENABLE_SWAPPER = bool(args.use_swaper)
    SWAPPER_BLEND = args.swaper_weigh
    RESTORE_WEIGHT = args.restore_weigh
    RESTORE_BLEND = args.restore_blend

    PROVIDER_ALL = _parse_provider(args.provider_all, "PROVIDER_ALL")
    if PROVIDER_ALL == "auto":
        PROVIDER_ALL = "trt"

    PROVIDER_POLICY = {
        "detect": _parse_provider(args.provider_detect, "PROVIDER_DETECT"),
        "swap": _parse_provider(args.provider_swaper, "PROVIDER_SWAPER"),
        "restore": _parse_provider(args.provider_restore, "PROVIDER_RESTORE"),
        "parse": _parse_provider(args.provider_parser, "PROVIDER_PARSER"),
    }
    for _stage, _provider in PROVIDER_POLICY.items():
        if _provider == "auto":
            PROVIDER_POLICY[_stage] = PROVIDER_ALL

    requested_project_path = args.PROJECT_PATH.strip()
    if requested_project_path:
        PROJECT_PATH = normalize_project_root(requested_project_path)
    else:
        if is_site_packages_path(BASE_DIR) and validate_paths:
            parser.error(
                "Detected site-packages runtime. Please specify --project-path to avoid writing assets into venv."
            )
        PROJECT_PATH = _resolve_default_project_root(BASE_DIR)

    layout = ensure_project_layout(PROJECT_PATH, SOURCE_ASSETS_DIR)
    ASSETS_HOME = layout.assets_dir
    MODELS_DIR = layout.models_dir
    FACES_DIR = layout.faces_dir
    TEMP_AUDIO_DIR = layout.temp_audio_dir
    TENSORRT_HOME = layout.tensorrt_home
    TRT_CACHE_DIR = layout.trt_cache_dir

    # Deprecated compatibility (kept for transition period).
    explicit_model_home = any(
        token.startswith("--model-home") or token.startswith("--model_home")
        for token in sys.argv[1:]
    )
    explicit_assets_home = any(
        token.startswith("--assets-home") or token.startswith("--assets_home")
        for token in sys.argv[1:]
    )

    MODEL_HOME = args.MODEL_HOME.strip() if explicit_model_home else ""
    ASSETS_HOME_OVERRIDE = args.ASSETS_HOME.strip() if explicit_assets_home else ""
    if MODEL_HOME:
        MODELS_DIR = os.path.abspath(MODEL_HOME)
        if validate_paths:
            sys.stderr.write("[WARN] --model-home is deprecated; prefer --project-path.\n")
    if ASSETS_HOME_OVERRIDE:
        ASSETS_HOME = os.path.abspath(ASSETS_HOME_OVERRIDE)
        if validate_paths:
            sys.stderr.write("[WARN] --assets-home is deprecated; prefer --project-path.\n")

    TRT_CACHE_DIR = os.path.join(ASSETS_HOME, "trt_cache")
    TRT_CACHE_DETECT_DIR = os.path.join(TRT_CACHE_DIR, "trt_cache_detect")
    TRT_CACHE_SWAP_DIR = os.path.join(TRT_CACHE_DIR, "trt_cache_swap")
    TRT_CACHE_RESTORE_DIR = os.path.join(TRT_CACHE_DIR, "trt_cache_restore")
    TRT_CACHE_PARSER_DIR = os.path.join(TRT_CACHE_DIR, "trt_cache_parser")
    INSIGHTFACE_ROOT = os.path.join(MODELS_DIR, "insightface_models")
    default_source_face_path = os.path.join(FACES_DIR, f"{FACE_NAME}.safetensors")
    SOURCE_FACE_PATH = args.face_model_path if args.face_model_path else default_source_face_path

    SWAPPER_MODEL = os.path.join(MODELS_DIR, "inswapper_128.onnx")

    if PARSER_CHOICE == "2":
        PARSER_TYPE = "segformer"
        PARSER_MODEL = os.path.join(MODELS_DIR, "Segformer_CelebAMask-HQ.onnx")
    else:
        PARSER_TYPE = "bisenet"
        PARSER_MODEL = os.path.join(MODELS_DIR, "faceparser_resnet34.onnx")

    if RESTORE_CHOICE == "2":
        RESTORE_MODEL_NAME = "GPEN-BFR-512.onnx"
        RESTORE_SIZE = 512
    elif RESTORE_CHOICE == "3":
        RESTORE_MODEL_NAME = "GPEN-BFR-1024.onnx"
        RESTORE_SIZE = 1024
    elif RESTORE_CHOICE == "4":
        RESTORE_MODEL_NAME = "codeformer.onnx"
        RESTORE_SIZE = 512
    else:
        RESTORE_MODEL_NAME = "GFPGANv1.4.onnx"
        RESTORE_SIZE = 512

    RESTORE_MODEL_PATH = os.path.join(MODELS_DIR, RESTORE_MODEL_NAME)
    manifest_file = _ensure_model_manifest(MODELS_DIR)
    manifest = _load_model_manifest(manifest_file)

    if args.output_path:
        OUTPUT_DIR = os.path.abspath(args.output_path)
    else:
        if FORMAT_IS_IMAGE:
            OUTPUT_DIR = os.path.join(layout.output_dir, "image", FACE_NAME)
        else:
            OUTPUT_DIR = os.path.join(layout.output_dir, "video", FACE_NAME)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TRT_CACHE_DIR, exist_ok=True)
    os.makedirs(TRT_CACHE_DETECT_DIR, exist_ok=True)
    os.makedirs(TRT_CACHE_SWAP_DIR, exist_ok=True)
    os.makedirs(TRT_CACHE_RESTORE_DIR, exist_ok=True)
    os.makedirs(TRT_CACHE_PARSER_DIR, exist_ok=True)
    os.makedirs(FACES_DIR, exist_ok=True)
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

    # ── Cross-platform ffmpeg ─────────────────────────────────────────────────
    FFMPEG_CMD = os.path.join(MODELS_DIR, _platform_ffmpeg_name())

    # ── Validation ────────────────────────────────────────────────────────────
    if WORKERS_PER_STAGE < 1 or WORKERS_PER_STAGE > 128:
        parser.error("WORKERS_PER_STAGE must be in range [1, 128].")
    if WORKER_QUEUE_SIZE < 4 or WORKER_QUEUE_SIZE > 4096:
        parser.error("WORKER_QUEUE_SIZE must be in range [4, 4096].")
    if OUT_QUEUE_SIZE < 8 or OUT_QUEUE_SIZE > 8192:
        parser.error("OUT_QUEUE_SIZE must be in range [8, 8192].")
    if GPU_TARGET_UTIL < 50 or GPU_TARGET_UTIL > 100:
        parser.error("GPU_TARGET_UTIL must be in range [50, 100].")
    if HIGH_WATERMARK < 1 or HIGH_WATERMARK > 4096:
        parser.error("HIGH_WATERMARK must be in range [1, 4096].")
    if LOW_WATERMARK < 0 or LOW_WATERMARK > 4096:
        parser.error("LOW_WATERMARK must be in range [0, 4096].")
    if LOW_WATERMARK >= HIGH_WATERMARK:
        parser.error("LOW_WATERMARK must be less than HIGH_WATERMARK.")
    if SWITCH_COOLDOWN_S < 0.0 or SWITCH_COOLDOWN_S > 60.0:
        parser.error("SWITCH_COOLDOWN_S must be in range [0.0, 60.0].")
    if PARSER_MASK_BLUR < 1 or PARSER_MASK_BLUR > 255:
        parser.error("PARSER_MASK_BLUR must be in range [1, 255].")
    if PARSER_MASK_BLUR % 2 == 0:
        parser.error("PARSER_MASK_BLUR must be an odd number.")
    if MAX_RETRIES < 1 or MAX_RETRIES > 20:
        parser.error("MAX_RETRIES must be in range [1, 20].")

    for name, value in [
        ("SWAPER_WEIGH", SWAPPER_BLEND),
        ("RESTORE_WEIGH", RESTORE_WEIGHT),
        ("RESTORE_BLEND", RESTORE_BLEND),
    ]:
        if value < 0.0 or value > 1.0:
            parser.error(f"{name} must be in range [0.0, 1.0].")

    if validate_paths:
        required_models = {_platform_ffmpeg_name()}
        if ENABLE_SWAPPER:
            required_models.add("inswapper_128.onnx")
        if ENABLE_RESTORE:
            required_models.add(RESTORE_MODEL_NAME)
        if ENABLE_PARSER:
            required_models.add(os.path.basename(PARSER_MODEL))
        _sync_models_from_manifest(
            models_dir=MODELS_DIR,
            manifest=manifest,
            preload_models=PRELOAD_MODELS,
            required_filenames=required_models,
        )

        if not FACE_NAME:
            parser.error("FACE_MODEL_NAME is required.")
        if not INPUT_PATH:
            parser.error("INPUT_PATH is required.")
        if not os.path.isdir(INPUT_PATH):
            parser.error(f"INPUT_PATH does not exist or is not a directory: {INPUT_PATH}")
        if ENABLE_SWAPPER and not os.path.isfile(SOURCE_FACE_PATH):
            parser.error(f"Source face model not found: {SOURCE_FACE_PATH}")
        if ENABLE_SWAPPER and not os.path.isfile(SWAPPER_MODEL):
            parser.error(f"Swapper model not found: {SWAPPER_MODEL}")
        if ENABLE_RESTORE and not os.path.isfile(RESTORE_MODEL_PATH):
            parser.error(f"Restore model not found: {RESTORE_MODEL_PATH}")
        if ENABLE_PARSER and not os.path.isfile(PARSER_MODEL):
            parser.error(f"Parser model not found: {PARSER_MODEL}")
        if not os.path.isfile(FFMPEG_CMD):
            parser.error(f"ffmpeg executable not found: {FFMPEG_CMD}")

    TENSORRT_DIR = os.path.join(TENSORRT_HOME, "bin")
    if not os.path.exists(TENSORRT_DIR):
        legacy_tensorrt_dir = os.path.join(MODELS_DIR, "TensorRT-10.15.1.29", "bin")
        if os.path.exists(legacy_tensorrt_dir):
            TENSORRT_DIR = legacy_tensorrt_dir
        elif validate_paths and PROVIDER_ALL == "trt":
            sys.stderr.write(
                "[WARN] TensorRT runtime not found in assets. "
                f"Download here: {TENSORRT_DOWNLOAD_URL}\n"
            )
    if hasattr(os, "add_dll_directory") and os.path.exists(TENSORRT_DIR):
        os.add_dll_directory(TENSORRT_DIR)
    os.environ["PATH"] = TENSORRT_DIR + os.pathsep + os.environ.get("PATH", "")


def initialize_from_cli(argv=None):
    global _CLI_INITIALIZED
    parsed = parser.parse_args(argv)
    _apply_parsed_args(parsed, validate_paths=True)
    _CLI_INITIALIZED = True
    return parsed


def ensure_cli_initialized(argv=None):
    if not _CLI_INITIALIZED:
        initialize_from_cli(argv=argv)


def is_cli_initialized():
    return bool(_CLI_INITIALIZED)


_apply_parsed_args(parser.parse_args([]), validate_paths=False)


def get_effective_config():
    return {
        "PROJECT_PATH": PROJECT_PATH,
        "FACE_MODEL_NAME": FACE_NAME,
        "MODEL_HOME": MODELS_DIR,
        "ASSETS_HOME": ASSETS_HOME,
        "PRELOAD_MODELS": PRELOAD_MODELS,
        "MODEL_MANIFEST_PATH": _manifest_path(MODELS_DIR),
        "FORMAT_IS_IMAGE": FORMAT_IS_IMAGE,
        "INPUT_PATH": INPUT_PATH,
        "OUTPUT_DIR": OUTPUT_DIR,
        "USE_SWAPER": ENABLE_SWAPPER,
        "USE_RESTORE": ENABLE_RESTORE,
        "USE_PARSER": ENABLE_PARSER,
        "RESTORE_CHOICE": RESTORE_CHOICE,
        "PARSER_CHOICE": PARSER_CHOICE,
        "WORKERS_PER_STAGE": WORKERS_PER_STAGE,
        "WORKER_QUEUE_SIZE": WORKER_QUEUE_SIZE,
        "OUT_QUEUE_SIZE": OUT_QUEUE_SIZE,
        "TUNER_MODE": TUNER_MODE,
        "GPU_TARGET_UTIL": GPU_TARGET_UTIL,
        "HIGH_WATERMARK": HIGH_WATERMARK,
        "LOW_WATERMARK": LOW_WATERMARK,
        "SWITCH_COOLDOWN_S": SWITCH_COOLDOWN_S,
        "MAX_FRAMES": MAX_FRAMES,
        "MAX_RETRIES": MAX_RETRIES,
        "SKIP_EXISTING": SKIP_EXISTING,
        "PRINT_EFFECTIVE_CONFIG": PRINT_EFFECTIVE_CONFIG,
        "PROVIDER_ALL": PROVIDER_ALL,
        "PROVIDER_POLICY": dict(PROVIDER_POLICY),
        "TRT_CACHE_DIR": TRT_CACHE_DIR,
        "TRT_CACHE_DETECT_DIR": TRT_CACHE_DETECT_DIR,
        "TRT_CACHE_SWAP_DIR": TRT_CACHE_SWAP_DIR,
        "TRT_CACHE_RESTORE_DIR": TRT_CACHE_RESTORE_DIR,
        "TRT_CACHE_PARSER_DIR": TRT_CACHE_PARSER_DIR,
        "FACES_DIR": FACES_DIR,
        "TEMP_AUDIO_DIR": TEMP_AUDIO_DIR,
        "TENSORRT_DIR": TENSORRT_DIR,
        "TENSORRT_DOWNLOAD_URL": TENSORRT_DOWNLOAD_URL,
        "SWAPER_WEIGH": SWAPPER_BLEND,
        "RESTORE_WEIGH": RESTORE_WEIGHT,
        "RESTORE_BLEND": RESTORE_BLEND,
        "PRESERVE_SWAP_EYES": PRESERVE_SWAP_EYES,
        "PARSER_MASK_BLUR": PARSER_MASK_BLUR,
        "DRY_RUN": DRY_RUN,
        "LOG_LEVEL": LOG_LEVEL,
    }
