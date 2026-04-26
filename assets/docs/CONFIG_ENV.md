# Env Configuration

Project now supports layered defaults via files in `assets/`:

1. `assets/.env` (official defaults, versioned)
2. `assets/.env_user` (user-local overrides, git-ignored)
3. CLI arguments (highest priority)

Precedence:

`CLI > assets/.env_user > assets/.env > code fallback`

## How to use

1. Keep `assets/.env` as project baseline.
2. Copy `assets/.env_user.example` to `assets/.env_user`.
3. Set only values you want to override in `assets/.env_user`.
4. Run normally with `.bat` or CLI.

## Notes

- Required values for real run are still:
  - `FACE_MODEL_NAME`
  - `FORMAT` (`image` or `video`)
  - `INPUT_PATH`
- You can set these in `.env_user` to avoid typing every run.
- Legacy keys are still accepted:
  - `FACE_NAME` -> `FACE_MODEL_NAME`
  - `INPUT_DIR` -> `INPUT_PATH`
- New tuner controls:
  - `HIGH_WATERMARK` (default `12`)
  - `LOW_WATERMARK` (default `4`)
  - `SWITCH_COOLDOWN_S` (default `0.35`)
- Debug helper:
  - `PRINT_EFFECTIVE_CONFIG=true` or CLI `--print-effective-config true`
- Workspace:
  - Use `PROJECT_PATH` as the only root selector.
  - Runtime creates/uses `PROJECT_PATH\\q1-FACESWAP\\...` structure automatically.
  - `PRELOAD_MODELS=true` auto-downloads missing required files from `model_manifest.json`.
- Safety:
  - If running from `site-packages`, provide `--project-path` (or set `PROJECT_PATH`) to avoid writing into venv.
  - If TensorRT runtime is missing while provider is `trt`, runtime prints download link:
    - `https://developer.nvidia.com/tensorrt/download/10x`

## Quick example (`assets/.env_user`)

```env
FACE_MODEL_NAME=earn
PROJECT_PATH=D:\Porject_folder\project_code
FORMAT=image
INPUT_PATH=D:\Porject_folder\project_code\q1-FACESWAP\input\test\image
OUTPUT_PATH=D:\Porject_folder\project_code\q1-FACESWAP\output\test
PRELOAD_MODELS=true
USE_RESTORE=false
USE_PARSER=false
WORKER_QUEUE_SIZE=64
OUT_QUEUE_SIZE=128
HIGH_WATERMARK=12
LOW_WATERMARK=4
SWITCH_COOLDOWN_S=0.35
PRINT_EFFECTIVE_CONFIG=true
```
