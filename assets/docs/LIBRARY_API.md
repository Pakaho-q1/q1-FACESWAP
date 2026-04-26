# Library API Contract

## Public Entry Points

- `core.run_pipeline(cfg_module=None, runtime_ctx=None, get_gpu_utilization=None, runtime_ui=None) -> dict`
- `core.run_image_job(config, get_gpu_utilization=None, runtime_ui=None) -> dict`
- `core.run_video_job(config, get_gpu_utilization=None, runtime_ui=None) -> dict`
- `core.resume_pipeline_job(config, get_gpu_utilization=None, runtime_ui=None) -> dict`

## Public Types

- `core.RunConfig`
- `core.RuntimeContext`
- `core.ProviderPolicy`

## Public Errors

- `core.FaceSwapError`
- `core.ConfigError`
- `core.ModelInitError`
- `core.PipelineError`
- `core.RecoveryError`

## Compatibility

- Backward-compatible changes are delivered in MINOR/PATCH versions.
- Breaking signature or behavior changes are delivered in MAJOR versions.
- CLI remains a thin adapter over the same runtime pipeline.
