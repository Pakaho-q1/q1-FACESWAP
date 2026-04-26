# Library Usage

This project now exposes a library-style runtime entrypoint:

- `core.run_pipeline(...)`

## 1) CLI mode (existing behavior)

```powershell
python face_swap_unified.py --face-model-name earn --format 1 --input-path "D:\Porject_folder\project_code\q1-FACESWAP\input\test\image"
```

## 2) Module mode

```powershell
python -m core --face-model-name earn --format 1 --input-path "D:\Porject_folder\project_code\q1-FACESWAP\input\test\image"
```

## 3) Embedded mode (same process)

```python
from core import run_pipeline

metrics = run_pipeline()
print(metrics)
```

Notes:
- `run_pipeline()` still uses current config module/CLI args as runtime source.
- For advanced embedding, pass a custom `RuntimeContext` and probe callbacks:
  - `run_pipeline(runtime_ctx=..., get_gpu_utilization=..., runtime_ui=...)`
- This keeps CLI as a thin adapter while centralizing runtime orchestration in one API.
