Param(
    [string]$PythonExe = ".\env\Scripts\python.exe",
    [string]$FaceModelName = "earn",
    [string]$InputPath = ".\input\test\image",
    [string]$OutputPath = ".\output\test"
)

$ErrorActionPreference = "Stop"

Write-Host "[SMOKE] Running unit tests..."
& $PythonExe -m unittest discover -s tests -p "test_*.py"

Write-Host "[SMOKE] Running dry-run config validation..."
& $PythonExe face_swap_unified.py `
  --face-model-name $FaceModelName `
  --format 1 `
  --input-path $InputPath `
  --output-path $OutputPath `
  --use-swaper false `
  --use-restore false `
  --use-parser false `
  --dry-run true

Write-Host "[SMOKE] Running 1-frame image smoke..."
& $PythonExe face_swap_unified.py `
  --face-model-name $FaceModelName `
  --format 1 `
  --input-path $InputPath `
  --output-path $OutputPath `
  --max-frames 1 `
  --workers-per-stage 2 `
  --worker-queue-size 16 `
  --out-queue-size 32 `
  --use-restore false `
  --use-parser false `
  --log-level warning

Write-Host "[SMOKE] All checks completed."
