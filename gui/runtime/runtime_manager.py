from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from core.project_layout import build_layout, normalize_project_root


def runtime_control_dir(project_path: str) -> Path:
    root = normalize_project_root(project_path)
    layout = build_layout(root)
    return Path(layout.assets_dir) / "runtime"


def runtime_stop_signal_path(project_path: str) -> Path:
    return runtime_control_dir(project_path) / "stop.signal.json"


def clear_runtime_stop_signal(project_path: str) -> None:
    path = runtime_stop_signal_path(project_path)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        return


def write_runtime_stop_signal(project_path: str, reason: str = "gui_stop") -> Path:
    path = runtime_stop_signal_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "reason": str(reason or "gui_stop"),
        "ts": float(time.time()),
        "pid": int(os.getpid()),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp_path), str(path))
    return path
