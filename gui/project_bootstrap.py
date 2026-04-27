from __future__ import annotations

import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Any

from core.project_layout import (
    build_layout,
    ensure_project_layout,
    is_site_packages_path,
    normalize_project_root,
)
try:
    from gui.services.output_index_service import OutputIndexService
except ImportError:
    from services.output_index_service import OutputIndexService  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ASSETS_DIR = PROJECT_ROOT / "assets"
USER_HOME = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))).expanduser()
FALLBACK_SETTINGS_ROOT = USER_HOME / "q1-faceswap"
FALLBACK_SETTINGS_PATH = FALLBACK_SETTINGS_ROOT / "settings.json"
_OUTPUT_INDEX_SERVICES: dict[str, OutputIndexService] = {}


def _get_output_index_service(project_root: str) -> OutputIndexService:
    normalized_root = normalize_project_root(str(project_root or ""))
    svc = _OUTPUT_INDEX_SERVICES.get(normalized_root)
    if svc is None:
        layout = build_layout(normalized_root)
        runtime_dir = Path(layout.assets_dir) / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        db_path = runtime_dir / "output_index.sqlite3"
        svc = OutputIndexService(str(db_path))
        _OUTPUT_INDEX_SERVICES[normalized_root] = svc
    return svc


def shutdown_output_index_service(project_root: str = "") -> None:
    if project_root:
        normalized_root = normalize_project_root(str(project_root))
        svc = _OUTPUT_INDEX_SERVICES.pop(normalized_root, None)
        if svc is not None:
            svc.shutdown()
        return

    for key in list(_OUTPUT_INDEX_SERVICES.keys()):
        svc = _OUTPUT_INDEX_SERVICES.pop(key, None)
        if svc is not None:
            svc.shutdown()


def detect_requires_project_path() -> bool:
    return is_site_packages_path(str(PROJECT_ROOT))


def settings_path_for(project_root: str) -> Path:
    return Path(project_root) / "assets" / "settings.json"


def load_last_project_root() -> str:
    if FALLBACK_SETTINGS_PATH.is_file():
        try:
            data = json.loads(FALLBACK_SETTINGS_PATH.read_text(encoding="utf-8"))
            value = str(data.get("project_root", "")).strip()
            if value:
                return value
        except Exception:
            return ""
    return ""


def save_last_project_root(project_root: str) -> None:
    FALLBACK_SETTINGS_ROOT.mkdir(parents=True, exist_ok=True)
    FALLBACK_SETTINGS_PATH.write_text(
        json.dumps({"project_root": project_root}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_workspace(project_path: str) -> str:
    root = normalize_project_root(project_path)
    ensure_project_layout(root, str(SOURCE_ASSETS_DIR))
    save_last_project_root(root)
    return root


def validate_project_path(project_path: str) -> tuple[bool, str]:
    try:
        root = normalize_project_root(project_path)
        parent = Path(root).parent
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(parent),
            prefix="q1fs_write_test_",
            delete=True,
        ) as f:
            f.write("ok")
            f.flush()
    except Exception as exc:  # noqa: BLE001
        return False, f"Cannot write to selected location: {exc}"
    return True, ""


def disk_space_gb(project_path: str) -> float:
    root = normalize_project_root(project_path)
    parent = Path(root).parent
    try:
        usage = shutil.disk_usage(str(parent))
        return float(usage.free) / (1024.0**3)
    except Exception:  # noqa: BLE001
        return -1.0


def list_face_names(project_root: str) -> list[str]:
    layout = build_layout(project_root)
    faces_dir = Path(layout.faces_dir)
    if not faces_dir.is_dir():
        return []
    dedup: dict[str, str] = {}
    for item in faces_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ".safetensors":
            key = item.stem.casefold()
            if key not in dedup:
                dedup[key] = item.stem
    names = list(dedup.values())
    names.sort(key=lambda s: s.casefold())
    return names


def load_project_settings(project_root: str) -> dict[str, Any]:
    path = settings_path_for(project_root)
    if not path.is_file():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def save_project_settings(project_root: str, settings: dict[str, Any]) -> None:
    path = settings_path_for(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(settings, ensure_ascii=False, indent=2)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(str(tmp_path), str(path))


def _read_model_manifest_names(project_root: str) -> list[str]:
    layout = build_layout(project_root)
    manifest_path = Path(layout.models_dir) / "model_manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = payload.get("models", [])
        names = [str(row.get("filename", "")).strip() for row in rows if str(row.get("filename", "")).strip()]
        return sorted(set(names), key=lambda n: n.casefold())
    except Exception:
        return []


def read_model_manifest_status(project_root: str) -> list[dict[str, Any]]:
    layout = build_layout(project_root)
    models_dir = Path(layout.models_dir)
    manifest_path = models_dir / "model_manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = payload.get("models", [])
        status_rows: list[dict[str, Any]] = []
        for row in rows:
            filename = str(row.get("filename", "")).strip()
            if not filename:
                continue
            url = str(row.get("url", "")).strip()
            local_path = models_dir / filename
            status_rows.append(
                {
                    "filename": filename,
                    "url": url,
                    "path": str(local_path),
                    "present": local_path.is_file(),
                }
            )
        status_rows.sort(key=lambda x: str(x["filename"]).casefold())
        return status_rows
    except Exception:
        return []


def check_tensorrt_status(project_root: str) -> dict[str, Any]:
    layout = build_layout(project_root)
    trt_home = Path(layout.tensorrt_home)
    trt_bin = trt_home / "bin"
    required = [
        "trtexec.exe",
        "nvinfer_10.dll",
        "nvinfer_plugin_10.dll",
        "nvonnxparser_10.dll",
    ]
    missing: list[str] = []
    for name in required:
        p = trt_bin / name
        if not p.is_file():
            missing.append(name)
    return {
        "ok": len(missing) == 0,
        "home": str(trt_home),
        "bin": str(trt_bin),
        "required": required,
        "missing": missing,
    }


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path),
            prefix="q1fs_gui_write_test_",
            delete=True,
        ) as f:
            f.write("ok")
            f.flush()
        return True
    except Exception:
        return False


def collect_runtime_health(project_root: str, output_path: str = "") -> dict[str, Any]:
    layout = build_layout(project_root)
    models_dir = Path(layout.models_dir)
    model_status = read_model_manifest_status(project_root)
    model_names = [str(x.get("filename", "")) for x in model_status]
    found = 0
    missing_names: list[str] = []
    for row in model_status:
        name = str(row.get("filename", ""))
        if bool(row.get("present", False)):
            found += 1
        else:
            missing_names.append(name)

    ffmpeg_local = models_dir / "ffmpeg.exe"
    ffmpeg_ok = ffmpeg_local.is_file() or shutil.which("ffmpeg") is not None
    trt_status = check_tensorrt_status(project_root)
    trt_ok = bool(trt_status.get("ok", False))

    target_output = Path(output_path).expanduser() if output_path.strip() else Path(layout.output_dir)
    writable = _is_writable_dir(target_output)
    free_gb = disk_space_gb(project_root)

    return {
        "models_found": found,
        "models_total": len(model_names),
        "missing_models": missing_names,
        "model_status": model_status,
        "ffmpeg_ok": ffmpeg_ok,
        "tensorrt_ok": trt_ok,
        "tensorrt_missing": list(trt_status.get("missing", [])),
        "tensorrt_bin_path": str(trt_status.get("bin", "")),
        "free_disk_gb": free_gb,
        "writable_output": writable,
        "writable_output_path": str(target_output),
    }


def preview_job_queue(input_path: str) -> dict[str, int]:
    p = Path(input_path).expanduser()
    if not p.is_dir():
        return {"image": 0, "video": 0, "all": 0}
    image_ext = {".png", ".jpg", ".jpeg"}
    video_ext = {".mp4", ".mkv", ".avi", ".mov"}
    image_count = 0
    video_count = 0
    try:
        with os.scandir(str(p)) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                _, ext = os.path.splitext(entry.name)
                ext_l = ext.lower()
                if ext_l in image_ext:
                    image_count += 1
                elif ext_l in video_ext:
                    video_count += 1
    except OSError:
        return {"image": 0, "video": 0, "all": 0}
    return {"image": image_count, "video": video_count, "all": image_count + video_count}


def list_latest_outputs(output_path: str, limit: int = 12) -> list[dict[str, str]]:
    page = list_output_gallery_page(
        project_root=str(PROJECT_ROOT),
        output_path=output_path,
        page=1,
        page_size=max(1, int(limit)),
        force_scan=False,
    )
    return list(page.get("rows", []))


def list_output_gallery_page(
    project_root: str,
    output_path: str,
    page: int = 1,
    page_size: int = 50,
    force_scan: bool = False,
) -> dict[str, Any]:
    service = _get_output_index_service(project_root)
    return service.get_page(
        output_path=output_path,
        page=page,
        page_size=page_size,
        force_refresh=force_scan,
    )
