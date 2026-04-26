from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import FileResponse
from nicegui import app

MEDIA_ROUTE_PATH = "/__q1_media"
_MEDIA_ROOTS: set[Path] = set()
_ROUTE_REGISTERED = False


def _normalize_path(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def _is_within_root(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def register_media_route() -> None:
    global _ROUTE_REGISTERED
    if _ROUTE_REGISTERED:
        return

    @app.get(MEDIA_ROUTE_PATH)
    def q1_media(path: str) -> FileResponse:
        try:
            candidate = _normalize_path(path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid path: {exc}") from exc
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        allowed = any(_is_within_root(candidate, root) for root in _MEDIA_ROOTS)
        if not allowed:
            raise HTTPException(status_code=403, detail="Path is outside allowed media roots")
        return FileResponse(candidate)

    _ROUTE_REGISTERED = True


def register_media_root(path_value: str) -> None:
    raw = str(path_value or "").strip()
    if not raw:
        return
    try:
        root = _normalize_path(raw)
    except Exception:  # noqa: BLE001
        return
    _MEDIA_ROOTS.add(root)


def to_media_url(path_value: str) -> str:
    resolved = _normalize_path(path_value)
    return f"{MEDIA_ROUTE_PATH}?path={quote(str(resolved))}"
