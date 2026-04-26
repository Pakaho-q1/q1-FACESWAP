from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from core.face_model_builder import (
    FaceModelBuildError,
    FaceModelBuildResult,
    FaceModelBuildStats,
    build_face_model_with_insightface,
)
from core.project_layout import build_layout


class FaceModelService:
    def __init__(
        self,
        project_root: str,
        build_face_model_fn: Callable[..., FaceModelBuildResult] = build_face_model_with_insightface,
    ) -> None:
        self.project_root = str(project_root)
        self._build_face_model_fn = build_face_model_fn
        self.build_state_lock = threading.Lock()
        self.build_state: dict[str, dict[str, Any]] = {}

    def snapshot_build_state(self) -> dict[str, dict[str, Any]]:
        with self.build_state_lock:
            return {k: dict(v) for k, v in self.build_state.items()}

    def has_active_builds(self) -> bool:
        with self.build_state_lock:
            return any(str(v.get("status", "")) in {"queued", "running"} for v in self.build_state.values())

    def clear_finished(self) -> None:
        with self.build_state_lock:
            to_delete = [
                key
                for key, value in self.build_state.items()
                if str(value.get("status", "")) in {"done", "error"}
            ]
            for key in to_delete:
                self.build_state.pop(key, None)

    def _set_state(self, face_name: str, **fields: Any) -> None:
        key = str(face_name).strip()
        with self.build_state_lock:
            current = dict(self.build_state.get(key, {}))
            current.update(fields)
            self.build_state[key] = current

    def _count_supported_images(self, input_dir: str) -> int:
        path = Path(str(input_dir))
        if not path.is_dir():
            return 0
        allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        count = 0
        for item in path.iterdir():
            if item.is_file() and item.suffix.lower() in allowed:
                count += 1
        return count

    def _build_worker(
        self,
        face_name: str,
        input_dir: str,
        provider: str,
        min_accepted_images: int,
        overwrite: bool,
    ) -> None:
        layout = build_layout(self.project_root)
        total_images = self._count_supported_images(input_dir)

        def on_progress(stats: FaceModelBuildStats) -> None:
            scanned = int(stats.scanned_images)
            progress = 0.0
            if total_images > 0:
                progress = max(0.0, min(1.0, float(scanned) / float(total_images)))
            detail = (
                f"scanned={stats.scanned_images} accepted={stats.accepted_images} "
                f"no_face={stats.skipped_no_face} multi_face={stats.skipped_multi_face}"
            )
            self._set_state(
                face_name,
                status="running",
                progress=progress,
                detail=detail,
                scanned=scanned,
                total=total_images,
                accepted=int(stats.accepted_images),
            )

        try:
            self._set_state(
                face_name,
                status="running",
                progress=0.0,
                detail="Preparing face model build",
                scanned=0,
                total=total_images,
                accepted=0,
            )
            result = self._build_face_model_fn(
                face_name=face_name,
                input_dir=input_dir,
                faces_dir=layout.faces_dir,
                insightface_root=str(Path(layout.models_dir) / "insightface_models"),
                requested_provider=provider,
                min_accepted_images=min_accepted_images,
                overwrite=overwrite,
                progress_callback=on_progress,
            )
            self._set_state(
                face_name,
                status="done",
                progress=1.0,
                detail=(
                    f"Done: accepted={result.accepted_images}, no_face={result.skipped_no_face}, "
                    f"multi_face={result.skipped_multi_face}, filtered={result.outlier_filtered}"
                ),
                output_path=result.output_path,
            )
        except FaceModelBuildError as exc:
            self._set_state(
                face_name,
                status="error",
                progress=0.0,
                detail=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            self._set_state(
                face_name,
                status="error",
                progress=0.0,
                detail=f"unexpected_error: {exc}",
            )

    def start_build(
        self,
        *,
        face_name: str,
        input_dir: str,
        provider: str = "trt",
        min_accepted_images: int = 1,
        overwrite: bool = True,
        force: bool = False,
    ) -> bool:
        safe_face = str(face_name or "").strip()
        src_dir = str(input_dir or "").strip()
        if not safe_face or not src_dir:
            return False

        with self.build_state_lock:
            current = dict(self.build_state.get(safe_face, {}))
            if (not force) and str(current.get("status", "")) in {"queued", "running"}:
                return False
            self.build_state[safe_face] = {
                "status": "queued",
                "progress": 0.0,
                "detail": "Queued",
                "input_dir": src_dir,
            }

        thread = threading.Thread(
            target=self._build_worker,
            args=(safe_face, src_dir, str(provider or "trt"), int(min_accepted_images), bool(overwrite)),
            daemon=True,
            name=f"face-model-build-{safe_face}",
        )
        thread.start()
        return True
