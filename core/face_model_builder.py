from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import cv2
import numpy as np
from safetensors.numpy import save_file

from core.provider_policy import build_ort_providers, resolve_provider


class FaceModelBuildError(Exception):
    """Raised when source face model build fails."""


@dataclass(frozen=True)
class FaceModelBuildResult:
    face_name: str
    output_path: str
    scanned_images: int
    accepted_images: int
    skipped_no_face: int
    skipped_multi_face: int
    skipped_invalid_embedding: int
    outlier_filtered: int


@dataclass(frozen=True)
class FaceModelBuildStats:
    scanned_images: int = 0
    accepted_images: int = 0
    skipped_no_face: int = 0
    skipped_multi_face: int = 0
    skipped_invalid_embedding: int = 0


class FaceEmbeddingExtractor(Protocol):
    def extract_embeddings(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        """Return one embedding per detected face in a frame."""


def _validate_face_name(raw_name: str) -> str:
    name = str(raw_name or "").strip()
    if not name:
        raise FaceModelBuildError("face_name is required")
    if len(name) > 128:
        raise FaceModelBuildError("face_name must be <= 128 characters")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise FaceModelBuildError("face_name may contain only letters, numbers, dot, underscore, dash")
    return name


def _list_input_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FaceModelBuildError(f"input_dir does not exist or is not a directory: {input_dir}")
    allowed_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    rows = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_ext
    ]
    rows.sort(key=lambda p: p.name.casefold())
    return rows


def _normalize_embedding(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise FaceModelBuildError("empty embedding")
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        raise FaceModelBuildError("zero-norm embedding")
    return arr / norm


def _aggregate_embeddings(
    normalized_embeddings: Iterable[np.ndarray],
    outlier_cosine_threshold: float,
) -> tuple[np.ndarray, int]:
    rows = [np.asarray(e, dtype=np.float32).reshape(-1) for e in normalized_embeddings]
    if not rows:
        raise FaceModelBuildError("no valid embeddings to aggregate")

    dim = rows[0].size
    for idx, row in enumerate(rows):
        if row.size != dim:
            raise FaceModelBuildError(f"inconsistent embedding dimensions at index {idx}")
    matrix = np.vstack(rows).astype(np.float32)

    center = matrix.mean(axis=0)
    center_norm = float(np.linalg.norm(center))
    if center_norm <= 1e-8:
        raise FaceModelBuildError("failed to compute stable center embedding")
    center = center / center_norm

    cosine = matrix @ center
    keep_mask = cosine >= float(outlier_cosine_threshold)
    kept = matrix[keep_mask]
    filtered_count = int(matrix.shape[0] - kept.shape[0])
    if kept.shape[0] == 0:
        kept = matrix
        filtered_count = 0

    final = kept.mean(axis=0)
    final_norm = float(np.linalg.norm(final))
    if final_norm <= 1e-8:
        raise FaceModelBuildError("failed to build final face embedding")
    return (final / final_norm).astype(np.float32), filtered_count


def _atomic_save_safetensors(output_path: Path, payload: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{output_path.stem}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        save_file(payload, str(tmp_path))
        os.replace(str(tmp_path), str(output_path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


class InsightFaceEmbeddingExtractor:
    """Embedding extractor backed by InsightFace buffalo_l model pack."""

    def __init__(
        self,
        insightface_root: str,
        requested_provider: str = "trt",
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        try:
            import onnxruntime
            from insightface.app import FaceAnalysis
        except Exception as exc:  # noqa: BLE001
            raise FaceModelBuildError(f"missing inference dependencies: {exc}") from exc

        try:
            available = set(onnxruntime.get_available_providers())
        except Exception:
            available = set()
        provider = resolve_provider(requested_provider, available)
        providers = build_ort_providers(
            selected=provider.selected,
            cache_prefix="face_builder",
            trt_cache_dir=str(Path(insightface_root).parent / "trt_cache" / "face_builder"),
            enable_fp16=True,
        )
        try:
            self._app = FaceAnalysis(
                name="buffalo_l",
                root=insightface_root,
                allowed_modules=["detection", "recognition"],
                providers=providers,
            )
            self._app.prepare(ctx_id=0, det_size=det_size)
        except Exception as exc:  # noqa: BLE001
            raise FaceModelBuildError(f"failed initializing insightface: {exc}") from exc

    def extract_embeddings(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        try:
            faces = self._app.get(image_bgr)
        except Exception as exc:  # noqa: BLE001
            raise FaceModelBuildError(f"face detection failed: {exc}") from exc
        rows: list[np.ndarray] = []
        for face in faces:
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = getattr(face, "embedding", None)
            if emb is None:
                continue
            rows.append(np.asarray(emb, dtype=np.float32).reshape(-1))
        return rows


def build_face_model(
    *,
    face_name: str,
    input_dir: str,
    faces_dir: str,
    extractor: FaceEmbeddingExtractor,
    outlier_cosine_threshold: float = 0.35,
    min_accepted_images: int = 1,
    overwrite: bool = True,
    progress_callback: Callable[[FaceModelBuildStats], None] | None = None,
) -> FaceModelBuildResult:
    safe_name = _validate_face_name(face_name)
    source_dir = Path(str(input_dir))
    target_dir = Path(str(faces_dir))
    if min_accepted_images < 1:
        raise FaceModelBuildError("min_accepted_images must be >= 1")
    if outlier_cosine_threshold < -1.0 or outlier_cosine_threshold > 1.0:
        raise FaceModelBuildError("outlier_cosine_threshold must be in range [-1.0, 1.0]")

    image_files = _list_input_images(source_dir)
    if not image_files:
        raise FaceModelBuildError(f"no supported images found in input_dir: {source_dir}")

    output_path = target_dir / f"{safe_name}.safetensors"
    if output_path.exists() and not overwrite:
        raise FaceModelBuildError(f"face model already exists: {output_path}")

    accepted_embeddings: list[np.ndarray] = []
    stats = FaceModelBuildStats()

    for image_path in image_files:
        frame = cv2.imread(str(image_path))
        if frame is None or frame.size == 0:
            stats = FaceModelBuildStats(
                scanned_images=stats.scanned_images + 1,
                accepted_images=stats.accepted_images,
                skipped_no_face=stats.skipped_no_face + 1,
                skipped_multi_face=stats.skipped_multi_face,
                skipped_invalid_embedding=stats.skipped_invalid_embedding,
            )
            if progress_callback is not None:
                progress_callback(stats)
            continue

        embeddings = extractor.extract_embeddings(frame)
        if len(embeddings) == 0:
            stats = FaceModelBuildStats(
                scanned_images=stats.scanned_images + 1,
                accepted_images=stats.accepted_images,
                skipped_no_face=stats.skipped_no_face + 1,
                skipped_multi_face=stats.skipped_multi_face,
                skipped_invalid_embedding=stats.skipped_invalid_embedding,
            )
            if progress_callback is not None:
                progress_callback(stats)
            continue
        if len(embeddings) > 1:
            stats = FaceModelBuildStats(
                scanned_images=stats.scanned_images + 1,
                accepted_images=stats.accepted_images,
                skipped_no_face=stats.skipped_no_face,
                skipped_multi_face=stats.skipped_multi_face + 1,
                skipped_invalid_embedding=stats.skipped_invalid_embedding,
            )
            if progress_callback is not None:
                progress_callback(stats)
            continue

        try:
            normalized = _normalize_embedding(embeddings[0])
        except FaceModelBuildError:
            stats = FaceModelBuildStats(
                scanned_images=stats.scanned_images + 1,
                accepted_images=stats.accepted_images,
                skipped_no_face=stats.skipped_no_face,
                skipped_multi_face=stats.skipped_multi_face,
                skipped_invalid_embedding=stats.skipped_invalid_embedding + 1,
            )
            if progress_callback is not None:
                progress_callback(stats)
            continue

        accepted_embeddings.append(normalized)
        stats = FaceModelBuildStats(
            scanned_images=stats.scanned_images + 1,
            accepted_images=stats.accepted_images + 1,
            skipped_no_face=stats.skipped_no_face,
            skipped_multi_face=stats.skipped_multi_face,
            skipped_invalid_embedding=stats.skipped_invalid_embedding,
        )
        if progress_callback is not None:
            progress_callback(stats)

    if len(accepted_embeddings) < min_accepted_images:
        raise FaceModelBuildError(
            f"not enough valid single-face images: accepted={len(accepted_embeddings)} required={min_accepted_images}"
        )

    final_embedding, filtered_count = _aggregate_embeddings(
        accepted_embeddings,
        outlier_cosine_threshold=outlier_cosine_threshold,
    )
    payload = {
        "embedding": final_embedding.reshape(1, -1),
        "version": np.array([1], dtype=np.int32),
    }
    _atomic_save_safetensors(output_path, payload)
    return FaceModelBuildResult(
        face_name=safe_name,
        output_path=str(output_path),
        scanned_images=stats.scanned_images,
        accepted_images=stats.accepted_images,
        skipped_no_face=stats.skipped_no_face,
        skipped_multi_face=stats.skipped_multi_face,
        skipped_invalid_embedding=stats.skipped_invalid_embedding,
        outlier_filtered=filtered_count,
    )


def build_face_model_with_insightface(
    *,
    face_name: str,
    input_dir: str,
    faces_dir: str,
    insightface_root: str,
    requested_provider: str = "trt",
    outlier_cosine_threshold: float = 0.35,
    min_accepted_images: int = 1,
    overwrite: bool = True,
    progress_callback: Callable[[FaceModelBuildStats], None] | None = None,
) -> FaceModelBuildResult:
    extractor = InsightFaceEmbeddingExtractor(
        insightface_root=insightface_root,
        requested_provider=requested_provider,
    )
    return build_face_model(
        face_name=face_name,
        input_dir=input_dir,
        faces_dir=faces_dir,
        extractor=extractor,
        outlier_cosine_threshold=outlier_cosine_threshold,
        min_accepted_images=min_accepted_images,
        overwrite=overwrite,
        progress_callback=progress_callback,
    )
