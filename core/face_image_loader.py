"""face_image_loader.py
======================
Load a source-face embedding from a plain image file instead of a
``.safetensors`` face model.

The detected embedding is held **in RAM** for the lifetime of the job --
there is no file written to disk.  The returned SourceFaceEmbedding object
is duck-type compatible with DummySourceFace and therefore transparent to
the swapper stage.
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager

import cv2
import numpy as np

from core.errors import ModelInitError
from core.provider_policy import build_ort_providers, resolve_provider


logger = logging.getLogger(__name__)


@contextmanager
def _suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


class SourceFaceEmbedding:
    """Minimal face-embedding object compatible with the insightface swapper API.

    Both ``embedding`` and ``normed_embedding`` are set to the same
    L2-normalised 512-d vector so that callers need not distinguish between
    the two attributes.
    """

    __slots__ = ("embedding", "normed_embedding")

    def __init__(self, embedding: np.ndarray) -> None:
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-8:
            arr = arr / norm
        self.embedding: np.ndarray = arr
        self.normed_embedding: np.ndarray = arr


def load_source_face_from_image(
    image_path: str,
    insightface_root: str,
    provider: str = "trt",
) -> SourceFaceEmbedding:
    """Detect the single face in *image_path* and return its normed embedding.

    Parameters
    ----------
    image_path:
        Absolute path to a source-face image (.jpg/.jpeg/.png/.webp/.bmp).
    insightface_root:
        Root directory used by ``insightface.app.FaceAnalysis`` for model
        discovery (same as ``RunConfig.insightface_root``).
    provider:
        Requested OrtProvider string (``"trt"``, ``"cuda"``, ``"cpu"``).
        Automatically falls back when the requested provider is unavailable.

    Returns
    -------
    SourceFaceEmbedding
        In-RAM face embedding ready for the swapper stage.

    Raises
    ------
    ModelInitError
        If the image cannot be read, contains no face, or contains multiple
        faces (caller should provide a clean single-face portrait).
    """
    # -- Load image -----------------------------------------------------------
    frame = cv2.imread(image_path)
    if frame is None or frame.size == 0:
        raise ModelInitError(
            f"Cannot read source face image (file missing or unsupported format): {image_path}"
        )

    # -- Resolve provider -----------------------------------------------------
    try:
        import onnxruntime
        available = set(onnxruntime.get_available_providers())
    except Exception:
        available = set()

    resolved = resolve_provider(provider, available)
    # Reuse the detect TRT-cache dir for this one-shot init
    trt_cache = os.path.join(os.path.dirname(insightface_root), "trt_cache", "trt_cache_detect")
    providers = build_ort_providers(
        selected=resolved.selected,
        cache_prefix="face_image_src",
        trt_cache_dir=trt_cache,
        enable_fp16=True,
    )

    # -- Initialise FaceAnalysis with recognition module ----------------------
    try:
        from insightface.app import FaceAnalysis

        with _suppress_stdout():
            app = FaceAnalysis(
                name="buffalo_l",
                root=insightface_root,
                allowed_modules=["detection", "recognition"],
                providers=providers,
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as exc:
        raise ModelInitError(
            f"Failed to initialise InsightFace for source-face image loading: {exc}"
        ) from exc

    # -- Detect ---------------------------------------------------------------
    try:
        with _suppress_stdout():
            faces = app.get(frame)
    except Exception as exc:
        raise ModelInitError(
            f"Face detection failed on source face image '{image_path}': {exc}"
        ) from exc

    if len(faces) == 0:
        raise ModelInitError(
            f"No face detected in source face image: {image_path}\n"
            "Please use a clear, front-facing portrait with exactly one face."
        )
    if len(faces) > 1:
        raise ModelInitError(
            f"Multiple faces ({len(faces)}) detected in source face image: {image_path}\n"
            "Please use an image containing exactly one face."
        )

    # -- Extract embedding ----------------------------------------------------
    face = faces[0]
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    if emb is None:
        raise ModelInitError(
            f"InsightFace returned a face without an embedding from: {image_path}"
        )

    logger.info(
        "source_face_image_loaded",
        extra={"image_path": image_path, "provider": resolved.selected},
    )
    return SourceFaceEmbedding(np.asarray(emb, dtype=np.float32))
