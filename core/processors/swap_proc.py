from __future__ import annotations

import numpy as np

from core.processor_config import ProcessorConfig


def _blend_frames(base_frame, swapped_frame, alpha):
    if alpha >= 0.999:
        return swapped_frame
    if alpha <= 0.001:
        return base_frame

    blended = (
        (alpha * swapped_frame.astype(np.float32))
        + ((1.0 - alpha) * base_frame.astype(np.float32))
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def run_swap(orig_frame, faces, model_manager, proc_cfg: ProcessorConfig):
    """Swap faces with Inswapper (optional).

    Parameters
    ----------
    orig_frame:
        Source BGR frame.
    faces:
        Detected face objects from InsightFace.
    model_manager:
        Initialised ModelManager instance.
    proc_cfg:
        Immutable processor configuration — replaces global cfg reads.
    """
    swapped_faces_data = []
    res_frame = orig_frame.copy()
    parser_or_restore_enabled = proc_cfg.enable_parser or proc_cfg.enable_restore

    if not proc_cfg.enable_swapper:
        # Keep compatible data shape for restore/parser-only workflows.
        for face in faces:
            swapped_faces_data.append((face, orig_frame.copy()))
        return res_frame, swapped_faces_data

    for face in faces:
        if parser_or_restore_enabled:
            swapped_full_img = model_manager.infer_swap(
                orig_frame, face, source=model_manager.source_face, paste_back=True
            )
            swapped_full_img = _blend_frames(orig_frame, swapped_full_img, proc_cfg.swapper_blend)
            swapped_faces_data.append((face, swapped_full_img))
        else:
            swapped_once = model_manager.infer_swap(
                res_frame, face, source=model_manager.source_face, paste_back=True
            )
            res_frame = _blend_frames(res_frame, swapped_once, proc_cfg.swapper_blend)

    return res_frame, swapped_faces_data
