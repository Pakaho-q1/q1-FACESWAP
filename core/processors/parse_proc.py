from __future__ import annotations

import cv2
import numpy as np

from core.processor_config import ProcessorConfig
from core.utils import FFHQ_KPS_512, get_similarity_matrix

# SegFormer face-class indices used to build the parser mask.
_SEGFORMER_FACE_CLASSES = [1, 2, 4, 5, 7, 10, 11, 12]


def _build_eye_priority_mask(face, frame_shape):
    mask = np.zeros(frame_shape[:2], dtype=np.float32)
    if not hasattr(face, "kps") or face.kps is None or len(face.kps) < 2:
        return np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    left_eye = face.kps[0]
    right_eye = face.kps[1]
    eye_dist = float(np.linalg.norm(left_eye - right_eye))

    radius_x = max(10, int(eye_dist * 0.42))
    radius_y = max(8, int(radius_x * 0.65))

    for eye in (left_eye, right_eye):
        center = (int(eye[0]), int(eye[1]))
        cv2.ellipse(mask, center, (radius_x, radius_y), 0, 0, 360, 1.0, -1, cv2.LINE_AA)

    mask = cv2.GaussianBlur(mask, (31, 31), 6)
    mask = np.clip(mask, 0.0, 1.0)
    return np.repeat(mask[:, :, np.newaxis], 3, axis=2)


def run_parse(orig_frame, previous_faces_data, model_manager, proc_cfg: ProcessorConfig):
    """Apply SegFormer face-parser mask to blend swapped regions cleanly.

    Parameters
    ----------
    orig_frame:
        Original BGR frame.
    previous_faces_data:
        List of ``(face, img_to_parse)`` tuples from the restore (or swap) stage.
    model_manager:
        Initialised ModelManager instance.
    proc_cfg:
        Immutable processor configuration — replaces global cfg reads.
    """
    res_frame = orig_frame.copy()

    for face, img_to_parse in previous_faces_data:
        M_parse = get_similarity_matrix(face.kps, FFHQ_KPS_512)
        crop_512 = cv2.warpAffine(
            img_to_parse, M_parse, (512, 512), borderMode=cv2.BORDER_REPLICATE
        )

        # SegFormer expects 224×224 input.
        prs_in_img = cv2.resize(crop_512, (224, 224))
        prs_in = cv2.cvtColor(prs_in_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        prs_in = (prs_in - mean) / std
        prs_in = np.transpose(prs_in, (2, 0, 1))
        prs_in = np.expand_dims(prs_in, axis=0).astype(np.float32)

        parsed_logits = model_manager.infer_parse(prs_in)
        parsed_mask = np.argmax(parsed_logits[0], axis=0)
        mask = np.isin(parsed_mask, _SEGFORMER_FACE_CLASSES).astype(np.float32)

        # SegFormer output is 224×224, resize back to 512×512.
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_LINEAR)

        mask_512 = np.expand_dims(mask, axis=-1)
        mask_512 = cv2.GaussianBlur(
            mask_512,
            (proc_cfg.parser_mask_blur, proc_cfg.parser_mask_blur),
            7,
        )

        if mask_512.ndim == 2:
            mask_512 = mask_512[:, :, np.newaxis]
        mask_512 = np.repeat(mask_512, 3, axis=2)

        IM_parse = cv2.invertAffineTransform(M_parse)
        inv_mask = cv2.warpAffine(
            mask_512,
            IM_parse,
            (res_frame.shape[1], res_frame.shape[0]),
            flags=cv2.INTER_LINEAR,
        )
        res_frame = ((inv_mask * img_to_parse) + ((1 - inv_mask) * res_frame)).astype(np.uint8)

        if proc_cfg.preserve_swap_eyes:
            eye_mask = _build_eye_priority_mask(face, res_frame.shape)
            res_frame = ((eye_mask * img_to_parse) + ((1 - eye_mask) * res_frame)).astype(np.uint8)

    return res_frame

