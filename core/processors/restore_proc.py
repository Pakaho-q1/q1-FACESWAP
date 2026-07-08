from __future__ import annotations

import cv2
import numpy as np

from core.processor_config import ProcessorConfig
from core.utils import FFHQ_KPS_512, get_similarity_matrix


def run_restore(orig_frame, swapped_faces_data, model_manager, proc_cfg: ProcessorConfig):
    """Restore / enhance swapped faces.

    Parameters
    ----------
    orig_frame:
        Original BGR frame (used as base when parser is disabled).
    swapped_faces_data:
        List of ``(face, swapped_full_img)`` tuples from the swap stage.
    model_manager:
        Initialised ModelManager instance.
    proc_cfg:
        Immutable processor configuration — replaces global cfg reads.
    """
    restored_faces_data = []
    res_frame = orig_frame.copy() if not proc_cfg.enable_parser else None

    kps_target = FFHQ_KPS_512
    if proc_cfg.restore_size != 512:
        kps_target = FFHQ_KPS_512 * (proc_cfg.restore_size / 512.0)

    for face, swapped_full_img in swapped_faces_data:
        M = get_similarity_matrix(face.kps, kps_target)
        cropped_face = cv2.warpAffine(
            swapped_full_img,
            M,
            (proc_cfg.restore_size, proc_cfg.restore_size),
            borderMode=cv2.BORDER_REPLICATE,
        )

        input_face = (
            cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        )
        input_face = np.expand_dims(
            np.transpose((input_face - 0.5) / 0.5, (2, 0, 1)), axis=0
        )

        out_face = model_manager.infer_restore(input_face, restore_weight=proc_cfg.restore_weight)
        out_face = np.clip(
            (np.transpose(np.squeeze(out_face), (1, 2, 0)) * 0.5 + 0.5) * 255.0, 0, 255
        ).astype(np.uint8)
        out_face = cv2.cvtColor(out_face, cv2.COLOR_RGB2BGR)

        mask_radius = int(210 * (proc_cfg.restore_size / 512))
        blur_size = int(81 * (proc_cfg.restore_size / 512)) | 1
        blur_sigma = int(21 * (proc_cfg.restore_size / 512))

        mask = cv2.circle(
            np.zeros((proc_cfg.restore_size, proc_cfg.restore_size, 3), dtype=np.float32),
            (proc_cfg.restore_size // 2, proc_cfg.restore_size // 2),
            mask_radius,
            (1, 1, 1),
            -1,
            cv2.LINE_AA,
        )
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), blur_sigma)

        IM = cv2.invertAffineTransform(M)
        if IM is not None:
            base_frame = swapped_full_img if proc_cfg.enable_parser else res_frame
            img_h, img_w = base_frame.shape[:2]
            inv_mask = cv2.warpAffine(mask, IM, (img_w, img_h), flags=cv2.INTER_LINEAR)
            inv_face = cv2.warpAffine(
                out_face,
                IM,
                (img_w, img_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT,
            )
            blend_weight = proc_cfg.restore_blend
            restored_full_img = (
                (inv_mask * blend_weight) * inv_face
                + (1 - (inv_mask * blend_weight)) * base_frame.astype(np.float32)
            ).astype(np.uint8)
        else:
            restored_full_img = swapped_full_img if proc_cfg.enable_parser else res_frame

        if not proc_cfg.enable_parser:
            res_frame = restored_full_img
        restored_faces_data.append((face, restored_full_img))

    return res_frame, restored_faces_data
