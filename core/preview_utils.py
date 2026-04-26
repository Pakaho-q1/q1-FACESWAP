from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np


def encode_preview_data_url(
    image_bgr: np.ndarray,
    max_width: int = 640,
    jpeg_quality: int = 75,
) -> Optional[str]:
    """Encode a BGR image into a compact JPEG data URL for GUI preview."""
    if image_bgr is None or image_bgr.size == 0:
        return None

    h, w = image_bgr.shape[:2]
    if w > max_width and w > 0:
        scale = float(max_width) / float(w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        return None

    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
