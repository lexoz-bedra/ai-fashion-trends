
from __future__ import annotations

import logging
from PIL import Image

logger = logging.getLogger(__name__)


def _center_crop_box(w: int, h: int, frac: float = 0.72) -> tuple[int, int, int, int]:
    cw = int(w * frac)
    ch = int(h * frac)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return x0, y0, x0 + cw, y0 + ch


def detect_face_bbox(
    image: Image.Image,
    *,
    min_confidence: float = 0.5,
) -> tuple[tuple[int, int, int, int], tuple[float, float, float, float]] | None:
    try:
        import mediapipe as mp
    except ImportError:
        logger.debug("mediapipe не установлен — используйте pip install 'ai-fashion-trends[vision]'")
        return None

    import numpy as np

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=min_confidence
    ) as face_det:
        result = face_det.process(rgb)

    if not result.detections:
        return None

    best = None
    best_area = 0.0
    for det in result.detections:
        bbox = det.location_data.relative_bounding_box
        area = bbox.width * bbox.height
        if area > best_area:
            best_area = area
            best = bbox

    if best is None:
        return None

    x0 = max(0, int(best.xmin * w))
    y0 = max(0, int(best.ymin * h))
    x1 = min(w, int((best.xmin + best.width) * w))
    y1 = min(h, int((best.ymin + best.height) * h))
    if x1 <= x0 or y1 <= y0:
        return None

    return (x0, y0, x1, y1), (best.xmin, best.ymin, best.width, best.height)


def extract_face_or_center_crop(
    image: Image.Image,
    *,
    try_mediapipe: bool = True,
    center_frac: float = 0.72,
) -> tuple[Image.Image, bool, tuple[float, float, float, float] | None]:
    w, h = image.size
    if try_mediapipe:
        det = detect_face_bbox(image)
        if det is not None:
            box, norm = det
            crop = image.crop(box)
            return crop, True, norm

    box = _center_crop_box(w, h, center_frac)
    crop = image.crop(box)
    x0, y0, x1, y1 = box
    norm = (x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)
    return crop, False, norm
