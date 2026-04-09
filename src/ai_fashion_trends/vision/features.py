
from __future__ import annotations

import numpy as np
from PIL import Image


def compute_makeup_features(crop: Image.Image) -> np.ndarray:
    im = crop.convert("RGB")
    w0, h0 = im.size
    if h0 < 8 or w0 < 8:
        im = im.resize((max(w0, 64), max(h0, 64)), Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    h, w, _ = arr.shape

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    def _stats(a: np.ndarray) -> list[float]:
        a = a.reshape(-1)
        return [float(a.mean()), float(a.std())]

    feats: list[float] = []
    feats += [float(r.mean()), float(g.mean()), float(b.mean())]
    feats += [float(r.std()), float(g.std()), float(b.std())]
    feats += _stats(lum)

    uh = max(1, h // 2)
    up = arr[:uh, :, :]
    ur, ug, ub = up[..., 0], up[..., 1], up[..., 2]
    ulum = 0.299 * ur + 0.587 * ug + 0.114 * ub
    blue_lead = ub - np.maximum(ur, ug)
    feats += [float(ur.mean()), float(ug.mean()), float(ub.mean())]
    feats += [float(ur.std()), float(ug.std()), float(ub.std())]
    feats += _stats(ulum)
    feats += [float(np.mean(blue_lead)), float(np.std(blue_lead))]
    feats += [float((blue_lead > 0.06).mean())]
    feats += [float((ulum < 0.48).mean())]

    lt = max(1, h * 2 // 3)
    low = arr[lt:, :, :]
    feats += [
        float(low[..., 0].mean()),
        float(low[..., 1].mean()),
        float(low[..., 2].mean()),
    ]


    hist, _ = np.histogram(ulum.reshape(-1), bins=8, range=(0.0, 1.0))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
    feats.extend(hist.tolist())

    out = np.array(feats, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def features_from_uploaded_pil(image: Image.Image, *, use_mediapipe: bool) -> np.ndarray:
    from ai_fashion_trends.vision.face import extract_face_or_center_crop

    crop, _, _ = extract_face_or_center_crop(image, try_mediapipe=use_mediapipe)
    return compute_makeup_features(crop)
