
from __future__ import annotations

import hashlib
import numpy as np
from PIL import Image

from ai_fashion_trends.vision.schema import MakeupStyle

_SCORABLE = [s for s in MakeupStyle if s != MakeupStyle.UNKNOWN]

_FOCUS = (MakeupStyle.SMOKY_EYES, MakeupStyle.BLUE_EYESHADOW)


def _image_fingerprint(img: Image.Image, max_side: int = 96) -> tuple[np.ndarray, int]:
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    h = int.from_bytes(hashlib.sha256(arr.tobytes()).digest()[:8], "little")
    return arr, h


class MockMakeupClassifier:

    name = "mock_heuristic"

    def predict_proba(self, crop: Image.Image) -> dict[MakeupStyle, float]:
        arr, seed = _image_fingerprint(crop)
        rng = np.random.default_rng(seed % (2**32))

        r = float(arr[..., 0].mean())
        g = float(arr[..., 1].mean())
        b = float(arr[..., 2].mean())
        lum = float(0.299 * r + 0.587 * g + 0.114 * b)
        h, w, _ = arr.shape
        upper = arr[: max(1, int(h * 0.42)), :, :]
        ur = float(upper[..., 0].mean())
        ug = float(upper[..., 1].mean())
        ub = float(upper[..., 2].mean())
        eye_dark = 1.0 - float(upper.mean())
        blue_lead = ub - max(ur, ug)
        pix = upper.reshape(-1, 3)
        sat_upper = float(np.mean(np.std(pix, axis=1)))

        bonus_smoky = 0.0
        bonus_blue = 0.0
        if eye_dark > 0.52 and lum < 0.56:
            bonus_smoky += 0.5
        if blue_lead > 0.055 and sat_upper > 0.042:
            bonus_blue += 0.45
        if blue_lead > 0.09:
            bonus_blue += 0.16
        if bonus_smoky > 0.25 and bonus_blue > 0.35:
            bonus_smoky *= 0.62


        raw: dict[MakeupStyle, float] = {s: 1.0 for s in _SCORABLE}
        raw[MakeupStyle.SMOKY_EYES] += bonus_smoky
        raw[MakeupStyle.BLUE_EYESHADOW] += bonus_blue
        raw[MakeupStyle.SMOKY_EYES] += float(rng.uniform(0.0, 0.06))
        raw[MakeupStyle.BLUE_EYESHADOW] += float(rng.uniform(0.0, 0.06))
        for s in _SCORABLE:
            raw[s] += float(rng.uniform(0.0, 0.003))

        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}

    def predict_top(
        self, crop: Image.Image, top_k: int = 3
    ) -> tuple[MakeupStyle, float, list[tuple[MakeupStyle, float]]]:
        proba = self.predict_proba(crop)
        ranked = sorted(proba.items(), key=lambda x: -x[1])[:top_k]
        best, conf = ranked[0]
        if conf < 0.13 or best not in _FOCUS:
            return MakeupStyle.UNKNOWN, 1.0 - conf, ranked
        return best, conf, ranked
