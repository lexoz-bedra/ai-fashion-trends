
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from ai_fashion_trends.vision.features import features_from_uploaded_pil

Outcome = Literal["smoky_eyes", "blue_eyeshadow", "other", "not_face"]


class SmokyBluePrediction(BaseModel):
    outcome: Outcome
    p_smoky: float = Field(..., ge=0.0, le=1.0)
    p_blue: float = Field(..., ge=0.0, le=1.0)
    p_other: float = Field(0.0, ge=0.0, le=1.0)
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Максимальная вероятность среди классов модели",
    )
    face_detected: bool = False


def default_model_path(project_root: Path) -> Path:
    return project_root / "data" / "models" / "makeup_smoky_blue.joblib"


def default_torch_model_path(project_root: Path) -> Path:
    return project_root / "data" / "models" / "makeup_resnet18.pt"


class SmokyBlueClassifier:

    def __init__(
        self,
        pipeline: Any,
        *,
        meta: dict[str, Any],
    ) -> None:
        self.pipeline = pipeline
        self.meta = meta
        self.classes_: np.ndarray = np.asarray(meta.get("classes", [0, 1, 2]))
        self.prob_threshold: float = float(meta.get("prob_threshold", 0.45))

    @classmethod
    def load(cls, path: Path | str) -> SmokyBlueClassifier:
        path = Path(path)
        raw = joblib.load(path)
        if isinstance(raw, dict) and "pipeline" in raw:
            loaded = cls(raw["pipeline"], meta=raw.get("meta", {}))

            clf = loaded.pipeline.steps[-1][1]
            if hasattr(clf, "classes_"):
                loaded.classes_ = np.asarray(clf.classes_)
            return loaded
        raise ValueError(f"Неизвестный формат модели: {path}")

    def predict_pil(
        self,
        image: Image.Image,
        *,
        use_mediapipe: bool = False,
    ) -> SmokyBluePrediction:
        from ai_fashion_trends.vision.face import extract_face_or_center_crop

        _, face_ok, _ = extract_face_or_center_crop(
            image.convert("RGB"), try_mediapipe=use_mediapipe
        )
        x = features_from_uploaded_pil(image, use_mediapipe=use_mediapipe).reshape(1, -1)
        proba = self.pipeline.predict_proba(x)[0]
        clf = self.pipeline.steps[-1][1]
        cls_order = [int(c) for c in clf.classes_]
        idx = {c: i for i, c in enumerate(cls_order)}

        p_tri = np.zeros(3, dtype=np.float64)
        for k in (0, 1, 2):
            if k in idx:
                p_tri[k] = proba[idx[k]]
        p_smoky = float(p_tri[0])
        p_blue = float(p_tri[1])
        p_other = float(p_tri[2])

        conf = float(p_tri.max())
        if conf < self.prob_threshold:
            return SmokyBluePrediction(
                outcome="not_face",
                p_smoky=p_smoky,
                p_blue=p_blue,
                p_other=p_other,
                confidence=conf,
                face_detected=face_ok,
            )

        best_k = int(p_tri.argmax())
        out_map: dict[int, Outcome] = {
            0: "smoky_eyes",
            1: "blue_eyeshadow",
            2: "other",
        }
        return SmokyBluePrediction(
            outcome=out_map[best_k],
            p_smoky=p_smoky,
            p_blue=p_blue,
            p_other=p_other,
            confidence=float(p_tri[best_k]),
            face_detected=face_ok,
        )


def save_classifier_bundle(
    path: Path,
    pipeline: Any,
    *,
    classes: list[int],
    prob_threshold: float,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "classes": classes,
        "prob_threshold": prob_threshold,
        "feature_module": "ai_fashion_trends.vision.features",
        "labels": {"0": "smoky_eyes", "1": "blue_eyeshadow", "2": "other"},
    }
    if extra_meta:
        meta.update(extra_meta)
    joblib.dump({"pipeline": pipeline, "meta": meta}, path)
    side = path.with_suffix(".json")
    side.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
