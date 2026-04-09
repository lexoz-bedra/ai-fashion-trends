
from __future__ import annotations

from pathlib import Path

from PIL import Image

from ai_fashion_trends.vision.face import extract_face_or_center_crop
from ai_fashion_trends.vision.mock_classifier import MockMakeupClassifier
from ai_fashion_trends.vision.schema import MakeupAnalysisResult, MakeupStyle, STYLE_DESCRIPTIONS


class FaceMakeupPipeline:

    def __init__(
        self,
        *,
        use_mediapipe: bool = True,
        classifier: MockMakeupClassifier | None = None,
    ) -> None:
        self.use_mediapipe = use_mediapipe
        self.classifier = classifier or MockMakeupClassifier()

    def analyze(self, image_path: Path | str) -> MakeupAnalysisResult:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        image = Image.open(path).convert("RGB")
        crop, face_ok, norm = extract_face_or_center_crop(
            image, try_mediapipe=self.use_mediapipe
        )

        proba = self.classifier.predict_proba(crop)
        ranked = sorted(proba.items(), key=lambda x: -x[1])
        top, conf = ranked[0]
        if conf < 0.11:
            top = MakeupStyle.UNKNOWN
            conf = 1.0 - conf

        scores = {k.value: round(float(v), 4) for k, v in proba.items()}
        notes = (
            "Мок-классификатор: эвристики по цвету + хэш изображения. "
            "Для продакшена нужна размеченная выборка и дообучение (см. vision/schema.py)."
        )
        if not face_ok:
            notes += " Лицо не детектировано — использован центральный crop."

        return MakeupAnalysisResult(
            image_path=str(path.resolve()),
            top_style=top,
            confidence=round(float(conf), 4),
            style_scores=scores,
            face_detected=face_ok,
            face_bbox_norm=norm,
            classifier=self.classifier.name,
            notes=notes,
        )


def describe_style(style: MakeupStyle) -> str:
    return STYLE_DESCRIPTIONS.get(style, "")
