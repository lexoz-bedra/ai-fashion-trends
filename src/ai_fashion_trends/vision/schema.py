from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class MakeupStyle(str, Enum):

    SMOKY_EYES = "smoky_eyes"
    BLUE_EYESHADOW = "blue_eyeshadow"
    GRAPHIC_LINER = "graphic_liner"
    GLASS_SKIN_DEWY = "glass_skin_dewy"
    BOLD_LIP = "bold_lip"
    NATURAL_NO_MAKEUP = "natural_no_makeup"
    CONTOUR_SCULPTED = "contour_sculpted"
    BLUSH_DRAPING = "blush_draping"
    GLITTER_EUPHORIA = "glitter_euphoria"
    FOX_EYE_WING = "fox_eye_wing"
    UNKNOWN = "unknown"


STYLE_DESCRIPTIONS: dict[MakeupStyle, str] = {
    MakeupStyle.SMOKY_EYES: "Тёмные тени, дымчатая растушёвка вокруг глаз",
    MakeupStyle.BLUE_EYESHADOW: (
        "Холодные акцентные тени: голубой, бирюза, лаванда, серебро (тренд «blue eyeshadow»)"
    ),
    MakeupStyle.GRAPHIC_LINER: "Графичные стрелки, негативное пространство",
    MakeupStyle.GLASS_SKIN_DEWY: "Влажный финиш, сияние кожи (glass / dewy)",
    MakeupStyle.BOLD_LIP: "Акцент на губах, насыщенный цвет",
    MakeupStyle.NATURAL_NO_MAKEUP: "Натуральный / no-makeup makeup",
    MakeupStyle.CONTOUR_SCULPTED: "Скульптор, чёткий контур лица",
    MakeupStyle.BLUSH_DRAPING: "Румяна / draping, цвет на скулах и веках",
    MakeupStyle.GLITTER_EUPHORIA: "Глиттер, декоративные акценты на глазах",
    MakeupStyle.FOX_EYE_WING: "Вытянутое крыло, fox / siren eye",
    MakeupStyle.UNKNOWN: "Не удалось уверенно отнести к классу",
}


class MakeupAnalysisResult(BaseModel):

    image_path: str = Field(..., description="Исходный файл")
    top_style: MakeupStyle
    confidence: float = Field(..., ge=0.0, le=1.0)
    style_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Нормализованные веса по всем классам (для отладки)",
    )
    face_detected: bool = Field(
        False, description="True если найден бокс лица (MediaPipe или fallback)"
    )
    face_bbox_norm: tuple[float, float, float, float] | None = Field(
        None,
        description="Относительные координаты лица (x_min, y_min, w, h) в [0,1], если есть",
    )
    classifier: str = Field("mock_heuristic", description="Имя использованного классификатора")
    notes: str = Field("", description="Пояснение / ограничения мока")

    def model_dump_json_pretty(self) -> str:
        return self.model_dump_json(indent=2)
