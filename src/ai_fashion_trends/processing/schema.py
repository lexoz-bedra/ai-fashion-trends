from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ExtractedTrend(BaseModel):

    source_record_id: str = Field(..., description="id_or_url исходного PostRecord")
    source: str = Field(..., description="Источник: rss, web_reddit_fashion, google_trends")

    category: str = Field(
        ...,
        description="Категория: clothing, footwear, accessory, fabric, color, pattern, aesthetic, brand",
    )
    item: str = Field(..., description="Конкретная сущность: 'linen shorts', 'red shoes', 'gorpcore'")
    sentiment: str = Field(
        default="neutral",
        description="Отношение: positive, negative, neutral",
    )
    confidence: float = Field(
        default=0.5,
        description="Уверенность модели 0.0–1.0",
    )
    context_snippet: str = Field(
        default="",
        description="Фрагмент текста, откуда извлечён тренд (до 200 символов)",
    )

    published_at: datetime | None = Field(None, description="Дата публикации исходного поста")
    processed_at: datetime = Field(default_factory=datetime.utcnow)

    tags: list[str] = Field(default_factory=list, description="Сопутствующие теги")
    extra: dict[str, Any] = Field(default_factory=dict)
