from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Engagement(BaseModel):

    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    saves: int | None = None
    views: int | None = None
    repins: int | None = None


class MediaMeta(BaseModel):

    type: str | None = None
    url: str | None = None
    width: int | None = None
    height: int | None = None
    duration_sec: float | None = None


class PostRecord(BaseModel):

    source: str = Field(..., description="Название источника: pinterest, tiktok, google_trends и т.д.")
    source_type: str = Field(..., description="Тип: social, search, marketplace")
    published_at: datetime | None = Field(None, description="Дата публикации (UTC)")
    id_or_url: str = Field(..., description="Уникальный идентификатор записи (ID поста или URL)")
    title_or_caption: str | None = Field(None, description="Заголовок или подпись")
    text: str | None = Field(None, description="Полный текст / описание")
    language: str | None = Field(None, description="Код языка ISO 639-1 (en, ru, ...)")
    engagement: Engagement = Field(default_factory=Engagement)
    media_meta: MediaMeta = Field(default_factory=MediaMeta)
    tags_raw: list[str] = Field(default_factory=list, description="Сырые теги / хештеги")
    ingested_at: datetime = Field(default_factory=datetime.utcnow, description="Момент сбора")
    extra: dict[str, Any] = Field(default_factory=dict, description="Доп. данные, не вошедшие в схему")
