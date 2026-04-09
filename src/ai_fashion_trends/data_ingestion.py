from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class RawRecord:
    source: str
    source_type: str
    published_at: str
    record_id: str
    title: str
    text: str
    language: str
    engagement: int
    tags_raw: str


TREND_LIBRARY: dict[str, list[str]] = {
    "glass_skin": ["glass skin", "dewy skin", "glowy base"],
    "clean_girl_makeup": ["clean girl", "minimal makeup", "no makeup makeup"],
    "euphoria_glitter_liner": ["euphoria makeup", "glitter liner", "rhinestone eyes"],
    "fox_eye_siren_eye": ["fox eye", "siren eyes", "winged liner lift"],
    "cherry_cola_lips": ["cherry cola lips", "dark liner lips", "90s lip combo"],
    "latte_makeup": ["latte makeup", "warm neutrals", "soft brown eyes"],
    "strawberry_girl_blush": ["strawberry girl", "fruit blush", "douyin blush"],
    "graphic_editorial_liner": ["graphic liner", "floating liner", "negative space liner"],
}

SOURCES = [
    ("pinterest", "social"),
    ("instagram", "social"),
    ("tiktok", "social"),
    ("vogue", "news"),
    ("business_of_fashion", "news"),
]


def generate_mock_raw_dataset(
    output_path: Path,
    seed: int = 42,
    weeks: int = 32,
    records_per_week: int = 50,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    base = datetime.now(tz=timezone.utc) - timedelta(weeks=weeks)

    fieldnames = [
        "source",
        "source_type",
        "published_at",
        "record_id",
        "title",
        "text",
        "language",
        "engagement",
        "tags_raw",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        idx = 0
        for week in range(weeks):
            for _ in range(records_per_week):
                source, source_type = random.choice(SOURCES)
                date = base + timedelta(weeks=week, days=random.randint(0, 6))
                trend = random.choice(list(TREND_LIBRARY.keys()))
                synonyms = TREND_LIBRARY[trend]
                season_boost = 1.0
                if trend == "berry_lips" and date.month in (10, 11, 12):
                    season_boost = 1.5
                if trend == "glass_skin" and date.month in (5, 6, 7):
                    season_boost = 1.4
                engagement = int(random.randint(20, 400) * season_boost)

                title = f"{random.choice(synonyms).title()} ideas #{week}"
                text = (
                    f"{random.choice(synonyms)} trend is growing. "
                    f"Look from {source} with seasonal beauty focus."
                )
                tags_raw = ",".join(random.sample(synonyms, k=min(2, len(synonyms))))
                row = RawRecord(
                    source=source,
                    source_type=source_type,
                    published_at=date.isoformat(),
                    record_id=f"mock_{idx}",
                    title=title,
                    text=text,
                    language="en",
                    engagement=engagement,
                    tags_raw=tags_raw,
                )
                writer.writerow(row.__dict__)
                idx += 1
    return output_path
