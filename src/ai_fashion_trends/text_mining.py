from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from ai_fashion_trends.data_ingestion import TREND_LIBRARY


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _canonical_tag(tag_candidate: str) -> str | None:
    cleaned = _normalize_text(tag_candidate)
    for canonical, variants in TREND_LIBRARY.items():
        if cleaned == canonical:
            return canonical
        if cleaned in {_normalize_text(v) for v in variants}:
            return canonical
    return None


def clean_and_extract_tags(raw_path: Path, cleaned_path: Path) -> Path:
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()

    fieldnames = [
        "record_id",
        "source",
        "source_type",
        "published_at",
        "week_start",
        "language",
        "engagement",
        "trend_tag",
        "text_clean",
    ]
    with raw_path.open("r", encoding="utf-8", newline="") as src, cleaned_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            record_id = row["record_id"].strip()
            if not record_id or record_id in seen_ids:
                continue
            seen_ids.add(record_id)

            text_clean = _normalize_text(f'{row.get("title", "")} {row.get("text", "")}')
            dt = datetime.fromisoformat(row["published_at"]).astimezone(timezone.utc)
            week_start = (dt.date().toordinal() - dt.weekday())
            week_start_dt = datetime.fromordinal(week_start).date().isoformat()

            tags_raw = row.get("tags_raw", "")
            tag_candidates = [x.strip() for x in tags_raw.split(",") if x.strip()]
            inferred = [_canonical_tag(x) for x in tag_candidates]
            inferred = [x for x in inferred if x is not None]

            if not inferred:

                for canonical, variants in TREND_LIBRARY.items():
                    if any(_normalize_text(v) in text_clean for v in variants):
                        inferred.append(canonical)

            for tag in sorted(set(inferred)):
                writer.writerow(
                    {
                        "record_id": record_id,
                        "source": row["source"],
                        "source_type": row["source_type"],
                        "published_at": row["published_at"],
                        "week_start": week_start_dt,
                        "language": row.get("language", "unknown"),
                        "engagement": int(float(row.get("engagement", 0) or 0)),
                        "trend_tag": tag,
                        "text_clean": text_clean,
                    }
                )
    return cleaned_path
