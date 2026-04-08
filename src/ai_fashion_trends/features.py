from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def build_weekly_features(cleaned_path: Path, features_path: Path) -> Path:
    """Aggregate cleaned records into weekly trend counts and engagement."""
    features_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    engagement_sum: dict[tuple[str, str], int] = defaultdict(int)

    with cleaned_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            key = (row["trend_tag"], row["week_start"])
            counts[key] += 1
            engagement_sum[key] += int(row.get("engagement", 0) or 0)

    rows = []
    for (trend_tag, week_start), cnt in counts.items():
        avg_eng = engagement_sum[(trend_tag, week_start)] / max(cnt, 1)
        rows.append(
            {
                "trend_tag": trend_tag,
                "week_start": week_start,
                "count": cnt,
                "avg_engagement": round(avg_eng, 2),
            }
        )
    rows.sort(key=lambda x: (x["trend_tag"], x["week_start"]))

    with features_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["trend_tag", "week_start", "count", "avg_engagement"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return features_path


def _parse_dt(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    s = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _week_start_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    d = dt.astimezone(timezone.utc).date()
    week_start_ord = d.toordinal() - d.weekday()
    return datetime.fromordinal(week_start_ord).date().isoformat()


def _trend_key_from_record(rec: dict) -> str | None:
    item = (rec.get("item") or "").strip().lower()
    if item:
        return item[:200]
    tags = rec.get("tags") or []
    if tags:
        return "|".join(sorted(str(t).lower() for t in tags if t))[:200]
    cat = (rec.get("category") or "").strip().lower()
    if cat:
        return cat[:200]
    return None


def build_weekly_features_from_trends_jsonl(jsonl_path: Path, features_path: Path) -> Path:
    """Агрегация по неделям из выхода `process` (ExtractedTrend в JSONL)."""
    features_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    conf_sum: dict[tuple[str, str], float] = defaultdict(float)

    with jsonl_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tag = _trend_key_from_record(rec)
            if not tag:
                continue
            dt = _parse_dt(rec.get("published_at")) or _parse_dt(rec.get("processed_at"))
            if dt is None:
                continue
            ws = _week_start_iso(dt)
            key = (tag, ws)
            counts[key] += 1
            conf_sum[key] += float(rec.get("confidence") or 0.0)

    rows: list[dict[str, str | int | float]] = []
    for (trend_tag, week_start), cnt in counts.items():
        avg_eng = (conf_sum[(trend_tag, week_start)] / cnt) * 100.0
        rows.append(
            {
                "trend_tag": trend_tag,
                "week_start": week_start,
                "count": cnt,
                "avg_engagement": round(avg_eng, 2),
            }
        )
    rows.sort(key=lambda x: (x["trend_tag"], x["week_start"]))

    with features_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["trend_tag", "week_start", "count", "avg_engagement"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return features_path
