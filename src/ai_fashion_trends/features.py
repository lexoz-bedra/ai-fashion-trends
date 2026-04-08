from __future__ import annotations

import csv
from collections import defaultdict
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
