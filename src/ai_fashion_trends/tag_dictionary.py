from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9]+")
_SPACE_RE = re.compile(r"\s+")
_HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")


def _norm_tag(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = _SPACE_RE.sub(" ", s).strip()
    words = _WORD_RE.findall(s)
    return " ".join(words).strip()


def _canonical_from_tag(tag: str) -> str:
    return _norm_tag(tag).replace(" ", "_")


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass
class TagDictionaryStats:
    entries: int
    trends_read: int
    output_path: Path


def build_tag_dictionary(
    trends_jsonl: Path,
    output_jsonl: Path,
    *,
    min_count: int = 2,
) -> TagDictionaryStats:
    trends_jsonl = Path(trends_jsonl)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    bucket: dict[str, dict[str, Any]] = {}
    trends_read = 0

    with trends_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            trends_read += 1

            raw_item = str(obj.get("item", "")).strip()
            if not raw_item:
                continue
            norm_item = _norm_tag(raw_item)
            if not norm_item:
                continue
            canonical = _canonical_from_tag(norm_item)

            row = bucket.setdefault(
                canonical,
                {
                    "canonical_tag": canonical,
                    "display_counter": Counter(),
                    "aliases_counter": Counter(),
                    "hashtag_counter": Counter(),
                    "categories": Counter(),
                    "sources": Counter(),
                    "sentiment": Counter(),
                    "count_mentions": 0,
                    "sum_conf": 0.0,
                    "first_seen": None,
                    "last_seen": None,
                },
            )

            row["display_counter"][norm_item] += 1
            row["aliases_counter"][norm_item] += 1
            row["count_mentions"] += 1

            conf = obj.get("confidence", 0.0)
            try:
                row["sum_conf"] += float(conf)
            except (TypeError, ValueError):
                pass

            cat = _norm_tag(str(obj.get("category", "")))
            if cat:
                row["categories"][cat] += 1

            src = _norm_tag(str(obj.get("source", "")))
            if src:
                row["sources"][src] += 1

            sent = _norm_tag(str(obj.get("sentiment", "")))
            if sent:
                row["sentiment"][sent] += 1

            for t in obj.get("tags", []) or []:
                nt = _norm_tag(str(t))
                if nt:
                    row["aliases_counter"][nt] += 1

            snippet = str(obj.get("context_snippet", ""))
            for h in _HASHTAG_RE.findall(snippet):
                nh = _canonical_from_tag(h)
                if nh:
                    row["aliases_counter"][nh.replace("_", " ")] += 1
                    row["hashtag_counter"][f"#{nh}"] += 1

            p_dt = _parse_dt(obj.get("published_at"))
            if p_dt is not None:
                fs = row["first_seen"]
                ls = row["last_seen"]
                if fs is None or p_dt < fs:
                    row["first_seen"] = p_dt
                if ls is None or p_dt > ls:
                    row["last_seen"] = p_dt

    entries: list[dict[str, Any]] = []
    for key, row in bucket.items():
        count = int(row["count_mentions"])
        if count < min_count:
            continue
        display = row["display_counter"].most_common(1)[0][0] if row["display_counter"] else key.replace("_", " ")
        aliases = [a for a, _ in row["aliases_counter"].most_common(20)]
        hashtags = [h for h, _ in row["hashtag_counter"].most_common(20)]
        categories = [c for c, _ in row["categories"].most_common()]
        sources = dict(row["sources"].most_common())
        sentiment = dict(row["sentiment"].most_common())
        first_seen = row["first_seen"].isoformat() if row["first_seen"] is not None else None
        last_seen = row["last_seen"].isoformat() if row["last_seen"] is not None else None
        avg_conf = row["sum_conf"] / max(count, 1)

        entries.append(
            {
                "canonical_tag": key,
                "display_name": display,
                "aliases": aliases,
                "hashtags": hashtags,
                "categories": categories,
                "count_mentions": count,
                "avg_confidence": round(float(avg_conf), 4),
                "sentiment_distribution": sentiment,
                "sources": sources,
                "first_seen": first_seen,
                "last_seen": last_seen,
            }
        )

    entries.sort(key=lambda x: (-x["count_mentions"], x["canonical_tag"]))
    with output_jsonl.open("w", encoding="utf-8") as out:
        for e in entries:
            out.write(json.dumps(e, ensure_ascii=False) + "\n")

    return TagDictionaryStats(
        entries=len(entries),
        trends_read=trends_read,
        output_path=output_jsonl,
    )
