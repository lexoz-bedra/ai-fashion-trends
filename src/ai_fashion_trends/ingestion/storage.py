from __future__ import annotations

import json
from pathlib import Path

from .schema import PostRecord

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


class JsonlStorage:

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or _DEFAULT_DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._seen: dict[str, set[str]] = {}

    def append(self, records: list[PostRecord]) -> int:
        written = 0
        by_source: dict[str, list[PostRecord]] = {}
        for rec in records:
            by_source.setdefault(rec.source, []).append(rec)

        for source, recs in by_source.items():
            seen = self._load_seen(source)
            path = self._file_path(source)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                for rec in recs:
                    if rec.id_or_url in seen:
                        continue
                    line = rec.model_dump_json()
                    f.write(line + "\n")
                    seen.add(rec.id_or_url)
                    written += 1
        return written

    def count(self, source: str) -> int:
        return len(self._load_seen(source))

    def has(self, source: str, id_or_url: str) -> bool:
        return id_or_url in self._load_seen(source)

    def _file_path(self, source: str) -> Path:
        return self._data_dir / source / "posts.jsonl"

    def _load_seen(self, source: str) -> set[str]:
        if source in self._seen:
            return self._seen[source]

        seen: set[str] = set()
        path = self._file_path(source)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        seen.add(obj["id_or_url"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        self._seen[source] = seen
        return seen
