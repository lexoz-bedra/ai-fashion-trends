from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "data" / "checkpoints"

_MAX_IDS = 5000


class CheckpointManager:

    def __init__(self, source: str, checkpoint_dir: Path | None = None) -> None:
        self._source = source
        self._dir = checkpoint_dir or _DEFAULT_CHECKPOINT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{source}.json"
        self._state: dict[str, Any] = self._load()

    @property
    def cursor(self) -> str | None:
        return self._state.get("cursor")

    @property
    def last_run_at(self) -> datetime | None:
        raw = self._state.get("last_run_at")
        if raw is None:
            return None
        return datetime.fromisoformat(raw)

    @property
    def processed_ids(self) -> set[str]:
        return set(self._state.get("processed_ids", []))

    def get_extra(self, key: str, default: Any = None) -> Any:
        return self._state.get("extra", {}).get(key, default)


    def update(
        self,
        *,
        new_ids: list[str] | None = None,
        cursor: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._state["last_run_at"] = datetime.utcnow().isoformat()

        if cursor is not None:
            self._state["cursor"] = cursor

        if new_ids:
            existing: list[str] = self._state.get("processed_ids", [])
            existing.extend(new_ids)
            self._state["processed_ids"] = existing[-_MAX_IDS:]

        if extra:
            self._state.setdefault("extra", {}).update(extra)

        self._save()

    def reset(self) -> None:
        self._state = {"source": self._source}
        self._save()


    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {"source": self._source}
        return {"source": self._source}

    def _save(self) -> None:
        self._state["source"] = self._source
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)
        tmp.replace(self._path)
