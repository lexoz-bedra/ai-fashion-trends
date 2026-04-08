from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .checkpoint import CheckpointManager
from .sources.base import BaseSource
from .storage import JsonlStorage

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Запуск источников с общим storage и per-source чекпоинтами."""

    def __init__(
        self,
        data_dir: Path | None = None,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.storage = JsonlStorage(data_dir=data_dir)
        self._checkpoint_dir = checkpoint_dir
        self._sources: list[BaseSource] = []

    def add_source(self, source: BaseSource) -> None:
        self._sources.append(source)

    def make_checkpoint(self, source_name: str) -> CheckpointManager:
        return CheckpointManager(source_name, checkpoint_dir=self._checkpoint_dir)

    def run(self) -> dict[str, int]:
        """Запустить все источники. Возвращает {source_name: new_records_count}."""
        results: dict[str, int] = {}
        for source in self._sources:
            name = source.source_name
            logger.info("=== Starting source: %s ===", name)
            try:
                count = source.run()
                results[name] = count
                logger.info("=== Source %s done: %d new records ===", name, count)
            except Exception:
                logger.exception("=== Source %s FAILED ===", name)
                results[name] = -1
        return results
