"""Абстрактный базовый класс для всех источников данных.

Каждый источник (Google Trends, RSS, веб-скрапер и т.д.) наследует BaseSource
и реализует методы fetch_batch() и source_name/source_type.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterator

from ..checkpoint import CheckpointManager
from ..schema import PostRecord
from ..storage import JsonlStorage

logger = logging.getLogger(__name__)


class BaseSource(ABC):
    """Интерфейс источника данных."""

    def __init__(
        self,
        storage: JsonlStorage,
        checkpoint: CheckpointManager,
        batch_size: int = 50,
    ) -> None:
        self.storage = storage
        self.checkpoint = checkpoint
        self.batch_size = batch_size

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Уникальное имя источника: google_trends, rss, forum и т.д."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Тип источника: search, feed, forum, website."""

    @abstractmethod
    def fetch_batch(self) -> Iterator[list[PostRecord]]:
        """Генератор батчей записей.

        Каждый yield — список PostRecord размером <= batch_size.
        Реализация должна учитывать checkpoint для возобновления.
        """

    def run(self) -> int:
        """Запустить полный цикл: fetch → save → checkpoint. Возвращает кол-во новых записей."""
        total = 0
        for batch in self.fetch_batch():
            if not batch:
                continue
            written = self.storage.append(batch)
            new_ids = [r.id_or_url for r in batch]
            self.checkpoint.update(new_ids=new_ids)
            total += written
            logger.info(
                "%s: batch saved, written=%d (total=%d)",
                self.source_name, written, total,
            )
        logger.info("%s: run complete, total new records=%d", self.source_name, total)
        return total
