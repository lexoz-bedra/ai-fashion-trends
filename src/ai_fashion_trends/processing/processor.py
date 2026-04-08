"""Процессор: читает сырые данные, пропускает через LLM, сохраняет тренды.

Полный цикл:
    1. Загрузить чекпоинт (какие id_or_url уже обработаны)
    2. Прочитать data/raw/*/*.jsonl
    3. Отфильтровать уже обработанные
    4. Батчами отправить в LLM
    5. Распарсить ответ → ExtractedTrend
    6. Сохранить в data/processed/trends.jsonl
    7. Обновить чекпоинт
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ..ingestion.checkpoint import CheckpointManager
from ..ingestion.llm import call_model
from .prompts import build_prompt
from .schema import ExtractedTrend

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TrendProcessor:
    """Обрабатывает сырые посты через LLM, извлекает тренды."""

    def __init__(
        self,
        raw_dir: Path | None = None,
        output_dir: Path | None = None,
        checkpoint_dir: Path | None = None,
        model_name: str = "gemma",
        batch_size: int = 10,
    ) -> None:
        self.raw_dir = raw_dir or _PROJECT_ROOT / "data" / "raw"
        self.output_dir = output_dir or _PROJECT_ROOT / "data" / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.batch_size = batch_size
        self.checkpoint = CheckpointManager(
            "processor", checkpoint_dir=checkpoint_dir
        )
        self._output_path = self.output_dir / "trends.jsonl"

    def run(self) -> int:
        """Основной цикл. Возвращает число извлечённых трендов."""
        posts = self._load_unprocessed_posts()
        if not posts:
            logger.info("Нет новых постов для обработки")
            return 0

        logger.info("Найдено %d необработанных постов", len(posts))
        total_trends = 0

        for i in range(0, len(posts), self.batch_size):
            batch = posts[i : i + self.batch_size]
            batch_trends: list[ExtractedTrend] = []
            processed_ids: list[str] = []

            for post in batch:
                post_id = post["id_or_url"]
                title = post.get("title_or_caption") or ""
                text = post.get("text") or ""

                if not title.strip() and not text.strip():
                    processed_ids.append(post_id)
                    continue

                if post.get("source") == "google_trends" and post.get("extra", {}).get("metric") == "interest_over_time":
                    processed_ids.append(post_id)
                    continue

                trends = self._process_single(post)
                batch_trends.extend(trends)
                processed_ids.append(post_id)

            # Сохраняем результаты батча
            if batch_trends:
                self._save_trends(batch_trends)
                total_trends += len(batch_trends)

            # Обновляем чекпоинт ПОСЛЕ сохранения
            self.checkpoint.update(new_ids=processed_ids)
            logger.info(
                "Batch %d–%d: извлечено %d трендов (total: %d)",
                i, i + len(batch), len(batch_trends), total_trends,
            )

        logger.info("Обработка завершена: %d трендов из %d постов", total_trends, len(posts))
        return total_trends

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _load_unprocessed_posts(self) -> list[dict[str, Any]]:
        """Загрузить все сырые посты, которых ещё нет в чекпоинте."""
        processed_ids = self.checkpoint.processed_ids
        posts: list[dict[str, Any]] = []

        for jsonl_path in sorted(self.raw_dir.rglob("*.jsonl")):
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if obj["id_or_url"] not in processed_ids:
                            posts.append(obj)
                    except (json.JSONDecodeError, KeyError):
                        continue
        return posts

    def _process_single(self, post: dict[str, Any]) -> list[ExtractedTrend]:
        """Отправить один пост в LLM, распарсить ответ."""
        title = post.get("title_or_caption") or ""
        text = post.get("text") or ""
        prompt = build_prompt(title, text)

        try:
            raw_response = call_model(prompt, model_name=self.model_name)
        except Exception:
            logger.exception("LLM call failed for %s", post["id_or_url"])
            raise

        return self._parse_response(raw_response, post)

    def _parse_response(
        self, raw: str, post: dict[str, Any]
    ) -> list[ExtractedTrend]:
        """Извлечь JSON-массив трендов из ответа модели."""
        # убираем reasoning в <think>...</think> (ну на всякий, пусть мы и через гемму делали)
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        cleaned = re.sub(r"```json\s*", "", cleaned)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()

        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            if cleaned == "[]" or not cleaned:
                return []
            logger.warning(
                "Не удалось найти JSON-массив в ответе LLM для %s: %.100s",
                post["id_or_url"], cleaned,
            )
            return []

        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            logger.warning(
                "Невалидный JSON от LLM для %s: %.200s",
                post["id_or_url"], match.group(),
            )
            return []

        if not isinstance(items, list):
            items = [items]

        trends: list[ExtractedTrend] = []
        published_at = None
        if post.get("published_at"):
            try:
                published_at = datetime.fromisoformat(str(post["published_at"]))
            except (ValueError, TypeError):
                pass

        valid_categories = {
            "clothing", "footwear", "accessory", "fabric",
            "color", "pattern", "aesthetic", "brand",
        }

        for item in items:
            if not isinstance(item, dict):
                continue
            cat = str(item.get("category", "")).lower()
            name = str(item.get("item", "")).strip()
            if not name or cat not in valid_categories:
                continue

            trends.append(
                ExtractedTrend(
                    source_record_id=post["id_or_url"],
                    source=post.get("source", "unknown"),
                    category=cat,
                    item=name.lower(),
                    sentiment=str(item.get("sentiment", "neutral")).lower(),
                    confidence=min(1.0, max(0.0, float(item.get("confidence", 0.5)))),
                    context_snippet=str(item.get("context_snippet", ""))[:200],
                    published_at=published_at,
                    tags=[str(t) for t in item.get("tags", [])],
                )
            )
        return trends

    def _save_trends(self, trends: list[ExtractedTrend]) -> None:
        """Дописать тренды в JSONL-файл."""
        with self._output_path.open("a", encoding="utf-8") as f:
            for t in trends:
                f.write(t.model_dump_json() + "\n")
