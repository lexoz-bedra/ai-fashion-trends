"""Источник: RSS-ленты.

Принимает список URL RSS/Atom фидов, парсит через feedparser,
нормализует каждую запись в PostRecord.

Чекпоинт — по id записи (link или guid). При рестарте пропускает уже
обработанные записи.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterator

from ..checkpoint import CheckpointManager
from ..schema import PostRecord
from ..storage import JsonlStorage
from .base import BaseSource

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")


class RssSource(BaseSource):

    def __init__(
        self,
        storage: JsonlStorage,
        checkpoint: CheckpointManager,
        feed_urls: list[str],
        batch_size: int = 50,
    ) -> None:
        super().__init__(storage, checkpoint, batch_size)
        self.feed_urls = feed_urls

    @property
    def source_name(self) -> str:
        return "rss"

    @property
    def source_type(self) -> str:
        return "feed"

    def fetch_batch(self) -> Iterator[list[PostRecord]]:
        import feedparser 

        processed = self.checkpoint.processed_ids

        for feed_url in self.feed_urls:
            logger.info("Fetching RSS feed: %s", feed_url)
            try:
                feed = feedparser.parse(feed_url)
            except Exception:
                logger.exception("Failed to parse feed: %s", feed_url)
                continue

            batch: list[PostRecord] = []
            for entry in feed.entries:
                rec_id = _entry_id(entry, feed_url)
                if rec_id in processed:
                    continue

                record = _entry_to_record(entry, feed_url, rec_id)
                batch.append(record)

                if len(batch) >= self.batch_size:
                    yield batch
                    self.checkpoint.update(
                        new_ids=[r.id_or_url for r in batch],
                        extra={"last_feed": feed_url},
                    )
                    batch = []

            # остаток
            if batch:
                yield batch
                self.checkpoint.update(
                    new_ids=[r.id_or_url for r in batch],
                    extra={"last_feed": feed_url},
                )


def _entry_id(entry: Any, feed_url: str) -> str:
    """Стабильный ID записи: guid > link > hash(title+feed)."""
    if getattr(entry, "id", None):
        return entry.id
    if getattr(entry, "link", None):
        return entry.link
    raw = f"{getattr(entry, 'title', '')}|{feed_url}"
    return "rss_" + hashlib.md5(raw.encode()).hexdigest()[:16]


def _parse_date(entry: Any) -> datetime | None:
    for field in ("published", "updated"):
        raw = getattr(entry, field, None)
        if raw:
            try:
                return parsedate_to_datetime(raw).astimezone(timezone.utc)
            except Exception:
                pass
    return None


def _strip_html(text: str) -> str:
    return _TAG_RE.sub("", text).strip()


def _extract_tags(entry: Any) -> list[str]:
    tags: list[str] = []
    for tag_info in getattr(entry, "tags", []):
        term = getattr(tag_info, "term", None)
        if term:
            tags.append(term)
    return tags


def _entry_to_record(entry: Any, feed_url: str, rec_id: str) -> PostRecord:
    title = getattr(entry, "title", None) or ""
    summary = getattr(entry, "summary", None) or ""
    content_blocks = getattr(entry, "content", [])
    full_text = ""
    if content_blocks:
        full_text = " ".join(
            getattr(block, "value", "") for block in content_blocks
        )
    text = _strip_html(full_text or summary)

    return PostRecord(
        source="rss",
        source_type="feed",
        published_at=_parse_date(entry),
        id_or_url=rec_id,
        title_or_caption=_strip_html(title),
        text=text,
        language=getattr(entry, "language", None),
        tags_raw=_extract_tags(entry),
        extra={"feed_url": feed_url, "link": getattr(entry, "link", None)},
    )
