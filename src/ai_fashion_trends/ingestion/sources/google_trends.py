"""Источник: Google Trends.

Использует pytrends для получения:
- trending searches (daily/realtime)
- interest over time по ключевым словам

Конфиг передаётся при инициализации — список ключевых слов (keywords)
и регион (geo).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Iterator

from ..checkpoint import CheckpointManager
from ..schema import Engagement, PostRecord
from ..storage import JsonlStorage
from .base import BaseSource

logger = logging.getLogger(__name__)


class GoogleTrendsSource(BaseSource):
    """Сбор данных из Google Trends через pytrends."""

    def __init__(
        self,
        storage: JsonlStorage,
        checkpoint: CheckpointManager,
        keywords: list[str],
        geo: str = "",
        timeframe: str = "today 3-m",
        batch_size: int = 5,
    ) -> None:
        super().__init__(storage, checkpoint, batch_size)
        self.keywords = keywords
        self.geo = geo
        self.timeframe = timeframe
        self._pytrends: Any = None

    @property
    def source_name(self) -> str:
        return "google_trends"

    @property
    def source_type(self) -> str:
        return "search"

    def _get_client(self) -> Any:
        if self._pytrends is None:
            from pytrends.request import TrendReq 
            self._pytrends = TrendReq(hl="en-US", tz=360)
        return self._pytrends

    def fetch_batch(self) -> Iterator[list[PostRecord]]:
        """Получаем interest_over_time + related_queries батчами по batch_size ключевых слов."""
        import time

        processed = self.checkpoint.processed_ids
        pt = self._get_client()

        for i in range(0, len(self.keywords), self.batch_size):
            kw_batch = self.keywords[i : i + self.batch_size]
            batch_key = _batch_id(kw_batch, self.geo, self.timeframe)
            if batch_key in processed:
                logger.debug("Skipping already processed batch: %s", kw_batch)
                continue

            records: list[PostRecord] = []
            
            max_retries = 3 # 429 error
            for attempt in range(max_retries):
                try:
                    logger.info("Fetching interest over time for %s (attempt %d)", kw_batch, attempt + 1)
                    pt.build_payload(kw_batch, cat=0, timeframe=self.timeframe, geo=self.geo)
                    iot = pt.interest_over_time()
                    if not iot.empty:
                        records.extend(self._parse_interest_over_time(iot, kw_batch))
                    
                    time.sleep(2)

                    logger.info("Fetching related queries for %s (attempt %d)", kw_batch, attempt + 1)
                    related = pt.related_queries()
                    records.extend(self._parse_related_queries(related, kw_batch))
                    
                    break
                    
                except Exception as e:
                    import pytrends.exceptions
                    if isinstance(e, pytrends.exceptions.TooManyRequestsError) or "429" in str(e):
                        wait_time = 60 * (attempt + 1)
                        logger.warning("Google Trends 429 Too Many Requests -> sleep %d seconds", wait_time)
                        time.sleep(wait_time)
                        if attempt == max_retries - 1:
                            logger.error("Skipping %s after max retries", kw_batch)
                    else:
                        logger.exception("Google Trends failed for %s", kw_batch)
                        break

            if records:
                yield records
                self.checkpoint.update(
                    new_ids=[batch_key],
                    extra={"last_keywords_batch": kw_batch},
                )
            
            time.sleep(5)

    def _parse_interest_over_time(self, df: Any, keywords: list[str]) -> list[PostRecord]:
        records: list[PostRecord] = []
        for kw in keywords:
            if kw not in df.columns:
                continue
            for ts, row in df[[kw]].iterrows():
                rec_id = f"gt_iot_{kw}_{ts.isoformat()}"
                records.append(
                    PostRecord(
                        source=self.source_name,
                        source_type=self.source_type,
                        published_at=ts.to_pydatetime().replace(tzinfo=timezone.utc),
                        id_or_url=rec_id,
                        title_or_caption=kw,
                        text=f"Interest score: {int(row[kw])}",
                        language=None,
                        engagement=Engagement(views=int(row[kw])),
                        tags_raw=[kw],
                        extra={"metric": "interest_over_time", "score": int(row[kw]), "geo": self.geo},
                    )
                )
        return records

    def _parse_related_queries(self, related: dict[str, Any], keywords: list[str]) -> list[PostRecord]:
        records: list[PostRecord] = []
        now = datetime.now(tz=timezone.utc)
        for kw in keywords:
            kw_data = related.get(kw, {})
            for qtype in ("top", "rising"):
                df = kw_data.get(qtype)
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    query = str(row.get("query", ""))
                    value = row.get("value", 0)
                    rec_id = f"gt_rq_{kw}_{qtype}_{query}"
                    records.append(
                        PostRecord(
                            source=self.source_name,
                            source_type=self.source_type,
                            published_at=now,
                            id_or_url=rec_id,
                            title_or_caption=query,
                            text=f"Related {qtype} query for '{kw}': {query} (value={value})",
                            language=None,
                            engagement=Engagement(views=int(value) if str(value).isdigit() else None),
                            tags_raw=[kw, query],
                            extra={"metric": f"related_{qtype}", "parent_keyword": kw, "value": str(value)},
                        )
                    )
        return records


def _batch_id(keywords: list[str], geo: str, timeframe: str) -> str:
    raw = f"{'|'.join(sorted(keywords))}|{geo}|{timeframe}"
    return "gtbatch_" + hashlib.md5(raw.encode()).hexdigest()[:12]
