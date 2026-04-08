"""Источник: веб-скрапер для форумов и сайтов.

Универсальный скрапер: принимает список конфигов сайтов с CSS-селекторами
для извлечения контента. Поддерживает:
- форумы (темы/посты)
- блоги / журналы (статьи)
- любые страницы с повторяющейся структурой

Каждый сайт описывается через SiteScrapeConfig.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator
from urllib.parse import urljoin

from ..checkpoint import CheckpointManager
from ..schema import PostRecord
from ..storage import JsonlStorage
from .base import BaseSource

logger = logging.getLogger(__name__)


@dataclass
class SiteScrapeConfig:
    """Конфиг для одного сайта/форума.

    CSS-селекторы указывают, где на странице находятся нужные элементы.
    """

    name: str                           # человекочитаемое имя: "fashionista_forum"
    start_urls: list[str]               # начальные URL (список страниц / разделов)
    item_selector: str                  # CSS-селектор одного поста / статьи на странице
    title_selector: str = ""            # заголовок внутри item
    text_selector: str = ""             # текст / контент внутри item
    date_selector: str = ""             # дата публикации
    date_attr: str = ""                 # атрибут с датой (datetime, content и т.д.), если пусто — берём text
    link_selector: str = ""             # ссылка на полный пост (href)
    tags_selector: str = ""             # теги / категории
    next_page_selector: str = ""        # кнопка / ссылка «следующая страница»
    max_pages: int = 10                 # лимит пагинации
    request_delay: float = 1.0          # задержка между запросами (сек)
    source_type: str = "forum"          # forum, website, blog
    extra_headers: dict[str, str] = field(default_factory=dict)


class WebScraperSource(BaseSource):
    """Скрапинг форумов / сайтов по конфигам."""

    def __init__(
        self,
        storage: JsonlStorage,
        checkpoint: CheckpointManager,
        configs: list[SiteScrapeConfig],
        batch_size: int = 20,
    ) -> None:
        super().__init__(storage, checkpoint, batch_size)
        self.configs = configs

    @property
    def source_name(self) -> str:
        return "web_scraper"

    @property
    def source_type(self) -> str:
        return "website"

    def fetch_batch(self) -> Iterator[list[PostRecord]]:
        import requests
        from bs4 import BeautifulSoup

        processed = self.checkpoint.processed_ids

        for cfg in self.configs:
            logger.info("Scraping site: %s", cfg.name)
            urls_to_scrape = list(cfg.start_urls)
            pages_scraped = 0

            while urls_to_scrape and pages_scraped < cfg.max_pages:
                url = urls_to_scrape.pop(0)
                page_key = _page_id(cfg.name, url)
                if page_key in processed:
                    pages_scraped += 1
                    continue

                logger.debug("Fetching page: %s", url)
                try:
                    headers = {"User-Agent": "Mozilla/5.0 (compatible; FashionTrendBot/0.1)"}
                    headers.update(cfg.extra_headers)
                    resp = requests.get(url, headers=headers, timeout=15)
                    resp.raise_for_status()
                except Exception:
                    logger.exception("Failed to fetch: %s", url)
                    pages_scraped += 1
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                records = self._parse_page(soup, url, cfg)

                new_records = [r for r in records if r.id_or_url not in processed]

                if new_records:
                    for j in range(0, len(new_records), self.batch_size):
                        batch = new_records[j : j + self.batch_size]
                        yield batch
                        self.checkpoint.update(
                            new_ids=[r.id_or_url for r in batch] + [page_key],
                            extra={"last_site": cfg.name, "last_url": url},
                        )

                if cfg.next_page_selector:
                    next_link = soup.select_one(cfg.next_page_selector)
                    if next_link:
                        href = next_link.get("href")
                        if href:
                            next_url = urljoin(url, str(href))
                            urls_to_scrape.append(next_url)

                pages_scraped += 1
                if cfg.request_delay > 0:
                    time.sleep(cfg.request_delay)

    def _parse_page(
        self, soup: Any, page_url: str, cfg: SiteScrapeConfig
    ) -> list[PostRecord]:
        items = soup.select(cfg.item_selector)
        records: list[PostRecord] = []

        for item in items:
            title = _extract_text(item, cfg.title_selector) if cfg.title_selector else ""
            text = _extract_text(item, cfg.text_selector) if cfg.text_selector else item.get_text(strip=True)
            link = _extract_href(item, cfg.link_selector, page_url) if cfg.link_selector else page_url
            pub_date = _extract_date(item, cfg.date_selector, cfg.date_attr) if cfg.date_selector else None
            tags = _extract_tags(item, cfg.tags_selector) if cfg.tags_selector else []

            rec_id = link or _content_id(cfg.name, title, text)

            records.append(
                PostRecord(
                    source=f"web_{cfg.name}",
                    source_type=cfg.source_type,
                    published_at=pub_date,
                    id_or_url=rec_id,
                    title_or_caption=title or None,
                    text=text or None,
                    language=None,
                    tags_raw=tags,
                    extra={"site": cfg.name, "page_url": page_url},
                )
            )
        return records


def _extract_text(parent: Any, selector: str) -> str:
    el = parent.select_one(selector)
    return el.get_text(strip=True) if el else ""


def _extract_href(parent: Any, selector: str, base_url: str) -> str:
    el = parent.select_one(selector)
    if el:
        href = el.get("href", "")
        return urljoin(base_url, href) if href else ""
    return ""


def _extract_date(parent: Any, selector: str, date_attr: str) -> datetime | None:
    el = parent.select_one(selector)
    if not el:
        return None
    raw = el.get(date_attr) if date_attr else el.get_text(strip=True)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw)).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _extract_tags(parent: Any, selector: str) -> list[str]:
    elements = parent.select(selector)
    return [el.get_text(strip=True) for el in elements if el.get_text(strip=True)]


def _page_id(site_name: str, url: str) -> str:
    return f"page_{site_name}_{hashlib.md5(url.encode()).hexdigest()[:12]}"


def _content_id(site_name: str, title: str, text: str) -> str:
    raw = f"{site_name}|{title}|{text[:200]}"
    return f"web_{site_name}_{hashlib.md5(raw.encode()).hexdigest()[:16]}"
