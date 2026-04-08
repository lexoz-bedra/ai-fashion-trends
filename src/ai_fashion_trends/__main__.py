from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="ai-fashion-trends")
    sub = parser.add_subparsers(dest="command")

    # --- ingest ---
    ingest_p = sub.add_parser("ingest", help="Запуск ingestion pipeline")
    ingest_p.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config.yaml"),
        help="Путь к YAML-конфигу (по умолчанию: config.yaml)",
    )
    ingest_p.add_argument(
        "--source",
        "-s",
        type=str,
        default=None,
        help="Запустить только указанный источник: google_trends, rss, web_scraper",
    )
    ingest_p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Подробный вывод (DEBUG)",
    )

    # --- process ---
    process_p = sub.add_parser("process", help="AI-обработка: извлечение трендов через LLM")
    process_p.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config.yaml"),
        help="Путь к YAML-конфигу",
    )
    process_p.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Имя модели из конфига (по умолчанию: gemma)",
    )
    process_p.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="Размер батча (по умолчанию: 10)",
    )
    process_p.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Макс. количество постов для обработки (для теста)",
    )
    process_p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Подробный вывод (DEBUG)",
    )

    # --- mock-pipeline: синтетика без ingest/process ---
    mock_p = sub.add_parser(
        "mock-pipeline",
        help="Полный ML-пайплайн на синтетических CSV (без сбора данных)",
    )

    # --- forecast: фичи + обучение из trends.jsonl после process ---
    forecast_p = sub.add_parser(
        "forecast",
        help="Недельные ряды и прогноз из data/processed/trends.jsonl",
    )
    forecast_p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/processed/trends.jsonl"),
        help="Путь к JSONL с извлечёнными трендами",
    )
    forecast_p.add_argument(
        "--holdout-weeks",
        type=int,
        default=6,
        help="Недель в holdout для оценки (для коротких рядов уменьшается автоматически)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "ingest":
        _run_ingest(args)
    elif args.command == "process":
        _run_process(args)
    elif args.command == "mock-pipeline":
        _run_mock_pipeline()
    elif args.command == "forecast":
        _run_forecast(args)


def _run_mock_pipeline() -> None:
    from ai_fashion_trends.pipeline import run_mock_pipeline

    artifacts = run_mock_pipeline(Path.cwd())
    print("Пайплайн (мок) завершён. Артефакты:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


def _run_forecast(args: argparse.Namespace) -> None:
    from ai_fashion_trends.pipeline import run_forecast_from_trends_jsonl

    artifacts = run_forecast_from_trends_jsonl(
        Path.cwd(),
        trends_jsonl=args.input,
        holdout_weeks=args.holdout_weeks,
    )
    print("Forecast завершён. Артефакты:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


def _run_ingest(args: argparse.Namespace) -> None:
    import yaml

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("ai_fashion_trends")

    config_path: Path = args.config
    if not config_path.exists():
        logger.error("Конфиг не найден: %s", config_path)
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_root = Path.cwd()
    data_dir = project_root / cfg.get("data_dir", "data/raw")
    checkpoint_dir = project_root / cfg.get("checkpoint_dir", "data/checkpoints")

    from .ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(data_dir=data_dir, checkpoint_dir=checkpoint_dir)
    only = args.source

    # Google Trends
    gt_cfg = cfg.get("google_trends", {})
    if gt_cfg.get("enabled") and only in (None, "google_trends"):
        from .ingestion.sources.google_trends import GoogleTrendsSource

        source = GoogleTrendsSource(
            storage=pipeline.storage,
            checkpoint=pipeline.make_checkpoint("google_trends"),
            keywords=gt_cfg["keywords"],
            geo=gt_cfg.get("geo", ""),
            timeframe=gt_cfg.get("timeframe", "today 3-m"),
            batch_size=gt_cfg.get("batch_size", 5),
        )
        pipeline.add_source(source)

    # RSS
    rss_cfg = cfg.get("rss", {})
    if rss_cfg.get("enabled") and only in (None, "rss"):
        from .ingestion.sources.rss import RssSource

        source = RssSource(
            storage=pipeline.storage,
            checkpoint=pipeline.make_checkpoint("rss"),
            feed_urls=rss_cfg["feeds"],
            batch_size=rss_cfg.get("batch_size", 50),
        )
        pipeline.add_source(source)

    # Web scraper
    ws_cfg = cfg.get("web_scraper", {})
    if ws_cfg.get("enabled") and only in (None, "web_scraper"):
        from .ingestion.sources.web_scraper import SiteScrapeConfig, WebScraperSource

        configs = []
        for site in ws_cfg.get("sites", []):
            configs.append(
                SiteScrapeConfig(
                    name=site["name"],
                    start_urls=site["start_urls"],
                    item_selector=site["item_selector"],
                    title_selector=site.get("title_selector", ""),
                    text_selector=site.get("text_selector", ""),
                    date_selector=site.get("date_selector", ""),
                    date_attr=site.get("date_attr", ""),
                    link_selector=site.get("link_selector", ""),
                    tags_selector=site.get("tags_selector", ""),
                    next_page_selector=site.get("next_page_selector", ""),
                    max_pages=site.get("max_pages", 10),
                    request_delay=site.get("request_delay", 1.0),
                    source_type=site.get("source_type", "forum"),
                )
            )
        source = WebScraperSource(
            storage=pipeline.storage,
            checkpoint=pipeline.make_checkpoint("web_scraper"),
            configs=configs,
            batch_size=ws_cfg.get("batch_size", 20),
        )
        pipeline.add_source(source)

    llm_cfg = cfg.get("llm", {})
    if llm_cfg.get("models"):
        from .ingestion.llm import register_model

        for name, model_cfg in llm_cfg["models"].items():
            register_model(name, model_cfg)

    if not pipeline._sources:
        logger.warning("Нет активных источников. Проверьте конфиг и флаг --source.")
        sys.exit(0)

    logger.info("Запуск ingestion pipeline (%d источников)", len(pipeline._sources))
    results = pipeline.run()

    logger.info("=" * 50)
    logger.info("РЕЗУЛЬТАТЫ:")
    for name, count in results.items():
        status = f"{count} новых записей" if count >= 0 else "ОШИБКА"
        logger.info("  %s: %s", name, status)
    logger.info("=" * 50)


def _run_process(args: argparse.Namespace) -> None:
    import yaml

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("ai_fashion_trends")

    config_path: Path = args.config
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    llm_cfg = cfg.get("llm", {})
    if llm_cfg.get("models"):
        from .ingestion.llm import register_model

        for name, model_cfg in llm_cfg["models"].items():
            register_model(name, model_cfg)

    project_root = Path.cwd()
    raw_dir = project_root / cfg.get("data_dir", "data/raw")
    checkpoint_dir = project_root / cfg.get("checkpoint_dir", "data/checkpoints")
    output_dir = project_root / "data" / "processed"
    model_name = args.model or llm_cfg.get("default_model", "gemma")

    from .processing.processor import TrendProcessor

    processor = TrendProcessor(
        raw_dir=raw_dir,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        batch_size=args.batch_size,
    )

    if args.limit:
        original_load = processor._load_unprocessed_posts

        def limited_load():
            posts = original_load()
            return posts[: args.limit]

        processor._load_unprocessed_posts = limited_load

    logger.info("Запуск AI-обработки (модель: %s, batch_size: %d)", model_name, args.batch_size)
    total = processor.run()

    logger.info("=" * 50)
    logger.info("РЕЗУЛЬТАТ: извлечено %d трендов", total)
    logger.info("Данные сохранены в: %s", output_dir / "trends.jsonl")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
