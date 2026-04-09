from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="ai-fashion-trends")
    sub = parser.add_subparsers(dest="command")


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


    mock_p = sub.add_parser(
        "mock-pipeline",
        help="Полный ML-пайплайн на синтетических CSV (без сбора данных)",
    )


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


    syn_p = sub.add_parser(
        "synthetic-forecast",
        help="Синтетические ~2y недельные ряды и прогноз ETS/Theta на 6–12 мес",
    )
    syn_p.add_argument(
        "--history-weeks",
        type=int,
        default=104,
        help="Недель истории (~2 года = 104)",
    )
    syn_p.add_argument(
        "--future-weeks",
        type=int,
        default=26,
        help="Горизонт прогноза в неделях (26 ≈ 6 мес, 52 ≈ 12 мес)",
    )
    syn_p.add_argument("--seed", type=int, default=42, help="Seed генератора")

    daily_p = sub.add_parser(
        "synthetic-daily",
        help="Дневная статистика по 8 трендам макияжа (тяжёлый CSV, по умолчанию 2 года)",
    )
    daily_p.add_argument(
        "--days",
        type=int,
        default=730,
        help="Число дней на тренд (730 ≈ 2×365)",
    )
    daily_p.add_argument("--seed", type=int, default=42, help="Seed генератора")
    daily_p.add_argument(
        "--output-name",
        type=str,
        default="daily_trends_2y.csv",
        help="Имя файла в data/synthetic/",
    )

    vision_p = sub.add_parser(
        "analyze-makeup",
        help="Лицо на фото + мок-классификатор стиля макияжа (без связи с forecast)",
    )
    v_src = vision_p.add_mutually_exclusive_group(required=True)
    v_src.add_argument(
        "--image",
        "-i",
        type=Path,
        help="Одно изображение (jpg/png/webp)",
    )
    v_src.add_argument(
        "--dataset-dir",
        type=Path,
        help="Папка с подпапками по стилю (напр. smoky_eyes, blue_eyeshadow, unknown)",
    )
    vision_p.add_argument(
        "--no-mediapipe",
        action="store_true",
        help="Не использовать MediaPipe, только центральный crop",
    )
    vision_p.add_argument(
        "--json",
        action="store_true",
        help="JSON: одно фото — объект; датасет — JSONL по строке на файл",
    )
    vision_p.add_argument(
        "--metrics",
        action="store_true",
        help="С --dataset-dir: top-1 совпадение с именем подпапки (для мока ожидайте низкий процент)",
    )

    train_m = sub.add_parser(
        "train-makeup-classifier",
        help="Обучить классификатор smoky / blue / other (папки с фото)",
    )
    train_m.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/makeup_dataset"),
        help="Каталог: smoky_eyes, blue-eyeshadow (или blue_eyeshadow), other",
    )
    train_m.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/makeup_smoky_blue.joblib"),
        help="Куда сохранить модель (.joblib)",
    )
    train_m.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Доля hold-out для отчёта accuracy",
    )
    train_m.add_argument("--seed", type=int, default=42)
    train_m.add_argument(
        "--no-mediapipe",
        action="store_true",
        help="Только центральный crop при обучении",
    )
    train_m.add_argument(
        "--prob-threshold",
        type=float,
        default=0.45,
        help="Порог max(P) по классам; ниже — в демо «нет лица на фото»",
    )
    train_m.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Регуляризация LogisticRegression",
    )

    train_cnn = sub.add_parser(
        "train-makeup-cnn",
        help="Дообучить ResNet18 (ImageNet) под smoky/blue/other — нужен extra torch",
    )
    train_cnn.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/makeup_dataset"),
        help="Каталог: smoky_eyes, blue-eyeshadow, other",
    )
    train_cnn.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/makeup_resnet18.pt"),
        help="Файл весов .pt (рядом запишется .meta.json)",
    )
    train_cnn.add_argument("--epochs", type=int, default=30)
    train_cnn.add_argument("--batch-size", type=int, default=8)
    train_cnn.add_argument("--lr", type=float, default=3e-4)
    train_cnn.add_argument("--test-size", type=float, default=0.2)
    train_cnn.add_argument("--seed", type=int, default=42)
    train_cnn.add_argument(
        "--no-mediapipe",
        action="store_true",
        help="Центральный crop при обучении",
    )
    train_cnn.add_argument(
        "--prob-threshold",
        type=float,
        default=0.45,
        help="Порог max(P) для ответа «не лицо» в демо",
    )

    tagdict_p = sub.add_parser(
        "build-tag-dictionary",
        help="Собрать отдельный словарь тегов из data/processed/trends.jsonl",
    )
    tagdict_p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/processed/trends.jsonl"),
        help="Путь к trends.jsonl",
    )
    tagdict_p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/processed/tag_dictionary.jsonl"),
        help="Куда записать словарь тегов (JSONL)",
    )
    tagdict_p.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Минимум упоминаний для попадания в словарь",
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
    elif args.command == "synthetic-forecast":
        _run_synthetic_forecast(args)
    elif args.command == "synthetic-daily":
        _run_synthetic_daily(args)
    elif args.command == "analyze-makeup":
        _run_analyze_makeup(args)
    elif args.command == "train-makeup-classifier":
        _run_train_makeup_classifier(args)
    elif args.command == "train-makeup-cnn":
        _run_train_makeup_cnn(args)
    elif args.command == "build-tag-dictionary":
        _run_build_tag_dictionary(args)


def _run_build_tag_dictionary(args: argparse.Namespace) -> None:
    from ai_fashion_trends.tag_dictionary import build_tag_dictionary

    stats = build_tag_dictionary(
        Path(args.input),
        Path(args.output),
        min_count=args.min_count,
    )
    print("Словарь тегов записан:", stats.output_path)
    print("Прочитано трендов:", stats.trends_read)
    print("Записано записей:", stats.entries)


def _run_train_makeup_cnn(args: argparse.Namespace) -> None:
    try:
        from ai_fashion_trends.vision.torch_makeup import train_makeup_resnet
    except ImportError:
        print(
            "Нужны PyTorch и torchvision: uv sync --extra torch",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    r = train_makeup_resnet(
        Path(args.data_dir),
        Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        test_size=args.test_size,
        seed=args.seed,
        use_mediapipe=not args.no_mediapipe,
        prob_threshold=args.prob_threshold,
    )
    print("Веса:", r["output_pt"])
    print("Мета:", r["meta_path"])
    print("Лучшая val accuracy:", f"{r['best_val_accuracy']:.3f}")
    print("train / val:", r["n_train"], "/", r["n_val"])
    print(r["classification_report"])


def _run_train_makeup_classifier(args: argparse.Namespace) -> None:
    from ai_fashion_trends.vision.train_makeup_pair import train_makeup_pair_classifier

    out = Path(args.output)
    report = train_makeup_pair_classifier(
        Path(args.data_dir),
        out,
        test_size=args.test_size,
        seed=args.seed,
        use_mediapipe=not args.no_mediapipe,
        prob_threshold=args.prob_threshold,
        C=args.C,
    )
    print("Модель сохранена:", report["output_path"])
    print("Классы в модели:", report.get("classes"))
    print("Образцов:", report["n_samples"], "| train:", report["n_train"], "| test:", report["n_test"])
    print("Accuracy (hold-out):", f"{report['accuracy_holdout']:.3f}")
    print(report["classification_report"])


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


def _run_synthetic_forecast(args: argparse.Namespace) -> None:
    from ai_fashion_trends.pipeline import run_synthetic_long_forecast_eval

    artifacts = run_synthetic_long_forecast_eval(
        Path.cwd(),
        history_weeks=args.history_weeks,
        future_weeks=args.future_weeks,
        seed=args.seed,
    )
    print("Synthetic forecast (ETS/Theta) завершён. Артефакты:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")
    met = artifacts["metrics_ets"]
    import pandas as pd

    df = pd.read_csv(met)
    print("\nСводка метрик (среднее по трендам):")
    print(f"  MAE:  {df['mae'].mean():.4f}")
    print(f"  RMSE: {df['rmse'].mean():.4f}")
    print(f"  MAPE: {df['mape_pct'].mean():.2f}%")


def _run_synthetic_daily(args: argparse.Namespace) -> None:
    from ai_fashion_trends.synthetic_series import write_daily_two_year_csv

    path, n = write_daily_two_year_csv(
        Path.cwd(),
        days=args.days,
        seed=args.seed,
        filename=args.output_name,
    )
    print(f"Дневной CSV записан: {path}")
    print(f"Строк: {n} (ожидаемо 8 × {args.days} = {8 * args.days})")


def _run_analyze_makeup(args: argparse.Namespace) -> None:
    if args.dataset_dir is not None:
        _run_analyze_makeup_dataset(args)
        return

    from ai_fashion_trends.vision import FaceMakeupPipeline, describe_style

    assert args.image is not None
    pipe = FaceMakeupPipeline(use_mediapipe=not args.no_mediapipe)
    result = pipe.analyze(args.image)
    if args.json:
        print(result.model_dump_json(indent=2))
        return
    print("Изображение:", result.image_path)
    print("Лицо детектировано:", result.face_detected)
    if result.face_bbox_norm:
        print("BBox (норм.):", tuple(round(x, 4) for x in result.face_bbox_norm))
    print("Классификатор:", result.classifier)
    print()
    print("Топ стиль:", result.top_style.value)
    print("Описание:", describe_style(result.top_style))
    print("Уверенность:", result.confidence)
    print()
    print("Распределение (мок):")
    for k, v in sorted(result.style_scores.items(), key=lambda x: -x[1])[:5]:
        print(f"  {k}: {v}")
    print()
    print(result.notes)


def _run_analyze_makeup_dataset(args: argparse.Namespace) -> None:
    import json

    from ai_fashion_trends.vision import FaceMakeupPipeline
    from ai_fashion_trends.vision.schema import MakeupStyle

    root: Path = args.dataset_dir
    if not root.is_dir():
        print(f"Не папка: {root}", file=sys.stderr)
        sys.exit(1)

    pipe = FaceMakeupPipeline(use_mediapipe=not args.no_mediapipe)
    valid_labels = {s.value for s in MakeupStyle}
    exts = {".jpg", ".jpeg", ".png", ".webp"}

    rows: list[dict] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        expected = sub.name if sub.name in valid_labels else None
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() not in exts:
                continue
            try:
                result = pipe.analyze(f)
            except OSError as e:
                rows.append(
                    {
                        "path": str(f),
                        "error": str(e),
                        "expected_label": expected,
                    }
                )
                continue
            top = result.top_style.value
            rows.append(
                {
                    "path": str(f),
                    "expected_label": expected,
                    "top_style": top,
                    "confidence": result.confidence,
                    "face_detected": result.face_detected,
                    "classifier": result.classifier,
                    "match": expected is not None and top == expected,
                }
            )

    labeled = [r for r in rows if r.get("expected_label") is not None and "error" not in r]
    if args.metrics and labeled:
        ok = sum(1 for r in labeled if r.get("match"))
        print(
            f"Top-1 по имени папки: {ok}/{len(labeled)} "
            f"({100.0 * ok / len(labeled):.1f}%) — мок не обучен на разметке, "
            f"цифра ориентир для будущей модели.\n"
        )

    if args.json:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
        return

    for r in rows:
        if "error" in r:
            print(f"ERR {r['path']}: {r['error']}")
            continue
        exp = r.get("expected_label") or "—"
        m = "✓" if r.get("match") else " "
        print(f"{m} [{exp}] -> {r['top_style']} ({r['confidence']:.3f}) {r['path']}")

    n_err = sum(1 for r in rows if "error" in r)
    print(f"\nВсего: {len(rows)} файлов, ошибок чтения: {n_err}.")
    if any("error" not in r for r in rows):
        print(
            "Детальный JSON по одному файлу: "
            "python -m ai_fashion_trends analyze-makeup -i <путь> --json"
        )


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
