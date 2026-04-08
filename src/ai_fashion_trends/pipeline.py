from __future__ import annotations

from pathlib import Path

from ai_fashion_trends.data_ingestion import generate_mock_raw_dataset
from ai_fashion_trends.evaluation import evaluate_predictions
from ai_fashion_trends.features import (
    build_weekly_features,
    build_weekly_features_from_trends_jsonl,
)
from ai_fashion_trends.forecasting import train_and_predict
from ai_fashion_trends.text_mining import clean_and_extract_tags


def run_mock_pipeline(base_dir: Path) -> dict[str, Path]:
    """Run full synthetic data pipeline and return produced artifact paths."""
    raw_path = base_dir / "data" / "raw" / "news" / "mock_raw.csv"
    cleaned_path = base_dir / "data" / "processed" / "news_cleaned" / "cleaned.csv"
    features_path = base_dir / "data" / "features" / "trend_timeseries" / "features.csv"
    predictions_path = base_dir / "data" / "predictions" / "predictions.csv"
    metrics_path = base_dir / "data" / "predictions" / "metrics.csv"

    generate_mock_raw_dataset(raw_path)
    clean_and_extract_tags(raw_path, cleaned_path)
    build_weekly_features(cleaned_path, features_path)
    train_and_predict(features_path, predictions_path)
    evaluate_predictions(predictions_path, metrics_path)

    return {
        "raw": raw_path,
        "cleaned": cleaned_path,
        "features": features_path,
        "predictions": predictions_path,
        "metrics": metrics_path,
    }


def run_forecast_from_trends_jsonl(
    base_dir: Path,
    trends_jsonl: Path | None = None,
    holdout_weeks: int = 6,
) -> dict[str, Path]:
    """
    Цепочка после ingest + process: недельные ряды из trends.jsonl → прогноз → метрики.
    """
    trends_path = trends_jsonl or (base_dir / "data" / "processed" / "trends.jsonl")
    if not trends_path.exists():
        raise FileNotFoundError(
            f"Нет файла {trends_path}. Сначала: ingest → process, либо укажи --input."
        )

    features_path = (
        base_dir / "data" / "features" / "trend_timeseries" / "features_from_processed.csv"
    )
    predictions_path = base_dir / "data" / "predictions" / "predictions_from_processed.csv"
    metrics_path = base_dir / "data" / "predictions" / "metrics_from_processed.csv"

    build_weekly_features_from_trends_jsonl(trends_path, features_path)
    train_and_predict(features_path, predictions_path, holdout_weeks=holdout_weeks)
    evaluate_predictions(predictions_path, metrics_path)

    return {
        "trends_input": trends_path,
        "features": features_path,
        "predictions": predictions_path,
        "metrics": metrics_path,
    }
