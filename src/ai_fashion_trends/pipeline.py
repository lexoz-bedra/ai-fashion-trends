from __future__ import annotations

from pathlib import Path

from ai_fashion_trends.data_ingestion import generate_mock_raw_dataset
from ai_fashion_trends.evaluation import evaluate_predictions
from ai_fashion_trends.features import build_weekly_features
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
