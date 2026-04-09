
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

DatasetVariant = Literal["processed", "synthetic", "mock_legacy"]


@dataclass(frozen=True)
class DemoPaths:
    root: Path
    variant: DatasetVariant

    @property
    def features(self) -> Path:
        if self.variant == "processed":
            return (
                self.root
                / "data"
                / "features"
                / "trend_timeseries"
                / "features_from_processed.csv"
            )
        if self.variant == "synthetic":
            return self.root / "data" / "synthetic" / "history_features.csv"
        return self.root / "data" / "features" / "trend_timeseries" / "features.csv"

    @property
    def predictions(self) -> Path:
        if self.variant == "processed":
            return (
                self.root
                / "data"
                / "predictions"
                / "predictions_from_processed.csv"
            )
        if self.variant == "synthetic":
            return self.root / "data" / "synthetic" / "predictions_ets.csv"
        return self.root / "data" / "predictions" / "predictions.csv"

    @property
    def metrics(self) -> Path:
        if self.variant == "processed":
            return self.root / "data" / "predictions" / "metrics_from_processed.csv"
        if self.variant == "synthetic":
            return self.root / "data" / "synthetic" / "metrics_ets.csv"
        return self.root / "data" / "predictions" / "metrics.csv"

    @property
    def synthetic_future_truth(self) -> Path:
        return self.root / "data" / "synthetic" / "future_truth.csv"


def load_synthetic_combined_features(root: Path) -> pd.DataFrame | None:
    hist_p = root / "data" / "synthetic" / "history_features.csv"
    fut_p = root / "data" / "synthetic" / "future_truth.csv"
    if not hist_p.is_file():
        return None
    hist = pd.read_csv(hist_p)
    if fut_p.is_file():
        fut = pd.read_csv(fut_p)
        df = pd.concat([hist, fut], ignore_index=True)
    else:
        df = hist
    if df.empty:
        return df
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df.sort_values(["trend_tag", "week_start"]).reset_index(drop=True)


def load_features(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def load_features_for_paths(paths: DemoPaths) -> pd.DataFrame | None:
    if paths.variant == "synthetic":
        return load_synthetic_combined_features(paths.root)
    return load_features(paths.features)


def load_predictions(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def load_metrics(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def is_ets_predictions(paths: DemoPaths) -> bool:
    return paths.variant == "synthetic"


def trend_ranking(features: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    g = (
        features.groupby("trend_tag", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
    )
    return g.head(top_n)


def history_for_trend(features: pd.DataFrame, trend_tag: str) -> pd.DataFrame:
    h = features.loc[features["trend_tag"] == trend_tag].sort_values("week_start")
    return h[["week_start", "count", "avg_engagement"]].copy()


def predictions_for_trend(predictions: pd.DataFrame, trend_tag: str) -> pd.DataFrame:
    p = predictions.loc[predictions["trend_tag"] == trend_tag].sort_values("week_start")
    return p


def metrics_row(metrics: pd.DataFrame, trend_tag: str) -> pd.Series | None:
    row = metrics.loc[metrics["trend_tag"] == trend_tag]
    if row.empty:
        return None
    return row.iloc[0]
