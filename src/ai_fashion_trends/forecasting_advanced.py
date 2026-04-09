
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _seasonal_period(n: int) -> int:
    if n >= 96:
        return 52
    if n >= 40:
        return 26
    if n >= 16:
        return 13
    return max(2, n // 4)


def forecast_multistep(
    y: np.ndarray,
    horizon: int,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    horizon = int(horizon)
    if horizon <= 0:
        return np.array([])
    if n == 0:
        return np.zeros(horizon)
    if n == 1:
        return np.full(horizon, y[0])

    sp = _seasonal_period(n)

    if n >= 2 * sp:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            model = ExponentialSmoothing(
                y,
                seasonal_periods=sp,
                trend="add",
                seasonal="add",
                initialization_method="estimated",
            ).fit(optimized=True)
            pred = model.forecast(horizon)
            return np.maximum(0.0, np.asarray(pred, dtype=float))
        except Exception as e:
            logger.debug("Holt-Winters не подошёл (n=%d, sp=%d): %s", n, sp, e)

    try:
        from statsmodels.tsa.forecasting.theta import ThetaModel

        period = min(sp, max(2, n // 3))
        tm = ThetaModel(y, period=period).fit()
        pred = tm.forecast(horizon)
        return np.maximum(0.0, np.asarray(pred, dtype=float))
    except Exception as e:
        logger.debug("Theta не подошёл: %s", e)

    tail = min(12, n)
    if tail >= 2:
        drift = (y[-1] - y[-tail]) / (tail - 1)
        steps = np.arange(1, horizon + 1, dtype=float)
        return np.maximum(0.0, y[-1] + drift * steps)
    return np.full(horizon, y[-1])


def run_ets_evaluation(
    history_path: Path,
    future_truth_path: Path,
    predictions_path: Path,
    metrics_path: Path,
) -> pd.DataFrame:
    hist = pd.read_csv(history_path)
    truth = pd.read_csv(future_truth_path)
    hist["week_start"] = pd.to_datetime(hist["week_start"])
    truth["week_start"] = pd.to_datetime(truth["week_start"])

    horizon = truth.groupby("trend_tag").size().iloc[0]
    if not truth.groupby("trend_tag").size().eq(horizon).all():
        raise ValueError("В future_truth разное число недель по трендам")

    pred_rows: list[dict] = []
    metric_rows: list[dict] = []

    for tag in sorted(hist["trend_tag"].unique()):
        h = hist.loc[hist["trend_tag"] == tag].sort_values("week_start")
        t = truth.loc[truth["trend_tag"] == tag].sort_values("week_start")
        if h.empty or t.empty:
            continue
        y = h["count"].astype(float).values
        y_true = t["count"].astype(float).values
        h_steps = len(y_true)
        y_pred = forecast_multistep(y, h_steps)
        if len(y_pred) != h_steps:
            y_pred = y_pred[:h_steps]

        for i, row in t.reset_index(drop=True).iterrows():
            pred_rows.append(
                {
                    "trend_tag": tag,
                    "week_start": row["week_start"].date().isoformat(),
                    "y_true": y_true[i],
                    "y_pred_ets": round(float(y_pred[i]), 4),
                }
            )

        err = y_true - y_pred
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        mape = float(np.mean(np.abs(err / np.maximum(y_true, 1.0))) * 100.0)
        metric_rows.append(
            {
                "trend_tag": tag,
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape_pct": round(mape, 4),
                "horizon_weeks": h_steps,
            }
        )

    pred_df = pd.DataFrame(pred_rows)
    met_df = pd.DataFrame(metric_rows)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(predictions_path, index=False)
    met_df.to_csv(metrics_path, index=False)
    return met_df
