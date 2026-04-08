from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def _linear_regression_predict(train_y: list[float], predict_steps: int) -> list[float]:
    if not train_y:
        return [0.0] * predict_steps
    n = len(train_y)
    if n == 1:
        return [train_y[0]] * predict_steps
    x_vals = list(range(n))
    x_mean = sum(x_vals) / n
    y_mean = sum(train_y) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, train_y))
    den = sum((x - x_mean) ** 2 for x in x_vals)
    slope = num / den if den else 0.0
    intercept = y_mean - slope * x_mean
    return [max(0.0, intercept + slope * (n + i)) for i in range(predict_steps)]


def _rolling_mean_baseline(train_y: list[float], predict_steps: int, window: int = 4) -> list[float]:
    if not train_y:
        return [0.0] * predict_steps
    ref = train_y[-window:] if len(train_y) >= window else train_y
    value = sum(ref) / len(ref)
    return [value] * predict_steps


def _effective_holdout(n: int, requested: int) -> int | None:
    """Need at least 2 train points and 1 test week: n >= h + 2, h >= 1."""
    if n < 3:
        return None
    h = min(max(requested, 1), n - 2)
    return h


def train_and_predict(
    features_path: Path, predictions_path: Path, holdout_weeks: int = 6
) -> Path:
    """Train baseline + regression per trend and write holdout predictions."""
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with features_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            grouped[row["trend_tag"]].append(row)

    out_rows: list[dict[str, str | float]] = []
    for trend_tag, rows in grouped.items():
        rows.sort(key=lambda x: x["week_start"])
        y = [float(r["count"]) for r in rows]
        w = [r["week_start"] for r in rows]
        h = _effective_holdout(len(y), holdout_weeks)
        if h is None:
            continue
        split = len(y) - h
        y_train, y_test = y[:split], y[split:]
        test_weeks = w[split:]

        pred_baseline = _rolling_mean_baseline(y_train, len(y_test))
        pred_reg = _linear_regression_predict(y_train, len(y_test))

        for week_start, y_true, y_b, y_r in zip(test_weeks, y_test, pred_baseline, pred_reg):
            out_rows.append(
                {
                    "trend_tag": trend_tag,
                    "week_start": week_start,
                    "y_true": y_true,
                    "y_pred_baseline": round(y_b, 4),
                    "y_pred_regression": round(y_r, 4),
                }
            )

    with predictions_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "trend_tag",
                "week_start",
                "y_true",
                "y_pred_baseline",
                "y_pred_regression",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    return predictions_path
