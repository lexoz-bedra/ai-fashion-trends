from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


def _mae(y_true: list[float], y_pred: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)


def _rmse(y_true: list[float], y_pred: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / max(len(y_true), 1))


def _mape(y_true: list[float], y_pred: list[float]) -> float:
    vals = [abs((a - b) / a) for a, b in zip(y_true, y_pred) if a != 0]
    if not vals:
        return 0.0
    return sum(vals) / len(vals) * 100.0


def evaluate_predictions(predictions_path: Path, metrics_path: Path) -> Path:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_true: dict[str, list[float]] = defaultdict(list)
    grouped_base: dict[str, list[float]] = defaultdict(list)
    grouped_reg: dict[str, list[float]] = defaultdict(list)

    with predictions_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            trend = row["trend_tag"]
            grouped_true[trend].append(float(row["y_true"]))
            grouped_base[trend].append(float(row["y_pred_baseline"]))
            grouped_reg[trend].append(float(row["y_pred_regression"]))

    out_rows: list[dict[str, str | float]] = []
    for trend in sorted(grouped_true):
        y = grouped_true[trend]
        b = grouped_base[trend]
        r = grouped_reg[trend]
        out_rows.append(
            {
                "trend_tag": trend,
                "mae_baseline": round(_mae(y, b), 4),
                "rmse_baseline": round(_rmse(y, b), 4),
                "mape_baseline": round(_mape(y, b), 4),
                "mae_regression": round(_mae(y, r), 4),
                "rmse_regression": round(_rmse(y, r), 4),
                "mape_regression": round(_mape(y, r), 4),
            }
        )

    with metrics_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "trend_tag",
                "mae_baseline",
                "rmse_baseline",
                "mape_baseline",
                "mae_regression",
                "rmse_regression",
                "mape_regression",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    return metrics_path
