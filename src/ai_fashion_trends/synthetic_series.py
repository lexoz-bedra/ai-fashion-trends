from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrendProfile:
    trend_tag: str
    level: float
    slope_per_week: float
    annual_amp: float
    quarterly_amp: float
    noise_sigma: float
    spike_prob: float
    spike_mult: float
    phase_y: float
    phase_q: float


DEFAULT_PROFILES: list[TrendProfile] = [

    TrendProfile(
        "glass_skin",
        16.0,
        0.03,
        5.5,
        2.0,
        1.2,
        0.02,
        1.7,
        0.2,
        1.0,
    ),

    TrendProfile(
        "clean_girl_makeup",
        24.0,
        -0.11,
        5.0,
        2.2,
        1.8,
        0.03,
        2.0,
        2.4,
        0.6,
    ),

    TrendProfile(
        "euphoria_glitter_liner",
        20.0,
        -0.16,
        6.0,
        3.5,
        2.4,
        0.04,
        2.2,
        3.9,
        1.2,
    ),

    TrendProfile(
        "fox_eye_siren_eye",
        15.0,
        -0.09,
        4.5,
        2.8,
        1.9,
        0.03,
        2.1,
        5.1,
        1.4,
    ),

    TrendProfile(
        "cherry_cola_lips",
        11.0,
        0.13,
        4.0,
        2.0,
        1.6,
        0.035,
        2.3,
        4.5,
        0.8,
    ),

    TrendProfile(
        "latte_makeup",
        14.0,
        0.05,
        3.5,
        1.8,
        1.3,
        0.02,
        1.8,
        1.0,
        2.2,
    ),

    TrendProfile(
        "strawberry_girl_blush",
        19.0,
        -0.06,
        6.5,
        3.0,
        2.1,
        0.04,
        2.4,
        0.8,
        2.6,
    ),

    TrendProfile(
        "graphic_editorial_liner",
        17.0,
        0.01,
        4.0,
        5.0,
        2.2,
        0.025,
        1.9,
        2.0,
        0.3,
    ),
]


def _series_for_profile_daily(
    profile: TrendProfile,
    n_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.arange(n_days, dtype=float)
    base = max(0.8, profile.level / 5.5)
    drift = (profile.slope_per_week / 7.0) * t
    annual = (profile.annual_amp / 2.75) * np.sin(2 * np.pi * t / 365.25 + profile.phase_y)
    quarterly = (profile.quarterly_amp / 3.1) * np.sin(2 * np.pi * t / 91.0 + profile.phase_q)
    weekly_pattern = 0.32 * np.sin(2 * np.pi * t / 7.0 + profile.phase_y * 0.65)
    noise = rng.normal(0.0, profile.noise_sigma * 0.42, size=n_days)
    y = base + drift + annual + quarterly + weekly_pattern + noise
    spikes = rng.random(n_days) < (profile.spike_prob * 0.4)
    mult = 1.15 + (profile.spike_mult - 1.0) * 0.45
    y = np.where(spikes, y * mult, y)
    y = np.clip(np.round(y), 0.0, None)
    return y.astype(int)


def generate_daily_stats_two_years(
    *,
    days: int = 730,
    start_date: date | None = None,
    profiles: list[TrendProfile] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    if days < 14:
        raise ValueError("days слишком мало для осмысленной сезонности")
    profiles = profiles or DEFAULT_PROFILES
    rng = np.random.default_rng(seed + 17)
    if start_date is None:
        start_date = date.today() - timedelta(days=days)

    rows: list[dict] = []
    for p in profiles:
        mentions = _series_for_profile_daily(p, days, rng)
        for i in range(days):
            d = start_date + timedelta(days=i)
            m = int(mentions[i])
            if m <= 0:
                posts = 0
            else:
                posts = int(rng.binomial(m, rng.uniform(0.15, 0.48)))
            eng = float(
                np.clip(
                    42.0
                    + 0.9 * m
                    + rng.normal(0, 6.5)
                    + 5.0 * np.sin(2 * np.pi * i / 7.0),
                    5.0,
                    100.0,
                )
            )
            rows.append(
                {
                    "date": d.isoformat(),
                    "trend_tag": p.trend_tag,
                    "mention_count": m,
                    "post_count": posts,
                    "avg_engagement_proxy": round(eng, 2),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["trend_tag", "date"]).reset_index(drop=True)
    df["rolling_7d_mention_sum"] = df.groupby("trend_tag", sort=False)["mention_count"].transform(
        lambda s: s.rolling(7, min_periods=1).sum().astype(int)
    )
    return df


def write_daily_two_year_csv(
    base_dir: Path,
    *,
    days: int = 730,
    seed: int = 42,
    filename: str = "daily_trends_2y.csv",
) -> tuple[Path, int]:
    out_dir = base_dir / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df = generate_daily_stats_two_years(days=days, seed=seed)
    df.to_csv(path, index=False)
    return path, len(df)


def aggregate_daily_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["dt"] = pd.to_datetime(d["date"])
    d["week_start"] = d["dt"] - pd.to_timedelta(d["dt"].dt.weekday, unit="d")
    g = d.groupby(["trend_tag", "week_start"], as_index=False).agg(
        count=("mention_count", "sum"),
        avg_engagement=("avg_engagement_proxy", "mean"),
    )
    g["week_start"] = g["week_start"].dt.strftime("%Y-%m-%d")
    g["avg_engagement"] = g["avg_engagement"].round(2)
    return g


def weekly_history_and_future_from_daily(
    *,
    history_weeks: int,
    future_weeks: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if history_weeks < 24:
        raise ValueError("history_weeks должно быть достаточно большим для сезонности")
    total_weeks = history_weeks + future_weeks
    total_days = total_weeks * 7
    today = date.today()
    this_monday = today - timedelta(days=today.weekday())
    start_monday = this_monday - timedelta(weeks=total_weeks)

    daily = generate_daily_stats_two_years(
        days=total_days,
        start_date=start_monday,
        seed=seed,
    )
    weekly = aggregate_daily_to_weekly(daily)

    hist_parts: list[pd.DataFrame] = []
    fut_parts: list[pd.DataFrame] = []
    for tag in weekly["trend_tag"].unique():
        grp = weekly.loc[weekly["trend_tag"] == tag].sort_values("week_start")
        if len(grp) < total_weeks:
            raise RuntimeError(
                f"Тренд {tag}: ожидалось {total_weeks} недель, получилось {len(grp)}"
            )
        hist_parts.append(grp.iloc[:history_weeks])
        fut_parts.append(grp.iloc[history_weeks : history_weeks + future_weeks])

    return pd.concat(hist_parts, ignore_index=True), pd.concat(fut_parts, ignore_index=True)


def write_weekly_eval_bundle_from_daily(
    base_dir: Path,
    *,
    history_weeks: int = 104,
    future_weeks: int = 26,
    seed: int = 42,
) -> dict[str, Path]:
    out_dir = base_dir / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    hist, fut = weekly_history_and_future_from_daily(
        history_weeks=history_weeks,
        future_weeks=future_weeks,
        seed=seed,
    )
    hp = out_dir / "history_features.csv"
    fp = out_dir / "future_truth.csv"
    hist.to_csv(hp, index=False)
    fut.to_csv(fp, index=False)
    return {"history_features": hp, "future_truth": fp, "dir": out_dir}
