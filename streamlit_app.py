
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ai_fashion_trends import __version__
from ai_fashion_trends.streamlit_data import (
    DatasetVariant,
    DemoPaths,
    history_for_trend,
    is_ets_predictions,
    load_features_for_paths,
    load_metrics,
    load_predictions,
    metrics_row,
    predictions_for_trend,
    trend_ranking,
)

st.set_page_config(page_title="AI Fashion Trends", layout="wide")

st.title("AI Fashion Trends")
st.caption(f"Демо · пакет v{__version__}")
st.caption(
    "Страница **Makeup classifier** — в боковом меню Streamlit (multipage `pages/`)."
)

with st.sidebar:
    st.header("Данные")
    root = Path(st.text_input("Корень проекта", value=str(Path.cwd()))).expanduser()
    variant: DatasetVariant = st.radio(
        "Источник",
        options=["synthetic", "processed", "mock_legacy"],
        format_func=lambda x: {
            "synthetic": "Мок: ETS (synthetic-forecast, daily→weekly)",
            "processed": "После ingest + process + forecast",
            "mock_legacy": "Мок: baseline + регрессия (mock-pipeline)",
        }[x],
        horizontal=False,
    )
    paths = DemoPaths(root=root, variant=variant)

    st.subheader("Файлы")
    st.code(str(paths.features), language="text")
    if variant == "synthetic":
        st.caption("Также: `future_truth`, `predictions_ets`, `metrics_ets` в `data/synthetic/`")
    st.caption("При отсутствии файлов выполни команды из подсказки ниже.")

features = load_features_for_paths(paths)
predictions = load_predictions(paths.predictions)
metrics = load_metrics(paths.metrics)

if features is None or features.empty:
    hints = {
        "synthetic": "`python -m ai_fashion_trends synthetic-forecast`",
        "processed": "`ingest` → `process` → `python -m ai_fashion_trends forecast`",
        "mock_legacy": "`python -m ai_fashion_trends mock-pipeline`",
    }
    st.warning(
        "Нет файла признаков или он пустой. Сгенерируй артефакты:\n\n"
        f"- **{variant}:** {hints[variant]}"
    )
    st.stop()

ranking = trend_ranking(features, top_n=300)
options = ranking["trend_tag"].tolist()

col_main, col_meta = st.columns([2, 1])

with col_main:
    trend = st.selectbox("Тренд (по сумме упоминаний за период)", options=options)
    hist = history_for_trend(features, trend)

    st.subheader("Динамика упоминаний (недели)")
    if hist.empty:
        st.info("Для выбранного тега нет строк в признаках.")
    else:
        chart_h = hist.set_index("week_start")[["count"]]
        chart_h.columns = ["упоминаний (факт)"]
        st.line_chart(chart_h)

    st.subheader("Hold-out: факт vs прогноз")
    if predictions is None or predictions.empty:
        st.info(
            "Нет файла предсказаний — запусти соответствующую команду (см. подсказку выше)."
        )
    else:
        pred = predictions_for_trend(predictions, trend)
        if pred.empty:
            st.info(
                "Для этого тренда нет hold-out строк (ряд короткий или тренд не попал в оценку)."
            )
        elif is_ets_predictions(paths):
            chart_p = pred.set_index("week_start")[["y_true", "y_pred_ets"]]
            chart_p.columns = ["факт", "ETS (Holt–Winters / Theta)"]
            st.line_chart(chart_p)
        else:
            chart_p = pred.set_index("week_start")[
                ["y_true", "y_pred_baseline", "y_pred_regression"]
            ]
            chart_p.columns = ["факт", "baseline", "регрессия"]
            st.line_chart(chart_p)

with col_meta:
    st.subheader("Метрики (hold-out)")
    if metrics is None or metrics.empty:
        st.info("Нет файла метрик.")
    elif is_ets_predictions(paths):
        row = metrics_row(metrics, trend)
        if row is None:
            st.info("Нет метрик для этого тренда.")
        else:
            st.metric("MAE (ETS)", f"{row['mae']:.4f}")
            st.metric("RMSE", f"{row['rmse']:.4f}")
            st.metric("MAPE", f"{row['mape_pct']:.2f}%")
    else:
        row = metrics_row(metrics, trend)
        if row is None:
            st.info("Нет метрик для этого тренда.")
        else:
            st.metric("MAE baseline", f"{row['mae_baseline']:.4f}")
            st.metric("MAE regression", f"{row['mae_regression']:.4f}")
            st.metric("RMSE regression", f"{row['rmse_regression']:.4f}")
            st.caption("MAPE в процентах (осторожно при малых count).")

    st.subheader("Сводка по тренду")
    if not hist.empty:
        total = int(hist["count"].sum())
        weeks = len(hist)
        st.metric("Всего упоминаний", total)
        st.metric("Недель в ряду", weeks)
        eng = hist["avg_engagement"].mean()
        if pd.notna(eng):
            st.metric("Средн. proxy engagement", f"{eng:.1f}")

st.divider()
if variant == "synthetic":
    st.markdown(
        """
**Режим ETS:** дневная синтетика по 8 трендам макияжа → сумма по неделям → прогноз **Holt–Winters** или **Theta**
(`statsmodels`). История и hold-out на графике «Динамика» склеены из `history_features` + `future_truth`.
"""
    )
elif variant == "processed":
    st.markdown(
        """
**Processed:** из `trends.jsonl` строятся недельные ряды по полю `item`; на hold-out — baseline (скользящее среднее)
и линейная регрессия по времени.
"""
    )
else:
    st.markdown(
        """
**Мок legacy:** CSV из `mock-pipeline` — та же логика baseline + регрессия на недельных признаках после мок-очистки.
"""
    )
st.markdown("Подробности — в `README.md`.")
