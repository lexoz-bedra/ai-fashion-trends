"""
Демо-интерфейс на Streamlit.

Запуск из корня репозитория (после `pip install -e .`):

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from ai_fashion_trends import __version__

st.set_page_config(page_title="AI Fashion Trends", layout="wide")

st.title("AI Fashion Trends")
st.caption(f"Демо · версия пакета {__version__}")

st.info(
    "Интерфейсная часть проекта строится на **Streamlit**. "
    "Здесь появятся список трендов, карточка выбранного тренда и графики "
    "(упоминания по соцсетям, динамика во времени) — после подключения данных."
)
