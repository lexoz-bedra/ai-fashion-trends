
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from ai_fashion_trends.vision.pair_classifier import (
    SmokyBlueClassifier,
    default_model_path,
    default_torch_model_path,
)

st.set_page_config(page_title="Макияж: smoky / blue / other", layout="centered")

st.title("Классификатор макияжа")
st.caption("Смоки айс · голубые тени · другое. Низкая уверенность → «не лицо».")

root = Path(
    st.sidebar.text_input("Корень проекта", value=str(Path.cwd().resolve()))
).expanduser()
torch_path = default_torch_model_path(root)
sk_path = default_model_path(root)
st.sidebar.caption(f"CNN: {torch_path.name}")
st.sidebar.caption(f"Sklearn: {sk_path.name}")

use_mediapipe = st.sidebar.checkbox("Искать лицо (MediaPipe)", value=False)
if use_mediapipe:
    st.sidebar.caption("Нужен пакет: `uv sync --extra vision`")


def _pick_model_path() -> tuple[str, Path, float] | tuple[None, None, float]:
    meta_pt = torch_path.with_name(torch_path.stem + ".meta.json")
    if torch_path.is_file() and meta_pt.is_file():
        try:
            import torch
        except ImportError:
            st.sidebar.warning(
                "Есть CNN-веса, но PyTorch не установлен. Выполните: `uv sync --extra torch`"
            )
        else:
            mtime = max(torch_path.stat().st_mtime, meta_pt.stat().st_mtime)
            return "cnn", torch_path, mtime
    if sk_path.is_file():
        return "sklearn", sk_path, sk_path.stat().st_mtime
    return None, None, 0.0


backend, model_path, mtime = _pick_model_path()


@st.cache_resource
def _load_classifier(backend: str, path_str: str, mtime_key: float):
    del mtime_key
    path = Path(path_str)
    if backend == "cnn":
        from ai_fashion_trends.vision.torch_makeup import TorchMakeupClassifier

        return TorchMakeupClassifier.load(path)
    return SmokyBlueClassifier.load(path)


if backend is None or model_path is None:
    st.warning("Нет ни CNN (.pt + .meta.json), ни sklearn (.joblib). Обучите модель.")
    st.code(
        "uv sync --extra torch\n"
        "uv run python -m ai_fashion_trends train-makeup-cnn --no-mediapipe\n"
        "# или без PyTorch:\n"
        "uv run python -m ai_fashion_trends train-makeup-classifier --no-mediapipe",
        language="bash",
    )
    st.stop()

clf = _load_classifier(backend, str(model_path), mtime)
st.sidebar.success(f"Модель: **{backend}** ({clf.name})")
thr = clf.prob_threshold
st.sidebar.metric(
    "Порог уверенности",
    f"{thr:.2f}",
    help="Если max(P) по классам ниже — «не лицо на фото»",
)

uploaded = st.file_uploader(
    "Загрузите изображение",
    type=["jpg", "jpeg", "png", "webp"],
)

LABEL_RU = {
    "smoky_eyes": "**Смоки айс** (дымчатые тёмные тени)",
    "blue_eyeshadow": "**Голубые / холодные тени** (blue eyeshadow)",
    "other": "**Другое** — не смоки и не голубые тени (ваш класс «other»)",
    "not_face": "**Это не лицо на фотографии.**",
}

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)
    try:
        pred = clf.predict_pil(image, use_mediapipe=use_mediapipe)
    except Exception as e:
        st.error(f"Ошибка при распознавании: {e}")
        st.stop()

    st.subheader("Результат")
    st.markdown(LABEL_RU[pred.outcome])

    if pred.outcome != "not_face":
        if pred.face_detected:
            st.success("Лицо найдено (MediaPipe).")
        elif use_mediapipe:
            st.info("Лицо не найдено — использован центральный crop.")
        else:
            st.caption("Используется центральный crop (MediaPipe выключен).")

    cols = st.columns(4)
    cols[0].metric("P (смоки)", f"{pred.p_smoky:.2f}")
    cols[1].metric("P (голубые)", f"{pred.p_blue:.2f}")
    cols[2].metric("P (другое)", f"{pred.p_other:.2f}")
    cols[3].metric("max(P)", f"{pred.confidence:.2f}")

    st.bar_chart(
        pd.DataFrame(
            {
                "вероятность": [pred.p_smoky, pred.p_blue, pred.p_other],
            },
            index=["Смоки айс", "Голубые тени", "Другое"],
        )
    )
