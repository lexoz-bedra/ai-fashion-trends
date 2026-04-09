
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_fashion_trends.vision.features import compute_makeup_features
from ai_fashion_trends.vision.face import extract_face_or_center_crop
from ai_fashion_trends.vision.makeup_data import (
    BLUE_DIRS,
    OTHER_DIRS,
    SMOKY_DIRS,
    iter_labeled_makeup_paths,
)
from ai_fashion_trends.vision.pair_classifier import save_classifier_bundle


def train_makeup_pair_classifier(
    data_dir: Path,
    output_path: Path,
    *,
    test_size: float = 0.25,
    seed: int = 42,
    use_mediapipe: bool = False,
    prob_threshold: float = 0.45,
    C: float = 1.0,
) -> dict:
    data_dir = data_dir.resolve()
    pairs = iter_labeled_makeup_paths(data_dir)
    if len(pairs) < 4:
        raise ValueError(
            f"Нужно минимум несколько фото в {SMOKY_DIRS}, {BLUE_DIRS} и/или {OTHER_DIRS} "
            f"внутри {data_dir}, найдено файлов: {len(pairs)}"
        )

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for path, y in pairs:
        try:
            im = Image.open(path).convert("RGB")
        except OSError:
            continue
        crop, _, _ = extract_face_or_center_crop(im, try_mediapipe=use_mediapipe)
        X_list.append(compute_makeup_features(crop))
        y_list.append(y)

    if len(X_list) < 4:
        raise ValueError("Слишком мало удалось прочитать изображений.")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    ulabels = sorted(np.unique(y).tolist())
    if len(ulabels) < 2:
        raise ValueError(
            "Нужны как минимум два разных класса среди папок smoky_eyes, blue-eyeshadow, other."
        )

    _, counts = np.unique(y, return_counts=True)
    strat = y if counts.min() >= 2 and len(ulabels) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=C,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test, y_pred, labels=ulabels, zero_division=0
    )

    fitted_classes = [int(c) for c in pipe.named_steps["clf"].classes_]

    save_classifier_bundle(
        output_path,
        pipe,
        classes=fitted_classes,
        prob_threshold=prob_threshold,
        extra_meta={
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "accuracy_holdout": acc,
            "data_dir": str(data_dir),
        },
    )

    return {
        "output_path": str(output_path),
        "n_samples": len(y),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "accuracy_holdout": acc,
        "classification_report": report,
        "classes": fitted_classes,
    }
