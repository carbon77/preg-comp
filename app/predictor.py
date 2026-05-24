from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class MultiLabelBinaryModel:
    def __init__(self, base_estimator=None, label_columns=None, use_sample_weight=True):
        self.base_estimator = base_estimator
        self.label_columns = label_columns
        self.use_sample_weight = use_sample_weight
        self.models_ = {}

    def predict_proba(self, X):
        if not self.models_:
            raise RuntimeError("Model is not fitted")

        proba = {}

        for label, model in self.models_.items():
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)

                if p.ndim == 2 and p.shape[1] == 2:
                    proba[label] = p[:, 1]
                else:
                    proba[label] = np.asarray(p).ravel()

            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                proba[label] = 1 / (1 + np.exp(-scores))

            else:
                proba[label] = model.predict(X)

        return pd.DataFrame(proba)

    def predict(self, X, thresholds=None):
        proba = self.predict_proba(X)
        thresholds = thresholds or {label: 0.5 for label in self.label_columns}

        pred = pd.DataFrame(index=proba.index)

        for label in self.label_columns:
            pred[label] = (proba[label] >= thresholds.get(label, 0.5)).astype(int)

        return pred


DEFAULT_MODEL_PATH = Path("artifacts/screening_ocr.joblib")

LABEL_RU = {
    "label_fgr": "Задержка роста плода",
    "label_preterm": "Преждевременные роды",
    "label_fetal_distress": "Дистресс плода",
    "label_preeclampsia": "Преэклампсия",
    "label_large_for_ga": "Крупный плод",
    "label_normal": "Без осложнений",
}


FEATURE_COLUMNS = [
    "age",
    "height",
    "weight",
    "bmi",
    "gestational_weeks",
    "gestational_days",
    "gestational_age_decimal",
    "crl",
    "nt",
    "fhr",
    "pappa",
    "pappa_mom",
    "fbhcg",
    "fbhcg_mom",
    "ua_left_pi",
    "ua_right_pi",
    "uapi_mean",
]


def features_to_model_input(features: dict) -> pd.DataFrame:
    row = {key: features.get(key) for key in FEATURE_COLUMNS}
    return pd.DataFrame([row])


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    return joblib.load(model_path)


def predict_complications(features: dict, model_path: Path = DEFAULT_MODEL_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = load_model(model_path)
    X = features_to_model_input(features)

    proba = model.predict_proba(X)
    if isinstance(proba, dict):
        proba = pd.DataFrame([proba])

    pred = model.predict(X)
    if isinstance(pred, dict):
        pred = pd.DataFrame([pred])

    return proba, pred


def format_predictions(proba: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in proba.columns:
        rows.append(
            {
                "Код": label,
                "Осложнение": LABEL_RU.get(label, label),
                "Вероятность": float(proba.iloc[0][label]),
                "Предсказание": "Да" if int(pred.iloc[0][label]) == 1 else "Нет",
            }
        )

    return pd.DataFrame(rows)
