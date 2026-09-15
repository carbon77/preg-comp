from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from autogluon.tabular import TabularPredictor

DEFAULT_MODEL_PATH = Path("artifacts/autogluon_early_w14.joblib")
DEFAULT_BACKGROUND_PATH = Path("artifacts/autogluon_early_w14_background.csv")
MODEL_PREDICTORS_DIR = Path("artifacts/ag_predictors")

COMPLICATION_LABELS = [
    "label_fgr",
    "label_preterm",
    "label_fetal_distress",
    "label_preeclampsia",
    "label_large_for_ga",
]

LABEL_COLUMNS = COMPLICATION_LABELS + ["label_normal"]

LABEL_RU = {
    "label_fgr": "Задержка роста плода",
    "label_preterm": "Преждевременные роды",
    "label_fetal_distress": "Дистресс плода",
    "label_preeclampsia": "Преэклампсия",
    "label_large_for_ga": "Крупный плод",
    "label_normal": "Без осложнений",
}

# Поля, которые реально приходят из OCR/парсера скрининга.
# Модель AutoGluon обучалась на X_early, поэтому остальные признаки будут добавлены как NaN.
OCR_TO_X_EARLY_FEATURE_MAP = {
    "age": "age_final",
    "height": "height_final",
    "weight": "weight_final",
    "bmi": "bmi_final",
    "gestational_age_decimal": "screening_ga_weeks",
    "gestational_weeks": "screening_weeks",
    "gestational_days": "screening_days",
    "crl": "crl",
    "nt": "nt",
    "fhr": "fhr",
    "ua_left_pi": "ua_left_pi",
    "ua_right_pi": "ua_right_pi",
    "uapi_mean": "uapi_mean",
    "nasal_bone": "nasal_bone",
    "pappa": "pappa",
    "pappa_mom": "pappa_mom",
    "fbhcg": "fbhcg",
    "fbhcg_mom": "fbhcg_mom",
}

def _resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path)
    if path.exists() or path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def _load_autogluon_predictors(
        artifact_path: str | Path = DEFAULT_MODEL_PATH,
        ag_dir: str | Path = MODEL_PREDICTORS_DIR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Файл AutoGluon artifact не найден: {artifact_path}")

    artifact = joblib.load(artifact_path)

    ag_dir = Path(ag_dir)
    if not ag_dir.exists():
        raise FileNotFoundError(
            "Не найдена директория AutoGluon predictors. "
            f"Путь из artifact: {artifact['autogluon_dir']}. "
        )

    train_labels = artifact.get("train_labels") or COMPLICATION_LABELS
    train_labels = [label for label in train_labels if label != "label_normal"]

    predictors: dict[str, Any] = {}
    for label in train_labels:
        label_dir = ag_dir / label
        if not label_dir.exists():
            raise FileNotFoundError(f'Не найдена папка predictor для {label}: {label_dir}')
        predictors[label] = TabularPredictor.load(str(label_dir), require_py_version_match=False)

    artifact["resolved_autogluon_dir"] = str(ag_dir)
    artifact['thresholds'] = {
        label: float(predictor.decision_threshold)
        for label, predictor in predictors.items()
    }
    return predictors, artifact

def _positive_class_probability(predictor: Any, X: pd.DataFrame) -> np.ndarray:
    p = predictor.predict_proba(X)

    if isinstance(p, pd.DataFrame):
        if 1 in p.columns:
            return p[1].to_numpy(dtype=float)
        if "1" in p.columns:
            return p["1"].to_numpy(dtype=float)
        if True in p.columns:
            return p[True].to_numpy(dtype=float)
        return p.iloc[:, -1].to_numpy(dtype=float)

    p = np.asarray(p)
    if p.ndim == 2:
        return p[:, -1].astype(float)
    return p.ravel().astype(float)


def _autogluon_predict_proba_multilabel(
        predictors: dict[str, Any],
        X: pd.DataFrame,
) -> pd.DataFrame:
    proba = {
        label: _positive_class_probability(predictor, X)
        for label, predictor in predictors.items()
    }
    return pd.DataFrame(proba, index=X.index)


def _add_derived_normal(
        proba: pd.DataFrame,
        pred: pd.DataFrame,
        complication_labels: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    complication_labels = complication_labels or [c for c in COMPLICATION_LABELS if c in proba.columns]

    if complication_labels:
        normal_proba = np.prod(1 - proba[complication_labels].clip(0, 1), axis=1)
    else:
        normal_proba = np.zeros(len(proba), dtype=float)

    proba = proba.copy()
    pred = pred.copy()
    proba["label_normal"] = normal_proba
    pred["label_normal"] = (pred[complication_labels].sum(axis=1) == 0).astype(int)
    return proba, pred


def _coerce_numeric_like_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Мягко приводит очевидные числовые поля к float.
    Категориальные поля AutoGluon может принять как object/string.
    """
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object:
            converted = pd.to_numeric(
                X[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            # Если хоть часть значений стала числами, используем числовую версию.
            if converted.notna().sum() > 0:
                X[col] = converted
    return X


def features_to_model_input(
        features: dict[str, Any] | pd.DataFrame,
        feature_columns: list[str],
        ocr_feature_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Приводит вход OCR/парсера к формату X_early, на котором обучался AutoGluon.

    PDF скрининга содержит только часть признаков X_early.
    Все отсутствующие признаки добавляются как NaN.     
    """
    ocr_feature_map = ocr_feature_map or OCR_TO_X_EARLY_FEATURE_MAP

    if isinstance(features, pd.DataFrame):
        raw = features.copy()
    else:
        raw = pd.DataFrame([features])

    # Переименовываем OCR-ключи в имена колонок из X_early.
    rename_map = {src: dst for src, dst in ocr_feature_map.items() if src in raw.columns}
    raw = raw.rename(columns=rename_map)

    X = pd.DataFrame(index=raw.index)
    for col in feature_columns:
        if col in raw.columns:
            X[col] = raw[col]
        else:
            X[col] = np.nan

    X = _coerce_numeric_like_columns(X)
    return X[feature_columns]


def load_model(
        model_path: str | Path = DEFAULT_MODEL_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _load_autogluon_predictors(model_path)

def _autogluon_predict_multilabel(
    predictors: dict[str, Any],
    X: pd.DataFrame,
) -> pd.DataFrame:
    pred = {}

    for label, predictor in predictors.items():
        pred[label] = predictor.predict(X).astype(int).values
    return pd.DataFrame(pred, index=X.index)


def predict_complications(
        features: dict[str, Any] | pd.DataFrame,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        derive_normal: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    predictors, artifact = load_model(model_path)

    feature_columns = artifact.get("feature_columns")
    if not feature_columns:
        raise KeyError("В AutoGluon artifact нет 'feature_columns'. Сохрани artifact заново из ноутбука.")

    X = features_to_model_input(features, feature_columns=feature_columns)

    proba = _autogluon_predict_proba_multilabel(predictors, X)
    pred = _autogluon_predict_multilabel(predictors, X)

    thresholds = {
        label: float(predictor.decision_threshold)
        for label, predictor in predictors.items()
    }
    if derive_normal and "label_normal" not in proba.columns:
        proba, pred = _add_derived_normal(
            proba=proba,
            pred=pred,
            complication_labels=[
                c for c in artifact.get("train_labels", COMPLICATION_LABELS) 
                if c != "label_normal"
            ],
        )

    return proba, pred, artifact['thresholds']


def format_predictions(
        proba: pd.DataFrame,
        pred: pd.DataFrame,
        thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for label in proba.columns:
        if label == "label_normal":
            threshold = np.nan
        else:
            threshold = round(thresholds.get(label, np.nan), 4) 
        rows.append(
            {
                "Код": label,
                "Осложнение": LABEL_RU.get(label, label),
                "Вероятность": float(proba.iloc[0][label]),
                "Порог предсказания": threshold,
                "Предсказание": "Да" if int(pred.iloc[0][label]) == 1 else "Нет",
            }
        )
    return pd.DataFrame(rows)


def _load_background(
        background_path: str | Path | None,
        feature_columns: list[str],
        max_background_rows: int = 100,
) -> pd.DataFrame | None:
    if background_path is None:
        return None

    background_path = Path(background_path)
    if not background_path.exists():
        return None

    background = pd.read_csv(background_path, encoding="utf-8-sig")
    # background = features_to_model_input(background, feature_columns=feature_columns)

    if len(background) > max_background_rows:
        background = background.sample(max_background_rows, random_state=42)

    return background.reset_index(drop=True)


def _kernel_shap_for_label(
        predictor: Any,
        X_patient: pd.DataFrame,
        background: pd.DataFrame,
        nsamples: int | str = "auto",
) -> np.ndarray:
    def predict_positive(data: np.ndarray | pd.DataFrame) -> np.ndarray:
        data_df = pd.DataFrame(data, columns=X_patient.columns)
        return _positive_class_probability(predictor, data_df)

    explainer = shap.KernelExplainer(predict_positive, background)
    shap_values = explainer.shap_values(X_patient, nsamples=nsamples)
    return np.asarray(shap_values).reshape(-1)


def _permutation_delta_for_label(
        predictor: Any,
        X_patient: pd.DataFrame,
        background: pd.DataFrame | None,
) -> np.ndarray:
    """
    Fallback, если нет SHAP/background: локальная чувствительность.
    Это не настоящий SHAP, но лучше, чем возвращать случайную/глобальную importance.
    """
    base_prob = float(_positive_class_probability(predictor, X_patient)[0])
    values: list[float] = []

    for col in X_patient.columns:
        X_tmp = X_patient.copy()
        if background is not None and col in background.columns and background[col].notna().any():
            replacement = background[col].median() if pd.api.types.is_numeric_dtype(background[col]) else background[col].mode(dropna=True).iloc[0]
        else:
            replacement = np.nan
        X_tmp[col] = replacement
        changed_prob = float(_positive_class_probability(predictor, X_tmp)[0])
        values.append(base_prob - changed_prob)

    return np.asarray(values, dtype=float)


def get_shap_interpretation(
        features: dict[str, Any] | pd.DataFrame,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        background_path: str | Path | None = DEFAULT_BACKGROUND_PATH,
        top_k: int = 3,
        only_predicted: bool = True,
        method: str = "auto",
        max_background_rows: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    """
    Возвращает top-k факторов по каждой метке для AutoGluon.

    method:
    - 'auto': настоящий Kernel SHAP, если есть background CSV и установлен shap; иначе fallback.
    - 'kernel_shap': требовать настоящий Kernel SHAP.
    - 'permutation': локальная чувствительность без SHAP.
    """
    predictors, artifact = load_model(model_path)
    feature_columns = artifact.get("feature_columns")
    if not feature_columns:
        raise KeyError("В AutoGluon artifact нет 'feature_columns'.")

    X_patient = features_to_model_input(features, feature_columns=feature_columns)
    if len(X_patient) != 1:
        raise ValueError("Для локального объяснения нужна ровно одна строка.")

    pred = _autogluon_predict_multilabel(predictors, X_patient)
    background = _load_background(background_path, feature_columns, max_background_rows=max_background_rows)

    explanations: dict[str, list[dict[str, Any]]] = {}

    for label, predictor in predictors.items():
        if only_predicted and int(pred.iloc[0][label]) != 1:
            continue

        used_method = method
        try:
            if method in {"auto", "kernel_shap"}:
                if background is None:
                    raise FileNotFoundError("Background CSV для Kernel SHAP не найден.")
                values = _kernel_shap_for_label(predictor, X_patient, background)
                used_method = "kernel_shap"
            elif method == "permutation":
                values = _permutation_delta_for_label(predictor, X_patient, background)
            else:
                raise ValueError("method должен быть: 'auto', 'kernel_shap' или 'permutation'.")
        except Exception:
            if method == "kernel_shap":
                raise
            values = _permutation_delta_for_label(predictor, X_patient, background)
            used_method = "permutation_delta"

        values = np.asarray(values, dtype=float).ravel()
        top_idx = np.argsort(np.abs(values))[::-1][:top_k]

        explanations[label] = [
            {
                "feature": feature_columns[i],
                "value": None if pd.isna(X_patient.iloc[0, i]) else X_patient.iloc[0, i].item() if hasattr(X_patient.iloc[0, i], "item") else X_patient.iloc[0, i],
                "shap_value": float(values[i]),
                "abs_shap_value": float(abs(values[i])),
                "impact": "увеличивает риск" if values[i] > 0 else "снижает риск",
                "method": used_method,
            }
            for i in top_idx
        ]

    return explanations


def predict_complications_with_shap(
        features: dict[str, Any] | pd.DataFrame,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        background_path: str | Path | None = DEFAULT_BACKGROUND_PATH,
        top_k: int = 3,
        only_predicted_shap: bool = True,
) -> dict[str, Any]:
    proba, pred, thresholds = predict_complications(features, model_path=model_path)
    formatted = format_predictions(proba, pred, thresholds)

    explanations = get_shap_interpretation(
        features=features,
        model_path=model_path,
        background_path=background_path,
        top_k=top_k,
        only_predicted=only_predicted_shap,
        method="auto",
    )

    result = []
    for _, row in formatted.iterrows():
        label = row["Код"]
        result.append(
            {
                "code": label,
                "label": row["Осложнение"],
                "probability": float(row["Вероятность"]),
                "threshold": None if pd.isna(row["Порог"]) else float(row["Порог"]),
                "prediction": row["Предсказание"],
                "top_features": explanations.get(label, []),
            }
        )

    return {"predictions": result}


def shap_to_dataframe(shap_explanations: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for label, items in shap_explanations.items():
        for rank, item in enumerate(items, start=1):
            rows.append(
                {
                    "Код метки": label,
                    "Осложнение": LABEL_RU.get(label, label),
                    "Ранг": rank,
                    "Признак": item.get("feature"),
                    "Значение": item.get("value"),
                    "SHAP value": item.get("shap_value"),
                    "Влияние": item.get("impact"),
                    "Метод": item.get("method"),
                }
            )
    return pd.DataFrame(rows)
