from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from pdf2image import convert_from_path
from PIL import Image
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.class_weight import compute_sample_weight

APP_TITLE = "OCR PDF скрининга"
POPPLER_PATH = None
TESSERACT_CMD = None
OCR_LANG = "rus+eng"
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
    "age_final", "height_final", "weight_final", "bmi_final", "screening_weeks", "screening_days",
    "screening_ga_weeks", "crl", "nt", "fhr", "pappa", "pappa_mom", "fbhcg", "fbhcg_mom",
    "ua_left_pi", "ua_right_pi", "uapi_mean", "nasal_bone",
]


class MultiLabelBinaryModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, label_columns=None, use_sample_weight=True):
        self.base_estimator = base_estimator
        self.label_columns = label_columns
        self.use_sample_weight = use_sample_weight

    def fit(self, X, y):
        self.models_ = {}
        for label in self.label_columns:
            y_label = y[label].astype(int).values
            model = clone(self.base_estimator[label]) if isinstance(self.base_estimator, dict) else clone(self.base_estimator)
            fit_kwargs = {}
            if self.use_sample_weight:
                try:
                    fit_kwargs["sample_weight"] = compute_sample_weight(class_weight="balanced", y=y_label)
                except Exception:
                    fit_kwargs = {}
            try:
                model.fit(X, y_label, **fit_kwargs)
            except TypeError:
                model.fit(X, y_label)
            self.models_[label] = model
        return self

    def predict_proba(self, X):
        if not hasattr(self, "models_") or not self.models_:
            raise RuntimeError("Model is not fitted")
        proba = {}
        for label, model in self.models_.items():
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)
                proba[label] = p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else np.asarray(p).ravel()
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                proba[label] = 1 / (1 + np.exp(-scores))
            else:
                proba[label] = model.predict(X)
        return pd.DataFrame(proba)

    def predict(self, X, thresholds=None):
        proba = self.predict_proba(X)
        thresholds = thresholds or {label: 0.5 for label in proba.columns}
        pred = pd.DataFrame(index=proba.index)
        for label in proba.columns:
            pred[label] = (proba[label] >= thresholds.get(label, 0.5)).astype(int)
        return pred

    def __sklearn_is_fitted__(self):
        return hasattr(self, "models_") and bool(self.models_)


def configure_external_tools() -> None:
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    image = np.array(pil_image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.fastNlMeansDenoising(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def save_uploaded_pdf_to_temp(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def convert_pdf_to_images(pdf_path: str, dpi: int) -> list[Image.Image]:
    return convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)


def image_to_text(image: Image.Image, page_number: int) -> str:
    processed_image = preprocess_image(image)
    text = pytesseract.image_to_string(processed_image, lang=OCR_LANG, config="--psm 6")
    return f"\n--- PAGE {page_number} ---\n{text}"


def read_pdf_with_ocr(uploaded_file, dpi: int = 300) -> str:
    pdf_path = save_uploaded_pdf_to_temp(uploaded_file)
    pages = convert_pdf_to_images(pdf_path, dpi=dpi)
    all_text = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    for page_number, page in enumerate(pages, start=1):
        status_text.write(f"Обрабатывается страница {page_number} из {len(pages)}...")
        all_text.append(image_to_text(page, page_number))
        progress_bar.progress(page_number / len(pages))
    status_text.write("OCR завершён.")
    return "\n".join(all_text)


def normalize_text(text: str) -> str:
    text = text.replace("ё", "е").replace("\xa0", " ")
    return re.sub(r"[ \t]+", " ", text)


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def find_number(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            parsed = parse_float(match.group("value"))
            if parsed is not None:
                return parsed
    return None


def find_text_value(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group("value").strip()
    return None


def extract_screening_features(ocr_text: str) -> dict[str, Any]:
    text = normalize_text(ocr_text)
    features = {
        "age_final": find_number(text, [r"возраст[^0-9]{0,30}(?P<value>\d{1,2})", r"age[^0-9]{0,30}(?P<value>\d{1,2})"]),
        "height_final": find_number(text, [r"рост[^0-9]{0,30}(?P<value>\d{2,3})", r"height[^0-9]{0,30}(?P<value>\d{2,3})"]),
        "weight_final": find_number(text, [r"вес[^0-9]{0,30}(?P<value>\d{2,3}(?:[\.,]\d+)?)", r"weight[^0-9]{0,30}(?P<value>\d{2,3}(?:[\.,]\d+)?)"]),
        "bmi_final": find_number(text, [r"имт[^0-9]{0,30}(?P<value>\d{1,2}(?:[\.,]\d+)?)", r"bmi[^0-9]{0,30}(?P<value>\d{1,2}(?:[\.,]\d+)?)"]),
        "screening_weeks": find_number(text, [r"срок[^0-9]{0,30}(?P<value>\d{1,2})\s*(?:нед|weeks)", r"гестационн\w*\s*возраст[^0-9]{0,30}(?P<value>\d{1,2})\s*(?:нед|weeks)"]),
        "screening_days": find_number(text, [r"срок[^\\n]{0,60}\d{1,2}\s*(?:нед|weeks)[^0-9]{0,20}(?P<value>\d)\s*(?:дн|days)", r"гестационн\w*\s*возраст[^\\n]{0,60}\d{1,2}\s*(?:нед|weeks)[^0-9]{0,20}(?P<value>\d)\s*(?:дн|days)"]),
        "crl": find_number(text, [r"(?:ктр|crl)[^0-9]{0,30}(?P<value>\d{1,3}(?:[\.,]\d+)?)"]),
        "nt": find_number(text, [r"(?:твп|nt|воротников\w*\s*пространств\w*)[^0-9]{0,30}(?P<value>\d{1,2}(?:[\.,]\d+)?)"]),
        "fhr": find_number(text, [r"(?:чсс|fhr)[^0-9]{0,30}(?P<value>\d{2,3})"]),
        "pappa": find_number(text, [r"papp[\-\s]?a(?![^\\n]{0,20}(?:mom|мом))[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)", r"папп[\-\s]?а(?![^\\n]{0,20}(?:mom|мом))[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)"]),
        "pappa_mom": find_number(text, [r"papp[\-\s]?a[^\\n]{0,40}(?:mom|мом)[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)", r"папп[\-\s]?а[^\\n]{0,40}(?:mom|мом)[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)"]),
        "fbhcg": find_number(text, [r"(?:fbhcg|free beta|свободн\w*\s*β?[\-\s]?хгч|бета[\-\s]?хгч)(?![^\\n]{0,20}(?:mom|мом))[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)"]),
        "fbhcg_mom": find_number(text, [r"(?:fbhcg|free beta|свободн\w*\s*β?[\-\s]?хгч|бета[\-\s]?хгч)[^\\n]{0,40}(?:mom|мом)[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)"]),
        "ua_left_pi": find_number(text, [r"(?:левая\s*маточн\w*\s*артери\w*|ua\s*left)[^\\n]{0,50}(?:pi|пи)[^0-9]{0,20}(?P<value>\d+(?:[\.,]\d+)?)"]),
        "ua_right_pi": find_number(text, [r"(?:правая\s*маточн\w*\s*артери\w*|ua\s*right)[^\\n]{0,50}(?:pi|пи)[^0-9]{0,20}(?P<value>\d+(?:[\.,]\d+)?)"]),
        "uapi_mean": find_number(text, [r"(?:средн\w*\s*pi|mean\s*pi|uapi\s*mean)[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)"]),
        "nasal_bone": find_text_value(text, [r"(?:носов\w*\s*кость|nasal\s*bone)[^:\n]{0,20}[:\-]?\s*(?P<value>[^\n\r]{2,80})"]),
    }
    if features["bmi_final"] is None and features["height_final"] and features["weight_final"]:
        height_m = features["height_final"] / 100
        features["bmi_final"] = round(features["weight_final"] / (height_m ** 2), 2)
    features["screening_ga_weeks"] = round(features["screening_weeks"] + (features["screening_days"] or 0) / 7, 2) if features["screening_weeks"] is not None else None
    return features


def features_to_dataframe(features: dict[str, Any]) -> pd.DataFrame:
    names = {
        "age_final": "Возраст", "height_final": "Рост, см", "weight_final": "Вес, кг", "bmi_final": "ИМТ",
        "screening_weeks": "Срок, недели", "screening_days": "Срок, дни", "screening_ga_weeks": "Срок, недель десятичный",
        "crl": "КТР / CRL, мм", "nt": "ТВП / NT, мм", "fhr": "ЧСС плода / FHR", "pappa": "PAPP-A",
        "pappa_mom": "PAPP-A MoM", "fbhcg": "Свободный β-ХГЧ", "fbhcg_mom": "Свободный β-ХГЧ MoM",
        "ua_left_pi": "PI левой маточной артерии", "ua_right_pi": "PI правой маточной артерии",
        "uapi_mean": "Средний PI маточных артерий", "nasal_bone": "Носовая кость",
    }
    rows = []
    for key, title in names.items():
        value = features.get(key)
        value_str = "не найдено" if value is None else (f"{value:.3f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value))
        rows.append({"Признак": title, "Код признака": key, "Значение": value_str})
    return pd.DataFrame(rows).astype(str)


def features_to_model_input(features: dict) -> pd.DataFrame:
    return pd.DataFrame([{key: features.get(key) for key in FEATURE_COLUMNS}])


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    return joblib.load(model_path)


def predict_complications(features: dict, model_path: Path = DEFAULT_MODEL_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifact = load_model(model_path)
    model = artifact["pipeline"]
    X = features_to_model_input(features)
    proba = model.predict_proba(X)
    pred = model.predict(X)
    if isinstance(proba, dict):
        proba = pd.DataFrame([proba])
    if isinstance(pred, dict):
        pred = pd.DataFrame([pred])
    return proba, pred


def format_predictions(proba: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Код": label,
            "Осложнение": LABEL_RU.get(label, label),
            "Вероятность": float(proba.iloc[0][label]),
            "Предсказание": "Да" if int(pred.iloc[0][label]) == 1 else "Нет",
        }
        for label in proba.columns
    ])


def main() -> None:
    configure_external_tools()
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    st.title(APP_TITLE)
    st.write("Загрузите PDF-файл скрининга. Приложение преобразует страницы PDF в изображения, выполнит OCR и покажет распознанный текст и извлечённые признаки.")

    dpi = st.slider("DPI для конвертации PDF", min_value=150, max_value=400, value=300, step=50)
    st.caption(f"OCR language: `{OCR_LANG}` | Poppler: `{POPPLER_PATH or 'PATH'}` | Tesseract: `{TESSERACT_CMD or 'PATH'}`")

    uploaded_file = st.file_uploader("PDF скрининга", type=["pdf"])
    if uploaded_file is None:
        st.info("Загрузите PDF-файл для запуска OCR.")
        return
    st.success(f"Файл загружен: {uploaded_file.name}")
    if not st.button("Запустить OCR", type="primary"):
        return

    try:
        with st.spinner("Выполняется OCR..."):
            ocr_text = read_pdf_with_ocr(uploaded_file, dpi=dpi)
        features = extract_screening_features(ocr_text)
        features_df = features_to_dataframe(features)

        tab_features, tab_predict, tab_text, tab_json = st.tabs(["Извлечённые признаки", "Предсказание осложнений", "Полный OCR-текст", "JSON"])
        with tab_features:
            st.subheader("Признаки, найденные в PDF")
            st.dataframe(features_df, width="stretch", hide_index=True)
            st.download_button("Скачать признаки CSV", features_df.to_csv(index=False).encode("utf-8-sig"), "ocr_features.csv", "text/csv")

        with tab_predict:
            st.subheader("Прогноз осложнений")
            if not DEFAULT_MODEL_PATH.exists():
                st.warning(f"Файл модели не найден: `{DEFAULT_MODEL_PATH}`")
            else:
                proba_df, pred_df = predict_complications(features, model_path=DEFAULT_MODEL_PATH)
                result_df = format_predictions(proba_df, pred_df)
                result_df["Вероятность"] = result_df["Вероятность"].map(lambda x: f"{x:.1%}")
                st.dataframe(result_df, width="stretch", hide_index=True)
                st.download_button("Скачать прогноз CSV", result_df.to_csv(index=False).encode("utf-8-sig"), "complications_prediction.csv", "text/csv")

        with tab_text:
            st.subheader("Распознанный OCR-текст")
            st.text_area(label="OCR text", value=ocr_text, height=600, label_visibility="collapsed")
            st.download_button("Скачать OCR-текст", ocr_text.encode("utf-8"), "ocr_text.txt", "text/plain")

        with tab_json:
            st.subheader("Извлечённые признаки в JSON")
            st.json(features)

    except Exception as exc:
        st.error("Не удалось выполнить OCR.")
        st.exception(exc)


if __name__ == "__main__":
    main()
