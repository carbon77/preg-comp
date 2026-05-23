# app.py

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from pdf2image import convert_from_path
from PIL import Image


# ============================================================
# 1. Настройки
# ============================================================

APP_TITLE = "OCR PDF скрининга"

# Для Windows можно указать пути явно.
# Если Poppler и Tesseract добавлены в PATH, можно оставить None.
POPPLER_PATH = None
TESSERACT_CMD = None

# Пример для Windows:
# POPPLER_PATH = r"C:\poppler\Library\bin"
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OCR_LANG = "rus+eng"


# ============================================================
# 2. Настройка внешних зависимостей
# ============================================================

def configure_external_tools() -> None:
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# ============================================================
# 3. OCR
# ============================================================

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Простая предобработка изображения перед OCR:
    - grayscale;
    - шумоподавление;
    - бинаризация.
    """
    image = np.array(pil_image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray = cv2.fastNlMeansDenoising(gray)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return binary


def save_uploaded_pdf_to_temp(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def convert_pdf_to_images(pdf_path: str, dpi: int) -> list[Image.Image]:
    return convert_from_path(
        pdf_path,
        dpi=dpi,
        poppler_path=POPPLER_PATH,
    )


def image_to_text(image: Image.Image, page_number: int) -> str:
    processed_image = preprocess_image(image)

    text = pytesseract.image_to_string(
        processed_image,
        lang=OCR_LANG,
        config="--psm 6",
    )

    return f"\n--- PAGE {page_number} ---\n{text}"


def read_pdf_with_ocr(uploaded_file, dpi: int = 300) -> str:
    pdf_path = save_uploaded_pdf_to_temp(uploaded_file)

    pages = convert_pdf_to_images(pdf_path, dpi=dpi)

    all_text = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for page_number, page in enumerate(pages, start=1):
        status_text.write(f"Обрабатывается страница {page_number} из {len(pages)}...")

        page_text = image_to_text(page, page_number)
        all_text.append(page_text)

        progress_bar.progress(page_number / len(pages))

    status_text.write("OCR завершён.")

    return "\n".join(all_text)


# ============================================================
# 4. Парсинг признаков из OCR-текста
# ============================================================

def normalize_text(text: str) -> str:
    text = text.replace("ё", "е")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None

    value = value.replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return None


def find_number(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)

        if match:
            value = match.group("value")
            parsed = parse_float(value)

            if parsed is not None:
                return parsed

    return None


def find_text_value(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)

        if match:
            value = match.group("value").strip()
            return value

    return None


def extract_screening_features(ocr_text: str) -> dict[str, Any]:
    """
    Базовое извлечение признаков из текста.
    Регулярки можно постепенно расширять под реальные PDF.
    """
    text = normalize_text(ocr_text)

    features = {
        "age": find_number(
            text,
            [
                r"возраст[^0-9]{0,30}(?P<value>\d{1,2})",
                r"age[^0-9]{0,30}(?P<value>\d{1,2})",
            ],
        ),
        "height": find_number(
            text,
            [
                r"рост[^0-9]{0,30}(?P<value>\d{2,3})",
                r"height[^0-9]{0,30}(?P<value>\d{2,3})",
            ],
        ),
        "weight": find_number(
            text,
            [
                r"вес[^0-9]{0,30}(?P<value>\d{2,3}(?:[\.,]\d+)?)",
                r"weight[^0-9]{0,30}(?P<value>\d{2,3}(?:[\.,]\d+)?)",
            ],
        ),
        "bmi": find_number(
            text,
            [
                r"имт[^0-9]{0,30}(?P<value>\d{1,2}(?:[\.,]\d+)?)",
                r"bmi[^0-9]{0,30}(?P<value>\d{1,2}(?:[\.,]\d+)?)",
            ],
        ),
        "gestational_weeks": find_number(
            text,
            [
                r"срок[^0-9]{0,30}(?P<value>\d{1,2})\s*(?:нед|weeks)",
                r"гестационн\w*\s*возраст[^0-9]{0,30}(?P<value>\d{1,2})\s*(?:нед|weeks)",
            ],
        ),
        "gestational_days": find_number(
            text,
            [
                r"срок[^\\n]{0,60}\d{1,2}\s*(?:нед|weeks)[^0-9]{0,20}(?P<value>\d)\s*(?:дн|days)",
                r"гестационн\w*\s*возраст[^\\n]{0,60}\d{1,2}\s*(?:нед|weeks)[^0-9]{0,20}(?P<value>\d)\s*(?:дн|days)",
            ],
        ),
        "crl": find_number(
            text,
            [
                r"(?:ктр|crl)[^0-9]{0,30}(?P<value>\d{1,3}(?:[\.,]\d+)?)",
            ],
        ),
        "nt": find_number(
            text,
            [
                r"(?:твп|nt|воротников\w*\s*пространств\w*)[^0-9]{0,30}(?P<value>\d{1,2}(?:[\.,]\d+)?)",
            ],
        ),
        "fhr": find_number(
            text,
            [
                r"(?:чсс|fhr)[^0-9]{0,30}(?P<value>\d{2,3})",
            ],
        ),
        "pappa": find_number(
            text,
            [
                r"papp[\-\s]?a(?![^\\n]{0,20}(?:mom|мом))[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)",
                r"папп[\-\s]?а(?![^\\n]{0,20}(?:mom|мом))[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),
        "pappa_mom": find_number(
            text,
            [
                r"papp[\-\s]?a[^\\n]{0,40}(?:mom|мом)[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)",
                r"папп[\-\s]?а[^\\n]{0,40}(?:mom|мом)[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),
        "fbhcg": find_number(
            text,
            [
                r"(?:fbhcg|free beta|свободн\w*\s*β?[\-\s]?хгч|бета[\-\s]?хгч)(?![^\\n]{0,20}(?:mom|мом))[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),
        "fbhcg_mom": find_number(
            text,
            [
                r"(?:fbhcg|free beta|свободн\w*\s*β?[\-\s]?хгч|бета[\-\s]?хгч)[^\\n]{0,40}(?:mom|мом)[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),
        "ua_left_pi": find_number(
            text,
            [
                r"(?:левая\s*маточн\w*\s*артери\w*|ua\s*left)[^\\n]{0,50}(?:pi|пи)[^0-9]{0,20}(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),
        "ua_right_pi": find_number(
            text,
            [
                r"(?:правая\s*маточн\w*\s*артери\w*|ua\s*right)[^\\n]{0,50}(?:pi|пи)[^0-9]{0,20}(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),
        "uapi_mean": find_number(
            text,
            [
                r"(?:средн\w*\s*pi|mean\s*pi|uapi\s*mean)[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),
        "nasal_bone": find_text_value(
            text,
            [
                r"(?:носов\w*\s*кость|nasal\s*bone)[^:\n]{0,20}[:\-]?\s*(?P<value>[^\n\r]{2,80})",
            ],
        ),
    }

    if features["bmi"] is None and features["height"] and features["weight"]:
        height_m = features["height"] / 100
        features["bmi"] = round(features["weight"] / (height_m ** 2), 2)

    if features["gestational_weeks"] is not None:
        days = features["gestational_days"] or 0
        features["gestational_age_decimal"] = round(
            features["gestational_weeks"] + days / 7,
            2,
        )
    else:
        features["gestational_age_decimal"] = None

    return features


def features_to_dataframe(features: dict[str, Any]) -> pd.DataFrame:
    rows = []

    names = {
        "age": "Возраст",
        "height": "Рост, см",
        "weight": "Вес, кг",
        "bmi": "ИМТ",
        "gestational_weeks": "Срок, недели",
        "gestational_days": "Срок, дни",
        "gestational_age_decimal": "Срок, недель десятичный",
        "crl": "КТР / CRL, мм",
        "nt": "ТВП / NT, мм",
        "fhr": "ЧСС плода / FHR",
        "pappa": "PAPP-A",
        "pappa_mom": "PAPP-A MoM",
        "fbhcg": "Свободный β-ХГЧ",
        "fbhcg_mom": "Свободный β-ХГЧ MoM",
        "ua_left_pi": "PI левой маточной артерии",
        "ua_right_pi": "PI правой маточной артерии",
        "uapi_mean": "Средний PI маточных артерий",
        "nasal_bone": "Носовая кость",
    }

    for key, title in names.items():
        value = features.get(key)

        if value is None:
            value_str = "не найдено"
        elif isinstance(value, float):
            value_str = f"{value:.3f}".rstrip("0").rstrip(".")
        else:
            value_str = str(value)

        rows.append(
            {
                "Признак": title,
                "Код признака": key,
                "Значение": value_str,
            }
        )

    df = pd.DataFrame(rows)

    # Важно: все колонки явно делаем строковыми,
    # чтобы Streamlit / PyArrow не пытался смешивать float и str.
    df = df.astype(str)

    return df


# ============================================================
# 5. Streamlit UI
# ============================================================

def main() -> None:
    configure_external_tools()

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📄",
        layout="wide",
    )

    st.title(APP_TITLE)

    st.write(
        "Загрузите PDF-файл скрининга. Приложение преобразует страницы PDF в изображения, "
        "выполнит OCR и покажет распознанный текст и извлечённые признаки."
    )

    with st.sidebar:
        st.header("Настройки OCR")

        dpi = st.slider(
            "DPI для конвертации PDF",
            min_value=150,
            max_value=400,
            value=300,
            step=50,
        )

        st.divider()

        st.caption("Текущие настройки")

        st.write(f"OCR language: `{OCR_LANG}`")

        if POPPLER_PATH:
            st.write(f"Poppler: `{POPPLER_PATH}`")
        else:
            st.write("Poppler: используется из PATH")

        if TESSERACT_CMD:
            st.write(f"Tesseract: `{TESSERACT_CMD}`")
        else:
            st.write("Tesseract: используется из PATH")

    uploaded_file = st.file_uploader(
        "PDF скрининга",
        type=["pdf"],
    )

    if uploaded_file is None:
        st.info("Загрузите PDF-файл для запуска OCR.")
        return

    st.success(f"Файл загружен: {uploaded_file.name}")

    run_ocr = st.button("Запустить OCR", type="primary")

    if not run_ocr:
        return

    try:
        with st.spinner("Выполняется OCR..."):
            ocr_text = read_pdf_with_ocr(uploaded_file, dpi=dpi)

        features = extract_screening_features(ocr_text)
        features_df = features_to_dataframe(features)

        tab_features, tab_text, tab_json = st.tabs(
            [
                "Извлечённые признаки",
                "Полный OCR-текст",
                "JSON",
            ]
        )

        with tab_features:
            st.subheader("Признаки, найденные в PDF")

            st.dataframe(
                features_df,
                width="stretch",
                hide_index=True,
            )

            csv_bytes = features_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="Скачать признаки CSV",
                data=csv_bytes,
                file_name="ocr_features.csv",
                mime="text/csv",
            )

        with tab_text:
            st.subheader("Распознанный OCR-текст")

            st.text_area(
                label="OCR text",
                value=ocr_text,
                height=600,
                label_visibility="collapsed",
            )

            st.download_button(
                label="Скачать OCR-текст",
                data=ocr_text.encode("utf-8"),
                file_name="ocr_text.txt",
                mime="text/plain",
            )

        with tab_json:
            st.subheader("Извлечённые признаки в JSON")

            st.json(features)

    except Exception as exc:
        st.error("Не удалось выполнить OCR.")
        st.exception(exc)

        st.warning(
            "Если ошибка связана с Poppler на Windows, укажи путь к Poppler в переменной "
            "`POPPLER_PATH`, например: `C:\\poppler\\Library\\bin`."
        )


if __name__ == "__main__":
    main()