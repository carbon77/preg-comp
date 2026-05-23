from __future__ import annotations

import streamlit as st

from .config import APP_TITLE, OCR_LANG, POPPLER_PATH, TESSERACT_CMD
from .ocr import configure_external_tools, read_pdf_with_ocr
from .parser import extract_screening_features, features_to_dataframe


def main() -> None:
    configure_external_tools()

    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    st.title(APP_TITLE)
    st.write(
        "Загрузите PDF-файл скрининга. Приложение преобразует страницы PDF в изображения, "
        "выполнит OCR и покажет распознанный текст и извлечённые признаки."
    )

    with st.sidebar:
        st.header("Настройки OCR")
        dpi = st.slider("DPI для конвертации PDF", min_value=150, max_value=400, value=300, step=50)
        st.divider()
        st.caption("Текущие настройки")
        st.write(f"OCR language: `{OCR_LANG}`")
        st.write(f"Poppler: `{POPPLER_PATH}`" if POPPLER_PATH else "Poppler: используется из PATH")
        st.write(f"Tesseract: `{TESSERACT_CMD}`" if TESSERACT_CMD else "Tesseract: используется из PATH")

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

        tab_features, tab_text, tab_json = st.tabs(["Извлечённые признаки", "Полный OCR-текст", "JSON"])

        with tab_features:
            st.subheader("Признаки, найденные в PDF")
            st.dataframe(features_df, width="stretch", hide_index=True)
            st.download_button(
                label="Скачать признаки CSV",
                data=features_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="ocr_features.csv",
                mime="text/csv",
            )

        with tab_text:
            st.subheader("Распознанный OCR-текст")
            st.text_area(label="OCR text", value=ocr_text, height=600, label_visibility="collapsed")
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
