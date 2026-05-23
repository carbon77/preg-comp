from __future__ import annotations

import tempfile

import cv2
import numpy as np
import pytesseract
import streamlit as st
from pdf2image import convert_from_path
from PIL import Image

from .config import OCR_LANG, POPPLER_PATH, TESSERACT_CMD


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
