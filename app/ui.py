from __future__ import annotations

import streamlit as st

from .config import APP_TITLE
from .ocr import read_pdf_with_ocr
from .parser import extract_screening_features, features_to_dataframe
from .predictor import (
    DEFAULT_MODEL_PATH,
    format_predictions,
    predict_complications,
    get_shap_interpretation,
)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    st.title(APP_TITLE)
    st.write(
        "Загрузите PDF-файл скрининга. Приложение покажет распознанный текст, извлечённые признаки и предсказания."
    )

    uploaded_file = st.file_uploader("PDF скрининга", type=["pdf"])
    if uploaded_file is None:
        st.info("Загрузите PDF-файл для запуска OCR.")
        return

    st.success(f"Файл загружен: {uploaded_file.name}")
    if not st.button("Запустить OCR", type="primary"):
        return

    try:
        with st.spinner("Выполняется OCR..."):
            ocr_text = read_pdf_with_ocr(uploaded_file)

        features = extract_screening_features(ocr_text)
        features_df = features_to_dataframe(features)

        tab_features, tab_predict, tab_text, tab_json = st.tabs(["Извлечённые признаки", "Предсказание осложнений", "Полный OCR-текст", "JSON"])

        with tab_features:
            st.subheader("Признаки, найденные в PDF")
            st.dataframe(features_df, width="stretch", hide_index=True)
            st.download_button(
                label="Скачать признаки CSV",
                data=features_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="ocr_features.csv",
                mime="text/csv",
            )


        with tab_predict:
            st.subheader("Прогноз осложнений")

            if not DEFAULT_MODEL_PATH.exists():
                st.warning(f"Файл модели не найден: `{DEFAULT_MODEL_PATH}`")
            else:
                try:
                    proba_df, pred_df, thresholds = predict_complications(
                        features,
                        model_path=DEFAULT_MODEL_PATH,
                    )

                    result_df = format_predictions(proba_df, pred_df, thresholds)

                    result_display_df = result_df.copy()
                    result_display_df["Вероятность"] = result_display_df["Вероятность"].map(
                        lambda x: f"{x:.1%}"
                    )

                    st.dataframe(
                        result_display_df,
                        width="stretch",
                        hide_index=True,
                    )

                    st.download_button(
                        label="Скачать прогноз CSV",
                        data=result_display_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="complications_prediction.csv",
                        mime="text/csv",
                    )

                    st.divider()
                    st.subheader("SHAP-интерпретация прогноза")

                    with st.spinner("Расчёт SHAP-интерпретации..."):
                        shap_explanations = get_shap_interpretation(
                            features=features,
                            model_path=DEFAULT_MODEL_PATH,
                            top_k=5,
                            only_predicted=False,
                        )

                    shap_rows = []

                    for _, pred_row in result_df.iterrows():
                        label = pred_row["Код"]
                        label_name = pred_row["Осложнение"]
                        probability = float(pred_row["Вероятность"])
                        prediction = pred_row["Предсказание"]

                        for rank, item in enumerate(shap_explanations.get(label, []), start=1):
                            shap_rows.append(
                                {
                                    "Код": label,
                                    "Осложнение": label_name,
                                    "Предсказание": prediction,
                                    "Вероятность": probability,
                                    "Ранг": rank,
                                    "Признак": item["feature"],
                                    "SHAP value": item["shap_value"],
                                    "Влияние": item["impact"],
                                }
                            )

                    import pandas as pd

                    shap_df = pd.DataFrame(shap_rows)

                    if shap_df.empty:
                        st.info("SHAP-признаки не найдены для текущего прогноза.")
                    else:
                        shap_display_df = shap_df.copy()
                        shap_display_df["Вероятность"] = shap_display_df["Вероятность"].map(
                            lambda x: f"{x:.1%}"
                        )
                        shap_display_df["SHAP value"] = shap_display_df["SHAP value"].map(
                            lambda x: round(float(x), 6)
                        )

                        st.dataframe(
                            shap_display_df,
                            width="stretch",
                            hide_index=True,
                        )

                        st.download_button(
                            label="Скачать SHAP CSV",
                            data=shap_display_df.to_csv(index=False).encode("utf-8-sig"),
                            file_name="shap_interpretation.csv",
                            mime="text/csv",
                        )

                        st.divider()
                        st.subheader("Топ признаков по каждой метке")

                        for label in result_df["Код"].tolist():
                            label_row = result_df[result_df["Код"] == label].iloc[0]
                            label_name = label_row["Осложнение"]
                            probability = float(label_row["Вероятность"])
                            prediction = label_row["Предсказание"]

                            label_shap_df = shap_df[shap_df["Код"] == label].copy()

                            if label_shap_df.empty:
                                continue

                            with st.expander(
                                    f"{label_name} — {probability:.1%}, прогноз: {prediction}",
                                    expanded=prediction == "Да",
                            ):
                                label_shap_display = label_shap_df[
                                    ["Ранг", "Признак", "SHAP value", "Влияние"]
                                ].copy()

                                label_shap_display["SHAP value"] = label_shap_display[
                                    "SHAP value"
                                ].map(lambda x: round(float(x), 6))

                                st.dataframe(
                                    label_shap_display,
                                    width="stretch",
                                    hide_index=True,
                                )

                                chart_df = label_shap_df[["Признак", "SHAP value"]].copy()
                                chart_df = chart_df.set_index("Признак")

                                st.bar_chart(chart_df)

                                st.caption(
                                    "Положительный SHAP value увеличивает риск по данной метке, "
                                    "отрицательный — снижает."
                                )

                except Exception as exc:
                    st.error("Не удалось выполнить предсказание осложнений или SHAP-интерпретацию.")
                    st.exception(exc)

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
