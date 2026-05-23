# OCR PDF скрининга

Простое Streamlit-приложение для:
- загрузки PDF скрининга;
- OCR распознавания текста;
- извлечения ключевых признаков;
- экспорта результатов в CSV/JSON.

## Запуск

```bash
uv run streamlit run main.py
```

## Структура

- `main.py` — точка входа.
- `app/config.py` — настройки приложения и OCR.
- `app/ocr.py` — OCR и обработка PDF.
- `app/parser.py` — парсинг признаков из текста.
- `app/ui.py` — Streamlit UI.
