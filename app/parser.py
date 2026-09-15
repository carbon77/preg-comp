from __future__ import annotations

import re
from typing import Any

import pandas as pd


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
        # ---------------------------------------------------------
        # Демография
        # ---------------------------------------------------------
        "age_final": find_number(
            text,
            [
                r"возраст[^0-9]{0,30}(?P<value>\d{1,2}(?:[\.,]\d+)?)",
                r"age[^0-9]{0,30}(?P<value>\d{1,2}(?:[\.,]\d+)?)",
            ],
        ),

        "height_final": find_number(
            text,
            [
                r"рост[^0-9]{0,30}(?P<value>\d{2,3})",
                r"height[^0-9]{0,30}(?P<value>\d{2,3})",
            ],
        ),

        "weight_final": find_number(
            text,
            [
                r"вес[^0-9]{0,30}(?P<value>\d{2,3}(?:[\.,]\d+)?)",
                r"weight[^0-9]{0,30}(?P<value>\d{2,3}(?:[\.,]\d+)?)",
            ],
        ),

        "bmi_final": find_number(
            text,
            [
                r"имт[^0-9]{0,30}(?P<value>\d{1,2}(?:[\.,]\d+)?)",
                r"bmi[^0-9]{0,30}(?P<value>\d{1,2}(?:[\.,]\d+)?)",
            ],
        ),

        # ---------------------------------------------------------
        # Гестационный возраст
        # ---------------------------------------------------------
        "screening_weeks": find_number(
            text,
            [
                r"срок[^0-9]{0,30}(?P<value>\d{1,2})\s*(?:нед|weeks)",
                r"гестационн\w*\s*возраст[^0-9]{0,30}(?P<value>\d{1,2})\s*(?:нед|weeks)",
            ],
        ),

        "screening_days": find_number(
            text,
            [
                r"срок[^\n]{0,80}\d{1,2}\s*(?:нед|weeks)[^0-9]{0,20}(?P<value>\d)\s*(?:дн|days)",
                r"гестационн\w*\s*возраст[^\n]{0,80}\d{1,2}\s*(?:нед|weeks)[^0-9]{0,20}(?P<value>\d)\s*(?:дн|days)",
            ],
        ),

        # ---------------------------------------------------------
        # Биометрия плода
        # ---------------------------------------------------------
        "crl": find_number(
            text,
            [
                r"(?:копчико-теменной\s*размер)[^0-9]*(?P<value>\d{1,3}(?:[\.,]\d+)?)"
            ],
        ),

        "nt": find_number(
            text,
            [
                r"(?:толщина воротникового\s*пространства)[^0-9]{0,30}(?P<value>\d{1,2}(?:[\.,]\d+)?)"
            ],
        ),

        "fhr": find_number(
            text,
            [
                r"(?:чсс|fhr)[^0-9]{0,30}(?P<value>\d{2,3})",
            ],
        ),

        "bpd": find_number(
            text,
            [
                r"(?:бипариетальный размер)[^0-9]*(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),

        "hc": find_number(
            text,
            [
                r"(?:окружность\s*головы)[^0-9]{0,20}(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),

        "ac": find_number(
            text,
            [
                r"(?:окружность\s*живота)[^0-9]{0,20}(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),

        "fl": find_number(
            text,
            [
                r"(?:дб|fl|длина\s*бедра)[^0-9]{0,20}(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),

        # ---------------------------------------------------------
        # Носовая кость
        # ---------------------------------------------------------
        "nasal_bone": find_text_value(
            text,
            [
                r"(?:кости\s*носа)\s*(?P<value>[^\n\r]{2,80})",
            ],
        ),

        # ---------------------------------------------------------
        # Венозный проток
        # ---------------------------------------------------------
        "dvpi": find_number(
            text,
            [
                r"венозный\s*проток,\s*пульсационный\s*индекс\s*(?P<value>\d+(?:[\.,]\d+)?)"
            ],
        ),

        # ---------------------------------------------------------
        # Маточные артерии
        # ---------------------------------------------------------
        "uapi_mean": find_number(
            text,
            [
                r"маточные\s*артерии\s*,\s*пульсационный\s*индекс\s*:\s*(?P<value>\d+(?:[\.,]\d+)?)"
            ],
        ),

        "uapi_mom": find_number(
            text,
            [
                r"маточные\s*артерии\s*,.*эквивалентно\s*(?P<value>\d+(?:[\.,]\d+)?)"
            ],
        ),

        # ---------------------------------------------------------
        # Артериальное давление
        # ---------------------------------------------------------
        "map": find_number(
            text,
            [
                r"артериальное\s*давление\s*,\s*среднее\s*:\s*(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),

        "map_mom": find_number(
            text,
            [
                r"артериальное\s*давление\s*,.*эквивалентно\s*(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),

        # ---------------------------------------------------------
        # Шейка матки
        # ---------------------------------------------------------
        "cervix_length": find_number(
            text,
            [
                r"длина\s*шейки\s*:\s*(?P<value>\d+(?:[\.,]\d+)?)"
            ],
        ),

        # ---------------------------------------------------------
        # Биохимия
        # ---------------------------------------------------------
        "pappa": find_number(
            text,
            [
                r"papp[\-\s]?a(?![^\n]{0,20}(?:mom|мом))[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)",
                r"папп[\-\s]?а(?![^\n]{0,20}(?:mom|мом))[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),

        "pappa_mom": find_number(
            text,
            [
                r"papp[\-\s]?a[^\n]{0,40}(?:mom|мом)[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)",
                r"папп[\-\s]?а[^\n]{0,40}(?:mom|мом)[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),

        "fbhcg": find_number(
            text,
            [
                r"(?:fbhcg|free beta|свободн\w*\s*β?[\-\s]?хгч|бета[\-\s]?хгч)(?![^\n]{0,20}(?:mom|мом))[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),

        "fbhcg_mom": find_number(
            text,
            [
                r"(?:fbhcg|free beta|свободн\w*\s*β?[\-\s]?хгч|бета[\-\s]?хгч)[^\n]{0,40}(?:mom|мом)[^0-9]{0,30}(?P<value>\d+(?:[\.,]\d+)?)",
            ],
        ),
    }

    if features["bmi_final"] is None and features["height_final"] and features["weight_final"]:
        height_m = features["height_final"] / 100
        features["bmi_final"] = round(features["weight_final"] / (height_m ** 2), 2)

    if features["screening_weeks"] is not None:
        days = features["screening_days"] or 0
        features["screening_ga_weeks"] = round(features["screening_weeks"] + days / 7, 2)
    else:
        features["screening_ga_weeks"] = None

    return features


def features_to_dataframe(features: dict[str, Any]) -> pd.DataFrame:
    rows = []
    names = {
        "age_final": "Возраст",
        "height_final": "Рост, см",
        "weight_final": "Вес, кг",
        "bmi_final": "ИМТ",
        "screening_weeks": "Срок, недели",
        "screening_days": "Срок, дни",
        "screening_ga_weeks": "Срок, недель десятичный",
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
        "dvpi": "DV PI",
        "uapi_mom": "UAPI MoM",
        "map": "Среднее АД (MAP)",
        "map_mom": "MAP MoM",
        "cervix_length": "Длина шейки матки, мм",
        "bpd": "БПР, мм",
        "hc": "ОГ, мм",
        "ac": "ОЖ, мм",
        "fl": "Длина бедра, мм",
    }

    for key, title in names.items():
        value = features.get(key)
        if value is None:
            value_str = "не найдено"
        elif isinstance(value, float):
            value_str = f"{value:.3f}".rstrip("0").rstrip(".")
        else:
            value_str = str(value)
        rows.append({"Признак": title, "Код признака": key, "Значение": value_str})

    return pd.DataFrame(rows).astype(str)
