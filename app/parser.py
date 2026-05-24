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
