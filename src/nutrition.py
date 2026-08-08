from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

FOOD_NAME_ALIASES = ["food", "food_name", "name", "nama", "nama_makanan", "menu"]

NUTRIENT_ALIASES = {
    "calories": ["calories", "calorie", "energy", "energy_kcal", "kalori", "energi"],
    "protein": ["protein", "proteins", "protein_g"],
    "fat": ["fat", "total_fat", "lemak", "fat_g"],
    "carbohydrate": ["carbohydrate", "carbohydrates", "carbs", "karbohidrat", "carbohydrate_g"],
    "iron": ["iron", "iron_mg", "zat_besi", "fe"],
    "calcium": ["calcium", "calcium_mg", "kalsium", "ca"],
    "vitamin_a": ["vitamin_a", "vitamin_a_mcg", "vita"],
    "vitamin_c": ["vitamin_c", "vitamin_c_mg", "vitc"],
}

NUTRIENT_REQUIREMENT_MAP = {
    "calories": "energy_kcal",
    "protein": "protein_g",
    "iron": "iron_mg",
    "calcium": "calcium_mg",
    "vitamin_a": "vitamin_a_mcg",
    "vitamin_c": "vitamin_c_mg",
}

DISPLAY_NAMES = {
    "calories": "Energi",
    "protein": "Protein",
    "fat": "Lemak",
    "carbohydrate": "Karbohidrat",
    "iron": "Zat Besi",
    "calcium": "Kalsium",
    "vitamin_a": "Vitamin A",
    "vitamin_c": "Vitamin C",
}

NUTRIENT_UNITS = {
    "calories": "kkal",
    "protein": "g",
    "fat": "g",
    "carbohydrate": "g",
    "iron": "mg",
    "calcium": "mg",
    "vitamin_a": "µg RE",
    "vitamin_c": "mg",
}


def normalize_colname(value: Any) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_colname(c) for c in out.columns]
    return out


def find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {normalize_colname(c): c for c in df.columns}

    for alias in aliases:
        key = normalize_colname(alias)
        if key in normalized:
            return normalized[key]

    for alias in aliases:
        key = normalize_colname(alias)
        for ncol, original in normalized.items():
            if key in ncol or ncol in key:
                return original

    return None


def clean_food_name(value: Any) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def build_nutrition_kb(df: pd.DataFrame) -> tuple[pd.DataFrame, str, dict[str, str]]:
    df = normalize_columns(df)
    food_col = find_column(df, FOOD_NAME_ALIASES)
    if food_col is None:
        raise ValueError(f"Kolom nama makanan tidak ditemukan. Kolom: {list(df.columns)}")

    nutrient_cols: dict[str, str] = {}
    used = set()

    for nutrient, aliases in NUTRIENT_ALIASES.items():
        col = find_column(df, aliases)
        if col and col != food_col and col not in used:
            nutrient_cols[nutrient] = col
            used.add(col)

    if not nutrient_cols:
        raise ValueError("Tidak ada kolom nutrisi yang berhasil dikenali.")

    kb = df.copy()
    kb["_food_key"] = kb[food_col].map(clean_food_name)
    kb = kb[kb["_food_key"].notna() & (kb["_food_key"] != "")].copy()

    return kb, food_col, nutrient_cols


def nutrient_lookup(
    food_name: str,
    kb: pd.DataFrame,
    food_col: str,
    nutrient_cols: dict[str, str],
    threshold: int = 72,
) -> dict:
    query = clean_food_name(food_name)
    choices = kb["_food_key"].dropna().astype(str).tolist()

    if not query or not choices:
        return {"status": "nutrition_data_not_found", "query": food_name}

    if query in set(choices):
        match, score = query, 100.0
    else:
        found = process.extractOne(query, choices, scorer=fuzz.token_sort_ratio)
        if not found:
            return {"status": "nutrition_data_not_found", "query": food_name}
        match, score, _ = found

    if score < threshold:
        return {
            "status": "nutrition_data_not_found",
            "query": food_name,
            "best_match": match,
            "similarity": float(score),
        }

    row = kb[kb["_food_key"] == match].iloc[0]
    nutrients = {}

    for key, col in nutrient_cols.items():
        val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        if not pd.isna(val):
            nutrients[key] = float(val)

    return {
        "status": "ok",
        "query": food_name,
        "matched_food": str(row[food_col]),
        "similarity": float(score),
        "nutrients": nutrients,
    }


def get_requirement(age_months: int, requirements: pd.DataFrame) -> dict:
    selected = requirements[
        (requirements["age_min_month"] <= age_months)
        & (requirements["age_max_month"] >= age_months)
    ]

    if selected.empty:
        raise ValueError(
            f"Umur {age_months} bulan belum tercakup pada tabel AKG aplikasi."
        )
    return selected.iloc[0].to_dict()


def contribution_status(percent: float) -> str:
    if percent < 25:
        return "Prioritas tinggi"
    if percent < 50:
        return "Prioritas sedang"
    return "Kontribusi relatif baik"


def analyze_nutrient_contribution(food_nutrients: dict, requirement: dict) -> dict:
    results = {}

    for nutrient, req_col in NUTRIENT_REQUIREMENT_MAP.items():
        food_value = food_nutrients.get(nutrient)
        req_value = requirement.get(req_col)

        if food_value is None or req_value is None:
            continue

        try:
            food_value = float(food_value)
            req_value = float(req_value)
        except (TypeError, ValueError):
            continue

        if req_value <= 0:
            continue

        pct = food_value / req_value * 100

        results[nutrient] = {
            "food_amount": food_value,
            "requirement": req_value,
            "contribution_percent": round(pct, 2),
            "gap_percent": round(max(100 - min(max(pct, 0), 100), 0), 2),
            "status": contribution_status(pct),
        }

    return results


def build_recommendation_table(
    kb: pd.DataFrame,
    food_col: str,
    nutrient_cols: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    cols = {k: v for k, v in nutrient_cols.items() if v in kb.columns}
    rec = kb[[food_col, "_food_key"] + list(cols.values())].copy()

    for col in cols.values():
        rec[col] = pd.to_numeric(rec[col], errors="coerce")

    rec = rec.dropna(subset=list(cols.values()), how="all").reset_index(drop=True)
    return rec, cols