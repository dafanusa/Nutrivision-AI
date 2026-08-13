from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process


# ============================================================
# KONFIGURASI NAMA KOLOM
# ============================================================

FOOD_NAME_ALIASES = [
    "food",
    "food_name",
    "name",
    "nama",
    "nama_makanan",
    "menu",
]


NUTRIENT_ALIASES = {
    "calories": [
        "calories",
        "calorie",
        "energy",
        "energy_kcal",
        "kalori",
        "energi",
    ],
    "protein": [
        "protein",
        "proteins",
        "protein_g",
    ],
    "fat": [
        "fat",
        "total_fat",
        "lemak",
        "fat_g",
    ],
    "carbohydrate": [
        "carbohydrate",
        "carbohydrates",
        "carbs",
        "karbohidrat",
        "carbohydrate_g",
    ],
    "iron": [
        "iron",
        "iron_mg",
        "zat_besi",
        "zatbesi",
        "fe",
    ],
    "calcium": [
        "calcium",
        "calcium_mg",
        "kalsium",
        "ca",
    ],
    "vitamin_a": [
        "vitamin_a",
        "vitamin_a_mcg",
        "vitamin_a_ug",
        "vita",
    ],
    "vitamin_c": [
        "vitamin_c",
        "vitamin_c_mg",
        "vitc",
    ],
}


# ============================================================
# PEMETAAN NUTRIEN KE KOLOM AKG
# ============================================================

NUTRIENT_REQUIREMENT_MAP = {
    "calories": "energy_kcal",
    "protein": "protein_g",
    "fat": "fat_g",
    "carbohydrate": "carbohydrate_g",
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


# ============================================================
# ALIAS LABEL MODEL -> NAMA MAKANAN PADA KNOWLEDGE BASE
# ============================================================
#
# Alias ini hanya membantu proses pencarian nutrisi.
# Label model dan indeks kelas TIDAK diubah.
#
# Tambahkan pasangan baru di sini jika ditemukan kelas model
# yang penamaannya berbeda dengan knowledge base.
# ============================================================

FOOD_NUTRITION_ALIASES = {
    "fried_rice": [
        "nasi goreng",
        "fried rice",
    ],
    "nasi_kuning": [
        "nasi kuning",
        "yellow rice",
    ],
    "yellow_rice": [
        "nasi kuning",
        "yellow rice",
    ],
    "chicken_soto": [
        "soto ayam",
        "chicken soto",
    ],
    "chicken_noodle": [
        "mie ayam",
        "mi ayam",
        "chicken noodle",
    ],
    "chicken_porridge": [
        "bubur ayam",
        "chicken porridge",
    ],
    "french_fries": [
        "kentang goreng",
        "french fries",
    ],
    "chicken_wings": [
        "sayap ayam",
        "chicken wings",
    ],
    "meatballs": [
        "bakso",
        "bakso sapi",
        "meatballs",
    ],
    "tempeh": [
        "tempe",
        "tempeh",
    ],
    "gado_gado": [
        "gado gado",
        "gado-gado",
    ],
}


# Threshold fuzzy dibuat lebih ketat dari versi lama (72)
# untuk menurunkan risiko salah mapping antar makanan yang berbeda.
DEFAULT_FUZZY_THRESHOLD = 82


# ============================================================
# NORMALISASI
# ============================================================

def _strip_accents(value: str) -> str:
    """
    Menghapus aksen Unicode agar pencocokan lebih stabil.
    """
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(
        char
        for char in normalized
        if not unicodedata.combining(char)
    )


def normalize_colname(value: Any) -> str:
    """
    Normalisasi nama kolom menjadi snake_case sederhana.
    """
    value = _strip_accents(str(value).strip().lower())
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengembalikan salinan DataFrame dengan nama kolom ternormalisasi.
    """
    out = df.copy()
    out.columns = [
        normalize_colname(column)
        for column in out.columns
    ]
    return out


def clean_food_name(value: Any) -> str | None:
    """
    Normalisasi nama makanan untuk matching internal.

    Contoh:
        "Nasi Goreng" -> "nasi_goreng"
        "gado-gado"   -> "gado_gado"
    """
    if pd.isna(value):
        return None

    value = _strip_accents(
        str(value).strip().lower()
    )

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    value = value.strip("_")

    return value or None


# ============================================================
# PENCARIAN KOLOM
# ============================================================

def find_column(
    df: pd.DataFrame,
    aliases: list[str],
) -> str | None:
    """
    Mencari kolom dengan urutan prioritas:
    1. exact normalized match
    2. containment match sebagai fallback
    """
    normalized = {
        normalize_colname(column): column
        for column in df.columns
    }

    # Exact match lebih aman.
    for alias in aliases:
        key = normalize_colname(alias)

        if key in normalized:
            return normalized[key]

    # Fallback untuk nama kolom seperti:
    # "protein_g_per_serving" atau "energy_kcal_100g".
    for alias in aliases:
        key = normalize_colname(alias)

        for normalized_col, original in normalized.items():
            if (
                key in normalized_col
                or normalized_col in key
            ):
                return original

    return None


# ============================================================
# KNOWLEDGE BASE
# ============================================================

def build_nutrition_kb(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str, dict[str, str]]:
    """
    Menyiapkan knowledge base nutrisi.

    Return:
        kb
        food_col
        nutrient_cols
    """
    if df is None or df.empty:
        raise ValueError(
            "Dataset nutrisi kosong."
        )

    df = normalize_columns(df)

    food_col = find_column(
        df,
        FOOD_NAME_ALIASES,
    )

    if food_col is None:
        raise ValueError(
            "Kolom nama makanan tidak ditemukan. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    nutrient_cols: dict[str, str] = {}
    used_columns = set()

    for nutrient, aliases in NUTRIENT_ALIASES.items():
        column = find_column(
            df,
            aliases,
        )

        if (
            column
            and column != food_col
            and column not in used_columns
        ):
            nutrient_cols[nutrient] = column
            used_columns.add(column)

    if not nutrient_cols:
        raise ValueError(
            "Tidak ada kolom nutrisi yang berhasil dikenali."
        )

    kb = df.copy()

    kb["_food_key"] = (
        kb[food_col]
        .map(clean_food_name)
    )

    kb = kb[
        kb["_food_key"].notna()
        & (kb["_food_key"] != "")
    ].copy()

    if kb.empty:
        raise ValueError(
            "Tidak ada nama makanan valid pada knowledge base."
        )

    return kb, food_col, nutrient_cols


# ============================================================
# FOOD MATCHING
# ============================================================

def _candidate_queries(
    food_name: str,
) -> list[str]:
    """
    Membuat daftar query kandidat:
    label asli + alias yang sudah ditentukan.
    """
    original = clean_food_name(food_name)

    if not original:
        return []

    queries = [original]

    alias_values = (
        FOOD_NUTRITION_ALIASES
        .get(original, [])
    )

    for alias in alias_values:
        alias_key = clean_food_name(alias)

        if (
            alias_key
            and alias_key not in queries
        ):
            queries.append(alias_key)

    return queries


def _find_best_food_match(
    food_name: str,
    kb: pd.DataFrame,
    threshold: int,
) -> dict:
    """
    Matching dengan urutan:
    1. exact label
    2. exact alias
    3. fuzzy label/alias

    Mengembalikan metadata matching agar audit lebih transparan.
    """
    choices = (
        kb["_food_key"]
        .dropna()
        .astype(str)
        .tolist()
    )

    if not choices:
        return {
            "status": "not_found",
            "query": food_name,
        }

    choice_set = set(choices)

    queries = _candidate_queries(
        food_name
    )

    if not queries:
        return {
            "status": "not_found",
            "query": food_name,
        }

    # 1. Exact label / alias.
    for index, query in enumerate(queries):
        if query in choice_set:
            return {
                "status": "ok",
                "match": query,
                "score": 100.0,
                "method": (
                    "exact"
                    if index == 0
                    else "alias_exact"
                ),
                "matched_query": query,
            }

    # 2. Fuzzy terhadap seluruh query alternatif.
    best_result = None

    for query in queries:
        found = process.extractOne(
            query,
            choices,
            scorer=fuzz.token_sort_ratio,
        )

        if not found:
            continue

        match, score, _ = found

        candidate = {
            "status": "ok",
            "match": match,
            "score": float(score),
            "method": "fuzzy",
            "matched_query": query,
        }

        if (
            best_result is None
            or candidate["score"] > best_result["score"]
        ):
            best_result = candidate

    if best_result is None:
        return {
            "status": "not_found",
            "query": food_name,
        }

    if best_result["score"] < threshold:
        return {
            "status": "below_threshold",
            "query": food_name,
            "best_match": best_result["match"],
            "similarity": best_result["score"],
            "matched_query": best_result["matched_query"],
            "threshold": threshold,
        }

    return best_result


def nutrient_lookup(
    food_name: str,
    kb: pd.DataFrame,
    food_col: str,
    nutrient_cols: dict[str, str],
    threshold: int = DEFAULT_FUZZY_THRESHOLD,
) -> dict:
    """
    Mengambil profil nutrisi makanan.

    Matching dilakukan secara konservatif:
    exact -> alias -> fuzzy.

    Return tetap kompatibel dengan inference.py lama,
    dengan metadata tambahan:
        match_method
        matched_query
        threshold
    """
    if kb is None or kb.empty:
        return {
            "status": "nutrition_data_not_found",
            "query": food_name,
            "reason": "knowledge_base_empty",
        }

    match_info = _find_best_food_match(
        food_name=food_name,
        kb=kb,
        threshold=int(threshold),
    )

    if match_info.get("status") != "ok":
        return {
            "status": "nutrition_data_not_found",
            "query": food_name,
            "best_match": match_info.get("best_match"),
            "similarity": match_info.get("similarity"),
            "matched_query": match_info.get("matched_query"),
            "threshold": int(threshold),
            "reason": match_info.get("status"),
        }

    match = match_info["match"]

    rows = kb[
        kb["_food_key"] == match
    ]

    if rows.empty:
        return {
            "status": "nutrition_data_not_found",
            "query": food_name,
            "best_match": match,
            "reason": "matched_row_missing",
        }

    # Jika ada duplikasi nama makanan, gunakan baris pertama
    # agar tidak melakukan agregasi tanpa dasar metodologis.
    row = rows.iloc[0]

    nutrients = {}

    for nutrient, column in nutrient_cols.items():
        if column not in row.index:
            continue

        value = pd.to_numeric(
            pd.Series([row[column]]),
            errors="coerce",
        ).iloc[0]

        if not pd.isna(value):
            nutrients[nutrient] = float(value)

    if not nutrients:
        return {
            "status": "nutrition_data_not_found",
            "query": food_name,
            "matched_food": str(row[food_col]),
            "similarity": float(match_info["score"]),
            "reason": "nutrient_values_empty",
        }

    return {
        "status": "ok",
        "query": food_name,
        "matched_food": str(row[food_col]),
        "similarity": float(match_info["score"]),
        "match_method": match_info["method"],
        "matched_query": match_info["matched_query"],
        "threshold": int(threshold),
        "nutrients": nutrients,
    }


# ============================================================
# AKG BERDASARKAN UMUR
# ============================================================

def get_requirement(
    age_months: int,
    requirements: pd.DataFrame,
) -> dict:
    """
    Memilih satu baris AKG yang mencakup umur anak.
    """
    if requirements is None or requirements.empty:
        raise ValueError(
            "Tabel AKG kosong."
        )

    required_columns = {
        "age_min_month",
        "age_max_month",
    }

    missing = (
        required_columns
        - set(requirements.columns)
    )

    if missing:
        raise ValueError(
            "Tabel AKG tidak memiliki kolom wajib: "
            + ", ".join(sorted(missing))
        )

    try:
        age_months = int(age_months)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Umur anak harus berupa jumlah bulan yang valid."
        ) from exc

    selected = requirements[
        (
            pd.to_numeric(
                requirements["age_min_month"],
                errors="coerce",
            )
            <= age_months
        )
        & (
            pd.to_numeric(
                requirements["age_max_month"],
                errors="coerce",
            )
            >= age_months
        )
    ]

    if selected.empty:
        raise ValueError(
            f"Umur {age_months} bulan belum tercakup "
            "pada tabel AKG aplikasi."
        )

    # Jika overlap terjadi, ambil interval paling sempit.
    if len(selected) > 1:
        selected = selected.copy()

        selected["_age_width"] = (
            pd.to_numeric(
                selected["age_max_month"],
                errors="coerce",
            )
            - pd.to_numeric(
                selected["age_min_month"],
                errors="coerce",
            )
        )

        selected = selected.sort_values(
            "_age_width",
            ascending=True,
        )

    return selected.iloc[0].to_dict()


# ============================================================
# ANALISIS KONTRIBUSI NUTRISI
# ============================================================

def contribution_status(
    percent: float,
) -> str:
    """
    Label internal untuk membantu prioritisasi dashboard.

    Batas 25% dan 50% adalah aturan aplikasi,
    bukan kategori diagnosis klinis.
    """
    if percent < 25:
        return "Prioritas tinggi"

    if percent < 50:
        return "Prioritas sedang"

    return "Kontribusi relatif baik"


def analyze_nutrient_contribution(
    food_nutrients: dict,
    requirement: dict,
) -> dict:
    """
    Menghitung kontribusi nilai nutrisi referensi makanan
    terhadap AKG kelompok umur.

    PENTING:
    Hasil ini bukan estimasi konsumsi aktual anak apabila
    berat/porsi makanan pada foto belum diketahui.
    """
    if not food_nutrients or not requirement:
        return {}

    results = {}

    for nutrient, req_col in NUTRIENT_REQUIREMENT_MAP.items():
        food_value = food_nutrients.get(
            nutrient
        )

        req_value = requirement.get(
            req_col
        )

        if (
            food_value is None
            or req_value is None
        ):
            continue

        try:
            food_value = float(food_value)
            req_value = float(req_value)
        except (TypeError, ValueError):
            continue

        if req_value <= 0:
            continue

        # Nilai negatif tidak masuk akal untuk kandungan nutrisi.
        if food_value < 0:
            continue

        contribution = (
            food_value
            / req_value
            * 100
        )

        contribution_clamped = min(
            max(contribution, 0.0),
            100.0,
        )

        gap = max(
            100.0 - contribution_clamped,
            0.0,
        )

        results[nutrient] = {
            "food_amount": round(
                food_value,
                4,
            ),
            "requirement": round(
                req_value,
                4,
            ),
            "contribution_percent": round(
                contribution,
                2,
            ),
            "gap_percent": round(
                gap,
                2,
            ),
            "status": contribution_status(
                contribution
            ),
            "unit": NUTRIENT_UNITS.get(
                nutrient,
                "",
            ),
            "display_name": DISPLAY_NAMES.get(
                nutrient,
                nutrient,
            ),
        }

    return results


# ============================================================
# TABEL KANDIDAT REKOMENDASI
# ============================================================

def build_recommendation_table(
    kb: pd.DataFrame,
    food_col: str,
    nutrient_cols: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Menyiapkan tabel kandidat untuk recommendation engine.

    Hanya kolom nutrisi yang benar-benar tersedia pada database
    yang diteruskan. Recommendation engine kemudian memakai irisan
    kolom tersebut dengan nutrient gap hasil analisis.
    """
    if kb is None or kb.empty:
        return (
            pd.DataFrame(),
            {},
        )

    if food_col not in kb.columns:
        raise ValueError(
            f"Kolom makanan '{food_col}' tidak ditemukan "
            "pada knowledge base."
        )

    columns = {
        nutrient: column
        for nutrient, column
        in nutrient_cols.items()
        if column in kb.columns
    }

    if not columns:
        return (
            pd.DataFrame(),
            {},
        )

    base_columns = [
        food_col,
        "_food_key",
    ]

    rec = kb[
        base_columns
        + list(columns.values())
    ].copy()

    for column in columns.values():
        rec[column] = pd.to_numeric(
            rec[column],
            errors="coerce",
        )

    rec = rec.dropna(
        subset=list(columns.values()),
        how="all",
    )

    rec = rec[
        rec["_food_key"].notna()
        & (rec["_food_key"] != "")
    ].copy()

    rec = rec.reset_index(
        drop=True
    )

    return rec, columns


# ============================================================
# DIAGNOSTIK OPSIONAL
# ============================================================

def nutrition_coverage_summary(
    nutrient_cols: dict[str, str],
) -> dict:
    """
    Ringkasan nutrien yang tersedia pada sebuah knowledge base.
    Berguna untuk audit/debugging recommendation engine.
    """
    available = [
        nutrient
        for nutrient in NUTRIENT_REQUIREMENT_MAP
        if nutrient in nutrient_cols
    ]

    missing = [
        nutrient
        for nutrient in NUTRIENT_REQUIREMENT_MAP
        if nutrient not in nutrient_cols
    ]

    return {
        "available": available,
        "missing": missing,
        "available_count": len(available),
        "expected_count": len(NUTRIENT_REQUIREMENT_MAP),
    }
