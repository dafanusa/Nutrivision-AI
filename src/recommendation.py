from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# METODE
# ============================================================

RECOMMENDATION_ENGINE_VERSION = "3.0"
RECOMMENDATION_METHOD_NAME = (
    "Age-First Food Suitability + Nutrient Gap Ranking"
)

# Umur bukan lagi sekadar bonus skor.
# Kandidat HARUS lolos filter umur/kelayakan terlebih dahulu.
# Bobot berikut hanya dipakai SETELAH kandidat lolos age gate.
RECOMMENDATION_SCORE_WEIGHTS = {
    "nutrient_gap_fit": 0.70,
    "coverage": 0.15,
    "food_quality": 0.15,
}


# ============================================================
# LABEL
# ============================================================

NUTRIENT_LABELS = {
    "calories": "energi",
    "protein": "protein",
    "fat": "lemak",
    "carbohydrate": "karbohidrat",
    "iron": "zat besi",
    "calcium": "kalsium",
    "vitamin_a": "vitamin A",
    "vitamin_c": "vitamin C",
}

CATEGORY_LABELS = {
    "animal_protein": "protein hewani",
    "plant_protein": "protein nabati",
    "fruit_vegetable": "sayur/buah",
    "staple": "sumber karbohidrat",
    "mixed_meal": "menu keluarga",
    "dairy": "produk susu",
}


# ============================================================
# FILTER NEGATIF
# ============================================================

# Bahan yang bukan rekomendasi menu.
NON_MENU_KEYWORDS = [
    "minyak", "oil",
    "margarin", "mentega", "butter",
    "bumbu", "saus", "sauce", "kecap",
    "sirup", "gula", "garam",
    "tepung", "pati", "seasoning",
    "extract", "ekstrak",
    "petis", "terasi", "sambal",
    "mayones", "mayonnaise", "dressing",
    "kaldu",
]

# Produk khusus/serbuk yang tidak digunakan sebagai rekomendasi otomatis.
SPECIAL_PRODUCT_KEYWORDS = [
    "bayi", "baby", "infant",
    "formula", "susu formula",
    "makanan bayi",
    "bubuk", "powder",
]

# Minuman stimulan/minuman manis bukan kandidat rekomendasi makanan.
BEVERAGE_EXCLUDED_KEYWORDS = [
    "kopi", "coffee",
    "teh", "tea",
    "soda",
    "energy drink",
    "minuman energi",
]

# Nama basis data yang menunjukkan fraksi bahan / bentuk yang ambigu.
AMBIGUOUS_INGREDIENT_KEYWORDS = [
    "ampas",
    "bagian yang larut",
    "biji",
    "kulit",
    "dedak",
    "bekatul",
    "bungkil",
    "pulp",
    "residu",
    "residue",
    "havermout",
]

# Bentuk yang belum siap direkomendasikan sebagai makanan anak.
UNREADY_FORM_KEYWORDS = [
    "mentah",
    "raw",
    "kering",
]

# Produk olahan/camilan dikeluarkan dari rekomendasi utama.
UNHEALTHY_OR_PROCESSED_KEYWORDS = [
    "opak", "keripik", "kerupuk", "chips",
    "snack", "wafer", "biskuit", "biscuit",
    "cookies", "cracker", "permen", "candy",
    "dodol", "emping", "rempeyek", "peyek",
    "rengginang", "pilus", "chiki", "sukro",
    "enting", "kacang telur", "kacang atom",
    "kue", "cake", "donat", "donut",
    "cokelat", "coklat",
    "sosis", "sausage", "nugget",
    "kornet", "corned", "abon", "dendeng",
    "instan", "instant",
    "ikan asin", "telur asin",
]

# Whole nuts tidak dipilih otomatis untuk anak <5 tahun.
WHOLE_NUT_KEYWORDS = [
    "kacang tanah",
    "peanut",
    "groundnut",
    "kacang negara",
]


# ============================================================
# POSITIVE FOOD GROUPS
# ============================================================

ANIMAL_PROTEIN_KEYWORDS = [
    "ikan", "fish",
    "ayam", "chicken",
    "daging", "sapi", "beef",
    "kambing", "goat",
    "telur", "egg",
    "hati", "liver",
    "udang", "shrimp",
    "teri", "tuna", "tongkol",
    "bandeng", "lele", "patin",
    "mujair", "gurame", "cakalang",
    "sarden", "sardine",
]

# Tidak menggunakan keyword generik "kacang".
# Itu sebelumnya menyebabkan "kacang panjang biji" salah masuk protein nabati.
PLANT_PROTEIN_KEYWORDS = [
    "tempe", "tempeh",
    "tahu", "tofu",
    "kedelai", "soy",
    "kacang hijau",
    "kacang merah",
    "edamame",
    "lentil",
    "oncom",
]

FRUIT_VEGETABLE_KEYWORDS = [
    "sayur", "vegetable",
    "bayam", "spinach",
    "wortel", "carrot",
    "brokoli", "broccoli",
    "labu", "pumpkin",
    "kangkung",
    "buncis",
    "kacang panjang",
    "tomat", "tomato",
    "pisang", "banana",
    "pepaya", "papaya",
    "mangga", "mango",
    "alpukat", "avocado",
    "jeruk", "orange",
    "apel", "apple",
    "buah", "fruit",
]

STAPLE_KEYWORDS = [
    "nasi", "rice",
    "bubur", "porridge",
    "kentang", "potato",
    "ubi",
    "singkong", "cassava",
    "jagung", "corn",
    "oatmeal",
    "roti", "bread",
    "mi", "mie", "noodle",
]

DAIRY_KEYWORDS = [
    "susu", "milk",
    "yogurt", "yoghurt",
    "keju", "cheese",
]

MIXED_MEAL_KEYWORDS = [
    "sup", "soup",
    "soto",
    "gulai",
    "kari", "curry",
    "tumis",
    "capcay",
    "gado gado",
    "pecel",
    "siomay",
    "bakso",
    "perkedel",
    "pepes",
    "semur",
    "opor",
    "rendang",
]

SOFT_FOOD_KEYWORDS = [
    "bubur", "porridge",
    "puree",
    "lumat",
    "halus",
    "lembut",
    "kukus",
    "rebus",
    "pisang",
    "banana",
    "alpukat",
    "avocado",
    "pepaya",
    "papaya",
    "labu",
    "pumpkin",
    "kentang",
    "potato",
    "ubi",
    "tahu",
    "tofu",
    "telur",
    "egg",
]


# ============================================================
# UTILITAS
# ============================================================

def _normalize_food_name(name: str) -> str:
    value = str(name).lower().strip()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _contains_keyword(
    name: str,
    keywords: list[str],
) -> bool:
    """
    Matching berbasis boundary frasa, bukan substring bebas.
    """
    normalized = _normalize_food_name(name)
    if not normalized:
        return False

    padded = f" {normalized} "

    for keyword in keywords:
        key = _normalize_food_name(keyword)
        if key and f" {key} " in padded:
            return True

    return False


def age_group_label(age_months: int) -> str:
    age = int(age_months)

    if age <= 5:
        return "0–5 bulan"
    if age <= 8:
        return "6–8 bulan"
    if age <= 11:
        return "9–11 bulan"
    if age <= 23:
        return "12–23 bulan"
    if age <= 47:
        return "2–3 tahun"
    if age <= 83:
        return "4–6 tahun"

    return f"{age} bulan"


def _food_category(name: str) -> str | None:
    """
    Positive categorization.

    Kandidat yang tidak cocok dengan kelompok makanan yang dikenali
    TIDAK otomatis disebut menu campuran.
    """
    if _contains_keyword(name, DAIRY_KEYWORDS):
        return "dairy"

    if _contains_keyword(name, ANIMAL_PROTEIN_KEYWORDS):
        return "animal_protein"

    if _contains_keyword(name, PLANT_PROTEIN_KEYWORDS):
        return "plant_protein"

    if _contains_keyword(name, FRUIT_VEGETABLE_KEYWORDS):
        return "fruit_vegetable"

    if _contains_keyword(name, STAPLE_KEYWORDS):
        return "staple"

    if _contains_keyword(name, MIXED_MEAL_KEYWORDS):
        return "mixed_meal"

    return None


def _passes_basic_food_gate(name: str) -> bool:
    """
    Gate tahap pertama:
    kandidat harus berupa makanan yang cukup jelas dan tidak berasal
    dari bahan/produk yang tidak sesuai untuk rekomendasi utama anak.
    """
    normalized = _normalize_food_name(name)

    if not normalized:
        return False

    blocked_groups = [
        NON_MENU_KEYWORDS,
        SPECIAL_PRODUCT_KEYWORDS,
        BEVERAGE_EXCLUDED_KEYWORDS,
        AMBIGUOUS_INGREDIENT_KEYWORDS,
        UNREADY_FORM_KEYWORDS,
        UNHEALTHY_OR_PROCESSED_KEYWORDS,
    ]

    return not any(
        _contains_keyword(normalized, group)
        for group in blocked_groups
    )


def _passes_age_gate(
    name: str,
    age_months: int,
    category: str,
) -> bool:
    """
    AGE GATE = filter keras, bukan bobot skor.

    Kandidat yang gagal di sini tidak boleh "diselamatkan" oleh skor nutrisi.
    """
    age = int(age_months)

    if age <= 5:
        return False

    # Whole nuts tidak dipilih otomatis untuk anak <5 tahun.
    if (
        age < 60
        and _contains_keyword(
            name,
            WHOLE_NUT_KEYWORDS,
        )
    ):
        return False

    allowed_categories = {
        "animal_protein",
        "plant_protein",
        "fruit_vegetable",
        "staple",
        "mixed_meal",
        "dairy",
    }

    if category not in allowed_categories:
        return False

    # 6–8 bulan: hanya kandidat yang cukup memungkinkan disajikan
    # sebagai makanan lembut/lumat atau memang secara nama bersifat lembut.
    if age <= 8:
        if category in {
            "animal_protein",
            "plant_protein",
            "fruit_vegetable",
            "staple",
            "dairy",
        }:
            return (
                _contains_keyword(name, SOFT_FOOD_KEYWORDS)
                or category in {
                    "fruit_vegetable",
                    "dairy",
                }
            )

        return False

    # 9–11 bulan: kelompok makanan padat gizi yang dikenali boleh masuk,
    # dengan penyajian tetap perlu dicincang/lumat sesuai kemampuan anak.
    if age <= 11:
        return True

    # >=12 bulan: makanan keluarga dari kelompok padat gizi yang dikenali.
    return True


def _serving_note(age_months: int) -> str:
    age = int(age_months)

    if age <= 5:
        return (
            "Tidak menampilkan rekomendasi makanan padat."
        )

    if age <= 8:
        return (
            "Sajikan dalam bentuk lumat, halus, atau semi-padat "
            "sesuai kemampuan makan anak."
        )

    if age <= 11:
        return (
            "Sajikan lunak, dicincang halus, atau sebagai finger food "
            "yang sesuai kemampuan anak."
        )

    if age <= 23:
        return (
            "Dapat berupa makanan keluarga dengan tekstur dan ukuran "
            "potongan yang disesuaikan untuk anak."
        )

    return (
        "Dapat berupa makanan keluarga bergizi seimbang dengan "
        "tekstur dan porsi yang sesuai anak."
    )


def _food_quality_score(category: str) -> float:
    """
    Skor kualitas kelompok makanan untuk ranking internal.

    Semua kandidat sudah lolos age gate sebelum skor ini dipakai.
    """
    return float(
        {
            "animal_protein": 100,
            "plant_protein": 96,
            "fruit_vegetable": 96,
            "dairy": 92,
            "mixed_meal": 88,
            "staple": 82,
        }.get(category, 0)
    )


def _priority_nutrient_weights(
    adequacy: dict,
    available: dict[str, str],
):
    gaps: dict[str, float] = {}

    for nutrient in available:
        info = adequacy.get(
            nutrient,
            {},
        )

        try:
            contribution = float(
                info.get(
                    "contribution_percent",
                    0,
                )
            )
        except (TypeError, ValueError):
            contribution = 0.0

        contribution = min(
            max(contribution, 0.0),
            100.0,
        )

        try:
            gap = float(
                info.get(
                    "gap_percent",
                    100.0 - contribution,
                )
            )
        except (TypeError, ValueError):
            gap = 100.0 - contribution

        gap = min(
            max(gap, 0.0),
            100.0,
        )

        if gap > 0:
            gaps[nutrient] = gap

    if not gaps:
        return {}, {}

    total_gap = sum(
        gaps.values()
    )

    weights = {
        nutrient: gap / total_gap
        for nutrient, gap
        in gaps.items()
    }

    return gaps, weights


def _select_diverse_top(
    food_scores: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """
    Diversifikasi ringan setelah scoring.

    Maksimal dua kandidat per kategori pada putaran pertama.
    Urutan akhir tetap descending berdasarkan score.
    """
    if (
        food_scores.empty
        or top_n <= 0
    ):
        return food_scores.head(0)

    ranked = food_scores.sort_values(
        [
            "score",
            "_quality_score",
            "_nutrient_fit",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    selected: list[int] = []
    counts: Counter = Counter()

    for idx, row in ranked.iterrows():
        category = str(
            row.get(
                "_category",
                "",
            )
        )

        if counts[category] >= 2:
            continue

        selected.append(idx)
        counts[category] += 1

        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        for idx in ranked.index:
            if idx in selected:
                continue

            selected.append(idx)

            if len(selected) >= top_n:
                break

    return (
        food_scores.loc[selected]
        .sort_values(
            [
                "score",
                "_quality_score",
                "_nutrient_fit",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
    )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def recommend_foods(
    adequacy: dict,
    rec_df: pd.DataFrame,
    food_col: str,
    rec_nutrient_cols: dict[str, str],
    age_months: int,
    top_n: int = 5,
    current_food: str | None = None,
) -> list[dict]:
    """
    Age-First Food Suitability + Nutrient Gap Ranking.

    TAHAP 1 — HARD AGE/FOOD GATE
    Kandidat harus:
    - merupakan kelompok makanan yang jelas,
    - bukan bahan/non-menu/camilan/produk olahan ambigu,
    - sesuai kelompok umur,
    - sesuai bentuk makanan yang dapat direkomendasikan secara konservatif.

    TAHAP 2 — NUTRIENT RANKING
    Hanya kandidat yang lolos gate yang diberi skor:
    - 70% nutrient-gap fit
    - 15% cakupan nutrien prioritas
    - 15% kualitas kelompok makanan

    Dengan desain ini kandidat yang tidak sesuai umur tidak bisa menang
    hanya karena angka energi/proteinnya tinggi.
    """
    if (
        not adequacy
        or rec_df is None
        or rec_df.empty
    ):
        return []

    age = int(
        age_months
    )

    if age <= 5:
        return []

    if food_col not in rec_df.columns:
        return []

    food_scores = (
        rec_df
        .copy()
    )

    # --------------------------------------------------------
    # 1. BASIC FOOD GATE
    # --------------------------------------------------------
    food_scores = food_scores[
        food_scores[
            food_col
        ].map(
            _passes_basic_food_gate
        )
    ].copy()

    if food_scores.empty:
        return []

    # --------------------------------------------------------
    # 2. POSITIVE CATEGORY
    # --------------------------------------------------------
    food_scores[
        "_category"
    ] = food_scores[
        food_col
    ].map(
        _food_category
    )

    food_scores = food_scores[
        food_scores[
            "_category"
        ].notna()
    ].copy()

    if food_scores.empty:
        return []

    # --------------------------------------------------------
    # 3. HARD AGE GATE
    # --------------------------------------------------------
    age_mask = [
        _passes_age_gate(
            name,
            age,
            category,
        )
        for name, category
        in zip(
            food_scores[
                food_col
            ],
            food_scores[
                "_category"
            ],
        )
    ]

    food_scores = food_scores[
        age_mask
    ].copy()

    if food_scores.empty:
        return []

    # --------------------------------------------------------
    # 4. DEDUPLIKASI + HINDARI MAKANAN YANG SAMA
    # --------------------------------------------------------
    food_scores[
        "_display_key"
    ] = food_scores[
        food_col
    ].astype(
        str
    ).map(
        _normalize_food_name
    )

    food_scores = (
        food_scores
        .drop_duplicates(
            "_display_key"
        )
    )

    if current_food:
        current_key = (
            _normalize_food_name(
                current_food
            )
        )

        if current_key:
            food_scores = food_scores[
                food_scores[
                    "_display_key"
                ] != current_key
            ].copy()

    if food_scores.empty:
        return []

    # --------------------------------------------------------
    # 5. NUTRIEN YANG TERSEDIA DI KEDUA SUMBER
    # --------------------------------------------------------
    available = {
        nutrient: column
        for nutrient, column
        in rec_nutrient_cols.items()
        if (
            column
            in food_scores.columns
            and nutrient
            in adequacy
        )
    }

    if not available:
        return []

    gaps, weights = (
        _priority_nutrient_weights(
            adequacy,
            available,
        )
    )

    if not gaps:
        return []

    # --------------------------------------------------------
    # 6. NUTRIENT-GAP FIT
    # --------------------------------------------------------
    food_scores[
        "_nutrient_fit"
    ] = 0.0

    fill_columns: list[str] = []

    for nutrient, weight in weights.items():
        column = available[
            nutrient
        ]

        values = pd.to_numeric(
            food_scores[
                column
            ],
            errors="coerce",
        ).fillna(
            0.0
        ).clip(
            lower=0.0
        )

        gap = gaps[
            nutrient
        ]

        requirement = (
            adequacy
            .get(
                nutrient,
                {},
            )
            .get(
                "requirement"
            )
        )

        try:
            requirement = float(
                requirement
            )
        except (
            TypeError,
            ValueError,
        ):
            requirement = None

        if (
            requirement is not None
            and np.isfinite(
                requirement
            )
            and requirement > 0
        ):
            candidate_contribution = (
                values
                / requirement
                * 100.0
            )

            useful = (
                candidate_contribution
                .clip(
                    upper=gap
                )
            )

            fill_ratio = (
                useful
                / gap
            ).clip(
                lower=0.0,
                upper=1.0,
            )

        else:
            upper = values.quantile(
                0.95
            )

            if (
                pd.isna(
                    upper
                )
                or upper <= 0
            ):
                fill_ratio = pd.Series(
                    0.0,
                    index=food_scores.index,
                )

            else:
                clipped = values.clip(
                    upper=upper
                )

                lo = float(
                    clipped.min()
                )

                hi = float(
                    clipped.max()
                )

                if hi > lo:
                    fill_ratio = (
                        (
                            clipped - lo
                        )
                        / (
                            hi - lo
                        )
                    ).clip(
                        0.0,
                        1.0,
                    )

                else:
                    fill_ratio = pd.Series(
                        0.0,
                        index=food_scores.index,
                    )

        fill_key = (
            f"_fill_{nutrient}"
        )

        food_scores[
            fill_key
        ] = fill_ratio

        fill_columns.append(
            fill_key
        )

        food_scores[
            "_nutrient_fit"
        ] += (
            weight
            * fill_ratio
        )

    # --------------------------------------------------------
    # 7. COVERAGE
    # --------------------------------------------------------
    food_scores[
        "_coverage_count"
    ] = (
        food_scores[
            fill_columns
        ]
        >= 0.25
    ).sum(
        axis=1
    )

    food_scores[
        "_coverage_ratio"
    ] = (
        food_scores[
            "_coverage_count"
        ]
        / max(
            len(
                fill_columns
            ),
            1,
        )
    )

    # --------------------------------------------------------
    # 8. FOOD QUALITY
    # --------------------------------------------------------
    food_scores[
        "_quality_score"
    ] = food_scores[
        "_category"
    ].map(
        _food_quality_score
    )

    # --------------------------------------------------------
    # 9. FINAL SCORE
    # --------------------------------------------------------
    food_scores[
        "score"
    ] = (
        RECOMMENDATION_SCORE_WEIGHTS[
            "nutrient_gap_fit"
        ]
        * food_scores[
            "_nutrient_fit"
        ]
        * 100.0

        + RECOMMENDATION_SCORE_WEIGHTS[
            "coverage"
        ]
        * food_scores[
            "_coverage_ratio"
        ]
        * 100.0

        + RECOMMENDATION_SCORE_WEIGHTS[
            "food_quality"
        ]
        * food_scores[
            "_quality_score"
        ]
    ).clip(
        lower=0.0,
        upper=100.0,
    )

    # --------------------------------------------------------
    # 10. RANKING + DIVERSITY
    # --------------------------------------------------------
    top = _select_diverse_top(
        food_scores,
        top_n=top_n,
    )

    # --------------------------------------------------------
    # 11. OUTPUT
    # --------------------------------------------------------
    output: list[dict] = []

    group_label = (
        age_group_label(
            age
        )
    )

    serving_note = (
        _serving_note(
            age
        )
    )

    for _, row in top.iterrows():
        support = []

        for nutrient, weight in weights.items():
            fill = float(
                row.get(
                    f"_fill_{nutrient}",
                    0.0,
                )
            )

            support.append(
                (
                    nutrient,
                    weight * fill,
                    fill,
                )
            )

        support.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        top_support_keys = [
            nutrient
            for nutrient, _, fill
            in support[:2]
            if fill >= 0.10
        ]

        top_support_labels = [
            NUTRIENT_LABELS.get(
                nutrient,
                nutrient,
            )
            for nutrient
            in top_support_keys
        ]

        category = str(
            row[
                "_category"
            ]
        )

        category_label = (
            CATEGORY_LABELS
            .get(
                category,
                category,
            )
        )

        if top_support_labels:
            support_text = (
                " dan ".join(
                    top_support_labels
                )
            )

            reason = (
                f"Lolos filter makanan untuk kelompok umur {group_label}; "
                f"diprioritaskan karena membantu melengkapi {support_text} "
                f"dari kelompok {category_label}."
            )

        else:
            reason = (
                f"Lolos filter makanan untuk kelompok umur {group_label} "
                f"dan termasuk kelompok {category_label}."
            )

        output.append(
            {
                "food": str(
                    row[
                        food_col
                    ]
                ),
                "score": round(
                    float(
                        row[
                            "score"
                        ]
                    ),
                    2,
                ),
                "coverage": int(
                    row[
                        "_coverage_count"
                    ]
                ),
                "coverage_total": int(
                    len(
                        fill_columns
                    )
                ),
                "coverage_score": round(
                    float(
                        row[
                            "_coverage_ratio"
                        ]
                        * 100.0
                    ),
                    2,
                ),
                "nutrient_fit_score": round(
                    float(
                        row[
                            "_nutrient_fit"
                        ]
                        * 100.0
                    ),
                    2,
                ),
                # Kompatibilitas UI lama.
                # 100 berarti kandidat SUDAH lolos hard age gate,
                # bukan skor probabilitas umur.
                "age_score": 100.0,
                "age_eligible": True,
                "quality_score": round(
                    float(
                        row[
                            "_quality_score"
                        ]
                    ),
                    2,
                ),
                "category": category,
                "category_label": category_label,
                "age_group": group_label,
                "support_nutrients": top_support_keys,
                "support_nutrient_labels": top_support_labels,
                "used_nutrients": list(
                    weights.keys()
                ),
                "serving_note": serving_note,
                "reason": reason,
                "method": RECOMMENDATION_METHOD_NAME,
                "engine_version": RECOMMENDATION_ENGINE_VERSION,
                "age_gate": "hard_filter",
            }
        )

    return output
