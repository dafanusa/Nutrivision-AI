from __future__ import annotations

import numpy as np
import pandas as pd

EXCLUDED_FOOD_KEYWORDS = [
    "minyak", "oil", "lemak", "margarin", "mentega", "butter",
    "bumbu", "saus", "sauce", "kecap", "sirup", "gula", "garam",
    "tepung", "pati", "seasoning", "extract", "ekstrak",
]


def _is_menu_candidate(name: str) -> bool:
    name = str(name).lower().strip()
    if not name:
        return False
    return not any(keyword in name for keyword in EXCLUDED_FOOD_KEYWORDS)


def recommend_foods(
    adequacy: dict,
    rec_df: pd.DataFrame,
    food_col: str,
    rec_nutrient_cols: dict[str, str],
    top_n: int = 5,
) -> list[dict]:
    """
    Ranking kandidat makanan berdasarkan nutrient gap.

    Catatan:
    Dataset Indonesian Food dari notebook hanya menyediakan makro nutrisi utama.
    Karena itu recommendation engine hanya memakai nutrient yang overlap dengan
    kolom dataset tersebut. Skor adalah ranking internal, bukan skor kesehatan.
    """
    if not adequacy or rec_df.empty:
        return []

    food_scores = rec_df.copy()
    food_scores = food_scores[food_scores[food_col].map(_is_menu_candidate)].copy()

    available = {
        k: v
        for k, v in rec_nutrient_cols.items()
        if v in food_scores.columns and k in adequacy
    }

    if not available:
        return []

    gaps = {}
    for nutrient in available:
        pct = float(adequacy[nutrient].get("contribution_percent", 0))
        pct = min(max(pct, 0), 100)
        gap = 100 - pct
        if gap > 0:
            gaps[nutrient] = gap

    if not gaps:
        gaps = {k: 1.0 for k in available}

    total_gap = sum(gaps.values())
    weights = {k: v / total_gap for k, v in gaps.items()}

    food_scores["_score"] = 0.0
    normalized_cols = []

    for nutrient, weight in weights.items():
        col = available[nutrient]
        values = pd.to_numeric(food_scores[col], errors="coerce").fillna(0)

        upper = values.quantile(0.95)
        if pd.isna(upper) or upper <= 0:
            norm = pd.Series(0.0, index=food_scores.index)
        else:
            clipped = values.clip(lower=0, upper=upper)
            lo, hi = clipped.min(), clipped.max()
            norm = (
                (clipped - lo) / (hi - lo)
                if hi > lo
                else pd.Series(0.0, index=food_scores.index)
            )

        key = f"_norm_{nutrient}"
        food_scores[key] = norm
        normalized_cols.append(key)
        food_scores["_score"] += weight * norm

    if normalized_cols:
        food_scores["_coverage"] = (food_scores[normalized_cols] >= 0.5).sum(axis=1)
        food_scores["_score"] += (
            0.10 * food_scores["_coverage"] / max(len(normalized_cols), 1)
        )
    else:
        food_scores["_coverage"] = 0

    lo, hi = food_scores["_score"].min(), food_scores["_score"].max()
    if hi > lo:
        food_scores["score"] = (food_scores["_score"] - lo) / (hi - lo) * 100
    else:
        food_scores["score"] = 0.0

    food_scores["_display_key"] = food_scores[food_col].astype(str).str.lower().str.strip()
    top = (
        food_scores.sort_values("score", ascending=False)
        .drop_duplicates("_display_key")
        .head(top_n)
    )

    output = []
    for _, row in top.iterrows():
        support = []
        for nutrient, weight in weights.items():
            nv = float(row.get(f"_norm_{nutrient}", 0))
            support.append((nutrient, weight * nv, nv))

        support.sort(key=lambda x: x[1], reverse=True)
        top_support = [x[0] for x in support[:2] if x[2] > 0]

        reason = (
            "Mendukung prioritas " + " dan ".join(top_support)
            if top_support
            else "Profil nutrisi relatif sesuai dengan prioritas sistem"
        )

        output.append(
            {
                "food": str(row[food_col]),
                "score": round(float(row["score"]), 2),
                "coverage": int(row["_coverage"]),
                "reason": reason,
            }
        )

    return output