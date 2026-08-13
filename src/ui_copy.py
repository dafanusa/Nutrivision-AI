from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go

from .nutrition import DISPLAY_NAMES, NUTRIENT_UNITS


def age_text(months: int) -> str:
    years, rem = divmod(int(months), 12)
    if years and rem:
        return f"{years} tahun {rem} bulan"
    if years:
        return f"{years} tahun"
    return f"{rem} bulan"


def confidence_label(predictions):
    if not predictions:
        return "Tidak tersedia"
    top1 = predictions[0]["confidence"]
    top2 = predictions[1]["confidence"] if len(predictions) > 1 else 0
    margin = top1 - top2

    if top1 >= 0.85 and margin >= 0.20:
        return "Tinggi"
    if top1 >= 0.60 and margin >= 0.10:
        return "Sedang"
    return "Perlu konfirmasi"


def contribution_dataframe(adequacy: dict) -> pd.DataFrame:
    rows = []
    for key, value in adequacy.items():
        pct = float(value["contribution_percent"])
        rows.append(
            {
                "key": key,
                "Nutrisi": DISPLAY_NAMES.get(key, key),
                "Kontribusi": pct,
                "Gap": max(100 - min(max(pct, 0), 100), 0),
                "Status": value.get("status", ""),
            }
        )
    return pd.DataFrame(rows)


def contribution_chart(adequacy: dict):
    df = contribution_dataframe(adequacy).sort_values("Kontribusi")
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["Kontribusi"],
            y=df["Nutrisi"],
            orientation="h",
            text=[f"{v:.1f}%" for v in df["Kontribusi"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Kontribusi: %{x:.1f}%<extra></extra>",
        )
    )

    for x, label in [(25, "25%"), (50, "50%"), (100, "100% AKG")]:
        fig.add_vline(x=x, line_dash="dash", opacity=0.45)
        fig.add_annotation(
            x=x, y=1.06, xref="x", yref="paper",
            text=label, showarrow=False, font=dict(size=11)
        )

    fig.update_layout(
        height=390,
        margin=dict(l=10, r=35, t=45, b=25),
        xaxis_title="Kontribusi terhadap acuan (%)",
        yaxis_title="",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, max(110, float(df["Kontribusi"].max()) + 15)]),
    )
    return fig


def gap_chart(adequacy: dict):
    df = contribution_dataframe(adequacy).sort_values("Gap")

    fig = go.Figure(
        go.Bar(
            x=df["Gap"],
            y=df["Nutrisi"],
            orientation="h",
            text=[f"{v:.1f}%" for v in df["Gap"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Gap makanan: %{x:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        height=390,
        margin=dict(l=10, r=35, t=30, b=25),
        xaxis_title="Gap dari makanan yang dianalisis (%)",
        yaxis_title="",
        xaxis=dict(range=[0, 105]),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def recommendation_chart(recommendations):
    df = pd.DataFrame(recommendations).sort_values("score")
    fig = go.Figure(
        go.Bar(
            x=df["score"],
            y=df["food"],
            orientation="h",
            text=[f"{v:.0f}" for v in df["score"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Skor kecocokan: %{x:.1f}/100<extra></extra>",
        )
    )

    fig.update_layout(
        height=330,
        margin=dict(l=10, r=35, t=25, b=25),
        xaxis_title="Skor kecocokan internal",
        yaxis_title="",
        xaxis=dict(range=[0, 105]),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_narrative(result):
    p = result["child_profile"]
    status = result["nutrition_status"]
    preds = result["food_prediction"]
    adequacy = result.get("nutrient_contribution", {})
    recs = result.get("recommendations", [])

    parts = [
        f"Untuk anak usia {age_text(p['age_months'])} dengan berat {p['weight_kg']:g} kg, "
        f"tinggi {p['height_cm']:g} cm, dan MUAC/LILA {p['muac_cm']:g} cm, "
        f"model antropometri memberikan hasil screening **{str(status['prediction']).title()}**"
        + (
            f" dengan confidence {status['confidence']*100:.1f}%."
            if status.get("confidence") is not None
            else "."
        )
    ]

    if preds:
        top = preds[0]
        parts.append(
            f"Foto makanan paling mungkin dikenali sebagai **{top['food']}** "
            f"({top['confidence']*100:.1f}%)."
        )

    if adequacy:
        ranked = sorted(
            adequacy.items(),
            key=lambda x: x[1]["contribution_percent"]
        )
        low = ranked[:3]
        low_text = ", ".join(
            f"{DISPLAY_NAMES.get(k, k)} ({v['contribution_percent']:.1f}%)"
            for k, v in low
        )
        parts.append(
            f"Dari makanan yang dianalisis, kontribusi relatif paling rendah terhadap "
            f"AKG kelompok umur terlihat pada **{low_text}**. "
            "Ini berarti makanan tersebut belum banyak menyumbang nutrien tersebut; "
            "bukan berarti anak terbukti mengalami defisiensi."
        )

    if recs:
        names = ", ".join(r["food"] for r in recs[:3])
        parts.append(
            f"Sebagai kandidat pelengkap profil makanan, sistem menempatkan **{names}** "
            "pada ranking teratas berdasarkan nutrien yang tersedia pada database rekomendasi."
        )

    parts.append(
        "Gunakan hasil ini sebagai screening dan edukasi. Satu foto makanan tidak mewakili "
        "seluruh asupan harian dan tidak menggantikan pemeriksaan tenaga kesehatan."
    )

    return "\n\n".join(parts)