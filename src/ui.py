from __future__ import annotations

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

    top1 = float(predictions[0].get("confidence", 0))
    top2 = (
        float(predictions[1].get("confidence", 0))
        if len(predictions) > 1
        else 0.0
    )
    margin = top1 - top2

    if top1 >= 0.85 and margin >= 0.20:
        return "Tinggi"
    if top1 >= 0.60 and margin >= 0.10:
        return "Sedang"
    return "Perlu konfirmasi"


def _empty_figure(
    message: str,
    *,
    height: int = 390,
) -> go.Figure:
    """
    Figure aman untuk kondisi ketika profil nutrisi belum tersedia.
    Mencegah KeyError saat DataFrame kosong.
    """
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=13,
            color="#8792A6",
        ),
        align="center",
    )

    fig.update_layout(
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=30,
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def contribution_dataframe(
    adequacy: dict,
) -> pd.DataFrame:
    """
    Bentuk tabel kontribusi dengan skema kolom yang stabil,
    bahkan jika adequacy kosong.
    """
    columns = [
        "key",
        "Nutrisi",
        "Kontribusi",
        "Gap",
        "Status",
    ]

    rows = []

    if not adequacy:
        return pd.DataFrame(
            columns=columns
        )

    for key, value in adequacy.items():
        if not isinstance(value, dict):
            continue

        try:
            pct = float(
                value.get(
                    "contribution_percent",
                    0,
                )
            )
        except (TypeError, ValueError):
            continue

        try:
            gap = float(
                value.get(
                    "gap_percent",
                    max(
                        100
                        - min(
                            max(pct, 0),
                            100,
                        ),
                        0,
                    ),
                )
            )
        except (TypeError, ValueError):
            gap = max(
                100
                - min(
                    max(pct, 0),
                    100,
                ),
                0,
            )

        rows.append(
            {
                "key": key,
                "Nutrisi": DISPLAY_NAMES.get(
                    key,
                    key,
                ),
                "Kontribusi": pct,
                "Gap": gap,
                "Status": value.get(
                    "status",
                    "",
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def contribution_chart(
    adequacy: dict,
):
    df = contribution_dataframe(
        adequacy
    )

    if df.empty:
        return _empty_figure(
            "Kontribusi nutrisi belum tersedia.<br>"
            "Profil nutrisi makanan belum berhasil dipetakan."
        )

    df = df.sort_values(
        "Kontribusi"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["Kontribusi"],
            y=df["Nutrisi"],
            orientation="h",
            text=[
                f"{v:.1f}%"
                for v
                in df["Kontribusi"]
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Kontribusi: %{x:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    for x, label in [
        (25, "25%"),
        (50, "50%"),
        (100, "100% AKG"),
    ]:
        fig.add_vline(
            x=x,
            line_dash="dash",
            opacity=0.45,
        )
        fig.add_annotation(
            x=x,
            y=1.06,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=11),
        )

    max_value = float(
        df["Kontribusi"].max()
    )

    fig.update_layout(
        height=390,
        margin=dict(
            l=10,
            r=35,
            t=45,
            b=25,
        ),
        xaxis_title=(
            "Kontribusi terhadap acuan AKG (%)"
        ),
        yaxis_title="",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[
                0,
                max(
                    110,
                    max_value + 15,
                ),
            ]
        ),
    )

    return fig


def gap_chart(
    adequacy: dict,
):
    df = contribution_dataframe(
        adequacy
    )

    if df.empty:
        return _empty_figure(
            "Gap nutrisi belum tersedia.<br>"
            "Profil nutrisi makanan belum berhasil dipetakan."
        )

    df = df.sort_values(
        "Gap"
    )

    fig = go.Figure(
        go.Bar(
            x=df["Gap"],
            y=df["Nutrisi"],
            orientation="h",
            text=[
                f"{v:.1f}%"
                for v
                in df["Gap"]
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Gap makanan: %{x:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=390,
        margin=dict(
            l=10,
            r=35,
            t=30,
            b=25,
        ),
        xaxis_title=(
            "Gap dari makanan yang dianalisis (%)"
        ),
        yaxis_title="",
        xaxis=dict(
            range=[0, 105]
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def recommendation_chart(
    recommendations,
):
    if not recommendations:
        return _empty_figure(
            "Rekomendasi belum tersedia.",
            height=330,
        )

    df = pd.DataFrame(
        recommendations
    )

    required = {
        "score",
        "food",
    }

    if (
        df.empty
        or not required.issubset(
            df.columns
        )
    ):
        return _empty_figure(
            "Rekomendasi belum tersedia.",
            height=330,
        )

    df = df.sort_values(
        "score"
    )

    fig = go.Figure(
        go.Bar(
            x=df["score"],
            y=df["food"],
            orientation="h",
            text=[
                f"{v:.0f}"
                for v
                in df["score"]
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Skor kecocokan: %{x:.1f}/100"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=330,
        margin=dict(
            l=10,
            r=35,
            t=25,
            b=25,
        ),
        xaxis_title=(
            "Skor kecocokan internal"
        ),
        yaxis_title="",
        xaxis=dict(
            range=[0, 105]
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def build_narrative(result):
    """
    Dipertahankan untuk kompatibilitas dengan app lama.
    App terbaru menggunakan build_narrative_id().
    """
    profile = result.get(
        "child_profile",
        {},
    )
    status = result.get(
        "nutrition_status",
        {},
    )
    preds = result.get(
        "food_prediction",
        [],
    )
    adequacy = result.get(
        "nutrient_contribution",
        {},
    )
    recs = result.get(
        "recommendations",
        [],
    )

    parts = [
        (
            f"Untuk anak usia "
            f"{age_text(profile.get('age_months', 0))} "
            f"dengan berat {profile.get('weight_kg', 0):g} kg, "
            f"tinggi {profile.get('height_cm', 0):g} cm, "
            f"dan MUAC/LILA {profile.get('muac_cm', 0):g} cm, "
            f"model antropometri memberikan hasil skrining "
            f"**{str(status.get('prediction', '-')).title()}**"
            + (
                f" dengan keyakinan "
                f"{float(status['confidence']) * 100:.1f}%."
                if status.get(
                    "confidence"
                ) is not None
                else "."
            )
        )
    ]

    if preds:
        top = preds[0]
        parts.append(
            f"Foto makanan paling mungkin dikenali sebagai "
            f"**{top.get('food', '-')}** "
            f"({float(top.get('confidence', 0)) * 100:.1f}%)."
        )

    if adequacy:
        ranked = sorted(
            adequacy.items(),
            key=lambda item: float(
                item[1].get(
                    "contribution_percent",
                    0,
                )
            ),
        )

        low = ranked[:3]

        low_text = ", ".join(
            (
                f"{DISPLAY_NAMES.get(key, key)} "
                f"({float(value.get('contribution_percent', 0)):.1f}%)"
            )
            for key, value
            in low
        )

        parts.append(
            "Dari makanan yang dianalisis, kontribusi relatif "
            "paling rendah terhadap AKG kelompok umur terlihat "
            f"pada **{low_text}**."
        )

    if recs:
        names = ", ".join(
            str(
                item.get(
                    "food",
                    "-",
                )
            )
            for item
            in recs[:3]
        )

        parts.append(
            f"Sistem menempatkan **{names}** "
            "sebagai kandidat pelengkap pada peringkat teratas."
        )

    if result.get("warning"):
        parts.append(
            str(
                result[
                    "warning"
                ]
            )
        )

    parts.append(
        "Gunakan hasil ini sebagai skrining dan edukasi. "
        "Satu foto makanan tidak mewakili seluruh asupan harian "
        "dan tidak menggantikan pemeriksaan tenaga kesehatan."
    )

    return "\n\n".join(
        parts
    )
