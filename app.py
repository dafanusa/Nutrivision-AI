from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

from src.inference import (
    analyze_child_and_food,
    load_food_model,
    load_malnutrition_artifact,
)
from src.nutrition import (
    DISPLAY_NAMES,
    NUTRIENT_UNITS,
    build_nutrition_kb,
    build_recommendation_table,
)
from src.ui import (
    age_text,
    build_narrative,
    confidence_label,
    contribution_chart,
    contribution_dataframe,
    gap_chart,
    recommendation_chart,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title="NutriVision AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "child_profile" not in st.session_state:
    st.session_state.child_profile = {
        "name": "Anak",
        "age_months": 36,
        "weight_kg": 11.0,
        "height_cm": 90.0,
        "muac_cm": 12.0,
    }

# ============================================================
# STYLE — mempertahankan UI putih-kuning yang sekarang
# ============================================================

st.markdown(
    """
<style>
    :root {
        --yellow: #FFC928;
        --yellow2: #FFD94A;
        --yellow-soft: #FFF8D7;
        --ink: #171717;
        --muted: #746E63;
        --line: #EDE9DD;
        --green: #1E9C54;
        --bg: #FBFBF8;
        --white: #FFFFFF;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--ink) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(251,251,248,.95) !important;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.0rem;
        padding-bottom: 3rem;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }

    .brand {
        display:flex;
        align-items:center;
        gap:10px;
        padding: 6px 2px 18px;
    }

    .brand-logo {
        width:42px;
        height:42px;
        display:grid;
        place-items:center;
        border-radius:13px;
        background:#FFF8D8;
        border:1px solid #F2D567;
        font-size:1.25rem;
    }

    .brand-name {
        font-weight:900;
        font-size:1.05rem;
    }

    .brand-sub {
        color:#8A8378;
        font-size:.72rem;
        margin-top:2px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 5px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label {
        padding: 10px 12px;
        border-radius: 10px;
        margin-bottom: 2px;
        transition: .15s ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
        background:#FFF9E7;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background:linear-gradient(90deg,#FFC315,#FFD83F);
        font-weight:850;
    }

    [data-testid="stSidebar"] [role="radiogroup"] input {
        display:none;
    }

    .about-card {
        margin-top:28px;
        padding:15px;
        border:1px solid #EED783;
        border-radius:15px;
        background:linear-gradient(180deg,#FFFDF4,#FFF8D9);
        color:#635A43;
        font-size:.77rem;
        line-height:1.6;
    }

    /* HEADER */
    .page-head {
        display:flex;
        align-items:flex-start;
        justify-content:space-between;
        gap:20px;
        margin-bottom:17px;
    }

    .hello {
        font-size:.92rem;
        font-weight:700;
        color:#5D584F;
    }

    .title {
        font-size:2.2rem;
        line-height:1;
        font-weight:950;
        letter-spacing:-.04em;
        color:#1C1C1C;
        margin-top:3px;
    }

    .title span { color:#FFBE00; }

    .subtitle {
        color:#716B61;
        font-size:.92rem;
        margin-top:8px;
    }

    .date-pill {
        padding:8px 12px;
        border-radius:10px;
        border:1px solid var(--line);
        background:white;
        color:#6E685E;
        font-size:.78rem;
        white-space:nowrap;
    }

    /* PANELS */
    .panel {
        background:#FFFFFF;
        border:1px solid var(--line);
        border-radius:16px;
        padding:17px 18px;
        margin-bottom:14px;
        box-shadow:0 5px 18px rgba(50,40,0,.03);
    }

    .panel-title {
        font-size:1rem;
        font-weight:900;
        margin-bottom:6px;
        color:#242424;
    }

    .panel-sub {
        font-size:.76rem;
        color:#827B70;
        margin-bottom:12px;
    }

    /* INPUT */
    [data-testid="stNumberInput"] input {
        background:#FFFDF7 !important;
        color:#242424 !important;
        border-color:#E9DFC0 !important;
    }

    [data-testid="stFileUploader"] {
        background:#FFFDF7;
        border:1px dashed #E3C45A;
        border-radius:12px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background:#FFFDF7 !important;
        border:none !important;
    }

    .cta-box {
        min-height:145px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        text-align:center;
        background:linear-gradient(135deg,#FFE46B,#FFC92A);
        border:1px solid #F0C126;
        border-radius:16px;
        padding:18px;
        margin-bottom:10px;
    }

    .cta-title {
        font-size:1.1rem;
        font-weight:950;
        color:#3D3000;
    }

    .cta-sub {
        color:#6F5A10;
        font-size:.78rem;
        margin-top:6px;
        line-height:1.5;
    }

    .stButton > button {
        width:100%;
        background:linear-gradient(90deg,#FFC315,#FFD83F) !important;
        color:#2A2200 !important;
        border:1px solid #E1AF00 !important;
        border-radius:10px !important;
        min-height:43px;
        font-weight:850 !important;
    }

    /* RESULT CARDS */
    .results-grid {
        display:grid;
        grid-template-columns:repeat(5,minmax(0,1fr));
        gap:12px;
    }

    .result-card {
        min-height:145px;
        padding:14px;
        border:1px solid var(--line);
        border-radius:14px;
        background:white;
    }

    .result-icon {
        margin-bottom:9px;
        font-size:1rem;
    }

    .result-label {
        font-size:.7rem;
        color:#746E64;
        margin-bottom:10px;
    }

    .result-value {
        font-size:1.03rem;
        font-weight:950;
        color:#232323;
    }

    .green { color:#159348; }
    .yellow { color:#D9A500; }

    .result-sub {
        font-size:.67rem;
        color:#8A8378;
        margin-top:4px;
    }

    .result-card .ring {
        margin-top: 2px;
        margin-bottom: 2px;
    }

    .results-grid > .result-card {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }

    .ring {
        width:62px;
        height:62px;
        border-radius:50%;
        display:grid;
        place-items:center;
        background:conic-gradient(#FFC31B calc(var(--v)*1%),#ECEBE6 0);
        position:relative;
        margin-top:2px;
    }

    .ring:before {
        content:"";
        width:44px;
        height:44px;
        border-radius:50%;
        background:#fff;
        position:absolute;
    }

    .ring span {
        position:relative;
        font-size:.78rem;
        font-weight:950;
        z-index:2;
    }

    /* TABS */
    [data-baseweb="tab-list"] {
        border-bottom:1px solid var(--line);
        gap:0;
    }

    button[data-baseweb="tab"] {
        background:transparent !important;
        color:#716B61 !important;
        border:none !important;
        border-radius:0 !important;
        padding-left:18px !important;
        padding-right:18px !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color:#DAA400 !important;
        border-bottom:2px solid #FFC21A !important;
    }

    div[data-testid="stPlotlyChart"] {
        background:#FFFFFF;
        border:1px solid var(--line);
        border-radius:14px;
        padding:4px;
    }

    /* RECOMMENDATIONS */
    .rec-card {
        display:grid;
        grid-template-columns:34px 1fr auto;
        align-items:center;
        gap:10px;
        padding:11px;
        border:1px solid var(--line);
        border-radius:12px;
        background:white;
        margin-bottom:8px;
    }

    .rec-rank {
        width:29px;
        height:29px;
        display:grid;
        place-items:center;
        border-radius:50%;
        background:#FFC51E;
        font-weight:950;
        color:#332700;
        font-size:.8rem;
    }

    .rec-name {
        font-size:.88rem;
        font-weight:900;
        color:#262626;
    }

    .rec-reason {
        font-size:.7rem;
        color:#7F786E;
        margin-top:3px;
        line-height:1.45;
    }

    .rec-score {
        color:#159348;
        font-size:.8rem;
        font-weight:900;
    }

    .insight {
        margin-top:11px;
        padding:16px 17px;
        border:1px solid #EFD67E;
        border-radius:14px;
        background:linear-gradient(90deg,#FFFDF4,#FFF8D6);
        color:#504939;
        font-size:.8rem;
        line-height:1.65;
    }

    .priority {
        display:inline-block;
        margin:8px 4px 0 0;
        padding:4px 8px;
        border-radius:999px;
        border:1px solid #EED168;
        background:#FFF8DA;
        color:#846600;
        font-weight:800;
        font-size:.68rem;
    }

    /* OTHER PAGES */
    .info-card {
        padding:17px;
        border:1px solid var(--line);
        border-radius:14px;
        background:#FFFFFF;
        margin-bottom:10px;
    }

    .info-card-title {
        font-size:.92rem;
        font-weight:900;
        color:#262626;
        margin-bottom:5px;
    }

    .info-card-text {
        color:#766F65;
        font-size:.79rem;
        line-height:1.6;
    }

    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}

    @media(max-width:1000px){
        .results-grid{
            grid-template-columns:repeat(2,minmax(0,1fr));
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LOADERS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():
    food_path = MODEL_DIR / "food_classifier_best.keras"
    mal_path = MODEL_DIR / "malnutrition_model.joblib"

    missing = [p.name for p in [food_path, mal_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Model belum tersedia: " + ", ".join(missing))

    return load_food_model(food_path), load_malnutrition_artifact(mal_path)


@st.cache_data(show_spinner=False)
def load_data():
    required = {
        "class_names": MODEL_DIR / "class_names.json",
        "nutricheck": DATA_DIR / "nutricheck_nutrition.csv",
        "indonesia": DATA_DIR / "indonesia_nutrition.csv",
        "requirements": DATA_DIR / "nutrition_requirements.csv",
    }

    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Data deployment belum lengkap: " + ", ".join(missing))

    class_names = json.loads(required["class_names"].read_text(encoding="utf-8"))

    return (
        class_names,
        pd.read_csv(required["nutricheck"]),
        pd.read_csv(required["indonesia"]),
        pd.read_csv(required["requirements"]),
    )


def fixed_preview(image, size=(720, 430)):
    return ImageOps.fit(
        image.convert("RGB"),
        size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


# ============================================================
# CHARTS
# ============================================================

def top3_chart(predictions):
    df = pd.DataFrame(predictions).copy()
    df["pct"] = df["confidence"] * 100
    df = df.sort_values("pct")

    fig = go.Figure(
        go.Bar(
            x=df["pct"],
            y=df["food"],
            orientation="h",
            marker_color="#FFC51D",
            text=[f"{x:.2f}%" for x in df["pct"]],
            textposition="outside",
            cliponaxis=False,
        )
    )

    fig.update_layout(
        title=dict(
            text="Top-3 Prediksi Kelas Makanan",
            x=.03,
            font=dict(size=15, color="#272727"),
        ),
        height=355,
        margin=dict(l=105, r=70, t=55, b=45),
        xaxis=dict(
            range=[0,105],
            title="Probabilitas (%)",
            gridcolor="#EFE9D8",
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            automargin=True,
            tickfont=dict(size=10),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def nutrient_radar(adequacy):
    """
    Radar diperbaiki agar label tidak nabrak.
    - legend dipindah ke bawah
    - domain radar diperkecil
    - margin diperbesar
    - label dibuat konsisten dan ringkas
    """

    if not adequacy:
        return go.Figure()

    pretty = {
        "calories": "Energi",
        "protein": "Protein",
        "carbohydrate": "Karbohidrat",
        "iron": "Zat Besi",
        "calcium": "Kalsium",
        "vitamin_a": "Vit A",
        "vitamin_c": "Vit C",
        "fat": "Lemak",
    }

    rows = []

    for key, info in adequacy.items():
        rows.append(
            (
                pretty.get(key, DISPLAY_NAMES.get(key, key)),
                min(max(float(info.get("contribution_percent", 0)), 0), 100),
            )
        )

    labels = [x[0] for x in rows]
    values = [x[1] for x in rows]

    if labels:
        labels = labels + [labels[0]]
        values = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            name="Makanan",
            line=dict(color="#FFC21A", width=2.7),
            fillcolor="rgba(255,194,26,.18)",
            hovertemplate="<b>%{theta}</b><br>%{r:.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[50] * len(values),
            theta=labels,
            name="Referensi 50%",
            line=dict(
                color="#8A8A86",
                width=1.15,
                dash="dash",
            ),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=dict(
            text="Profil Nutrisi Makanan",
            x=.03,
            font=dict(size=15, color="#272727"),
        ),
        height=355,
        margin=dict(l=65, r=65, t=62, b=62),
        polar=dict(
            domain=dict(x=[.14, .86], y=[.16, .90]),
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=[0,100],
                tickvals=[25,50,75,100],
                tickfont=dict(size=8, color="#948C81"),
                gridcolor="#EAE4D5",
                linecolor="#EAE4D5",
                angle=90,
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color="#6B655D"),
                gridcolor="#EEE7D8",
                linecolor="#EEE7D8",
            ),
        ),
        legend=dict(
            orientation="h",
            x=.5,
            xanchor="center",
            y=-.12,
            yanchor="top",
            font=dict(size=9),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def history_status_chart(history):
    if not history:
        return go.Figure()

    df = pd.DataFrame(history)
    counts = df["status_gizi"].value_counts().reset_index()
    counts.columns = ["status", "jumlah"]

    fig = go.Figure(
        go.Bar(
            x=counts["status"],
            y=counts["jumlah"],
            marker_color="#FFC51D",
            text=counts["jumlah"],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Distribusi Hasil Screening",
        height=330,
        xaxis_title="Status Gizi",
        yaxis_title="Jumlah Analisis",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=45,r=20,t=50,b=45),
        yaxis=dict(gridcolor="#EFE9D8"),
    )

    return fig


def history_food_chart(history):
    if not history:
        return go.Figure()

    df = pd.DataFrame(history)
    counts = df["makanan"].value_counts().head(8).sort_values().reset_index()
    counts.columns = ["makanan", "jumlah"]

    fig = go.Figure(
        go.Bar(
            x=counts["jumlah"],
            y=counts["makanan"],
            orientation="h",
            marker_color="#FFD85A",
            text=counts["jumlah"],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Makanan yang Paling Sering Terdeteksi",
        height=330,
        xaxis_title="Jumlah",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=115,r=45,t=50,b=45),
        xaxis=dict(gridcolor="#EFE9D8"),
    )

    return fig


def profile_radar(profile):
    """
    Visualisasi profil antropometri sederhana.
    Bukan z-score WHO; hanya normalisasi dashboard untuk ringkasan visual.
    """
    age = profile["age_months"]
    weight = profile["weight_kg"]
    height = profile["height_cm"]
    muac = profile["muac_cm"]
    bmi = weight / ((height/100)**2)

    labels = ["Umur", "Berat", "Tinggi", "MUAC", "BMI"]
    values = [
        min(age / 83 * 100, 100),
        min(weight / 40 * 100, 100),
        min(height / 140 * 100, 100),
        min(muac / 30 * 100, 100),
        min(bmi / 25 * 100, 100),
    ]

    labels += [labels[0]]
    values += [values[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            line=dict(color="#FFC21A",width=2.5),
            fillcolor="rgba(255,194,26,.18)",
            name="Profil",
        )
    )

    fig.update_layout(
        title="Ringkasan Profil Antropometri",
        height=360,
        polar=dict(
            radialaxis=dict(range=[0,100],gridcolor="#EAE4D5"),
            angularaxis=dict(gridcolor="#EEE7D8"),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=55,r=55,t=60,b=45),
    )

    return fig


def knowledge_chart():
    labels = ["Energi", "Protein", "Zat Besi", "Kalsium", "Vit A", "Vit C"]
    groups = ["Makronutrien", "Makronutrien", "Mikronutrien", "Mikronutrien", "Mikronutrien", "Mikronutrien"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=[1,1,1,1,1,1],
            marker_color=["#FFC51D","#FFC51D","#FFE17A","#FFE17A","#FFE17A","#FFE17A"],
            text=groups,
            textposition="inside",
            hovertemplate="<b>%{x}</b><extra></extra>",
        )
    )

    fig.update_layout(
        title="Kelompok Nutrisi yang Dianalisis NutriVision",
        height=300,
        yaxis=dict(visible=False),
        xaxis_title="",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20,r=20,t=55,b=45),
    )

    return fig


def nutrient_mean(adequacy):
    if not adequacy:
        return 0.0
    vals = [
        min(max(float(x.get("contribution_percent",0)),0),100)
        for x in adequacy.values()
    ]
    return float(np.mean(vals)) if vals else 0.0


def nutrient_priorities(adequacy):
    if not adequacy:
        return []
    ranked = sorted(
        adequacy.items(),
        key=lambda x: float(x[1].get("contribution_percent",0)),
    )
    return [DISPLAY_NAMES.get(k,k) for k,_ in ranked[:3]]


# ============================================================
# GRAD-CAM
# ============================================================

def find_gradcam_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for inner in reversed(layer.layers):
                try:
                    if len(inner.output.shape) == 4:
                        return layer.name, inner.name
                except Exception:
                    pass
    raise ValueError("Conv feature layer tidak ditemukan.")


def make_gradcam(image, model):
    img = image.convert("RGB").resize((224,224))
    x = tf.keras.utils.img_to_array(img)[None,...]

    parent_name, conv_name = find_gradcam_layer(model)
    backbone = model.get_layer(parent_name)
    conv_layer = backbone.get_layer(conv_name)

    grad_backbone = tf.keras.Model(
        backbone.input,
        [conv_layer.output, backbone.output]
    )

    bidx = next(i for i,l in enumerate(model.layers) if l.name == parent_name)

    with tf.GradientTape() as tape:
        z = x

        for layer in model.layers[:bidx]:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            z = layer(z, training=False)

        conv_out, z = grad_backbone(z, training=False)

        for layer in model.layers[bidx+1:]:
            z = layer(z, training=False)

        preds = z
        idx = int(tf.argmax(preds[0]).numpy())
        class_score = preds[:,idx]

    grads = tape.gradient(class_score, conv_out)

    if grads is None:
        raise RuntimeError("Gradient Grad-CAM gagal dihitung.")

    weights = tf.reduce_mean(grads, axis=(0,1,2))
    heat = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    heat = tf.maximum(heat,0)

    mx = tf.reduce_max(heat)
    if float(mx.numpy()) > 0:
        heat = heat / mx

    return heat.numpy(), idx, float(preds[0,idx].numpy()), (parent_name,conv_name)


def overlay_gradcam(image, heatmap):
    base = np.asarray(image.convert("RGB").resize((224,224))).astype(np.float32)
    heat = tf.image.resize(heatmap[...,None],(224,224)).numpy()[...,0]

    color = np.zeros((224,224,3),dtype=np.float32)
    color[...,0] = 255
    color[...,1] = 210*(1-.8*heat)
    color[...,2] = 25*(1-heat)

    opacity = heat[...,None]*.48
    out = base*(1-opacity)+color*opacity

    return Image.fromarray(np.clip(out,0,255).astype(np.uint8))


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-logo">🍎</div>
            <div>
                <div class="brand-name">NutriVision AI</div>
                <div class="brand-sub">AI Nutrition Screening</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigasi",
        [
            "🏠 Dashboard",
            "🗂️ Riwayat Analisis",
            "👤 Profil Anak",
            "🥗 Pengetahuan Gizi",
            "ℹ️ Tentang Aplikasi",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="about-card">
            <b>💡 NutriVision AI</b><br><br>
            Sistem AI untuk screening pola status gizi anak,
            pengenalan makanan, analisis nutrisi, dan rekomendasi
            makanan Indonesia.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

now = datetime.now().strftime("%d %B %Y • %H:%M")

st.markdown(
    f"""
    <div class="page-head">
        <div>
            <div class="hello">Halo! 👋</div>
            <div class="title">Nutri<span>Vision</span> AI</div>
            <div class="subtitle">
                Analisis cerdas status gizi anak dan kandungan gizi makanan
            </div>
        </div>
        <div class="date-pill">🗓 {now}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    profile = st.session_state.child_profile

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    input_col, image_col, action_col = st.columns(
        [1.25, 1.1, .95],
        gap="large",
    )

    with input_col:
        st.markdown(
            '<div class="panel-title">👤 Data Anak</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="panel-sub">Input langsung pada dashboard agar alur analisis lebih ringkas.</div>',
            unsafe_allow_html=True,
        )

        r1c1, r1c2 = st.columns(2)

        with r1c1:
            age_months = st.number_input(
                "Umur (bulan)",
                min_value=6,
                max_value=83,
                value=int(profile["age_months"]),
                step=1,
            )

        with r1c2:
            weight_kg = st.number_input(
                "Berat badan (kg)",
                min_value=3.0,
                max_value=40.0,
                value=float(profile["weight_kg"]),
                step=.1,
            )

        r2c1, r2c2 = st.columns(2)

        with r2c1:
            height_cm = st.number_input(
                "Tinggi badan (cm)",
                min_value=45.0,
                max_value=140.0,
                value=float(profile["height_cm"]),
                step=.5,
            )

        with r2c2:
            muac_cm = st.number_input(
                "MUAC / LILA (cm)",
                min_value=5.0,
                max_value=30.0,
                value=float(profile["muac_cm"]),
                step=.1,
            )

        bmi_preview = weight_kg / ((height_cm/100)**2)

        p1,p2,p3 = st.columns(3)
        p1.metric("Usia", age_text(age_months))
        p2.metric("BMI", f"{bmi_preview:.2f}")
        p3.metric("MUAC", f"{muac_cm:.1f} cm")

    with image_col:
        st.markdown(
            '<div class="panel-title">🖼 Foto Makanan</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="panel-sub">Gunakan foto dengan satu menu utama yang jelas.</div>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Upload foto",
            type=["jpg","jpeg","png","webp"],
            label_visibility="collapsed",
        )

        image = None

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            preview = fixed_preview(image)

            st.image(
                preview,
                use_container_width=True,
                caption="Preview makanan",
            )
        else:
            st.info("Upload foto makanan.")

    with action_col:
        st.markdown(
            """
            <div class="cta-box">
                <div class="cta-title">✨ Analisis Sekarang</div>
                <div class="cta-sub">
                    Jalankan model status gizi, EfficientNetB0,
                    analisis nutrisi, dan recommendation engine.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        analyze = st.button(
            "Jalankan Analisis",
            type="primary",
            use_container_width=True,
            disabled=image is None,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if analyze:

        st.session_state.child_profile.update(
            {
                "age_months": int(age_months),
                "weight_kg": float(weight_kg),
                "height_cm": float(height_cm),
                "muac_cm": float(muac_cm),
            }
        )

        try:
            with st.spinner("NutriVision sedang menganalisis..."):

                food_model, malnutrition_artifact = load_models()

                class_names, nutricheck_df, indonesia_df, requirements_df = load_data()

                nutricheck_kb, nutricheck_food_col, nutricheck_nutrient_cols = (
                    build_nutrition_kb(nutricheck_df)
                )

                indo_kb, indo_food_col, indo_nutrient_cols = (
                    build_nutrition_kb(indonesia_df)
                )

                indo_rec_df, indo_rec_cols = build_recommendation_table(
                    indo_kb,
                    indo_food_col,
                    indo_nutrient_cols,
                )

                result = analyze_child_and_food(
                    age_months=age_months,
                    weight_kg=weight_kg,
                    height_cm=height_cm,
                    muac_cm=muac_cm,
                    image=image,
                    requirements=requirements_df,
                    food_model=food_model,
                    class_names=class_names,
                    malnutrition_artifact=malnutrition_artifact,
                    nutricheck_kb=nutricheck_kb,
                    nutricheck_food_col=nutricheck_food_col,
                    nutricheck_nutrient_cols=nutricheck_nutrient_cols,
                    indo_rec_df=indo_rec_df,
                    indo_food_col=indo_food_col,
                    indo_rec_cols=indo_rec_cols,
                )

        except Exception as exc:
            st.error("Inference gagal.")
            st.code(str(exc))
            st.stop()

        status = result["nutrition_status"]
        preds = result["food_prediction"]
        top_food = preds[0]
        certainty = confidence_label(preds)
        bmi = status.get("bmi")
        adequacy = result.get("nutrient_contribution",{})
        recommendations = result.get("recommendations",[])

        status_conf = float(status.get("confidence",0))*100
        food_conf = float(top_food.get("confidence",0))*100
        nutrient_score = nutrient_mean(adequacy)

        st.session_state.history.insert(
            0,
            {
                "waktu": datetime.now().strftime("%d-%m-%Y %H:%M"),
                "umur_bulan": int(age_months),
                "berat_kg": float(weight_kg),
                "tinggi_cm": float(height_cm),
                "muac_cm": float(muac_cm),
                "status_gizi": str(status["prediction"]),
                "makanan": str(top_food["food"]),
                "confidence": round(food_conf,2),
            }
        )

        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🧾 Hasil Analisis</div>',unsafe_allow_html=True)

        status_css = "green" if str(status["prediction"]).lower()=="normal" else "yellow"

        result_cards_html = (
            f'<div class="results-grid">'
            f'<div class="result-card">'
            f'<div class="result-icon">🛡️</div>'
            f'<div class="result-label">Status Gizi</div>'
            f'<div class="result-value {status_css}">{str(status["prediction"]).upper()}</div>'
            f'<div class="result-sub">Confidence {status_conf:.1f}%</div>'
            f'</div>'
            f'<div class="result-card">'
            f'<div class="result-icon">🍴</div>'
            f'<div class="result-label">Makanan Terprediksi</div>'
            f'<div class="result-value yellow">{top_food["food"]}</div>'
            f'<div class="result-sub">Kelas makanan</div>'
            f'</div>'
            f'<div class="result-card">'
            f'<div class="result-icon">🎯</div>'
            f'<div class="result-label">Confidence Makanan</div>'
            f'<div class="ring" style="--v:{food_conf:.1f}"><span>{food_conf:.1f}%</span></div>'
            f'<div class="result-sub">{certainty}</div>'
            f'</div>'
            f'<div class="result-card">'
            f'<div class="result-icon">⚕</div>'
            f'<div class="result-label">BMI</div>'
            f'<div class="result-value">{bmi:.2f}</div>'
            f'<div class="result-sub">Fitur model • bukan diagnosis</div>'
            f'</div>'
            f'<div class="result-card">'
            f'<div class="result-icon">❤️</div>'
            f'<div class="result-label">Rata-rata Kontribusi Nutrisi</div>'
            f'<div class="ring" style="--v:{nutrient_score:.1f}"><span>{nutrient_score:.0f}%</span></div>'
            f'<div class="result-sub">Ringkasan makanan</div>'
            f'</div>'
            f'</div>'
        )

        st.markdown(
            result_cards_html,
            unsafe_allow_html=True,
        )

        st.markdown("</div>",unsafe_allow_html=True)

        tab1,tab2,tab3,tab4,tab5 = st.tabs(
            ["Ringkasan","Nutrisi & Gap","Rekomendasi Makanan","Grad-CAM","Detail Model"]
        )

        with tab1:
            a,b,c = st.columns([1.02,1.08,1.22],gap="large")

            with a:
                st.plotly_chart(
                    nutrient_radar(adequacy),
                    use_container_width=True,
                    config={"displayModeBar":False},
                )

            with b:
                st.plotly_chart(
                    top3_chart(preds),
                    use_container_width=True,
                    config={"displayModeBar":False},
                )

            with c:
                st.markdown("#### 🔎 Rekomendasi Terbaik")

                if recommendations:
                    for i,rec in enumerate(recommendations[:3],1):
                        st.markdown(
                            f"""
                            <div class="rec-card">
                                <div class="rec-rank">{i}</div>
                                <div>
                                    <div class="rec-name">{rec["food"]}</div>
                                    <div class="rec-reason">{rec.get("reason","")}</div>
                                </div>
                                <div class="rec-score">{rec["score"]:.0f}/100</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("Belum ada rekomendasi.")

            priority_html = "".join(
                f'<span class="priority">✦ {x}</span>'
                for x in nutrient_priorities(adequacy)
            )

            st.markdown(
                f"""
                <div class="insight">
                    <b>⭐ Insight & Rekomendasi untuk Anak</b><br><br>
                    {build_narrative(result)}
                    <br>{priority_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tab2:
            x,y = st.columns(2,gap="large")

            with x:
                st.plotly_chart(
                    contribution_chart(adequacy),
                    use_container_width=True,
                    config={"displayModeBar":False},
                )

            with y:
                st.plotly_chart(
                    gap_chart(adequacy),
                    use_container_width=True,
                    config={"displayModeBar":False},
                )

            if result["food_nutrition"].get("status")=="ok":
                rows=[]

                for k,v in result["food_nutrition"]["nutrients"].items():
                    rows.append(
                        {
                            "Nutrisi":DISPLAY_NAMES.get(k,k),
                            "Jumlah":v,
                            "Satuan":NUTRIENT_UNITS.get(k,""),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(rows),
                    hide_index=True,
                    use_container_width=True,
                )

        with tab3:
            if recommendations:
                st.plotly_chart(
                    recommendation_chart(recommendations),
                    use_container_width=True,
                    config={"displayModeBar":False},
                )

                cols = st.columns(min(3,len(recommendations)))

                for i,rec in enumerate(recommendations[:3]):
                    with cols[i]:
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="result-icon">🥗 #{i+1}</div>
                                <div class="result-label">Rekomendasi</div>
                                <div class="result-value">{rec["food"]}</div>
                                <div class="result-value green"
                                     style="font-size:.9rem;margin-top:7px;">
                                    {rec["score"]:.0f}/100
                                </div>
                                <div class="result-sub">{rec.get("reason","")}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.info("Recommendation engine belum menghasilkan kandidat.")

        with tab4:
            st.markdown("#### Grad-CAM — Explainable AI")

            try:
                heat,idx,conf,layer_info = make_gradcam(image,food_model)
                overlay = overlay_gradcam(image,heat)

                heat_img = Image.fromarray(
                    np.uint8(np.clip(heat*255,0,255))
                ).resize((224,224))

                c1,c2,c3 = st.columns(3)

                with c1:
                    st.image(
                        image.resize((224,224)),
                        caption="Foto asli",
                        use_container_width=True,
                    )

                with c2:
                    st.image(
                        heat_img,
                        caption="Grad-CAM heatmap",
                        use_container_width=True,
                    )

                with c3:
                    st.image(
                        overlay,
                        caption="Grad-CAM overlay",
                        use_container_width=True,
                    )

                st.markdown(
                    f"""
                    <div class="insight">
                        <b>Explainable AI:</b> model memprediksi
                        <b>{class_names[idx]}</b> dengan confidence
                        <b>{conf*100:.2f}%</b>. Area aktivasi tinggi
                        menunjukkan region citra yang relatif lebih
                        berpengaruh terhadap prediksi.<br><br>
                        Layer: {layer_info}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as exc:
                st.warning("Grad-CAM belum dapat divisualisasikan.")
                st.code(str(exc))

        with tab5:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Kelas":p["food"],
                            "Confidence (%)":round(p["confidence"]*100,2),
                        }
                        for p in preds
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

            st.code(
                """Data antropometri
→ Model malnutrition
→ Screening status gizi

Foto makanan
→ EfficientNetB0
→ Klasifikasi makanan
→ NutriCheck nutrition mapping
→ Perbandingan AKG
→ Nutrient contribution / gap

Indonesian Food Nutrition
→ Recommendation engine
→ Ranking makanan"""
            )


# ============================================================
# RIWAYAT ANALISIS
# ============================================================

elif page == "🗂️ Riwayat Analisis":

    st.markdown("## Riwayat Analisis")
    st.caption(
        "Riwayat disimpan selama sesi Streamlit aktif."
    )

    if not st.session_state.history:
        st.info(
            "Belum ada riwayat analisis. Jalankan analisis dari Dashboard terlebih dahulu."
        )

    else:
        hist = pd.DataFrame(st.session_state.history)

        h1,h2 = st.columns(2,gap="large")

        with h1:
            st.plotly_chart(
                history_status_chart(st.session_state.history),
                use_container_width=True,
                config={"displayModeBar":False},
            )

        with h2:
            st.plotly_chart(
                history_food_chart(st.session_state.history),
                use_container_width=True,
                config={"displayModeBar":False},
            )

        st.markdown("#### Detail Riwayat")

        st.dataframe(
            hist.rename(
                columns={
                    "waktu":"Waktu",
                    "umur_bulan":"Umur (bulan)",
                    "berat_kg":"Berat (kg)",
                    "tinggi_cm":"Tinggi (cm)",
                    "muac_cm":"MUAC (cm)",
                    "status_gizi":"Status Gizi",
                    "makanan":"Makanan",
                    "confidence":"Confidence (%)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        if st.button("Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()


# ============================================================
# PROFIL ANAK
# ============================================================

elif page == "👤 Profil Anak":

    st.markdown("## Profil Anak")
    st.caption(
        "Profil ini menjadi nilai default pada halaman Dashboard."
    )

    p = st.session_state.child_profile

    left,right = st.columns([1,1],gap="large")

    with left:
        name = st.text_input(
            "Nama / Label Anak",
            value=str(p.get("name","Anak")),
        )

        age = st.number_input(
            "Umur (bulan)",
            6,
            83,
            int(p["age_months"]),
            1,
        )

        weight = st.number_input(
            "Berat badan (kg)",
            3.0,
            40.0,
            float(p["weight_kg"]),
            .1,
        )

    with right:
        height = st.number_input(
            "Tinggi badan (cm)",
            45.0,
            140.0,
            float(p["height_cm"]),
            .5,
        )

        muac = st.number_input(
            "MUAC / LILA (cm)",
            5.0,
            30.0,
            float(p["muac_cm"]),
            .1,
        )

        bmi = weight / ((height/100)**2)

        st.metric(
            "BMI terhitung",
            f"{bmi:.2f}",
            help="Digunakan sebagai fitur model, bukan diagnosis.",
        )

    preview_profile = {
        "age_months":age,
        "weight_kg":weight,
        "height_cm":height,
        "muac_cm":muac,
    }

    st.plotly_chart(
        profile_radar(preview_profile),
        use_container_width=True,
        config={"displayModeBar":False},
    )

    if st.button("Simpan Profil",type="primary"):
        st.session_state.child_profile = {
            "name":name,
            **preview_profile,
        }

        st.success("Profil anak berhasil disimpan.")


# ============================================================
# PENGETAHUAN GIZI
# ============================================================

elif page == "🥗 Pengetahuan Gizi":

    st.markdown("## Pengetahuan Gizi")
    st.caption(
        "Halaman edukasi untuk membantu memahami hasil analisis NutriVision."
    )

    st.plotly_chart(
        knowledge_chart(),
        use_container_width=True,
        config={"displayModeBar":False},
    )

    cards = [
        (
            "Energi",
            "Energi mendukung aktivitas dan pertumbuhan. Pada dashboard, persentase kontribusi menunjukkan seberapa besar makanan yang dianalisis berkontribusi terhadap acuan kelompok umur.",
        ),
        (
            "Protein",
            "Protein berperan dalam pembentukan dan pemeliharaan jaringan. Nilai rendah pada satu foto makanan tidak berarti anak terbukti kekurangan protein.",
        ),
        (
            "Zat Besi",
            "Zat besi berperan dalam pembentukan hemoglobin. Recommendation engine dapat memprioritaskan makanan yang lebih kaya zat besi bila kontribusinya rendah.",
        ),
        (
            "Kalsium",
            "Kalsium berkaitan dengan tulang dan gigi. Analisis NutriVision menggunakan data nutrisi makanan dan reference layer AKG.",
        ),
        (
            "Vitamin A",
            "Vitamin A berperan dalam fungsi penglihatan dan imunitas. Persentase pada dashboard adalah kontribusi makanan terhadap acuan, bukan hasil laboratorium anak.",
        ),
        (
            "Vitamin C",
            "Vitamin C berperan dalam berbagai fungsi metabolik dan membantu penyerapan zat besi non-heme.",
        ),
    ]

    c1,c2 = st.columns(2)

    for i,(title,text) in enumerate(cards):
        with (c1 if i%2==0 else c2):
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-card-title">{title}</div>
                    <div class="info-card-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# TENTANG APLIKASI
# ============================================================

else:

    st.markdown("## Tentang Aplikasi")

    a1,a2 = st.columns([1.2,1],gap="large")

    with a1:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">NutriVision AI</div>
                <div class="info-card-text">
                    NutriVision AI adalah prototype multimodal yang menggabungkan
                    data antropometri anak dan citra makanan untuk screening status
                    gizi, klasifikasi makanan, analisis kontribusi nutrisi terhadap
                    AKG, serta recommendation engine makanan Indonesia.
                </div>
            </div>

            <div class="info-card">
                <div class="info-card-title">Model & Explainability</div>
                <div class="info-card-text">
                    Klasifikasi citra menggunakan EfficientNetB0. Grad-CAM dipakai
                    untuk memvisualisasikan region citra yang relatif lebih
                    berpengaruh terhadap keputusan model.
                </div>
            </div>

            <div class="info-card">
                <div class="info-card-title">Batasan</div>
                <div class="info-card-text">
                    Sistem adalah prototype edukasi dan screening. Satu foto
                    makanan tidak mewakili asupan 24 jam dan rekomendasi bukan
                    terapi klinis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:
        flow_fig = go.Figure()

        nodes = [
            ("Data Anak", 5),
            ("Model Gizi", 4),
            ("Foto Makanan", 3),
            ("EfficientNetB0", 2),
            ("Analisis Nutrisi", 1),
            ("Rekomendasi", 0),
        ]

        for label,y in nodes:
            flow_fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[y],
                    mode="markers+text",
                    marker=dict(size=34,color="#FFC928"),
                    text=[label],
                    textposition="middle right",
                    textfont=dict(size=13,color="#292929"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        for i in range(len(nodes)-1):
            flow_fig.add_shape(
                type="line",
                x0=0,
                y0=nodes[i][1]-.22,
                x1=0,
                y1=nodes[i+1][1]+.22,
                line=dict(color="#D7C26A",width=2),
            )

        flow_fig.update_layout(
            title="Alur Sistem NutriVision",
            height=430,
            xaxis=dict(visible=False,range=[-.2,1.5]),
            yaxis=dict(visible=False,range=[-.5,5.5]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30,r=30,t=60,b=20),
        )

        st.plotly_chart(
            flow_fig,
            use_container_width=True,
            config={"displayModeBar":False},
        )
