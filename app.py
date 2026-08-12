from pathlib import Path
import base64
import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

from src.inference import (
    DEFAULT_ENSEMBLE_WEIGHTS,
    analyze_child_and_food,
    load_convnext_model,
    load_noisyvit_model,
    load_malnutrition_artifact,
    make_gradcam,
    overlay_gradcam,
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
ASSET_DIR = BASE_DIR / "assets"
LOGO_ICON_PATH = ASSET_DIR / "nutrivision_logo_icon.png"


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


LOGO_ICON_URI = image_data_uri(LOGO_ICON_PATH)

def _sidebar_initial_state():
    """
    Streamlit >= 1.60 mendukung state `locked`.
    Pada desktop sidebar akan selalu terbuka, sedangkan pada mobile
    perilakunya tetap aman sebagai drawer.
    """
    try:
        parts = str(st.__version__).split(".")
        major = int(parts[0])
        minor = int(parts[1])
        if (major, minor) >= (1, 60):
            return "locked"
    except Exception:
        pass

    return "auto"


st.set_page_config(
    page_title="NutriVision AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state=_sidebar_initial_state(),
)

# Minimal toolbar keeps essential app controls available while removing
# unnecessary developer chrome.
try:
    st.set_option("client.toolbarMode", "minimal")
except Exception:
    pass

# ============================================================
# SESSION STATE
# ============================================================

NAV_OPTIONS = [
    "Dashboard",
    "Riwayat Analisis",
    "Profil Anak",
    "Pengetahuan Gizi",
    "Tentang Aplikasi",
]

PAGE_TO_SLUG = {
    "Dashboard": "dashboard",
    "Riwayat Analisis": "riwayat",
    "Profil Anak": "profil",
    "Pengetahuan Gizi": "gizi",
    "Tentang Aplikasi": "tentang",
}

SLUG_TO_PAGE = {
    slug: page
    for page, slug in PAGE_TO_SLUG.items()
}


def _read_nav_query():
    try:
        value = st.query_params.get("nav")
    except Exception:
        try:
            params = st.experimental_get_query_params()
            value = params.get("nav", [None])
            if isinstance(value, list):
                value = value[0] if value else None
        except Exception:
            value = None

    if isinstance(value, list):
        value = value[0] if value else None

    return str(value).strip().lower() if value else None


def _write_nav_query(page_name):
    slug = PAGE_TO_SLUG.get(page_name, "dashboard")
    try:
        st.query_params["nav"] = slug
    except Exception:
        try:
            st.experimental_set_query_params(nav=slug)
        except Exception:
            pass


query_nav = _read_nav_query()

if "nav_page" not in st.session_state:
    st.session_state.nav_page = SLUG_TO_PAGE.get(
        query_nav,
        "Dashboard",
    )
elif query_nav in SLUG_TO_PAGE:
    st.session_state.nav_page = SLUG_TO_PAGE[query_nav]


def _sync_sidebar_navigation():
    st.session_state.nav_page = st.session_state.sidebar_nav
    _write_nav_query(st.session_state.nav_page)


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

if "food_image_bytes" not in st.session_state:
    st.session_state.food_image_bytes = None

if "food_image_name" not in st.session_state:
    st.session_state.food_image_name = None

# ============================================================
# STYLE — mempertahankan UI putih-kuning yang sekarang
# ============================================================


st.markdown(
    """
<style>
:root {
    --gold:#FFC928;
    --gold-2:#F5B800;
    --gold-soft:#FFF7D8;
    --navy:#0D1B34;
    --navy-2:#112748;
    --navy-3:#172C4F;
    --ink:#14203A;
    --muted:#7E899E;
    --line:#E6EAF1;
    --bg:#F6F8FC;
    --white:#FFFFFF;
    --blue-soft:#EEF5FF;
    --purple:#7B3DBA;
    --shadow:0 10px 28px rgba(20,32,58,.065);
    --shadow-lg:0 18px 46px rgba(20,32,58,.10);
}

* { box-sizing:border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 92% 5%, rgba(255,201,40,.07), transparent 25rem),
        linear-gradient(180deg,#FBFCFE 0%,#F5F7FB 100%) !important;
    color:var(--ink) !important;
    font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}

/* Keep a tiny transparent header so Streamlit can expose the
   sidebar-open control when needed. Other toolbar chrome stays hidden. */
[data-testid="stHeader"] {
    height:46px !important;
    min-height:46px !important;
    background:rgba(247,249,252,.90) !important;
    backdrop-filter:blur(10px);
    border-bottom:1px solid rgba(231,235,241,.65);
}

[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu,
footer {
    display:none !important;
    visibility:hidden !important;
}

/* Do not hide stToolbar itself. On narrow screens Streamlit can place
   sidebar controls inside/near this header structure. toolbarMode=minimal
   removes unnecessary items without destroying the sidebar trigger. */
[data-testid="stToolbar"] {
    visibility:visible !important;
}

/* Keep the collapsed/open-sidebar control available.
   This is especially important on mobile and on older Streamlit versions. */
[data-testid="stSidebarCollapsedControl"] {
    display:flex !important;
    visibility:visible !important;
    opacity:1 !important;
    z-index:10000 !important;
}

.block-container {
    max-width:1480px;
    padding:1.65rem 1.85rem 3.5rem 1.85rem !important;
}

/* ============================================================
   SIDEBAR
   ============================================================ */
[data-testid="stSidebar"] {
    min-width:248px !important;
    max-width:248px !important;
    background:
        radial-gradient(circle at 30% 6%,rgba(34,67,116,.34),transparent 15rem),
        linear-gradient(180deg,#0E1D37 0%,#0B1A31 100%) !important;
    border-right:1px solid rgba(255,255,255,.055);
}
[data-testid="stSidebar"] > div:first-child {
    padding:1.05rem .9rem .85rem !important;
    overflow-y:auto !important;
    overflow-x:hidden !important;
}
[data-testid="stSidebar"] * { color:#F6F8FC; }

.brand {
    display:flex;
    flex-direction:column;
    align-items:center;
    text-align:center;
    padding:2px 4px 15px;
}
.brand-logo-img {
    width:62px;
    height:62px;
    object-fit:contain;
    filter:drop-shadow(0 10px 18px rgba(255,201,40,.16));
    margin-bottom:4px;
}
.brand-name {
    font-size:1.05rem;
    font-weight:950;
    letter-spacing:-.035em;
    color:#FFF !important;
}
.brand-name .vision { color:var(--gold) !important; }
.brand-sub {
    color:#9FAAC0 !important;
    font-size:.57rem;
    margin-top:3px;
}

/* Remove the native radio bullets completely. */
[data-testid="stSidebar"] [role="radiogroup"] {
    gap:3px !important;
    margin-top:2px;
}
[data-testid="stSidebar"] [role="radiogroup"] input,
[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child,
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child:not([data-testid="stMarkdownContainer"]) {
    display:none !important;
}
[data-testid="stSidebar"] [role="radiogroup"] > label {
    position:relative;
    display:flex !important;
    align-items:center;
    gap:9px;
    width:100%;
    min-height:43px;
    padding:6px 9px !important;
    margin:0 !important;
    border-radius:13px !important;
    border:1px solid transparent !important;
    background:transparent !important;
    transition:.16s ease;
    cursor:pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] > label::before {
    content:"";
    flex:0 0 29px;
    width:29px;
    height:29px;
    border-radius:9px;
    display:grid;
    place-items:center;
    color:#EDF2FA;
    font-size:.90rem;
    font-weight:800;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(1)::before { content:"▦"; }
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(2)::before { content:"◷"; }
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(3)::before { content:"♙"; }
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(4)::before { content:"▤"; }
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(5)::before { content:"ⓘ"; }
[data-testid="stSidebar"] [role="radiogroup"] > label p {
    color:#F0F3F9 !important;
    font-size:.75rem !important;
    font-weight:720 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background:rgba(255,255,255,.035) !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background:linear-gradient(100deg,rgba(255,201,40,.10),rgba(255,255,255,.025)) !important;
    border:1px solid rgba(255,201,40,.16) !important;
    box-shadow:inset 3px 0 0 var(--gold);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked)::before {
    color:#172039;
    background:linear-gradient(145deg,#FFD94D,#F6BE18);
    box-shadow:0 6px 14px rgba(255,201,40,.16);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
    color:#FFD84D !important;
    font-weight:850 !important;
}

.about-card {
    position:relative;
    margin-top:19px;
    padding:13px;
    border:1px solid rgba(255,255,255,.10);
    border-radius:16px;
    background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.025));
    color:#C2CAD8 !important;
    font-size:.63rem;
    line-height:1.55;
}
.about-card:after {
    content:"✦";
    position:absolute;
    right:14px;
    top:12px;
    color:var(--gold);
    font-size:1.25rem;
}
.about-card b { color:#FFF !important; font-size:.76rem; }
.about-mini {
    margin-top:9px;
    padding-top:8px;
    border-top:1px solid rgba(255,255,255,.10);
    color:#9BA7BA !important;
    font-size:.64rem;
}


/* Hide ALL native BaseWeb radio controls inside sidebar.
   Navigation uses our own pseudo-icon (::before), so the extra white/green
   radio circle must never be visible. */
[data-testid="stSidebar"] input[type="radio"],
[data-testid="stSidebar"] label[data-baseweb="radio"] input,
[data-testid="stSidebar"] [role="radiogroup"] label input {
    position:absolute !important;
    opacity:0 !important;
    pointer-events:none !important;
    width:0 !important;
    height:0 !important;
}

/* Streamlit/BaseWeb versions use slightly different wrappers. */
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child,
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child:has(input[type="radio"]),
[data-testid="stSidebar"] [role="radiogroup"] label div:has(> input[type="radio"]) {
    display:none !important;
    width:0 !important;
    min-width:0 !important;
    margin:0 !important;
    padding:0 !important;
}

/* If the control is represented by an SVG/circular mark, remove only the
   radio-control wrapper; our text/icons remain untouched. */
[data-testid="stSidebar"] [role="radiogroup"] label svg[aria-hidden="true"] {
    display:none !important;
}

/* ============================================================
   HERO
   ============================================================ */
.hero-shell {
    position:relative;
    overflow:hidden;
    min-height:250px;
    margin:0 0 24px 0;
    padding:28px 32px;
    border:1px solid #E7EBF1;
    border-radius:26px;
    background:
        radial-gradient(circle at 91% 15%,rgba(255,201,40,.20),transparent 18rem),
        radial-gradient(circle at 72% 102%,rgba(36,155,101,.07),transparent 12rem),
        linear-gradient(110deg,#FFFFFF 0%,#FFFFFF 56%,#FFFDF4 100%);
    box-shadow:var(--shadow-lg);
}
.hero-shell:before {
    content:"";
    position:absolute;
    width:250px;
    height:250px;
    right:-80px;
    bottom:-150px;
    border-radius:50%;
    border:25px solid rgba(255,201,40,.09);
}
.hero-inner {
    position:relative;
    z-index:2;
    display:grid;
    grid-template-columns:1.28fr .50fr .92fr;
    align-items:center;
    gap:22px;
    min-height:190px;
}
.hero-eyebrow {
    display:inline-flex;
    align-items:center;
    gap:7px;
    padding:7px 12px;
    border-radius:999px;
    background:#FFF8DC;
    border:1px solid #F2D77B;
    color:#876600;
    font-size:.66rem;
    font-weight:900;
    letter-spacing:.03em;
    margin-bottom:12px;
}
.hero-eyebrow:before { content:"✦"; color:#DBA300; }
.title {
    font-size:2.55rem;
    line-height:1;
    font-weight:950;
    letter-spacing:-.055em;
    color:#13203A;
}
.title span { color:#EEB200; }
.subtitle {
    max-width:650px;
    margin-top:13px;
    color:#758198;
    font-size:.88rem;
    line-height:1.65;
}
.hero-logo-wrap {
    display:grid;
    place-items:center;
}
.hero-logo-img {
    width:152px;
    height:152px;
    object-fit:contain;
    filter:drop-shadow(0 16px 25px rgba(255,201,40,.12));
}
.hero-meta {
    display:flex;
    flex-direction:column;
    align-items:flex-end;
    justify-content:center;
    gap:58px;
}
.date-pill {
    padding:9px 13px;
    border-radius:12px;
    border:1px solid #E1E6EF;
    background:rgba(255,255,255,.88);
    color:#536079;
    font-size:.72rem;
    font-weight:800;
    box-shadow:0 5px 16px rgba(20,32,58,.045);
    white-space:nowrap;
}
.tech-row {
    display:flex;
    justify-content:flex-end;
    gap:9px;
    flex-wrap:nowrap;
}
.tech-chip {
    padding:8px 11px;
    border-radius:999px;
    background:rgba(255,255,255,.88);
    border:1px solid #E3E7EF;
    color:#59657A;
    font-size:.61rem;
    font-weight:820;
    white-space:nowrap;
    box-shadow:0 5px 14px rgba(20,32,58,.035);
}

/* ============================================================
   WORKFLOW
   ============================================================ */
.section-kicker {
    color:#A47400;
    font-size:.67rem;
    font-weight:950;
    text-transform:uppercase;
    letter-spacing:.095em;
    margin-bottom:5px;
}
.section-title {
    font-size:1.30rem;
    font-weight:950;
    color:#152039;
    letter-spacing:-.03em;
}
.section-sub {
    color:#8791A4;
    font-size:.75rem;
    margin-top:3px;
}
.workflow-strip {
    display:grid;
    grid-template-columns:1fr 44px 1fr 44px 1fr;
    align-items:center;
    margin-top:14px;
    margin-bottom:18px;
    padding:12px 16px;
    border:1px solid #E6EAF1;
    border-radius:16px;
    background:#FFFFFF;
    box-shadow:var(--shadow);
}
.workflow-item {
    display:flex;
    align-items:center;
    gap:12px;
    padding:3px 0;
    min-width:0;
}
.workflow-number {
    width:38px;
    height:38px;
    flex:0 0 38px;
    display:grid;
    place-items:center;
    border-radius:50%;
    background:linear-gradient(145deg,#FFDB69,#FFC02D);
    color:#263146;
    font-size:.75rem;
    font-weight:950;
    box-shadow:0 6px 14px rgba(255,192,45,.17);
}
.workflow-title { color:#27334A; font-size:.75rem; font-weight:900; }
.workflow-sub { color:#929CAF; font-size:.62rem; margin-top:3px; }
.workflow-arrow {
    width:31px;
    height:31px;
    display:grid;
    place-items:center;
    justify-self:center;
    border-radius:50%;
    background:#F6F8FB;
    color:#17243C;
    font-size:1.12rem;
    font-weight:900;
}

/* ============================================================
   ANALYSIS CARDS
   ============================================================ */
.st-key-child_card,
.st-key-photo_card,
.st-key-action_card {
    border-radius:20px;
    height:455px;
    min-height:455px;
    max-height:455px;
    overflow:hidden;
}

/* Ketiga kartu kembali sejajar, tetapi isi kiri/kanan dibuat lebih proporsional. */
.st-key-child_card,
.st-key-photo_card {
    background:#FFFFFF;
    border:1px solid #E5E9F0;
    box-shadow:var(--shadow);
}

.st-key-child_card {
    padding:24px 22px 20px;
}

.st-key-photo_card {
    padding:19px 18px 16px;
}

.st-key-action_card {
    padding:24px 22px 20px;
    background:
        radial-gradient(circle at 87% 12%,rgba(255,255,255,.08),transparent 8rem),
        linear-gradient(145deg,#10284B,#081B37);
    border:1px solid rgba(18,42,78,.7);
    box-shadow:0 18px 40px rgba(11,28,54,.18);
}

/* Isi kartu tetap natural, dengan tinggi luar yang sejajar. */
.st-key-child_card [data-testid="stVerticalBlock"],
.st-key-photo_card [data-testid="stVerticalBlock"],
.st-key-action_card [data-testid="stVerticalBlock"] {
    justify-content:flex-start !important;
    align-content:flex-start !important;
}

/* Card foto tidak boleh tumbuh mengikuti ukuran gambar asli. */
.st-key-photo_card [data-testid="stImage"] {
    margin-top:10px;
}

.st-key-photo_card [data-testid="stImage"] img {
    width:100% !important;
    height:190px !important;
    max-height:190px !important;
    object-fit:cover !important;
    object-position:center !important;
    border-radius:14px !important;
}

.st-key-photo_card [data-testid="stImage"] [data-testid="stImageCaption"] {
    margin-top:5px !important;
    font-size:.67rem !important;
}

/* Action card dibuat seperti satu panel utuh dan tombol berada stabil. */
.st-key-action_card [data-testid="stVerticalBlock"] {
    height:100%;
}

.st-key-action_card .cta-box {
    min-height:244px;
    padding:3px 0 8px;
}

.st-key-action_card .action-note {
    margin-top:12px;
}
.panel-title {
    font-size:1.10rem;
    font-weight:950;
    color:#1A2740;
    margin-bottom:5px;
}
.panel-sub {
    font-size:.78rem;
    color:#8C96A8;
    line-height:1.5;
    margin-bottom:15px;
}
.st-key-action_card .panel-title,
.st-key-action_card .panel-sub { color:#FFFFFF !important; }

label[data-testid="stWidgetLabel"] p {
    color:#536079 !important;
    font-size:.80rem !important;
    font-weight:850 !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    min-height:46px;
    font-size:.86rem !important;
    background:#FFFFFF !important;
    color:#1E2B43 !important;
    border:1px solid #E2E6ED !important;
}
[data-testid="stNumberInput"] button {
    background:#FFFFFF !important;
    color:#22314D !important;
    border-color:#E2E6ED !important;
}

.st-key-child_card [data-testid="stHorizontalBlock"] {
    gap:14px !important;
    margin-bottom:10px !important;
}

.st-key-child_card [data-testid="stNumberInput"] {
    margin-bottom:4px !important;
}

.st-key-child_card [data-testid="stMetric"] {
    min-height:82px;
    padding:13px 14px;
}
[data-testid="stMetric"] {
    padding:10px 12px;
    border-radius:12px;
    background:#F7F9FC;
    border:1px solid #EEF1F5;
}
[data-testid="stMetricLabel"] {
    color:#8994A7 !important;
    font-size:.76rem !important;
    font-weight:700 !important;
}
[data-testid="stMetricValue"] {
    color:#14203A !important;
    font-size:1.24rem !important;
    font-weight:950 !important;
}

[data-testid="stFileUploader"] {
    min-height:132px;
    background:linear-gradient(180deg,#FFFDF8,#FFF9EC);
    border:1.5px dashed #E8B522;
    border-radius:16px;
    padding:6px;
}
[data-testid="stFileUploaderDropzone"] {
    min-height:118px;
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    background:transparent !important;
    border:none !important;
}
[data-testid="stFileUploaderDropzone"] button {
    border-radius:10px !important;
    border:1px solid #E4B62C !important;
    background:#FFF8DD !important;
    color:#725700 !important;
    font-weight:850 !important;
}
[data-testid="stImage"] img {
    border-radius:14px !important;
    border:1px solid #E7EAF0;
}

.st-key-photo_card .stButton > button {
    min-height:34px !important;
    height:34px !important;
    margin-top:2px !important;
    border-radius:10px !important;
    background:#F7F9FC !important;
    border:1px solid #E4E8EF !important;
    color:#56637A !important;
    font-size:.68rem !important;
    font-weight:800 !important;
    box-shadow:none !important;
}

.st-key-photo_card .stButton > button:hover {
    border-color:#E4B62C !important;
    color:#8A6900 !important;
    background:#FFF9E6 !important;
}

.photo-tip,
.action-note {
    display:flex;
    gap:10px;
    align-items:flex-start;
    padding:11px 12px;
    border-radius:12px;
    font-size:.67rem;
    line-height:1.45;
}
.photo-tip {
    margin-top:12px;
    color:#2C5E9F;
    background:#EEF5FF;
    border:1px solid #DFEAFA;
}
.photo-tip-icon,
.note-icon { font-size:1rem; flex:0 0 auto; }

/* CTA is transparent because its whole column is already dark. */
.cta-box {
    position:relative;
    min-height:176px;
    padding:3px 0 10px;
    background:transparent;
    border:none;
    box-shadow:none;
}
.cta-box:before {
    content:"✦";
    position:absolute;
    right:2px;
    top:-4px;
    color:var(--gold);
    font-size:2rem;
}
.cta-badge {
    display:inline-block;
    padding:7px 10px;
    border-radius:999px;
    background:rgba(255,201,40,.10);
    border:1px solid rgba(255,201,40,.16);
    color:#FFD347;
    font-size:.66rem;
    font-weight:900;
    margin-bottom:30px;
}
.cta-title {
    color:#FFFFFF;
    font-size:1.42rem;
    font-weight:950;
    letter-spacing:-.03em;
}
.cta-sub {
    margin-top:12px;
    color:#B9C4D5;
    font-size:.84rem;
    line-height:1.65;
    max-width:96%;
}
.st-key-action_card .stButton > button {
    width:100%;
    min-height:54px;
    font-size:.82rem !important;
    border-radius:11px !important;
    background:linear-gradient(100deg,#FFD85D,#FFC122) !important;
    border:1px solid #F4B900 !important;
    color:#16213B !important;
    font-weight:950 !important;
    box-shadow:0 9px 22px rgba(255,193,34,.18);
}
.st-key-action_card .stButton > button:disabled {
    opacity:.58 !important;
    color:#6A6143 !important;
}
.action-note {
    margin-top:16px;
    padding:13px 14px;
    min-height:58px;
    color:#59677E;
    background:#F8FAFD;
    border:1px solid #E9EDF3;
    font-size:.74rem;
}

/* Keep result-area styling from the same visual language. */
.results-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin-top:11px; }
.result-card { min-height:148px; padding:15px; border:1px solid #E7EAF0; border-radius:16px; background:#FFF; box-shadow:var(--shadow); }
.result-icon { width:34px; height:34px; display:grid; place-items:center; border-radius:11px; background:#FFF6D5; margin-bottom:10px; }
.result-label { color:#8994A7; font-size:.66rem; font-weight:760; margin-bottom:7px; }
.result-value { color:#1A2740; font-size:1rem; font-weight:950; }
.result-sub { color:#98A1B1; font-size:.64rem; margin-top:5px; }
.green { color:#198F5D !important; }
.yellow { color:#BF8B00 !important; }
.ring { width:64px; height:64px; border-radius:50%; display:grid; place-items:center; position:relative; background:conic-gradient(#F5C230 calc(var(--v)*1%),#EEF0F4 0); }
.ring:before { content:""; width:46px; height:46px; border-radius:50%; background:#FFF; position:absolute; }
.ring span { position:relative; z-index:2; color:#1A2740; font-size:.74rem; font-weight:950; }

[data-baseweb="tab-list"] { gap:6px !important; padding:5px; border:1px solid #E6EAF1; border-radius:13px; background:#F2F4F8; }
button[data-baseweb="tab"] { min-height:39px !important; border-radius:9px !important; color:#758096 !important; font-weight:760 !important; }
button[data-baseweb="tab"][aria-selected="true"] { background:#FFF !important; color:#1C2942 !important; box-shadow:0 4px 11px rgba(20,32,58,.055); }
div[data-testid="stPlotlyChart"] { background:#FFF; border:1px solid #E7EAF0; border-radius:17px; box-shadow:var(--shadow); padding:4px; }
[data-testid="stDataFrame"] { border:1px solid #E7EAF0; border-radius:15px; overflow:hidden; }
.rec-card { display:grid; grid-template-columns:34px 1fr auto; gap:10px; align-items:center; padding:11px; border:1px solid #E7EAF0; border-radius:13px; background:#FFF; margin-bottom:8px; }
.rec-rank { width:30px; height:30px; display:grid; place-items:center; border-radius:50%; background:#FFC928; color:#2B2B20; font-weight:950; }
.rec-name { color:#1A2740; font-size:.83rem; font-weight:900; }
.rec-reason { color:#8B95A7; font-size:.66rem; margin-top:3px; }
.rec-score { color:#158A59; font-size:.72rem; font-weight:900; }
.insight { margin-top:11px; padding:16px 17px; border:1px solid #EED889; border-radius:15px; background:linear-gradient(90deg,#FFFDF5,#FFF8DA); color:#5A523E; font-size:.77rem; line-height:1.65; }
.priority { display:inline-block; margin:8px 4px 0 0; padding:4px 8px; border-radius:999px; border:1px solid #EBD16D; background:#FFF; color:#826700; font-weight:820; font-size:.65rem; }
.info-card { padding:17px; border:1px solid #E7EAF0; border-radius:16px; background:#FFF; margin-bottom:10px; box-shadow:var(--shadow); }
.info-card-title { font-size:.90rem; font-weight:900; color:#1A2740; margin-bottom:5px; }
.info-card-text { color:#7D879B; font-size:.77rem; line-height:1.6; }

/* ============================================================
   MODERN SECTION COMPONENTS
   ============================================================ */
.section-card {
    padding:18px 18px 16px;
    border:1px solid #E7EAF0;
    border-radius:18px;
    background:#FFFFFF;
    box-shadow:var(--shadow);
}
.section-card-dark {
    padding:18px 18px 16px;
    border:1px solid rgba(18,42,78,.60);
    border-radius:18px;
    background:
        radial-gradient(circle at 90% 15%,rgba(255,255,255,.08),transparent 8rem),
        linear-gradient(145deg,#10284B,#081B37);
    box-shadow:0 18px 40px rgba(11,28,54,.18);
    color:#FFF;
}
.kpi-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
    gap:12px;
    margin:10px 0 18px;
}
.kpi-card {
    padding:14px 14px 13px;
    border:1px solid #E7EAF0;
    border-radius:16px;
    background:linear-gradient(180deg,#FFFFFF,#FBFCFE);
    box-shadow:var(--shadow);
}
.kpi-card.gold { background:linear-gradient(180deg,#FFF9E8,#FFFFFF); border-color:#F4E2A2; }
.kpi-card.navy { background:linear-gradient(180deg,#F6F9FE,#FFFFFF); }
.kpi-label {
    font-size:.75rem;
    color:#8A95A8;
    font-weight:800;
    margin-bottom:5px;
}
.kpi-value {
    font-size:1.35rem;
    line-height:1.05;
    color:#15213B;
    font-weight:950;
    letter-spacing:-.03em;
}
.kpi-sub {
    margin-top:5px;
    color:#7F8A9E;
    font-size:.72rem;
    line-height:1.45;
}
.tag-row {
    display:flex;
    flex-wrap:wrap;
    gap:8px;
    margin-top:10px;
}
.tag-chip {
    display:inline-flex;
    align-items:center;
    gap:6px;
    padding:8px 11px;
    border-radius:999px;
    background:#FFF8DC;
    border:1px solid #F1D97E;
    color:#8B6800;
    font-size:.68rem;
    font-weight:850;
}
.tile-grid-3 {
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:14px;
}
.tile-grid-2 {
    display:grid;
    grid-template-columns:repeat(2,minmax(0,1fr));
    gap:14px;
}
.nutrient-tile {
    padding:15px 15px 14px;
    border:1px solid #E7EAF0;
    border-radius:18px;
    background:#FFFFFF;
    box-shadow:var(--shadow);
}
.nutrient-icon {
    width:36px;
    height:36px;
    border-radius:12px;
    display:grid;
    place-items:center;
    margin-bottom:10px;
    background:#FFF3C6;
    color:#9B7400;
    font-size:1rem;
}
.nutrient-title {
    font-size:.94rem;
    font-weight:900;
    color:#1A2740;
    margin-bottom:4px;
}
.nutrient-meta {
    display:inline-flex;
    align-items:center;
    gap:7px;
    padding:5px 9px;
    border-radius:999px;
    margin-bottom:9px;
    background:#F6F8FC;
    color:#77839A;
    font-size:.63rem;
    font-weight:800;
}
.nutrient-text {
    color:#7D879B;
    font-size:.76rem;
    line-height:1.65;
}
.timeline-shell {
    position:relative;
    padding:14px 14px 12px 18px;
}
.timeline-item {
    position:relative;
    display:flex;
    gap:14px;
    align-items:flex-start;
    padding:0 0 18px 0;
}
.timeline-item:last-child { padding-bottom:0; }
.timeline-dot {
    width:34px;
    height:34px;
    flex:0 0 34px;
    border-radius:50%;
    display:grid;
    place-items:center;
    background:linear-gradient(145deg,#FFD958,#F5BE19);
    color:#172039;
    font-size:.78rem;
    font-weight:900;
    box-shadow:0 8px 16px rgba(255,201,40,.18);
}
.timeline-item:not(:last-child)::after {
    content:"";
    position:absolute;
    left:16px;
    top:34px;
    bottom:-2px;
    width:2px;
    background:linear-gradient(180deg,#F4CF56,#E7EDF6);
}
.timeline-title {
    font-size:.88rem;
    font-weight:900;
    color:#1A2740;
    margin-bottom:3px;
}
.timeline-text {
    color:#7D879B;
    font-size:.74rem;
    line-height:1.58;
}
.flow-grid {
    display:grid;
    grid-template-columns:repeat(2,minmax(0,1fr));
    gap:12px;
    margin-top:12px;
}
.flow-node {
    padding:14px 14px 12px;
    border:1px solid #E7EAF0;
    border-radius:16px;
    background:#FFFFFF;
    box-shadow:var(--shadow);
}
.flow-node .num {
    display:inline-grid;
    place-items:center;
    width:28px;
    height:28px;
    border-radius:10px;
    background:#FFF2BF;
    color:#8A6500;
    font-size:.72rem;
    font-weight:900;
    margin-bottom:9px;
}
.flow-node .title {
    font-size:.88rem;
    font-weight:900;
    color:#1A2740;
    margin-bottom:4px;
    line-height:1.2;
}
.flow-node .text {
    color:#7D879B;
    font-size:.73rem;
    line-height:1.56;
}
.model-pill-grid {
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:12px;
    margin:10px 0 12px;
}
.model-pill {
    padding:14px 14px 13px;
    border-radius:16px;
    border:1px solid #E7EAF0;
    background:#FFFFFF;
    box-shadow:var(--shadow);
}
.model-pill-label {
    color:#8893A6;
    font-size:.70rem;
    font-weight:800;
    margin-bottom:5px;
}
.model-pill-value {
    color:#15213B;
    font-size:1.08rem;
    font-weight:950;
    line-height:1.2;
}
.model-pill-note {
    color:#8A95A8;
    font-size:.66rem;
    line-height:1.45;
    margin-top:4px;
}
.rec-hero {
    padding:16px 18px;
    border:1px solid #F0E1A5;
    border-radius:18px;
    background:linear-gradient(120deg,#FFF8DF,#FFFFFF);
    box-shadow:var(--shadow);
    margin-bottom:16px;
}
.rec-hero-title {
    font-size:1rem;
    font-weight:900;
    color:#1A2740;
    margin-bottom:5px;
}
.rec-hero-text {
    color:#7D879B;
    font-size:.77rem;
    line-height:1.62;
}
.recommend-card-rich {
    padding:16px 16px 15px;
    border:1px solid #E7EAF0;
    border-radius:18px;
    background:#FFFFFF;
    box-shadow:var(--shadow);
    min-height:175px;
}
.rec-rank-badge {
    display:inline-flex;
    align-items:center;
    gap:7px;
    padding:6px 10px;
    border-radius:999px;
    background:#FFF5CF;
    color:#8A6700;
    border:1px solid #F1DA82;
    font-size:.66rem;
    font-weight:900;
}
.recommend-card-rich .food {
    margin-top:13px;
    font-size:1.02rem;
    font-weight:950;
    color:#15213B;
    line-height:1.3;
}
.recommend-card-rich .score {
    margin-top:10px;
    color:#0A9F76;
    font-size:1.05rem;
    font-weight:950;
}
.recommend-card-rich .reason {
    margin-top:8px;
    color:#8A95A8;
    font-size:.73rem;
    line-height:1.58;
}
.insight-soft {
    padding:15px 16px;
    border:1px solid #E8EDF5;
    border-radius:16px;
    background:#FBFCFE;
}
.dark-note {
    padding:14px 14px 13px;
    border:1px solid rgba(255,255,255,.10);
    border-radius:14px;
    background:rgba(255,255,255,.05);
    color:#C2CBDB;
    font-size:.72rem;
    line-height:1.58;
}

/* ============================================================
   ALIGNMENT & REFINED PANELS
   ============================================================ */

.st-key-profile_form_card,
.st-key-profile_visual_card {
    height:540px;
    min-height:540px;
    max-height:540px;
    overflow:hidden;
    padding:20px;
    border:1px solid #E7EAF0;
    border-radius:20px;
    background:#FFFFFF;
    box-shadow:var(--shadow);
}

.st-key-profile_form_card [data-testid="stVerticalBlock"],
.st-key-profile_visual_card [data-testid="stVerticalBlock"] {
    height:100%;
    justify-content:flex-start !important;
}

.st-key-profile_form_card .stButton > button {
    margin-top:10px;
}

.profile-form-head,
.profile-visual-head {
    margin-bottom:14px;
}

.profile-form-title,
.profile-visual-title {
    color:#17233D;
    font-size:1.05rem;
    font-weight:950;
    letter-spacing:-.02em;
}

.profile-form-sub,
.profile-visual-sub {
    margin-top:4px;
    color:#8994A7;
    font-size:.72rem;
    line-height:1.5;
}

.st-key-profile_visual_card div[data-testid="stPlotlyChart"] {
    box-shadow:none;
    border:none;
    padding:0;
    margin-top:4px;
}

.profile-mini-grid {
    display:grid;
    grid-template-columns:repeat(2,minmax(0,1fr));
    gap:10px;
    margin-bottom:8px;
}

.profile-mini {
    padding:11px 12px;
    border:1px solid #E8ECF2;
    border-radius:13px;
    background:#F9FAFC;
}

.profile-mini .label {
    color:#8D97A9;
    font-size:.64rem;
    font-weight:800;
}

.profile-mini .value {
    margin-top:4px;
    color:#17233D;
    font-size:.88rem;
    font-weight:950;
}

/* About page: keep left and right sections visually aligned. */
.st-key-about_capabilities,
.st-key-about_flow {
    height:510px;
    min-height:510px;
    max-height:510px;
    overflow:hidden;
}

.st-key-about_capabilities {
    padding:0;
}

.st-key-about_capabilities .tile-grid-2 {
    height:100%;
    grid-template-rows:1fr 1fr;
}

.st-key-about_capabilities .info-card {
    height:100%;
    margin-bottom:0;
    padding:18px;
}

.st-key-about_flow {
    padding:20px;
    border:1px solid #E7EAF0;
    border-radius:20px;
    background:#FFFFFF;
    box-shadow:var(--shadow);
}

.st-key-about_flow .timeline-shell {
    height:440px;
    overflow:hidden;
    box-shadow:none;
    border:none;
    padding:8px 4px 0;
}

/* Recommendation visualization */
.rec-visual-head {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:16px;
    margin-bottom:10px;
}

.rec-visual-title {
    color:#17233D;
    font-size:1rem;
    font-weight:950;
}

.rec-visual-sub {
    color:#8994A7;
    font-size:.70rem;
    margin-top:3px;
}

.rec-legend {
    display:flex;
    gap:8px;
    flex-wrap:wrap;
}

.rec-legend span {
    padding:6px 9px;
    border-radius:999px;
    background:#F7F9FC;
    border:1px solid #E7EBF1;
    color:#748096;
    font-size:.62rem;
    font-weight:800;
}

/* Grad-CAM */
.xai-hero {
    display:grid;
    grid-template-columns:1.35fr repeat(3,.55fr);
    gap:12px;
    margin-bottom:16px;
}

.xai-intro,
.xai-stat {
    border:1px solid #E7EAF0;
    border-radius:17px;
    background:#FFFFFF;
    box-shadow:var(--shadow);
}

.xai-intro {
    padding:16px 17px;
    background:
        radial-gradient(circle at 92% 5%,rgba(255,201,40,.13),transparent 8rem),
        #FFFFFF;
}

.xai-intro-title {
    color:#17233D;
    font-size:1rem;
    font-weight:950;
}

.xai-intro-text {
    margin-top:5px;
    color:#7F899D;
    font-size:.72rem;
    line-height:1.55;
}

.xai-stat {
    padding:13px;
}

.xai-stat-label {
    color:#8D97A9;
    font-size:.62rem;
    font-weight:800;
    margin-bottom:5px;
}

.xai-stat-value {
    color:#17233D;
    font-size:.93rem;
    line-height:1.25;
    font-weight:950;
    word-break:break-word;
}

.st-key-grad_original,
.st-key-grad_heatmap,
.st-key-grad_overlay {
    padding:13px 13px 10px;
    border:1px solid #E7EAF0;
    border-radius:18px;
    background:#FFFFFF;
    box-shadow:var(--shadow);
}

.grad-card-head {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:10px;
    margin-bottom:10px;
}

.grad-card-title {
    color:#17233D;
    font-size:.82rem;
    font-weight:900;
}

.grad-card-badge {
    padding:5px 8px;
    border-radius:999px;
    background:#FFF5CF;
    border:1px solid #F1DC8D;
    color:#8A6700;
    font-size:.58rem;
    font-weight:850;
}

.st-key-grad_original [data-testid="stImage"] img,
.st-key-grad_heatmap [data-testid="stImage"] img,
.st-key-grad_overlay [data-testid="stImage"] img {
    border-radius:14px !important;
    aspect-ratio:1 / 1;
    object-fit:cover;
}

.xai-explain-grid {
    display:grid;
    grid-template-columns:1.3fr .7fr;
    gap:12px;
    margin-top:14px;
}

.xai-explain,
.xai-layer {
    padding:15px 16px;
    border-radius:16px;
}

.xai-explain {
    background:linear-gradient(120deg,#FFF9E5,#FFFFFF);
    border:1px solid #EED889;
    color:#5C543E;
    font-size:.74rem;
    line-height:1.65;
}

.xai-layer {
    background:#10284B;
    border:1px solid #193862;
    color:#C3CDDC;
}

.xai-layer .label {
    color:#8FA2BE;
    font-size:.62rem;
    font-weight:800;
}

.xai-layer .value {
    margin-top:6px;
    color:#FFFFFF;
    font-size:.75rem;
    line-height:1.5;
    font-weight:800;
    word-break:break-all;
}


/* ============================================================
   RECOMMENDATION RANKING BOARD
   ============================================================ */
.ranking-board {
    padding:16px;
    border:1px solid #E7EAF0;
    border-radius:19px;
    background:#FFFFFF;
    box-shadow:var(--shadow);
    margin-bottom:16px;
}
.ranking-head {
    display:grid;
    grid-template-columns:52px 1.7fr .9fr .8fr;
    gap:12px;
    padding:0 12px 9px;
    color:#929CAF;
    font-size:.63rem;
    font-weight:850;
    text-transform:uppercase;
    letter-spacing:.045em;
}
.ranking-row {
    display:grid;
    grid-template-columns:52px 1.7fr .9fr .8fr;
    gap:12px;
    align-items:center;
    padding:12px;
    border:1px solid #EBEEF3;
    border-radius:15px;
    background:#FBFCFE;
    margin-bottom:9px;
}
.ranking-row:last-child { margin-bottom:0; }
.ranking-row.top {
    background:linear-gradient(100deg,#FFF8DC,#FFFFFF);
    border-color:#EFD477;
    box-shadow:0 8px 18px rgba(255,201,40,.08);
}
.rank-number {
    width:36px;
    height:36px;
    display:grid;
    place-items:center;
    border-radius:12px;
    background:#EEF2F8;
    color:#536179;
    font-size:.78rem;
    font-weight:950;
}
.ranking-row.top .rank-number {
    background:linear-gradient(145deg,#FFD85A,#FFC224);
    color:#172039;
}
.rank-food { min-width:0; }
.rank-food-name {
    color:#17233D;
    font-size:.85rem;
    font-weight:950;
    line-height:1.3;
}
.rank-reason {
    margin-top:3px;
    color:#8994A7;
    font-size:.64rem;
    line-height:1.45;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
}
.rank-score-label {
    display:flex;
    justify-content:space-between;
    gap:8px;
    color:#718097;
    font-size:.62rem;
    margin-bottom:5px;
}
.rank-score-label b {
    color:#17233D;
    font-size:.72rem;
}
.rank-progress {
    height:8px;
    overflow:hidden;
    border-radius:999px;
    background:#E9EDF3;
}
.rank-progress > span {
    display:block;
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg,#F2B900,#FFD75A);
}
.rank-coverage {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    min-height:34px;
    padding:7px 9px;
    border-radius:11px;
    background:#F3F6FA;
    color:#617087;
    font-size:.68rem;
    font-weight:850;
    text-align:center;
}
.ranking-row.top .rank-coverage {
    background:#FFF2B9;
    color:#836300;
}

/* ============================================================
   ABOUT — TRUE EQUAL HEIGHT
   ============================================================ */
.about-equal-panel {
    height:500px;
    min-height:500px;
    max-height:500px;
    overflow:hidden;
    border:1px solid #E7EAF0;
    border-radius:20px;
    background:#FFFFFF;
    box-shadow:var(--shadow);
}
.about-cap-panel { padding:14px; }
.about-cap-grid {
    height:100%;
    display:grid;
    grid-template-columns:repeat(2,minmax(0,1fr));
    grid-template-rows:repeat(2,minmax(0,1fr));
    gap:12px;
}
.about-cap-card {
    height:100%;
    padding:18px;
    border:1px solid #E8EBF1;
    border-radius:16px;
    background:linear-gradient(180deg,#FFFFFF,#FBFCFE);
}
.about-cap-icon {
    width:42px;
    height:42px;
    display:grid;
    place-items:center;
    border-radius:11px;
    background:#FFF4C7;
    margin-bottom:10px;
    font-size:1.08rem;
}
.about-cap-title {
    color:#17233D;
    font-size:.98rem;
    font-weight:950;
    margin-bottom:5px;
}
.about-cap-text {
    color:#6F7B91;
    font-size:.80rem;
    line-height:1.68;
}
.about-flow-panel { padding:20px 20px 14px; }
.about-flow-title {
    color:#17233D;
    font-size:1.28rem;
    font-weight:950;
    letter-spacing:-.02em;
    margin-bottom:18px;
}
.about-timeline { position:relative; }
.about-step {
    position:relative;
    display:grid;
    grid-template-columns:42px 1fr;
    gap:15px;
    padding-bottom:19px;
}
.about-step:last-child { padding-bottom:0; }
.about-step:not(:last-child)::before {
    content:"";
    position:absolute;
    left:19px;
    top:40px;
    bottom:0;
    width:2px;
    background:linear-gradient(180deg,#EEC744,#E7ECF4);
}
.about-step-num {
    width:40px;
    height:40px;
    display:grid;
    place-items:center;
    border-radius:50%;
    background:linear-gradient(145deg,#FFD854,#F7BE1C);
    color:#172039;
    font-size:.76rem;
    font-weight:950;
    z-index:1;
}
.about-step-title {
    color:#17233D;
    font-size:.92rem;
    font-weight:950;
    margin-top:2px;
}
.about-step-text {
    margin-top:5px;
    color:#78859A;
    font-size:.77rem;
    line-height:1.58;
}
@media(max-width:1100px) {
    .about-equal-panel {
        height:auto;
        min-height:0;
        max-height:none;
        overflow:visible;
    }
    .about-cap-grid {
        grid-template-columns:1fr;
        grid-template-rows:auto;
    }
    .ranking-head { display:none; }
    .ranking-row {
        grid-template-columns:44px 1fr;
    }
    .ranking-score,
    .rank-coverage {
        grid-column:2;
    }
}

@media(max-width:1100px) {
    .st-key-profile_form_card,
    .st-key-profile_visual_card,
    .st-key-about_capabilities,
    .st-key-about_flow {
        height:auto;
        min-height:0;
        max-height:none;
        overflow:visible;
    }

    .xai-hero {
        grid-template-columns:1fr 1fr;
    }

    .xai-explain-grid {
        grid-template-columns:1fr;
    }
}

@media(max-width:1100px) {
    .tile-grid-3, .tile-grid-2, .model-pill-grid, .flow-grid {
        grid-template-columns:1fr;
    }
}

@media(max-width:1200px) {
    .hero-inner { grid-template-columns:1.15fr .42fr .8fr; }
    .hero-logo-img { width:120px; height:120px; }
    .tech-row { flex-wrap:wrap; }
    .hero-meta { gap:36px; }
    .results-grid { grid-template-columns:repeat(3,minmax(0,1fr)); }
}
@media(max-width:900px) {
    .st-key-child_card,
    .st-key-photo_card,
    .st-key-action_card {
        height:auto;
        min-height:0;
        max-height:none;
        overflow:visible;
    }

    .st-key-photo_card [data-testid="stImage"] img {
        height:auto !important;
        max-height:320px !important;
    }

    [data-testid="stSidebar"] { min-width:240px !important; max-width:240px !important; }
    .block-container { padding-left:1rem !important; padding-right:1rem !important; }
    .hero-inner { grid-template-columns:1fr; }
    .hero-logo-wrap { display:none; }
    .hero-meta { align-items:flex-start; gap:12px; }
    .tech-row { justify-content:flex-start; flex-wrap:wrap; }
    .workflow-strip { grid-template-columns:1fr; gap:8px; }
    .workflow-arrow { display:none; }
    .results-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
@media(max-width:560px) {
    .title { font-size:1.9rem; }
    .hero-shell { padding:20px; }
    .results-grid { grid-template-columns:1fr; }
}

/* ============================================================
   RESPONSIVE — ALL DEVICES
   ============================================================ */

/* Large laptop / small desktop */
@media (max-width:1280px) {
    .block-container {
        max-width:1180px;
        padding-left:1.3rem !important;
        padding-right:1.3rem !important;
    }

    [data-testid="stSidebar"] {
        min-width:230px !important;
        max-width:230px !important;
    }

    .hero-shell {
        padding:24px 25px;
    }

    .hero-inner {
        grid-template-columns:1.25fr .42fr .92fr;
        gap:16px;
    }

    .hero-logo-img {
        width:116px;
        height:116px;
    }

    .tech-row {
        flex-wrap:wrap;
    }

    .results-grid {
        grid-template-columns:repeat(3,minmax(0,1fr));
    }
}

/* Tablet landscape / compact notebook */
@media (max-width:1024px) {
    [data-testid="stSidebar"] {
        min-width:220px !important;
        max-width:220px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding:.8rem .72rem !important;
    }

    .brand-logo-img {
        width:52px;
        height:52px;
    }

    .brand-name {
        font-size:.94rem;
    }

    .brand-sub {
        font-size:.58rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label {
        min-height:40px;
        padding:5px 7px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label::before {
        width:27px;
        height:27px;
        flex-basis:27px;
    }

    .about-card {
        margin-top:14px;
        padding:11px;
    }

    .hero-inner {
        grid-template-columns:1fr .38fr .85fr;
    }

    .title {
        font-size:2.15rem;
    }

    .subtitle {
        font-size:.80rem;
    }

    .workflow-strip {
        padding:10px 12px;
    }

    .tile-grid-3,
    .model-pill-grid {
        grid-template-columns:repeat(2,minmax(0,1fr));
    }
}

/* Tablet portrait */
@media (max-width:820px) {
    /* Sidebar becomes a compact drawer when opened. Streamlit's "auto"
       initial state keeps it collapsed on narrow screens. */
    [data-testid="stSidebar"] {
        min-width:245px !important;
        max-width:245px !important;
        width:245px !important;
    }

    .about-card {
        display:none !important;
    }

    .brand {
        padding-bottom:9px;
    }

    .block-container {
        padding:.85rem .85rem 2rem !important;
        max-width:100%;
    }

    .hero-shell {
        min-height:0;
        padding:19px;
        border-radius:20px;
        margin-bottom:16px;
    }

    .hero-inner {
        grid-template-columns:1fr;
        min-height:0;
        gap:12px;
    }

    .hero-logo-wrap {
        display:none !important;
    }

    .hero-meta {
        align-items:flex-start;
        gap:10px;
    }

    .tech-row {
        justify-content:flex-start;
        flex-wrap:wrap;
    }

    .title {
        font-size:1.85rem;
    }

    .subtitle {
        font-size:.76rem;
        line-height:1.55;
    }

    .date-pill {
        padding:7px 10px;
        font-size:.64rem;
    }

    .tech-chip {
        padding:6px 8px;
        font-size:.56rem;
    }

    .workflow-strip {
        grid-template-columns:1fr;
        gap:6px;
        padding:10px;
    }

    .workflow-arrow {
        display:none !important;
    }

    .workflow-item {
        padding:5px;
    }

    .st-key-child_card,
    .st-key-photo_card,
    .st-key-action_card,
    .st-key-profile_form_card,
    .st-key-profile_visual_card,
    .about-equal-panel {
        height:auto !important;
        min-height:0 !important;
        max-height:none !important;
        overflow:visible !important;
    }

    .results-grid {
        grid-template-columns:repeat(2,minmax(0,1fr));
    }

    .tile-grid-3,
    .tile-grid-2,
    .model-pill-grid,
    .flow-grid,
    .about-cap-grid,
    .xai-explain-grid {
        grid-template-columns:1fr !important;
        grid-template-rows:auto !important;
    }

    .xai-hero {
        grid-template-columns:repeat(2,minmax(0,1fr));
    }

    [data-testid="stHorizontalBlock"] {
        flex-wrap:wrap !important;
        gap:.75rem !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        min-width:calc(50% - .75rem) !important;
    }

    div[data-testid="stPlotlyChart"] {
        overflow:hidden;
    }

    [data-testid="stDataFrame"] {
        overflow-x:auto !important;
    }
}

/* Phones */
@media (max-width:600px) {
    [data-testid="stSidebar"] {
        min-width:228px !important;
        max-width:228px !important;
        width:228px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding:.65rem .6rem !important;
    }

    .brand {
        flex-direction:row;
        justify-content:flex-start;
        text-align:left;
        gap:9px;
        padding:0 4px 10px;
    }

    .brand-logo-img {
        width:42px;
        height:42px;
        margin:0;
        flex:0 0 42px;
    }

    .brand-name {
        font-size:.88rem;
    }

    .brand-sub {
        font-size:.53rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap:2px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label {
        min-height:38px;
        padding:4px 6px !important;
        border-radius:10px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label::before {
        width:25px;
        height:25px;
        flex-basis:25px;
        font-size:.80rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label p {
        font-size:.68rem !important;
    }

    .block-container {
        padding:.65rem .62rem 1.8rem !important;
    }

    .hero-shell {
        padding:15px;
        border-radius:17px;
    }

    .hero-eyebrow {
        padding:5px 8px;
        font-size:.55rem;
        margin-bottom:8px;
    }

    .title {
        font-size:1.55rem;
    }

    .subtitle {
        font-size:.69rem;
    }

    .section-title {
        font-size:1.08rem;
    }

    .section-sub {
        font-size:.68rem;
    }

    .results-grid,
    .kpi-grid,
    .profile-mini-grid,
    .xai-hero {
        grid-template-columns:1fr !important;
    }

    /* Every Streamlit column stacks vertically on phones. */
    [data-testid="stHorizontalBlock"] {
        display:flex !important;
        flex-direction:column !important;
        flex-wrap:nowrap !important;
        gap:.65rem !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        width:100% !important;
        flex:1 1 auto !important;
        min-width:100% !important;
    }

    .panel,
    .section-card,
    .kpi-card,
    .result-card,
    .info-card,
    .recommend-card-rich,
    .ranking-board,
    .about-equal-panel {
        border-radius:14px;
    }

    .ranking-head {
        display:none !important;
    }

    .ranking-row {
        grid-template-columns:38px 1fr !important;
        gap:8px;
        padding:10px;
    }

    .ranking-score,
    .rank-coverage {
        grid-column:2 !important;
    }

    .rank-reason {
        white-space:normal;
    }

    .rec-visual-head {
        align-items:flex-start;
        flex-direction:column;
    }

    .rec-legend {
        width:100%;
    }

    .about-cap-panel,
    .about-flow-panel {
        padding:11px;
    }

    .about-step {
        grid-template-columns:34px 1fr;
        gap:10px;
    }

    .about-step-num {
        width:32px;
        height:32px;
    }

    .about-step:not(:last-child)::before {
        left:15px;
        top:32px;
    }

    .xai-intro,
    .xai-stat {
        padding:12px;
    }

    .st-key-grad_original,
    .st-key-grad_heatmap,
    .st-key-grad_overlay {
        padding:10px;
    }

    button[data-baseweb="tab"] {
        padding-left:10px !important;
        padding-right:10px !important;
        min-width:max-content !important;
    }

    [data-baseweb="tab-list"] {
        overflow-x:auto !important;
        flex-wrap:nowrap !important;
        scrollbar-width:none;
    }

    [data-baseweb="tab-list"]::-webkit-scrollbar {
        display:none;
    }
}

/* Very small phones */
@media (max-width:390px) {
    [data-testid="stSidebar"] {
        min-width:214px !important;
        max-width:214px !important;
        width:214px !important;
    }

    .title {
        font-size:1.38rem;
    }

    .hero-shell {
        padding:13px;
    }

    .tech-chip {
        font-size:.50rem;
    }

    .result-value {
        font-size:.92rem;
    }

    .kpi-value {
        font-size:1.16rem;
    }
}

/* Short laptop screens: don't let lower sidebar card push navigation down. */
@media (max-height:780px) and (min-width:821px) {
    .about-card {
        display:none !important;
    }

    .brand {
        padding-bottom:10px;
    }

    .brand-logo-img {
        width:52px;
        height:52px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label {
        min-height:39px;
    }
}


/* ============================================================
   FINAL SIDEBAR OVERRIDE
   Desktop: visible + readable
   Mobile/tablet: compact drawer
   ============================================================ */

/* Desktop / laptop */
@media (min-width:1025px) {
    [data-testid="stSidebar"] {
        display:block !important;
        min-width:266px !important;
        max-width:266px !important;
        width:266px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding:1.15rem 1rem .9rem !important;
    }

    .brand {
        padding:4px 4px 16px !important;
    }

    .brand-logo-img {
        width:70px !important;
        height:70px !important;
        margin-bottom:5px !important;
    }

    .brand-name {
        font-size:1.12rem !important;
    }

    .brand-sub {
        font-size:.66rem !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap:5px !important;
        margin-top:3px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label {
        min-height:47px !important;
        padding:7px 10px !important;
        gap:10px !important;
        border-radius:12px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label::before {
        width:31px !important;
        height:31px !important;
        flex-basis:31px !important;
        font-size:.94rem !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label p {
        font-size:.82rem !important;
        line-height:1.2 !important;
        font-weight:760 !important;
    }

    .about-card {
        margin-top:22px !important;
        padding:14px !important;
        font-size:.66rem !important;
        line-height:1.58 !important;
    }

    .about-card b {
        font-size:.79rem !important;
    }

    .about-mini {
        font-size:.59rem !important;
    }
}

/* Tablet */
@media (min-width:601px) and (max-width:1024px) {
    [data-testid="stSidebar"] {
        min-width:252px !important;
        max-width:252px !important;
        width:252px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding:.9rem .82rem !important;
    }

    .brand-logo-img {
        width:60px !important;
        height:60px !important;
    }

    .brand-name {
        font-size:1rem !important;
    }

    .brand-sub {
        font-size:.60rem !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label {
        min-height:44px !important;
        padding:6px 8px !important;
        gap:9px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label::before {
        width:29px !important;
        height:29px !important;
        flex-basis:29px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label p {
        font-size:.76rem !important;
    }

    .about-card {
        margin-top:16px !important;
        padding:12px !important;
        font-size:.61rem !important;
    }
}

/* Phones: sidebar remains a drawer and does not become excessively wide. */
@media (max-width:600px) {
    [data-testid="stSidebar"] {
        min-width:246px !important;
        max-width:246px !important;
        width:246px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding:.7rem .65rem !important;
    }

    .brand {
        flex-direction:row !important;
        justify-content:flex-start !important;
        align-items:center !important;
        text-align:left !important;
        gap:10px !important;
        padding:2px 5px 11px !important;
    }

    .brand-logo-img {
        width:46px !important;
        height:46px !important;
        flex:0 0 46px !important;
        margin:0 !important;
    }

    .brand-name {
        font-size:.94rem !important;
    }

    .brand-sub {
        font-size:.56rem !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap:3px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label {
        min-height:42px !important;
        padding:5px 7px !important;
        gap:8px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label::before {
        width:28px !important;
        height:28px !important;
        flex-basis:28px !important;
        font-size:.84rem !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label p {
        font-size:.72rem !important;
    }

    /* Hide the bottom description card on phones so navigation stays short. */
    .about-card {
        display:none !important;
    }
}

/* Extra safety: never show native radio circles. */
[data-testid="stSidebar"] input[type="radio"],
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child,
[data-testid="stSidebar"] [role="radiogroup"] label > div:has(input[type="radio"]) {
    display:none !important;
    opacity:0 !important;
    width:0 !important;
    height:0 !important;
    min-width:0 !important;
    padding:0 !important;
    margin:0 !important;
}

/* Short desktop screens: keep menu visible and hide only the lower info card. */
@media (max-height:720px) and (min-width:1025px) {
    .about-card {
        display:none !important;
    }

    .brand-logo-img {
        width:58px !important;
        height:58px !important;
    }

    .brand {
        padding-bottom:10px !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label {
        min-height:43px !important;
    }
}


/* ============================================================
   SIDEBAR VISIBILITY — FINAL OVERRIDE
   ============================================================ */

/* Desktop: readable sidebar, always visible when Streamlit supports locked. */
@media (min-width:1025px) {
    section[data-testid="stSidebar"] {
        display:block !important;
        visibility:visible !important;
        opacity:1 !important;
        min-width:278px !important;
        max-width:278px !important;
        width:278px !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding:1.1rem 1rem .85rem !important;
    }

    .brand {
        padding:3px 4px 17px !important;
    }

    .brand-logo-img {
        width:72px !important;
        height:72px !important;
    }

    .brand-name {
        font-size:1.14rem !important;
    }

    .brand-sub {
        font-size:.67rem !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] {
        gap:5px !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label {
        min-height:49px !important;
        padding:8px 11px !important;
        gap:11px !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label::before {
        width:32px !important;
        height:32px !important;
        flex-basis:32px !important;
        font-size:.96rem !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label p {
        font-size:.84rem !important;
        font-weight:780 !important;
    }

    .about-card {
        margin-top:22px !important;
        padding:14px !important;
    }

    /* On locked/current Streamlit this normally does not exist.
       Hide it on desktop so the UI stays clean. */
    [data-testid="stSidebarCollapsedControl"] {
        display:none !important;
    }
}

/* Tablet: sidebar opens as a normal drawer. */
@media (min-width:601px) and (max-width:1024px) {
    [data-testid="stHeader"] {
        height:44px !important;
        min-height:44px !important;
    }

    section[data-testid="stSidebar"] {
        min-width:258px !important;
        max-width:258px !important;
        width:258px !important;
    }

    .brand-logo-img {
        width:60px !important;
        height:60px !important;
    }

    .brand-name {
        font-size:1rem !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label {
        min-height:45px !important;
        padding:6px 9px !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label p {
        font-size:.78rem !important;
    }

    [data-testid="stSidebarCollapsedControl"] {
        display:flex !important;
        position:fixed !important;
        top:7px !important;
        left:8px !important;
    }
}

/* Mobile: sidebar becomes a compact drawer and an open button stays visible. */
@media (max-width:600px) {
    [data-testid="stHeader"] {
        height:42px !important;
        min-height:42px !important;
    }

    section[data-testid="stSidebar"] {
        min-width:250px !important;
        max-width:250px !important;
        width:250px !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding:.72rem .68rem !important;
    }

    .brand {
        flex-direction:row !important;
        align-items:center !important;
        justify-content:flex-start !important;
        text-align:left !important;
        padding:1px 4px 11px !important;
        gap:10px !important;
    }

    .brand-logo-img {
        width:47px !important;
        height:47px !important;
        flex:0 0 47px !important;
        margin:0 !important;
    }

    .brand-name {
        font-size:.95rem !important;
    }

    .brand-sub {
        font-size:.56rem !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label {
        min-height:43px !important;
        padding:5px 8px !important;
        gap:9px !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label::before {
        width:29px !important;
        height:29px !important;
        flex-basis:29px !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label p {
        font-size:.73rem !important;
    }

    .about-card {
        display:none !important;
    }

    [data-testid="stSidebarCollapsedControl"] {
        display:flex !important;
        position:fixed !important;
        top:6px !important;
        left:7px !important;
    }
}

/* Short desktop screens: preserve navigation, only remove the info card. */
@media (min-width:1025px) and (max-height:760px) {
    .about-card {
        display:none !important;
    }

    .brand-logo-img {
        width:60px !important;
        height:60px !important;
    }

    .brand {
        padding-bottom:10px !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label {
        min-height:44px !important;
    }
}


/* ============================================================
   NAVIGATION SIZE + MOBILE MENU
   ============================================================ */

/* Native mobile menu is hidden on desktop. */
.st-key-mobile_navbar {
    display:none !important;
}

/* Make desktop sidebar text slightly larger without adding excess height. */
@media (min-width:1025px) {
    section[data-testid="stSidebar"] {
        min-width:284px !important;
        max-width:284px !important;
        width:284px !important;
    }

    .brand-logo-img {
        width:76px !important;
        height:76px !important;
    }

    .brand-name {
        font-size:1.20rem !important;
        line-height:1.15 !important;
    }

    .brand-sub {
        font-size:.70rem !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label {
        min-height:50px !important;
        padding:8px 11px !important;
        gap:11px !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label::before {
        width:33px !important;
        height:33px !important;
        flex-basis:33px !important;
        font-size:.98rem !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label p {
        font-size:.89rem !important;
        line-height:1.25 !important;
        font-weight:790 !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
        font-size:.91rem !important;
    }
}

/* Tablet and mobile: show our own reliable menu button. */
@media (max-width:820px) {
    .st-key-mobile_navbar {
        display:block !important;
        position:sticky !important;
        top:0 !important;
        z-index:9998 !important;
        margin:-.25rem 0 .75rem 0 !important;
        padding:6px 0 !important;
        background:rgba(247,249,252,.92) !important;
        backdrop-filter:blur(14px);
    }

    .st-key-mobile_navbar [data-testid="stPopover"] {
        width:max-content !important;
    }

    .st-key-mobile_navbar [data-testid="stPopover"] > button,
    .st-key-mobile_navbar button[kind="secondary"] {
        min-height:42px !important;
        width:auto !important;
        padding:8px 14px !important;
        border-radius:12px !important;
        border:1px solid #19375F !important;
        background:linear-gradient(145deg,#10284B,#0B1D38) !important;
        color:#FFFFFF !important;
        font-size:.78rem !important;
        font-weight:900 !important;
        box-shadow:0 8px 20px rgba(12,29,56,.16) !important;
    }

    /* The normal Streamlit sidebar can stay collapsed on mobile because
       this custom menu is always available. */
    [data-testid="stSidebarCollapsedControl"] {
        display:none !important;
    }

    /* Slightly increase mobile body typography. */
    .hero-eyebrow {
        font-size:.59rem !important;
    }

    .title {
        font-size:1.70rem !important;
    }

    .subtitle {
        font-size:.75rem !important;
        line-height:1.60 !important;
    }

    .section-kicker {
        font-size:.64rem !important;
    }

    .section-title {
        font-size:1.18rem !important;
    }

    .section-sub {
        font-size:.73rem !important;
    }

    .workflow-title {
        font-size:.80rem !important;
    }

    .workflow-sub {
        font-size:.68rem !important;
    }

    .panel-title {
        font-size:1.12rem !important;
    }

    .panel-sub {
        font-size:.78rem !important;
    }

    label[data-testid="stWidgetLabel"] p {
        font-size:.80rem !important;
    }
}

/* Extra-compact phones */
@media (max-width:480px) {
    .st-key-mobile_navbar {
        margin:-.15rem 0 .65rem 0 !important;
    }

    .st-key-mobile_navbar [data-testid="stPopover"] > button,
    .st-key-mobile_navbar button[kind="secondary"] {
        min-height:40px !important;
        padding:7px 12px !important;
        font-size:.75rem !important;
    }

    .title {
        font-size:1.58rem !important;
    }

    .subtitle {
        font-size:.72rem !important;
    }

    .workflow-title {
        font-size:.78rem !important;
    }

    .workflow-sub {
        font-size:.66rem !important;
    }
}

/* Popover content is rendered outside the source container in Streamlit.
   Style the navigation radio when the popover is open. */
div[data-baseweb="popover"] [role="radiogroup"] {
    min-width:220px !important;
    padding:4px !important;
}

div[data-baseweb="popover"] [role="radiogroup"] > label {
    min-height:42px !important;
    padding:7px 9px !important;
    border-radius:10px !important;
    margin-bottom:3px !important;
}

div[data-baseweb="popover"] [role="radiogroup"] > label p {
    font-size:.78rem !important;
    font-weight:800 !important;
    color:#26334A !important;
}

div[data-baseweb="popover"] [role="radiogroup"] label:has(input:checked) {
    background:#FFF5CF !important;
}

div[data-baseweb="popover"] [role="radiogroup"] label:has(input:checked) p {
    color:#836200 !important;
}

.mobile-menu-brand {
    display:flex;
    align-items:center;
    gap:9px;
    padding:8px 6px 11px;
    margin-bottom:5px;
    border-bottom:1px solid #EBEEF3;
}

.mobile-menu-brand img {
    width:38px;
    height:38px;
    object-fit:contain;
}

.mobile-menu-brand b {
    display:block;
    color:#17233D;
    font-size:.84rem;
}

.mobile-menu-brand span {
    display:block;
    margin-top:2px;
    color:#8B95A7;
    font-size:.62rem;
}

/* Do not show the lower sidebar information card on short screens. */
@media (max-height:760px) {
    .about-card {
        display:none !important;
    }
}


/* ============================================================
   MOBILE LEFT DRAWER — NUTRIVISION STYLE
   ============================================================ */

/* Tablet/mobile menu button: no negative margin, so it is never clipped. */
@media (max-width:820px) {
    .st-key-mobile_navbar {
        display:block !important;
        position:relative !important;
        top:auto !important;
        z-index:9998 !important;
        margin:.35rem 0 .85rem 0 !important;
        padding:5px 0 4px !important;
        background:transparent !important;
        backdrop-filter:none !important;
    }

    .st-key-mobile_navbar [data-testid="stPopover"] {
        width:max-content !important;
    }

    .st-key-mobile_navbar [data-testid="stPopover"] > button,
    .st-key-mobile_navbar button[kind="secondary"] {
        min-height:44px !important;
        width:auto !important;
        padding:9px 15px !important;
        border-radius:12px !important;
        border:1px solid #1A3A64 !important;
        background:linear-gradient(145deg,#112A4E,#0B1D38) !important;
        color:#FFFFFF !important;
        font-size:.82rem !important;
        font-weight:900 !important;
        box-shadow:0 8px 20px rgba(12,29,56,.18) !important;
    }

    .st-key-mobile_navbar [data-testid="stPopover"] > button:hover,
    .st-key-mobile_navbar button[kind="secondary"]:hover {
        border-color:#E4BD32 !important;
        background:linear-gradient(145deg,#17365F,#10284B) !important;
    }

    /* Body text a little larger on mobile. */
    .hero-eyebrow {
        font-size:.61rem !important;
    }

    .title {
        font-size:1.76rem !important;
    }

    .subtitle {
        font-size:.78rem !important;
        line-height:1.62 !important;
    }

    .section-kicker {
        font-size:.66rem !important;
    }

    .section-title {
        font-size:1.22rem !important;
    }

    .section-sub {
        font-size:.76rem !important;
    }

    .workflow-title {
        font-size:.84rem !important;
    }

    .workflow-sub {
        font-size:.70rem !important;
        line-height:1.45 !important;
    }

    .panel-title {
        font-size:1.15rem !important;
    }

    .panel-sub {
        font-size:.80rem !important;
    }

    label[data-testid="stWidgetLabel"] p {
        font-size:.82rem !important;
    }
}

/* The popover is rendered in a portal outside the source container.
   On tablet/mobile we reposition it into a compact LEFT drawer. */
@media (max-width:820px) {
    div[data-baseweb="popover"] {
        position:fixed !important;
        left:8px !important;
        right:auto !important;
        top:58px !important;
        bottom:auto !important;
        transform:none !important;
        width:260px !important;
        max-width:calc(100vw - 28px) !important;
        z-index:100000 !important;
    }

    /* Cover multiple Streamlit/BaseWeb DOM versions. */
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] [role="dialog"] {
        width:260px !important;
        max-width:calc(100vw - 28px) !important;
        max-height:calc(100vh - 78px) !important;
        overflow-y:auto !important;
        padding:13px !important;
        border:1px solid rgba(255,255,255,.10) !important;
        border-radius:0 18px 18px 0 !important;
        background:
            radial-gradient(circle at 22% 4%,rgba(39,78,132,.32),transparent 12rem),
            linear-gradient(180deg,#10233F,#0B1A31) !important;
        box-shadow:18px 12px 45px rgba(5,17,34,.28) !important;
    }

    /* Some BaseWeb versions wrap the actual surface one layer deeper. */
    div[data-baseweb="popover"] > div > div {
        background:transparent !important;
    }

    .mobile-menu-brand {
        display:flex !important;
        align-items:center !important;
        gap:10px !important;
        padding:7px 7px 13px !important;
        margin-bottom:8px !important;
        border-bottom:1px solid rgba(255,255,255,.10) !important;
    }

    .mobile-menu-brand img {
        width:48px !important;
        height:48px !important;
        object-fit:contain !important;
        flex:0 0 48px !important;
    }

    .mobile-menu-brand b {
        display:block !important;
        color:#FFFFFF !important;
        font-size:.92rem !important;
        line-height:1.2 !important;
    }

    .mobile-menu-brand span {
        display:block !important;
        margin-top:3px !important;
        color:#9EABC1 !important;
        font-size:.63rem !important;
    }

    div[data-baseweb="popover"] [role="radiogroup"] {
        min-width:0 !important;
        width:100% !important;
        padding:2px !important;
        gap:4px !important;
    }

    div[data-baseweb="popover"] [role="radiogroup"] > label {
        position:relative !important;
        display:flex !important;
        align-items:center !important;
        width:100% !important;
        min-height:46px !important;
        margin:0 0 4px !important;
        padding:7px 9px !important;
        border-radius:12px !important;
        border:1px solid transparent !important;
        background:transparent !important;
        gap:10px !important;
    }

    /* Hide native radio marker in mobile drawer. */
    div[data-baseweb="popover"] [role="radiogroup"] input[type="radio"],
    div[data-baseweb="popover"] [role="radiogroup"] label > div:first-child:has(input[type="radio"]),
    div[data-baseweb="popover"] [role="radiogroup"] label div:has(> input[type="radio"]) {
        display:none !important;
        width:0 !important;
        min-width:0 !important;
        height:0 !important;
        margin:0 !important;
        padding:0 !important;
        opacity:0 !important;
    }

    /* Use the same visual language as the desktop sidebar. */
    div[data-baseweb="popover"] [role="radiogroup"] > label::before {
        content:"";
        width:30px !important;
        height:30px !important;
        flex:0 0 30px !important;
        display:grid !important;
        place-items:center !important;
        border-radius:9px !important;
        background:rgba(255,255,255,.045) !important;
        color:#EAF0F9 !important;
        font-size:.88rem !important;
        font-weight:900 !important;
    }

    div[data-baseweb="popover"] [role="radiogroup"] > label:nth-child(1)::before { content:"▦"; }
    div[data-baseweb="popover"] [role="radiogroup"] > label:nth-child(2)::before { content:"◷"; }
    div[data-baseweb="popover"] [role="radiogroup"] > label:nth-child(3)::before { content:"♙"; }
    div[data-baseweb="popover"] [role="radiogroup"] > label:nth-child(4)::before { content:"▤"; }
    div[data-baseweb="popover"] [role="radiogroup"] > label:nth-child(5)::before { content:"ⓘ"; }

    div[data-baseweb="popover"] [role="radiogroup"] > label p {
        color:#F0F4FA !important;
        font-size:.80rem !important;
        line-height:1.2 !important;
        font-weight:790 !important;
    }

    div[data-baseweb="popover"] [role="radiogroup"] > label:hover {
        background:rgba(255,255,255,.045) !important;
    }

    div[data-baseweb="popover"] [role="radiogroup"] label:has(input:checked) {
        background:linear-gradient(90deg,rgba(255,201,40,.13),rgba(255,255,255,.03)) !important;
        border:1px solid rgba(255,201,40,.24) !important;
        box-shadow:inset 3px 0 0 #FFC928 !important;
    }

    div[data-baseweb="popover"] [role="radiogroup"] label:has(input:checked)::before {
        background:linear-gradient(145deg,#FFD955,#FFC225) !important;
        color:#172039 !important;
    }

    div[data-baseweb="popover"] [role="radiogroup"] label:has(input:checked) p {
        color:#FFD54A !important;
        font-weight:900 !important;
    }
}

/* Small phones: drawer is narrower but still readable. */
@media (max-width:480px) {
    .st-key-mobile_navbar {
        margin:.25rem 0 .75rem 0 !important;
    }

    .st-key-mobile_navbar [data-testid="stPopover"] > button,
    .st-key-mobile_navbar button[kind="secondary"] {
        min-height:42px !important;
        padding:8px 13px !important;
        font-size:.79rem !important;
    }

    div[data-baseweb="popover"] {
        left:6px !important;
        top:54px !important;
        width:244px !important;
        max-width:calc(100vw - 20px) !important;
    }

    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] [role="dialog"] {
        width:244px !important;
        max-width:calc(100vw - 20px) !important;
        border-radius:0 16px 16px 0 !important;
        padding:11px !important;
    }

    .mobile-menu-brand img {
        width:43px !important;
        height:43px !important;
        flex-basis:43px !important;
    }

    .mobile-menu-brand b {
        font-size:.88rem !important;
    }

    div[data-baseweb="popover"] [role="radiogroup"] > label {
        min-height:44px !important;
        padding:6px 8px !important;
    }

    div[data-baseweb="popover"] [role="radiogroup"] > label p {
        font-size:.77rem !important;
    }

    .title {
        font-size:1.64rem !important;
    }

    .subtitle {
        font-size:.75rem !important;
    }
}


/* ============================================================
   TRUE MOBILE LEFT SIDEBAR DRAWER
   ============================================================ */

.nv-mobile-drawer-root {
    display:none;
}

/* Desktop/laptop continues to use the real Streamlit sidebar. */
@media (min-width:821px) {
    .nv-mobile-drawer-root {
        display:none !important;
    }
}

/* Tablet + phone */
@media (max-width:820px) {
    /* Hide old custom Streamlit popover menu if legacy CSS remains. */
    .st-key-mobile_navbar {
        display:none !important;
    }

    /* No duplicate built-in sidebar toggle on mobile. */
    [data-testid="stSidebarCollapsedControl"] {
        display:none !important;
    }

    .nv-mobile-drawer-root {
        display:block !important;
        position:relative;
        z-index:100000;
        width:100%;
        height:0;
    }

    .nv-mobile-drawer-toggle {
        position:absolute !important;
        opacity:0 !important;
        width:1px !important;
        height:1px !important;
        pointer-events:none;
    }

    .nv-mobile-menu-button {
        position:fixed;
        top:10px;
        left:12px;
        z-index:100003;
        display:inline-flex;
        align-items:center;
        gap:8px;
        height:42px;
        padding:0 14px;
        border:1px solid #1A3B66;
        border-radius:12px;
        background:
            linear-gradient(145deg,#112B50,#0A1C37);
        color:#FFFFFF;
        box-shadow:
            0 8px 22px rgba(8,24,48,.24);
        font-size:.80rem;
        font-weight:900;
        cursor:pointer;
        user-select:none;
    }

    .nv-mobile-menu-button:hover {
        border-color:#E5BF36;
        background:
            linear-gradient(145deg,#173861,#10284A);
    }

    .nv-hamburger {
        font-size:1.05rem;
        line-height:1;
    }

    .nv-mobile-drawer-overlay {
        position:fixed;
        inset:0;
        z-index:100001;
        background:rgba(4,13,27,.45);
        backdrop-filter:blur(1.5px);
        opacity:0;
        visibility:hidden;
        transition:
            opacity .24s ease,
            visibility .24s ease;
        cursor:pointer;
    }

    .nv-mobile-drawer {
        position:fixed;
        top:0;
        bottom:0;
        left:0;
        z-index:100002;
        width:260px;
        max-width:76vw;
        padding:20px 14px 16px;
        overflow-y:auto;
        overflow-x:hidden;
        border-right:
            1px solid rgba(255,255,255,.08);
        background:
            radial-gradient(
                circle at 28% 5%,
                rgba(42,82,137,.35),
                transparent 14rem
            ),
            linear-gradient(
                180deg,
                #102441 0%,
                #0B1B32 100%
            );
        box-shadow:
            20px 0 48px rgba(4,15,31,.30);
        transform:translateX(-104%);
        transition:transform .28s cubic-bezier(.2,.8,.2,1);
    }

    .nv-mobile-drawer-toggle:checked
        ~ .nv-mobile-drawer-overlay {
        opacity:1;
        visibility:visible;
    }

    .nv-mobile-drawer-toggle:checked
        ~ .nv-mobile-drawer {
        transform:translateX(0);
    }

    .nv-mobile-drawer-toggle:checked
        ~ .nv-mobile-menu-button {
        opacity:0;
        pointer-events:none;
    }

    .nv-mobile-drawer-header {
        display:flex;
        align-items:flex-start;
        justify-content:space-between;
        gap:8px;
        padding:4px 3px 17px;
        border-bottom:
            1px solid rgba(255,255,255,.10);
    }

    .nv-mobile-brand {
        display:flex;
        align-items:center;
        gap:10px;
        min-width:0;
    }

    .nv-mobile-brand img {
        width:50px;
        height:50px;
        object-fit:contain;
        flex:0 0 50px;
        filter:
            drop-shadow(
                0 8px 15px rgba(255,201,40,.12)
            );
    }

    .nv-mobile-brand-name {
        color:#FFFFFF;
        font-size:.94rem;
        line-height:1.18;
        font-weight:950;
        letter-spacing:-.025em;
        white-space:nowrap;
    }

    .nv-mobile-brand-name span {
        color:#FFC928;
    }

    .nv-mobile-brand-sub {
        margin-top:3px;
        color:#98A7BE;
        font-size:.56rem;
        line-height:1.3;
    }

    .nv-mobile-close {
        display:grid;
        place-items:center;
        width:30px;
        height:30px;
        flex:0 0 30px;
        border-radius:9px;
        border:
            1px solid rgba(255,255,255,.10);
        background:
            rgba(255,255,255,.045);
        color:#FFFFFF;
        font-size:1.20rem;
        line-height:1;
        cursor:pointer;
    }

    .nv-mobile-nav {
        display:flex;
        flex-direction:column;
        gap:5px;
        margin-top:14px;
    }

    .nv-drawer-link {
        position:relative;
        display:flex;
        align-items:center;
        gap:10px;
        min-height:46px;
        padding:7px 9px;
        border:
            1px solid transparent;
        border-radius:12px;
        text-decoration:none !important;
        color:#F0F4FA !important;
        font-size:.78rem;
        font-weight:800;
        line-height:1.2;
        transition:
            background .16s ease,
            border .16s ease;
    }

    .nv-drawer-link:hover {
        background:
            rgba(255,255,255,.045);
    }

    .nv-drawer-icon {
        display:grid;
        place-items:center;
        width:30px;
        height:30px;
        flex:0 0 30px;
        border-radius:9px;
        background:
            rgba(255,255,255,.045);
        color:#EAF0F9;
        font-size:.87rem;
        font-weight:900;
    }

    .nv-drawer-link.active {
        color:#FFD548 !important;
        background:
            linear-gradient(
                90deg,
                rgba(255,201,40,.13),
                rgba(255,255,255,.025)
            );
        border:
            1px solid rgba(255,201,40,.22);
        box-shadow:
            inset 3px 0 0 #FFC928;
    }

    .nv-drawer-link.active
        .nv-drawer-icon {
        color:#172039;
        background:
            linear-gradient(
                145deg,
                #FFD957,
                #FFC225
            );
        box-shadow:
            0 7px 14px rgba(255,201,40,.14);
    }

    .nv-mobile-info {
        margin-top:24px;
        padding:13px;
        border:
            1px solid rgba(255,255,255,.09);
        border-radius:14px;
        background:
            rgba(255,255,255,.035);
    }

    .nv-mobile-info-title {
        color:#FFFFFF;
        font-size:.72rem;
        font-weight:900;
    }

    .nv-mobile-info-text {
        margin-top:7px;
        color:#9DAAC0;
        font-size:.58rem;
        line-height:1.55;
    }

    /* Reserve space for the fixed menu button.
       Prevents the hero from appearing clipped under it. */
    .block-container {
        padding-top:4.35rem !important;
    }

    /* Slightly larger mobile typography. */
    .hero-eyebrow {
        font-size:.62rem !important;
    }

    .title {
        font-size:1.76rem !important;
    }

    .subtitle {
        font-size:.78rem !important;
        line-height:1.62 !important;
    }

    .section-title {
        font-size:1.22rem !important;
    }

    .section-sub {
        font-size:.76rem !important;
    }

    .workflow-title {
        font-size:.84rem !important;
    }

    .workflow-sub {
        font-size:.70rem !important;
    }

    .panel-title {
        font-size:1.15rem !important;
    }

    .panel-sub {
        font-size:.80rem !important;
    }
}

/* Phones */
@media (max-width:480px) {
    .nv-mobile-menu-button {
        top:8px;
        left:9px;
        height:40px;
        padding:0 12px;
        font-size:.77rem;
        border-radius:11px;
    }

    .nv-mobile-drawer {
        width:248px;
        max-width:79vw;
        padding:
            16px 12px 14px;
    }

    .nv-mobile-brand img {
        width:45px;
        height:45px;
        flex-basis:45px;
    }

    .nv-mobile-brand-name {
        font-size:.89rem;
    }

    .nv-drawer-link {
        min-height:44px;
        padding:6px 8px;
        font-size:.76rem;
    }

    .nv-drawer-icon {
        width:28px;
        height:28px;
        flex-basis:28px;
    }

    .block-container {
        padding-top:4rem !important;
    }

    .title {
        font-size:1.64rem !important;
    }

    .subtitle {
        font-size:.75rem !important;
    }
}


/* ============================================================
   MOBILE DRAWER STABILITY FIX
   Menu button occupies real layout space; drawer still slides left.
   ============================================================ */

@media (max-width:820px) {
    /* The root itself owns visible height so Streamlit cannot clip it. */
    .nv-mobile-drawer-root {
        display:block !important;
        position:sticky !important;
        top:44px !important;
        z-index:99990 !important;
        width:100% !important;
        height:56px !important;
        min-height:56px !important;
        margin:0 0 12px 0 !important;
        padding:6px 0 !important;
        overflow:visible !important;
        background:
            linear-gradient(
                180deg,
                rgba(247,249,252,.98),
                rgba(247,249,252,.93)
            ) !important;
        backdrop-filter:blur(12px);
    }

    /* Button is no longer fixed to the viewport. */
    .nv-mobile-menu-button {
        position:absolute !important;
        top:6px !important;
        left:0 !important;
        z-index:100003 !important;
        display:inline-flex !important;
        align-items:center !important;
        gap:8px !important;
        height:43px !important;
        padding:0 14px !important;
        border:1px solid #1A3B66 !important;
        border-radius:12px !important;
        background:
            linear-gradient(145deg,#112B50,#0A1C37) !important;
        color:#FFFFFF !important;
        box-shadow:
            0 7px 18px rgba(8,24,48,.18) !important;
        font-size:.81rem !important;
        font-weight:900 !important;
        cursor:pointer !important;
        user-select:none !important;
    }

    .nv-mobile-menu-button:hover {
        border-color:#E5BF36 !important;
        background:
            linear-gradient(145deg,#173861,#10284A) !important;
    }

    /* Drawer remains a true left-side fixed panel. */
    .nv-mobile-drawer {
        top:0 !important;
        left:0 !important;
        bottom:0 !important;
        width:260px !important;
        max-width:76vw !important;
        border-radius:0 18px 18px 0 !important;
    }

    .nv-mobile-drawer-overlay {
        inset:0 !important;
    }

    /* No artificial giant space above hero anymore. */
    .block-container {
        padding-top:.9rem !important;
    }

    /* Keep Streamlit header above body but below the drawer when opened. */
    [data-testid="stHeader"] {
        height:44px !important;
        min-height:44px !important;
        z-index:9990 !important;
    }

    /* When drawer is open, hide menu button but keep its layout row. */
    .nv-mobile-drawer-toggle:checked
        ~ .nv-mobile-menu-button {
        opacity:0 !important;
        pointer-events:none !important;
    }
}

@media (max-width:480px) {
    .nv-mobile-drawer-root {
        top:42px !important;
        height:53px !important;
        min-height:53px !important;
        padding:5px 0 !important;
        margin-bottom:10px !important;
    }

    .nv-mobile-menu-button {
        top:5px !important;
        left:0 !important;
        height:41px !important;
        padding:0 12px !important;
        font-size:.78rem !important;
    }

    .nv-mobile-drawer {
        width:248px !important;
        max-width:79vw !important;
    }

    [data-testid="stHeader"] {
        height:42px !important;
        min-height:42px !important;
    }

    .block-container {
        padding-top:.7rem !important;
    }
}


/* ============================================================
   NATIVE STREAMLIT MOBILE SIDEBAR — FINAL
   ============================================================ */

/* Kill all previous custom drawer/popover remnants. */
.nv-mobile-drawer-root,
.nv-mobile-drawer,
.nv-mobile-drawer-overlay,
.nv-mobile-menu-button,
.st-key-mobile_navbar {
    display:none !important;
}

/* Desktop: keep the NutriVision sidebar permanently clean and readable. */
@media (min-width:1025px) {
    section[data-testid="stSidebar"] {
        min-width:284px !important;
        max-width:284px !important;
        width:284px !important;
        display:block !important;
        visibility:visible !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding:1.1rem 1rem .9rem !important;
    }

    .brand {
        padding:3px 4px 17px !important;
    }

    .brand-logo-img {
        width:76px !important;
        height:76px !important;
    }

    .brand-name {
        font-size:1.20rem !important;
    }

    .brand-sub {
        font-size:.70rem !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] {
        gap:5px !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label {
        min-height:50px !important;
        padding:8px 11px !important;
        gap:11px !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label::before {
        width:33px !important;
        height:33px !important;
        flex-basis:33px !important;
        font-size:.98rem !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label p {
        font-size:.89rem !important;
        line-height:1.25 !important;
        font-weight:790 !important;
    }

    /* Locked mode has no collapse button on desktop. */
    [data-testid="stSidebarCollapsedControl"] {
        display:none !important;
    }
}

/* Tablet + mobile: native Streamlit sidebar becomes the left drawer. */
@media (max-width:1024px) {
    /* Keep header just tall enough for Streamlit's native menu button. */
    [data-testid="stHeader"] {
        height:50px !important;
        min-height:50px !important;
        background:rgba(247,249,252,.96) !important;
        border-bottom:1px solid #E8ECF2 !important;
        backdrop-filter:blur(12px);
        z-index:9999 !important;
    }

    /* Native collapsed sidebar / hamburger control. */
    [data-testid="stSidebarCollapsedControl"] {
        display:flex !important;
        visibility:visible !important;
        opacity:1 !important;
        position:fixed !important;
        top:8px !important;
        left:9px !important;
        z-index:10001 !important;
    }

    [data-testid="stSidebarCollapsedControl"] button {
        width:40px !important;
        height:36px !important;
        min-height:36px !important;
        border-radius:10px !important;
        border:1px solid #1A3B66 !important;
        background:linear-gradient(145deg,#112B50,#0A1C37) !important;
        color:#FFFFFF !important;
        box-shadow:0 7px 18px rgba(8,24,48,.20) !important;
    }

    [data-testid="stSidebarCollapsedControl"] button svg {
        color:#FFFFFF !important;
        fill:#FFFFFF !important;
        width:19px !important;
        height:19px !important;
    }

    /* True native left drawer, not full-screen. */
    section[data-testid="stSidebar"] {
        min-width:260px !important;
        max-width:260px !important;
        width:260px !important;
        background:
            radial-gradient(circle at 28% 5%,rgba(42,82,137,.34),transparent 14rem),
            linear-gradient(180deg,#102441 0%,#0B1B32 100%) !important;
        box-shadow:18px 0 42px rgba(5,17,34,.26) !important;
        border-right:1px solid rgba(255,255,255,.08) !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding:1.05rem .88rem .9rem !important;
        overflow-y:auto !important;
        overflow-x:hidden !important;
    }

    /* Mobile brand follows the desktop sidebar visual language. */
    .brand {
        display:flex !important;
        flex-direction:column !important;
        align-items:center !important;
        text-align:center !important;
        padding:4px 4px 15px !important;
    }

    .brand-logo-img {
        width:64px !important;
        height:64px !important;
        margin-bottom:4px !important;
    }

    .brand-name {
        font-size:1.05rem !important;
        line-height:1.15 !important;
    }

    .brand-sub {
        font-size:.61rem !important;
        margin-top:3px !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] {
        gap:4px !important;
        margin-top:3px !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label {
        min-height:46px !important;
        padding:7px 9px !important;
        gap:10px !important;
        border-radius:12px !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label::before {
        width:31px !important;
        height:31px !important;
        flex-basis:31px !important;
        font-size:.91rem !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label p {
        font-size:.80rem !important;
        line-height:1.2 !important;
        font-weight:790 !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background:linear-gradient(
            90deg,
            rgba(255,201,40,.14),
            rgba(255,255,255,.025)
        ) !important;
        border:1px solid rgba(255,201,40,.22) !important;
        box-shadow:inset 3px 0 0 #FFC928 !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked)::before {
        background:linear-gradient(145deg,#FFD957,#FFC225) !important;
        color:#172039 !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
        color:#FFD548 !important;
        font-weight:900 !important;
    }

    /* Don't let the descriptive footer make mobile navigation too tall. */
    .about-card {
        display:none !important;
    }

    /* Main content starts below the native mobile header. */
    .block-container {
        padding-top:1.15rem !important;
    }

    /* Slightly larger mobile content typography. */
    .hero-eyebrow {
        font-size:.62rem !important;
    }

    .title {
        font-size:1.74rem !important;
    }

    .subtitle {
        font-size:.77rem !important;
        line-height:1.62 !important;
    }

    .section-title {
        font-size:1.20rem !important;
    }

    .section-sub {
        font-size:.75rem !important;
    }

    .workflow-title {
        font-size:.82rem !important;
    }

    .workflow-sub {
        font-size:.69rem !important;
    }

    .panel-title {
        font-size:1.14rem !important;
    }

    .panel-sub {
        font-size:.79rem !important;
    }
}

/* Phones */
@media (max-width:600px) {
    section[data-testid="stSidebar"] {
        min-width:248px !important;
        max-width:248px !important;
        width:248px !important;
    }

    [data-testid="stHeader"] {
        height:48px !important;
        min-height:48px !important;
    }

    [data-testid="stSidebarCollapsedControl"] {
        top:7px !important;
        left:8px !important;
    }

    [data-testid="stSidebarCollapsedControl"] button {
        width:39px !important;
        height:35px !important;
    }

    .brand-logo-img {
        width:58px !important;
        height:58px !important;
    }

    .brand-name {
        font-size:1rem !important;
    }

    .brand-sub {
        font-size:.58rem !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label {
        min-height:44px !important;
        padding:6px 8px !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] > label p {
        font-size:.77rem !important;
    }

    .title {
        font-size:1.62rem !important;
    }

    .subtitle {
        font-size:.74rem !important;
    }
}

/* Hard safety: native radio bullets never appear. */
section[data-testid="stSidebar"] input[type="radio"],
section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child,
section[data-testid="stSidebar"] [role="radiogroup"] label > div:has(input[type="radio"]) {
    display:none !important;
    opacity:0 !important;
    width:0 !important;
    min-width:0 !important;
    height:0 !important;
    margin:0 !important;
    padding:0 !important;
}


/* ============================================================
   MOBILE NATIVE SIDEBAR BUTTON — HARD FIX
   ============================================================ */

/* Desktop keeps a locked/expanded sidebar and does not need an opener. */
@media (min-width:1025px) {
    [data-testid="stToolbar"] {
        display:none !important;
    }
}

/* Narrow screens: preserve Streamlit's header/control layer. */
@media (max-width:1024px) {
    [data-testid="stHeader"] {
        display:flex !important;
        visibility:visible !important;
        opacity:1 !important;
        height:52px !important;
        min-height:52px !important;
        background:rgba(247,249,252,.98) !important;
        border-bottom:1px solid #E6EAF0 !important;
        z-index:99999 !important;
    }

    [data-testid="stToolbar"] {
        display:flex !important;
        visibility:visible !important;
        opacity:1 !important;
        min-height:44px !important;
        z-index:100000 !important;
    }

    /* Known Streamlit sidebar open-control selector. */
    [data-testid="stSidebarCollapsedControl"] {
        display:flex !important;
        visibility:visible !important;
        opacity:1 !important;
        pointer-events:auto !important;
        position:fixed !important;
        top:7px !important;
        left:9px !important;
        z-index:100005 !important;
    }

    [data-testid="stSidebarCollapsedControl"] > button,
    [data-testid="stSidebarCollapsedControl"] button {
        display:flex !important;
        visibility:visible !important;
        opacity:1 !important;
        align-items:center !important;
        justify-content:center !important;
        width:42px !important;
        min-width:42px !important;
        height:38px !important;
        min-height:38px !important;
        padding:0 !important;
        border-radius:11px !important;
        border:1px solid #1B3B65 !important;
        background:linear-gradient(145deg,#112B50,#0A1C37) !important;
        color:#FFFFFF !important;
        box-shadow:0 7px 18px rgba(8,24,48,.18) !important;
    }

    [data-testid="stSidebarCollapsedControl"] svg {
        color:#FFFFFF !important;
        fill:currentColor !important;
        width:20px !important;
        height:20px !important;
    }

    /* Fallback selectors for Streamlit releases whose test-id changed.
       These only affect sidebar-labelled buttons in the header. */
    [data-testid="stHeader"] button[aria-label*="sidebar" i],
    [data-testid="stHeader"] button[title*="sidebar" i],
    [data-testid="stToolbar"] button[aria-label*="sidebar" i],
    [data-testid="stToolbar"] button[title*="sidebar" i] {
        display:flex !important;
        visibility:visible !important;
        opacity:1 !important;
        pointer-events:auto !important;
        align-items:center !important;
        justify-content:center !important;
        position:fixed !important;
        top:7px !important;
        left:9px !important;
        z-index:100006 !important;
        width:42px !important;
        min-width:42px !important;
        height:38px !important;
        min-height:38px !important;
        padding:0 !important;
        border-radius:11px !important;
        border:1px solid #1B3B65 !important;
        background:linear-gradient(145deg,#112B50,#0A1C37) !important;
        color:#FFFFFF !important;
        box-shadow:0 7px 18px rgba(8,24,48,.18) !important;
    }

    /* Header starts above content; don't leave a giant blank band. */
    .block-container {
        padding-top:1rem !important;
    }

    /* Native mobile left drawer. */
    section[data-testid="stSidebar"] {
        min-width:260px !important;
        max-width:260px !important;
        width:260px !important;
        background:
            radial-gradient(circle at 28% 5%,rgba(42,82,137,.34),transparent 14rem),
            linear-gradient(180deg,#102441 0%,#0B1B32 100%) !important;
        border-right:1px solid rgba(255,255,255,.08) !important;
        box-shadow:18px 0 42px rgba(5,17,34,.28) !important;
        z-index:100004 !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding:1rem .88rem .85rem !important;
        overflow-y:auto !important;
    }

    .about-card {
        display:none !important;
    }
}

@media (max-width:600px) {
    [data-testid="stHeader"] {
        height:50px !important;
        min-height:50px !important;
    }

    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stHeader"] button[aria-label*="sidebar" i],
    [data-testid="stHeader"] button[title*="sidebar" i],
    [data-testid="stToolbar"] button[aria-label*="sidebar" i],
    [data-testid="stToolbar"] button[title*="sidebar" i] {
        top:6px !important;
        left:8px !important;
    }

    section[data-testid="stSidebar"] {
        min-width:248px !important;
        max-width:248px !important;
        width:248px !important;
    }
}


/* ============================================================
   SIDEBAR INFO CARD — REFINED
   ============================================================ */
@media (min-width:1025px) {
    .about-card {
        position:relative !important;
        width:100% !important;
        margin:17px 0 0 !important;
        padding:14px 14px 13px !important;
        border:1px solid rgba(255,255,255,.10) !important;
        border-radius:16px !important;
        background:
            radial-gradient(circle at 92% 10%,rgba(255,201,40,.08),transparent 6rem),
            linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.028)) !important;
        box-shadow:none !important;
        color:#C4CDDC !important;
        font-size:inherit !important;
        line-height:normal !important;
    }

    /* Disable the older generated star. */
    .about-card::after {
        display:none !important;
    }

    .about-card-head {
        display:flex !important;
        align-items:center !important;
        justify-content:space-between !important;
        gap:10px !important;
    }

    .about-card-name {
        color:#FFFFFF !important;
        font-size:.80rem !important;
        font-weight:900 !important;
        line-height:1.2 !important;
    }

    .about-card-star {
        color:#FFC928 !important;
        font-size:1.12rem !important;
        line-height:1 !important;
    }

    .about-card-text {
        margin-top:10px !important;
        color:#B8C3D5 !important;
        font-size:.66rem !important;
        line-height:1.62 !important;
    }

    .about-mini {
        margin-top:11px !important;
        padding-top:10px !important;
        border-top:1px solid rgba(255,255,255,.09) !important;
        color:inherit !important;
        font-size:inherit !important;
    }

    .about-mini-label {
        color:#8290A6 !important;
        font-size:.56rem !important;
        font-weight:700 !important;
        margin-bottom:7px !important;
    }

    .about-tech-row {
        display:flex !important;
        flex-wrap:wrap !important;
        gap:5px !important;
    }

    .about-tech-row span {
        display:inline-flex !important;
        align-items:center !important;
        min-height:24px !important;
        padding:4px 7px !important;
        border-radius:999px !important;
        border:1px solid rgba(255,255,255,.08) !important;
        background:rgba(255,255,255,.045) !important;
        color:#AEBACD !important;
        font-size:.52rem !important;
        font-weight:750 !important;
        line-height:1.1 !important;
    }
}

/* Compact desktop heights: keep the card but compress it instead of
   making the whole sidebar feel too long. */
@media (min-width:1025px) and (max-height:820px) {
    .about-card {
        margin-top:12px !important;
        padding:11px 12px !important;
    }

    .about-card-text {
        margin-top:7px !important;
        font-size:.61rem !important;
        line-height:1.48 !important;
    }

    .about-mini {
        margin-top:8px !important;
        padding-top:7px !important;
    }

    .about-tech-row span {
        min-height:21px !important;
        padding:3px 6px !important;
        font-size:.49rem !important;
    }
}

/* Mobile/tablet: keep the navigation focused and short. */
@media (max-width:1024px) {
    .about-card {
        display:none !important;
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
    convnext_path = MODEL_DIR / "convnextv2_tiny_best.pt"
    noisyvit_path = MODEL_DIR / "noisyvit_b16_best.pt"
    mal_path = MODEL_DIR / "malnutrition_model.joblib"

    required = [convnext_path, noisyvit_path, mal_path]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Model belum tersedia: " + ", ".join(missing)
        )

    convnext_model = load_convnext_model(
        convnext_path,
        num_classes=53,
    )
    noisyvit_model = load_noisyvit_model(
        noisyvit_path,
        num_classes=53,
    )
    malnutrition_artifact = load_malnutrition_artifact(
        mal_path
    )

    return (
        convnext_model,
        noisyvit_model,
        malnutrition_artifact,
    )


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

    if not isinstance(class_names, list):
        raise ValueError("class_names.json harus berupa list nama kelas.")
    if len(class_names) != 53:
        raise ValueError(
            f"class_names.json harus berisi 53 kelas ensemble, ditemukan {len(class_names)}."
        )

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


def knowledge_group_chart():
    groups = ["Makronutrien", "Mikronutrien"]
    values = [2, 4]
    colors = ["#FFC51D", "#F5D86E"]

    fig = go.Figure(
        go.Pie(
            labels=groups,
            values=values,
            hole=.58,
            marker=dict(colors=colors),
            textinfo="label+value",
            hovertemplate="%{label}: %{value} nutrien<extra></extra>",
        )
    )
    fig.update_layout(
        title="Komposisi Kelompok Nutrisi",
        height=320,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=20),
        annotations=[
            dict(
                text="<b>6</b><br>nutrien",
                x=.5, y=.5, showarrow=False,
                font=dict(size=18, color="#1A2740"),
            )
        ],
    )
    return fig


def model_compare_chart(preds):
    if not preds:
        return go.Figure()

    df = pd.DataFrame(
        [
            {
                "kelas": p["food"],
                "Ensemble": round(float(p.get("confidence", 0)) * 100, 2),
                "ConvNeXt": round(float(p.get("convnext_confidence", 0)) * 100, 2),
                "NoisyViT": round(float(p.get("noisyvit_confidence", 0)) * 100, 2),
            }
            for p in preds[:3]
        ]
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Ensemble", x=df["kelas"], y=df["Ensemble"], marker_color="#FFC928"))
    fig.add_trace(go.Bar(name="ConvNeXt", x=df["kelas"], y=df["ConvNeXt"], marker_color="#163A6C"))
    fig.add_trace(go.Bar(name="NoisyViT", x=df["kelas"], y=df["NoisyViT"], marker_color="#7E8FB0"))

    fig.update_layout(
        barmode="group",
        title="Perbandingan Confidence Top-3",
        height=330,
        yaxis_title="Confidence (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=15, t=50, b=40),
        legend=dict(orientation="h", y=1.12, x=.5, xanchor="center"),
        yaxis=dict(gridcolor="#EFE9D8", range=[0, 100]),
    )
    return fig


def ensemble_weight_chart(ensemble_info):
    conv = float(ensemble_info.get("convnext_weight", 0.5)) * 100
    vit = float(ensemble_info.get("noisyvit_weight", 0.5)) * 100

    fig = go.Figure(
        go.Pie(
            labels=["ConvNeXt", "NoisyViT"],
            values=[conv, vit],
            hole=.64,
            marker=dict(colors=["#163A6C", "#FFC928"]),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.0f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Komposisi Bobot Ensemble",
        height=300,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=20),
        annotations=[
            dict(
                text="<b>Soft</b><br>Voting",
                x=.5, y=.5, showarrow=False,
                font=dict(size=16, color="#1A2740"),
            )
        ],
    )
    return fig


def recommendation_bubble_chart(recommendations):
    """
    Visual alternatif recommendation engine:
    score pada sumbu X, coverage nutrisi pada sumbu Y.
    Ukuran bubble mengikuti ranking agar kandidat utama lebih menonjol.
    """
    if not recommendations:
        return go.Figure()

    rows = []
    max_cov = max(
        [int(x.get("coverage", 0)) for x in recommendations] + [1]
    )

    for rank, rec in enumerate(recommendations[:5], 1):
        rows.append(
            {
                "food": str(rec.get("food", "-")),
                "score": float(rec.get("score", 0)),
                "coverage": int(rec.get("coverage", 0)),
                "rank": rank,
                "size": max(26, 54 - rank * 5),
            }
        )

    df = pd.DataFrame(rows)

    fig = go.Figure(
        go.Scatter(
            x=df["score"],
            y=df["coverage"],
            mode="markers+text",
            text=[
                f"#{r}<br>{food}"
                for r, food in zip(df["rank"], df["food"])
            ],
            textposition="top center",
            marker=dict(
                size=df["size"],
                color=df["rank"],
                colorscale=[
                    [0.0, "#FFC928"],
                    [0.35, "#F4D66B"],
                    [1.0, "#163A6C"],
                ],
                reversescale=False,
                line=dict(color="#FFFFFF", width=2),
                opacity=.92,
                showscale=False,
            ),
            customdata=np.stack(
                [df["food"], df["rank"]],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Ranking: #%{customdata[1]}<br>"
                "Score: %{x:.1f}/100<br>"
                "Coverage: %{y}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Peta Kecocokan Rekomendasi",
        height=360,
        xaxis=dict(
            title="Skor kecocokan internal",
            range=[max(0, float(df["score"].min()) - 8), 104],
            gridcolor="#EEF1F5",
            zeroline=False,
        ),
        yaxis=dict(
            title="Coverage nutrisi prioritas",
            range=[-0.4, max_cov + .8],
            dtick=1,
            gridcolor="#EEF1F5",
            zeroline=False,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=55, r=25, t=55, b=50),
    )

    return fig


def colorize_gradcam(heatmap):
    """Buat heatmap Grad-CAM berwarna tanpa dependency tambahan."""
    h = np.clip(np.asarray(heatmap, dtype=np.float32), 0, 1)

    # Navy -> yellow -> orange/red heat gradient.
    r = np.clip(35 + 230 * h, 0, 255)
    g = np.clip(45 + 190 * np.sqrt(h), 0, 255)
    b = np.clip(75 - 65 * h, 0, 255)

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb).resize((224, 224), Image.Resampling.BILINEAR)


def nutrition_cards_payload():
    return [
        ("⚡", "Energi", "Makronutrien", "Mendukung aktivitas, pertumbuhan, dan membantu melihat seberapa besar makanan berkontribusi terhadap kebutuhan harian anak."),
        ("🥚", "Protein", "Makronutrien", "Berperan dalam pembentukan jaringan tubuh. Pada NutriVision, kontribusi protein membantu recommendation engine memilih makanan pendamping yang relevan."),
        ("🩸", "Zat Besi", "Mikronutrien", "Penting untuk pembentukan hemoglobin. Jika kontribusinya rendah, rekomendasi dapat diarahkan ke makanan yang lebih kaya zat besi."),
        ("🦴", "Kalsium", "Mikronutrien", "Berkaitan dengan tulang dan gigi. Analisis berbasis data nutrisi makanan dan acuan AKG, bukan pemeriksaan laboratorium."),
        ("👁️", "Vitamin A", "Mikronutrien", "Mendukung fungsi penglihatan dan imunitas. Persentase kontribusi menunjukkan peran makanan terhadap acuan harian."),
        ("🍊", "Vitamin C", "Mikronutrien", "Membantu berbagai proses metabolik dan dapat mendukung penyerapan zat besi non-heme.")
    ]


def history_overview(history):
    if not history:
        return {"total": 0, "latest_status": "-", "top_food": "-", "avg_conf": 0.0}

    df = pd.DataFrame(history)
    return {
        "total": int(len(df)),
        "latest_status": str(df.iloc[-1]["status_gizi"]),
        "top_food": str(df["makanan"].value_counts().index[0]),
        "avg_conf": float(df["confidence"].mean()),
    }


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
# SIDEBAR NAVIGATION
# ============================================================

# Sinkronkan sidebar desktop dengan page aktif.
if st.session_state.get("sidebar_nav") != st.session_state.nav_page:
    st.session_state.sidebar_nav = st.session_state.nav_page


with st.sidebar:
    st.markdown(
        f"""<div class="brand"><img class="brand-logo-img" src="{LOGO_ICON_URI}" alt="NutriVision logo"><div class="brand-name">Nutri<span class="vision">Vision</span> AI</div><div class="brand-sub">Intelligent Child Nutrition</div></div>""",
        unsafe_allow_html=True,
    )

    st.radio(
        "Navigasi",
        NAV_OPTIONS,
        key="sidebar_nav",
        label_visibility="collapsed",
        on_change=_sync_sidebar_navigation,
    )


# Mobile menggunakan sidebar native Streamlit.
# Pada layar sempit sidebar akan menjadi drawer dari sisi kiri.
page = st.session_state.nav_page


# ============================================================
# HEADER
# ============================================================

now = datetime.now().strftime("%d %B %Y • %H:%M")

st.markdown(
    f"""<div class="hero-shell"><div class="hero-inner"><div class="hero-copy"><div class="hero-eyebrow">AI-POWERED NUTRITION SCREENING</div><div class="title">Nutri<span>Vision</span> AI</div><div class="subtitle">Integrasi data antropometri dan ensemble ConvNeXt V2 Tiny + NoisyViT B/16 untuk screening status gizi, pengenalan makanan, analisis nutrisi, dan rekomendasi yang lebih terarah.</div></div><div class="hero-logo-wrap"><img class="hero-logo-img" src="{LOGO_ICON_URI}" alt="NutriVision AI"></div><div class="hero-meta"><div class="date-pill">▣ {now}</div><div class="tech-row"><span class="tech-chip">◈ Ensemble Deep Learning</span><span class="tech-chip">♜ 53 Food Classes</span><span class="tech-chip">◇ Soft Voting</span></div></div></div></div>""",
    unsafe_allow_html=True,
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    profile = st.session_state.child_profile


    st.markdown(
        """<div style="margin-bottom:13px"><div class="section-kicker">Smart Screening Workflow</div><div class="section-title">Mulai Analisis NutriVision</div><div class="section-sub">Lengkapi data anak, unggah foto makanan, kemudian jalankan analisis AI.</div></div><div class="workflow-strip"><div class="workflow-item"><div class="workflow-number">01</div><div><div class="workflow-title">Data Antropometri</div><div class="workflow-sub">Umur, BB, TB, dan MUAC/LILA</div></div></div><div class="workflow-arrow">→</div><div class="workflow-item"><div class="workflow-number">02</div><div><div class="workflow-title">Foto Makanan</div><div class="workflow-sub">Satu menu utama dengan objek jelas</div></div></div><div class="workflow-arrow">→</div><div class="workflow-item"><div class="workflow-number">03</div><div><div class="workflow-title">Analisis AI</div><div class="workflow-sub">Screening, nutrisi, dan rekomendasi</div></div></div></div>""",
        unsafe_allow_html=True,
    )

    input_col, image_col, action_col = st.columns(
        [1.12, .95, .96],
        gap="small",
    )

    with input_col:
        with st.container(key="child_card"):
            st.markdown(
                '<div class="panel-title">♙ &nbsp;Data Anak</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="panel-sub">Input langsung pada dashboard agar alur analisis lebih ringkas.</div>',
                unsafe_allow_html=True,
            )

            r1c1, r1c2 = st.columns(2)
            with r1c1:
                age_months = st.number_input(
                    "Umur (bulan)", min_value=6, max_value=83,
                    value=int(profile["age_months"]), step=1,
                )
            with r1c2:
                weight_kg = st.number_input(
                    "Berat badan (kg)", min_value=3.0, max_value=40.0,
                    value=float(profile["weight_kg"]), step=.1,
                )

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                height_cm = st.number_input(
                    "Tinggi badan (cm)", min_value=45.0, max_value=140.0,
                    value=float(profile["height_cm"]), step=.5,
                )
            with r2c2:
                muac_cm = st.number_input(
                    "MUAC / LILA (cm)", min_value=5.0, max_value=30.0,
                    value=float(profile["muac_cm"]), step=.1,
                )

            bmi_preview = weight_kg / ((height_cm / 100) ** 2)
            p1, p2, p3 = st.columns(3)
            p1.metric("Usia", age_text(age_months))
            p2.metric("BMI", f"{bmi_preview:.2f}")
            p3.metric("MUAC", f"{muac_cm:.1f} cm")

    with image_col:
        with st.container(key="photo_card"):
            st.markdown(
                '<div class="panel-title">▣ &nbsp;Foto Makanan</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="panel-sub">Gunakan foto dengan satu menu utama yang jelas.</div>',
                unsafe_allow_html=True,
            )

            image = None

            # Sebelum ada foto: tampilkan uploader kuning.
            # Sesudah foto dipilih: uploader disembunyikan dan diganti preview.
            if st.session_state.food_image_bytes is None:
                uploaded = st.file_uploader(
                    "Upload foto",
                    type=["jpg", "jpeg", "png", "webp"],
                    label_visibility="collapsed",
                    key="food_upload_widget",
                )

                if uploaded is not None:
                    st.session_state.food_image_bytes = uploaded.getvalue()
                    st.session_state.food_image_name = uploaded.name
                    st.rerun()

            else:
                try:
                    image = Image.open(
                        io.BytesIO(st.session_state.food_image_bytes)
                    ).convert("RGB")
                    preview = fixed_preview(image)

                    st.image(
                        preview,
                        use_container_width=True,
                        caption="Preview makanan",
                    )

                    if st.button(
                        "↻  Ganti foto makanan",
                        key="change_food_photo",
                        use_container_width=True,
                    ):
                        st.session_state.food_image_bytes = None
                        st.session_state.food_image_name = None
                        st.session_state.pop("food_upload_widget", None)
                        st.rerun()

                except Exception:
                    st.session_state.food_image_bytes = None
                    st.session_state.food_image_name = None
                    st.session_state.pop("food_upload_widget", None)
                    st.rerun()

            st.markdown(
                '<div class="photo-tip"><span class="photo-tip-icon">ⓘ</span><span>Pastikan foto diambil dari atas (top-down) dengan pencahayaan yang baik.</span></div>',
                unsafe_allow_html=True,
            )

    with action_col:
        with st.container(key="action_card"):
            st.markdown(
                """<div class="cta-box"><div class="cta-badge">AI ANALYSIS</div><div class="cta-title">Analisis dalam Satu Alur</div><div class="cta-sub">ConvNeXt V2 Tiny dan NoisyViT B/16 digabung dengan weighted soft voting, lalu diteruskan ke nutrition knowledge base dan recommendation engine.</div></div>""",
                unsafe_allow_html=True,
            )

            analyze = st.button(
                "🚀  Jalankan Analisis",
                type="primary",
                use_container_width=True,
                disabled=image is None,
            )

            st.markdown(
                '<div class="action-note"><span class="note-icon">◇</span><span>Hasil analisis bersifat informatif dan bukan pengganti konsultasi profesional kesehatan.</span></div>',
                unsafe_allow_html=True,
            )

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

                convnext_model, noisyvit_model, malnutrition_artifact = load_models()

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
                    convnext_model=convnext_model,
                    noisyvit_model=noisyvit_model,
                    class_names=class_names,
                    ensemble_weights=DEFAULT_ENSEMBLE_WEIGHTS,
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

        status_conf_raw = status.get("confidence")
        status_conf = (
            float(status_conf_raw) * 100
            if status_conf_raw is not None
            else 0.0
        )
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
        st.markdown(
            """
            <div class="section-kicker">AI Analysis Result</div>
            <div class="section-title">Ringkasan Hasil Analisis</div>
            <div class="section-sub">
                Hasil berikut menggabungkan screening antropometri, klasifikasi makanan,
                dan analisis kontribusi nutrisi.
            </div>
            """,
            unsafe_allow_html=True,
        )

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
                prio_html = "".join(
                    f'<span class="tag-chip">✦ {x}</span>'
                    for x in nutrient_priorities(adequacy)
                )

                st.markdown(
                    f"""
                    <div class="rec-hero">
                        <div class="rec-hero-title">Rekomendasi Makanan Prioritas</div>
                        <div class="rec-hero-text">
                            Recommendation engine menyusun kandidat makanan berdasarkan gap nutrisi
                            yang perlu diprioritaskan. Fokus saat ini: {", ".join(nutrient_priorities(adequacy)) or "kontribusi nutrisi umum"}.
                        </div>
                        <div class="tag-row">{prio_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <div class="rec-visual-head">
                        <div>
                            <div class="rec-visual-title">Ranking Kandidat Makanan</div>
                            <div class="rec-visual-sub">
                                Urutan dibuat berdasarkan skor kecocokan internal.
                                Coverage menunjukkan jumlah nutrien prioritas yang didukung setiap kandidat.
                            </div>
                        </div>
                        <div class="rec-legend">
                            <span>★ Ranking</span>
                            <span>Score 0–100</span>
                            <span>Coverage nutrisi</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                ranking_rows = []
                for rank, rec in enumerate(recommendations[:5], 1):
                    score = float(rec.get("score", 0))
                    coverage = int(rec.get("coverage", 0))
                    reason = str(rec.get("reason", ""))
                    top_class = " top" if rank == 1 else ""

                    ranking_rows.append(
                        f'<div class="ranking-row{top_class}">'
                        f'<div class="rank-number">#{rank}</div>'
                        f'<div class="rank-food">'
                        f'<div class="rank-food-name">{rec["food"]}</div>'
                        f'<div class="rank-reason">{reason}</div>'
                        f'</div>'
                        f'<div class="ranking-score">'
                        f'<div class="rank-score-label"><span>Skor</span><b>{score:.0f}/100</b></div>'
                        f'<div class="rank-progress"><span style="width:{max(0,min(score,100)):.1f}%"></span></div>'
                        f'</div>'
                        f'<div class="rank-coverage">{coverage} nutrien<br>tercakup</div>'
                        f'</div>'
                    )

                st.markdown(
                    '<div class="ranking-board">'
                    '<div class="ranking-head">'
                    '<div>Rank</div><div>Makanan</div><div>Skor</div><div>Coverage</div>'
                    '</div>'
                    + "".join(ranking_rows)
                    + '</div>',
                    unsafe_allow_html=True,
                )

                cols = st.columns(min(3, len(recommendations)))
                for i, rec in enumerate(recommendations[:3]):
                    with cols[i]:
                        st.markdown(
                            f"""
                            <div class="recommend-card-rich">
                                <div class="rec-rank-badge">🥗 #{i+1} &nbsp; Kandidat Utama</div>
                                <div class="food">{rec["food"]}</div>
                                <div class="score">{rec["score"]:.0f}/100</div>
                                <div class="reason">{rec.get("reason", "Direkomendasikan untuk membantu menutup gap nutrisi prioritas.")}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                if len(recommendations) > 3:
                    st.markdown("#### Kandidat Lain")
                    extra_cols = st.columns(2)
                    for i, rec in enumerate(recommendations[3:5]):
                        with extra_cols[i % 2]:
                            st.markdown(
                                f"""
                                <div class="insight-soft">
                                    <b>#{i+4} • {rec["food"]}</b><br>
                                    <span style="color:#0A9F76;font-weight:900;">{rec["score"]:.0f}/100</span><br>
                                    <span style="color:#8792A6;font-size:.74rem;line-height:1.58;">{rec.get("reason", "")}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                st.markdown(
                    '<div class="dark-note" style="color:#59677E;background:#FBFCFE;border:1px solid #E8EDF5;">'
                    'Skor rekomendasi adalah skor kecocokan internal berbasis gap nutrisi, bukan nilai mutu absolut suatu makanan.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Recommendation engine belum menghasilkan kandidat.")

        with tab4:
            try:
                heat, idx, conf, layer_info = make_gradcam(
                    image,
                    convnext_model,
                )
                overlay = overlay_gradcam(image, heat)
                colored_heat = colorize_gradcam(heat)

                pred_label = (
                    class_names[idx]
                    if 0 <= idx < len(class_names)
                    else f"Kelas {idx}"
                )

                st.markdown(
                    f"""
                    <div class="xai-hero">
                        <div class="xai-intro">
                            <div class="xai-intro-title">Grad-CAM • Explainable AI Workspace</div>
                            <div class="xai-intro-text">
                                Visualisasi ini membantu melihat area citra yang relatif lebih berpengaruh
                                pada keputusan <b>branch ConvNeXt V2 Tiny</b>. Ini bukan penjelasan penuh
                                keputusan ensemble ConvNeXt + NoisyViT.
                            </div>
                        </div>
                        <div class="xai-stat">
                            <div class="xai-stat-label">Prediksi Branch</div>
                            <div class="xai-stat-value">{pred_label}</div>
                        </div>
                        <div class="xai-stat">
                            <div class="xai-stat-label">Confidence</div>
                            <div class="xai-stat-value">{conf*100:.2f}%</div>
                        </div>
                        <div class="xai-stat">
                            <div class="xai-stat-label">Explainability</div>
                            <div class="xai-stat-value">ConvNeXt</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c1, c2, c3 = st.columns(3, gap="large")

                with c1:
                    with st.container(key="grad_original"):
                        st.markdown(
                            '<div class="grad-card-head"><div class="grad-card-title">Foto Input</div><div class="grad-card-badge">ORIGINAL</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.image(
                            image.resize((224, 224)),
                            use_container_width=True,
                        )

                with c2:
                    with st.container(key="grad_heatmap"):
                        st.markdown(
                            '<div class="grad-card-head"><div class="grad-card-title">Activation Map</div><div class="grad-card-badge">HEATMAP</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.image(
                            colored_heat,
                            use_container_width=True,
                        )

                with c3:
                    with st.container(key="grad_overlay"):
                        st.markdown(
                            '<div class="grad-card-head"><div class="grad-card-title">Model Attention Overlay</div><div class="grad-card-badge">OVERLAY</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.image(
                            overlay,
                            use_container_width=True,
                        )

                st.markdown(
                    f"""
                    <div class="xai-explain-grid">
                        <div class="xai-explain">
                            <b>Cara membaca:</b> area dengan aktivasi lebih kuat menunjukkan region yang
                            relatif lebih banyak memengaruhi keluaran branch ConvNeXt ketika memprediksi
                            <b>{pred_label}</b>. Heatmap sebaiknya dibaca bersama foto asli dan overlay,
                            bukan sebagai segmentasi objek atau bukti kausal.
                        </div>
                        <div class="xai-layer">
                            <div class="label">TARGET FEATURE LAYER</div>
                            <div class="value">{layer_info}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as exc:
                st.warning("Grad-CAM belum dapat divisualisasikan.")
                st.code(str(exc))

        with tab5:
            ensemble_info = result.get("ensemble", {})

            st.markdown(
                f"""
                <div class="model-pill-grid">
                    <div class="model-pill">
                        <div class="model-pill-label">Backbone 1</div>
                        <div class="model-pill-value">ConvNeXt V2 Tiny</div>
                        <div class="model-pill-note">Cabang CNN untuk pengenalan citra makanan.</div>
                    </div>
                    <div class="model-pill">
                        <div class="model-pill-label">Backbone 2</div>
                        <div class="model-pill-value">NoisyViT B/16</div>
                        <div class="model-pill-note">Cabang transformer untuk visual representation.</div>
                    </div>
                    <div class="model-pill">
                        <div class="model-pill-label">Metode Fusi</div>
                        <div class="model-pill-value">Weighted Soft Voting</div>
                        <div class="model-pill-note">Bobot saat ini ConvNeXt {ensemble_info.get('convnext_weight', 0.5):.0%} • NoisyViT {ensemble_info.get('noisyvit_weight', 0.5):.0%}.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            left_model, right_model = st.columns([1.35, .95], gap="large")
            with left_model:
                st.plotly_chart(
                    model_compare_chart(preds),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            with right_model:
                st.plotly_chart(
                    ensemble_weight_chart(ensemble_info),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Kelas": p["food"],
                            "Ensemble (%)": round(p["confidence"] * 100, 2),
                            "ConvNeXt (%)": round(p.get("convnext_confidence", 0) * 100, 2),
                            "NoisyViT (%)": round(p.get("noisyvit_confidence", 0) * 100, 2),
                        }
                        for p in preds
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

            st.markdown(
                '''
                <div class="flow-grid">
                    <div class="flow-node"><div class="num">01</div><div class="title">Input Antropometri</div><div class="text">Umur, berat, tinggi, dan MUAC/LILA diproses oleh model screening status gizi.</div></div>
                    <div class="flow-node"><div class="num">02</div><div class="title">Input Foto Makanan</div><div class="text">Satu foto makanan diproses paralel oleh ConvNeXt V2 Tiny dan NoisyViT B/16.</div></div>
                    <div class="flow-node"><div class="num">03</div><div class="title">Ensemble Prediction</div><div class="text">Dua probabilitas keluaran digabung menggunakan weighted soft voting untuk menghasilkan prediksi final 53 kelas.</div></div>
                    <div class="flow-node"><div class="num">04</div><div class="title">Nutrition Mapping</div><div class="text">Kelas makanan dipetakan ke knowledge base nutrisi, lalu dibandingkan dengan AKG kelompok umur.</div></div>
                    <div class="flow-node"><div class="num">05</div><div class="title">Gap Analysis</div><div class="text">Sistem menghitung kontribusi dan gap nutrisi prioritas dari makanan yang terdeteksi.</div></div>
                    <div class="flow-node"><div class="num">06</div><div class="title">Recommendation Engine</div><div class="text">Engine memberi ranking makanan pendamping yang lebih relevan dengan kebutuhan nutrisi.</div></div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

            st.caption(
                "Detail model menampilkan keluaran ensemble, kontribusi masing-masing branch, serta alur pemrosesan NutriVision AI secara ringkas."
            )


# ============================================================
# RIWAYAT ANALISIS
# ============================================================

elif page == "Riwayat Analisis":

    st.markdown("## Riwayat Analisis")
    st.caption("Riwayat analisis disimpan selama sesi Streamlit aktif.")

    if not st.session_state.history:
        st.info(
            "Belum ada riwayat analisis. Jalankan analisis dari Dashboard terlebih dahulu."
        )

    else:
        hist = pd.DataFrame(st.session_state.history)
        overview = history_overview(st.session_state.history)

        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card gold">
                    <div class="kpi-label">Total Analisis</div>
                    <div class="kpi-value">{overview["total"]}</div>
                    <div class="kpi-sub">Seluruh hasil pada sesi aktif.</div>
                </div>
                <div class="kpi-card navy">
                    <div class="kpi-label">Status Terakhir</div>
                    <div class="kpi-value">{str(overview["latest_status"]).upper()}</div>
                    <div class="kpi-sub">Hasil screening yang paling baru.</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Makanan Tersering</div>
                    <div class="kpi-value" style="font-size:1.10rem;">{overview["top_food"]}</div>
                    <div class="kpi-sub">Prediksi makanan yang paling sering muncul.</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Rata-rata Confidence</div>
                    <div class="kpi-value">{overview["avg_conf"]:.1f}%</div>
                    <div class="kpi-sub">Rata-rata confidence klasifikasi makanan.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        h1, h2 = st.columns(2, gap="large")
        with h1:
            st.plotly_chart(
                history_status_chart(st.session_state.history),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with h2:
            st.plotly_chart(
                history_food_chart(st.session_state.history),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.markdown("#### Detail Riwayat Analisis")
        st.dataframe(
            hist.rename(
                columns={
                    "waktu": "Waktu",
                    "umur_bulan": "Umur (bulan)",
                    "berat_kg": "Berat (kg)",
                    "tinggi_cm": "Tinggi (cm)",
                    "muac_cm": "MUAC (cm)",
                    "status_gizi": "Status Gizi",
                    "makanan": "Makanan",
                    "confidence": "Confidence (%)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown(
            '<div class="dark-note" style="color:#59677E;background:#FBFCFE;border:1px solid #E8EDF5;margin-top:10px;">Riwayat ini dapat digunakan sebagai demo monitoring penggunaan aplikasi pada sesi berjalan. Untuk penyimpanan permanen, riwayat dapat dihubungkan ke database.</div>',
            unsafe_allow_html=True,
        )

        if st.button("Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()


# ============================================================
# PROFIL ANAK
# ============================================================

elif page == "Profil Anak":

    st.markdown("## Profil Anak")
    st.caption(
        "Profil ini menjadi nilai default pada halaman Dashboard dan membantu mempercepat alur input analisis."
    )

    p = st.session_state.child_profile

    initial_name = str(p.get("name", "Anak"))
    initial_age = int(p["age_months"])
    initial_weight = float(p["weight_kg"])
    initial_height = float(p["height_cm"])
    initial_muac = float(p["muac_cm"])
    initial_bmi = initial_weight / ((initial_height / 100) ** 2)

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card gold">
                <div class="kpi-label">Nama Profil</div>
                <div class="kpi-value" style="font-size:1.12rem;">{initial_name}</div>
                <div class="kpi-sub">Profil default untuk proses analisis.</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Usia</div>
                <div class="kpi-value">{age_text(initial_age)}</div>
                <div class="kpi-sub">{initial_age} bulan</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Tinggi & Berat</div>
                <div class="kpi-value" style="font-size:1.12rem;">{initial_height:.0f} cm • {initial_weight:.1f} kg</div>
                <div class="kpi-sub">Parameter antropometri utama.</div>
            </div>
            <div class="kpi-card navy">
                <div class="kpi-label">BMI Terhitung</div>
                <div class="kpi-value">{initial_bmi:.2f}</div>
                <div class="kpi-sub">Fitur model, bukan diagnosis klinis.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(key="profile_form_card"):
            st.markdown(
                """
                <div class="profile-form-head">
                    <div class="profile-form-title">Form Profil Antropometri</div>
                    <div class="profile-form-sub">
                        Perbarui data dasar anak. Nilai yang disimpan akan otomatis
                        menjadi input awal pada Dashboard.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            name = st.text_input(
                "Nama / Label Anak",
                value=initial_name,
            )

            f1, f2 = st.columns(2)
            with f1:
                age = st.number_input(
                    "Umur (bulan)",
                    6,
                    83,
                    initial_age,
                    1,
                )
            with f2:
                weight = st.number_input(
                    "Berat badan (kg)",
                    3.0,
                    40.0,
                    initial_weight,
                    .1,
                )

            f3, f4 = st.columns(2)
            with f3:
                height = st.number_input(
                    "Tinggi badan (cm)",
                    45.0,
                    140.0,
                    initial_height,
                    .5,
                )
            with f4:
                muac = st.number_input(
                    "MUAC / LILA (cm)",
                    5.0,
                    30.0,
                    initial_muac,
                    .1,
                )

            bmi = weight / ((height / 100) ** 2)

            st.markdown(
                f"""
                <div class="profile-mini-grid">
                    <div class="profile-mini">
                        <div class="label">BMI terhitung</div>
                        <div class="value">{bmi:.2f}</div>
                    </div>
                    <div class="profile-mini">
                        <div class="label">Usia tampilan</div>
                        <div class="value">{age_text(age)}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            save_profile = st.button(
                "Simpan Profil",
                type="primary",
                use_container_width=True,
            )

    preview_profile = {
        "age_months": age,
        "weight_kg": weight,
        "height_cm": height,
        "muac_cm": muac,
    }

    with right:
        with st.container(key="profile_visual_card"):
            st.markdown(
                f"""
                <div class="profile-visual-head">
                    <div class="profile-visual-title">Ringkasan Profil Saat Ini</div>
                    <div class="profile-visual-sub">
                        Visualisasi ini merupakan ringkasan normalisasi dashboard,
                        bukan z-score WHO atau diagnosis klinis.
                    </div>
                </div>
                <div class="tag-row">
                    <span class="tag-chip">👶 {age_text(age)}</span>
                    <span class="tag-chip">⚖️ {weight:.1f} kg</span>
                    <span class="tag-chip">📏 {height:.1f} cm</span>
                    <span class="tag-chip">💪 MUAC {muac:.1f} cm</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.plotly_chart(
                profile_radar(preview_profile),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    if save_profile:
        st.session_state.child_profile = {
            "name": name,
            **preview_profile,
        }
        st.success("Profil anak berhasil disimpan.")


# ============================================================
# PENGETAHUAN GIZI
# ============================================================

elif page == "Pengetahuan Gizi":

    st.markdown("## Pengetahuan Gizi")
    st.caption("Halaman edukasi untuk membantu memahami nutrien yang dianalisis oleh NutriVision dan bagaimana hasilnya dibaca.")

    st.markdown(
        """
        <div class="kpi-grid">
            <div class="kpi-card gold"><div class="kpi-label">Total Nutrien</div><div class="kpi-value">6</div><div class="kpi-sub">Dua makronutrien dan empat mikronutrien.</div></div>
            <div class="kpi-card"><div class="kpi-label">Makronutrien</div><div class="kpi-value">2</div><div class="kpi-sub">Energi dan protein.</div></div>
            <div class="kpi-card"><div class="kpi-label">Mikronutrien</div><div class="kpi-value">4</div><div class="kpi-sub">Zat besi, kalsium, vitamin A, vitamin C.</div></div>
            <div class="kpi-card navy"><div class="kpi-label">Tujuan</div><div class="kpi-value" style="font-size:1.08rem;">Edukasi & Prioritas</div><div class="kpi-sub">Membantu memahami kontribusi makanan terhadap AKG.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2 = st.columns([1.2, .9], gap="large")
    with k1:
        st.plotly_chart(
            knowledge_chart(),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with k2:
        st.plotly_chart(
            knowledge_group_chart(),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    payload = nutrition_cards_payload()
    rows = [payload[:3], payload[3:]]
    for row in rows:
        cols = st.columns(3, gap="large")
        for col, item in zip(cols, row):
            icon, title, group, desc = item
            with col:
                st.markdown(
                    f"""
                    <div class="nutrient-tile">
                        <div class="nutrient-icon">{icon}</div>
                        <div class="nutrient-title">{title}</div>
                        <div class="nutrient-meta">{group}</div>
                        <div class="nutrient-text">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown(
        '<div class="dark-note" style="color:#59677E;background:#FBFCFE;border:1px solid #E8EDF5;margin-top:12px;">Persentase kontribusi pada dashboard menunjukkan seberapa besar makanan yang dianalisis membantu memenuhi acuan gizi kelompok umur. Nilai tersebut bukan hasil pemeriksaan laboratorium atau diagnosis klinis.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# TENTANG APLIKASI
# ============================================================

else:

    st.markdown("## Tentang Aplikasi")
    st.caption(
        "Ringkasan kapabilitas utama, pendekatan model, serta batasan penggunaan NutriVision AI."
    )

    st.markdown(
        """
        <div class="kpi-grid">
            <div class="kpi-card gold">
                <div class="kpi-label">Pendekatan</div>
                <div class="kpi-value" style="font-size:1.08rem;">Multimodal AI</div>
                <div class="kpi-sub">Antropometri + citra makanan.</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Food Classes</div>
                <div class="kpi-value">53</div>
                <div class="kpi-sub">Kelas pada image recognition.</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Ensemble</div>
                <div class="kpi-value" style="font-size:1.08rem;">CNN + ViT</div>
                <div class="kpi-sub">ConvNeXt V2 Tiny + NoisyViT B/16.</div>
            </div>
            <div class="kpi-card navy">
                <div class="kpi-label">Output</div>
                <div class="kpi-value" style="font-size:1.08rem;">Screening + Rekomendasi</div>
                <div class="kpi-sub">Status, nutrisi, dan saran makanan.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a1, a2 = st.columns(2, gap="large")

    with a1:
        st.markdown(
            """<div class="about-equal-panel about-cap-panel"><div class="about-cap-grid">
            <div class="about-cap-card">
                <div class="about-cap-icon">🔬</div>
                <div class="about-cap-title">NutriVision AI</div>
                <div class="about-cap-text">
                    Prototype multimodal yang menggabungkan data antropometri anak dan citra makanan
                    untuk screening status gizi, klasifikasi makanan, analisis kontribusi nutrisi
                    terhadap AKG, dan recommendation engine.
                </div>
            </div>
            <div class="about-cap-card">
                <div class="about-cap-icon">🧠</div>
                <div class="about-cap-title">Model & Explainability</div>
                <div class="about-cap-text">
                    Food recognition menggunakan ensemble ConvNeXt V2 Tiny dan NoisyViT B/16
                    dengan weighted soft voting pada 53 kelas. Grad-CAM menjelaskan branch ConvNeXt.
                </div>
            </div>
            <div class="about-cap-card">
                <div class="about-cap-icon">🎯</div>
                <div class="about-cap-title">Nilai Utama</div>
                <div class="about-cap-text">
                    Hasil model dihubungkan ke nutrition knowledge base, analisis kontribusi dan gap,
                    lalu recommendation engine menyusun ranking makanan pendamping.
                </div>
            </div>
            <div class="about-cap-card">
                <div class="about-cap-icon">⚠️</div>
                <div class="about-cap-title">Batasan</div>
                <div class="about-cap-text">
                    Sistem ditujukan untuk edukasi dan screening. Satu foto tidak mewakili asupan
                    24 jam dan rekomendasi bukan terapi klinis maupun pengganti tenaga kesehatan.
                </div>
            </div>
            </div></div>""",
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown(
            """<div class="about-equal-panel about-flow-panel">
            <div class="about-flow-title">Alur Sistem NutriVision</div>
            <div class="about-timeline">
                <div class="about-step">
                    <div class="about-step-num">01</div>
                    <div>
                        <div class="about-step-title">Data Anak</div>
                        <div class="about-step-text">Umur, berat, tinggi, dan MUAC/LILA menjadi input screening status gizi.</div>
                    </div>
                </div>
                <div class="about-step">
                    <div class="about-step-num">02</div>
                    <div>
                        <div class="about-step-title">Foto Makanan</div>
                        <div class="about-step-text">Foto menu utama diproses sebagai input klasifikasi makanan.</div>
                    </div>
                </div>
                <div class="about-step">
                    <div class="about-step-num">03</div>
                    <div>
                        <div class="about-step-title">Ensemble CNN + ViT</div>
                        <div class="about-step-text">Probabilitas ConvNeXt dan NoisyViT digabung melalui weighted soft voting.</div>
                    </div>
                </div>
                <div class="about-step">
                    <div class="about-step-num">04</div>
                    <div>
                        <div class="about-step-title">Analisis Nutrisi</div>
                        <div class="about-step-text">Prediksi makanan dipetakan ke knowledge base dan dibandingkan dengan AKG.</div>
                    </div>
                </div>
                <div class="about-step">
                    <div class="about-step-num">05</div>
                    <div>
                        <div class="about-step-title">Rekomendasi</div>
                        <div class="about-step-text">Engine menyusun ranking makanan berdasarkan gap nutrisi prioritas.</div>
                    </div>
                </div>
            </div>
            </div>""",
            unsafe_allow_html=True,
        )
