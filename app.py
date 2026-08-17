# =============================================================================
# NutriVision AI — Skrining Gizi Anak Berbasis AI
# © 2026 Tim Sang Surya 1 — Universitas Muhammadiyah Malang (UMM)
# Hak cipta dilindungi. Dikembangkan untuk keperluan lomba/edukasi.
# =============================================================================


from pathlib import Path
import base64
import io
import html
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import timm

from src.inference import (
    DEFAULT_ENSEMBLE_WEIGHTS,
    analyze_child_and_food,
    load_convnext_model,
    load_noisyvit_model,
    load_malnutrition_artifact,
    make_gradcam,
    overlay_gradcam,
)
from src import inference as _inference
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
    grid-template-columns:78px 1.7fr .9fr .8fr;
    gap:12px;
    padding:0 12px 9px;
    color:#929CAF;
    font-size:.63rem;
    font-weight:850;
    text-transform:uppercase;
    letter-spacing:.045em;
}
.ranking-head > div {
    white-space:nowrap;
    word-break:normal;
    overflow-wrap:normal;
    min-width:0;
}
.ranking-head > div:first-child {
    text-align:center;
}
.ranking-row {
    display:grid;
    grid-template-columns:78px 1.7fr .9fr .8fr;
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
# STYLE — PREMIUM REFINEMENT LAYER (loaded after base styles)
# Warna & struktur dipertahankan; hanya tampilan yang dipoles.
# ============================================================
st.markdown(
    """
<style>
/* ============================================================
   PREMIUM REFINEMENT LAYER  —  clean / premium re-skin
   Palette (gold + navy) and page structure are preserved.
   This layer is loaded AFTER the base styles, so it wins the
   cascade on the surfaces below without touching the base
   responsive rules or Streamlit-specific selectors.
   ============================================================ */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --pj:'Plus Jakarta Sans', Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --body:'Inter', var(--pj);
    --ink-1:#0F1B33;
    --ink-2:#3B475F;
    --ink-3:#6C7889;
    --hair:#ECEFF4;
    --hair-2:#E3E8EF;
    --paper:#FFFFFF;
    --r-lg:22px;
    --r-md:18px;
    --r-sm:13px;
    --e1:0 1px 2px rgba(16,27,51,.05), 0 2px 6px rgba(16,27,51,.04);
    --e2:0 2px 4px rgba(16,27,51,.03), 0 16px 38px rgba(16,27,51,.07);
    --e3:0 6px 12px rgba(16,27,51,.05), 0 28px 60px rgba(16,27,51,.11);
    --gold-grad:linear-gradient(105deg,#FFD75A 0%,#FFC01F 100%);
    --navy-grad:linear-gradient(150deg,#143260 0%,#0B1F3E 58%,#081A34 100%);
}

/* ---------- Base surface & typography ---------- */
html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1100px 480px at 100% -8%, rgba(255,201,40,.055), transparent 60%),
        radial-gradient(920px 520px at -8% 6%, rgba(20,50,96,.045), transparent 55%),
        linear-gradient(180deg,#FCFDFF 0%,#F3F6FB 100%) !important;
    color:var(--ink-1) !important;
    font-family:var(--body) !important;
    -webkit-font-smoothing:antialiased;
    text-rendering:optimizeLegibility;
}
.block-container { max-width:1440px !important; }

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    font-family:var(--pj) !important;
    color:var(--ink-1) !important;
    font-weight:800 !important;
    letter-spacing:-.025em !important;
}
[data-testid="stMarkdownContainer"] h2 { font-size:1.55rem !important; }
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color:var(--ink-3) !important;
    font-size:.82rem !important;
    line-height:1.6 !important;
}

/* ---------- Header height a touch airier ---------- */
[data-testid="stHeader"] {
    background:rgba(250,251,253,.82) !important;
    backdrop-filter:blur(12px) !important;
    border-bottom:1px solid rgba(236,239,244,.7) !important;
}

/* ============================================================
   SIDEBAR  —  premium product navigation
   ============================================================ */
[data-testid="stSidebar"] {
    background:
        radial-gradient(280px 220px at 28% 2%, rgba(46,88,150,.38), transparent 72%),
        var(--navy-grad) !important;
    border-right:1px solid rgba(255,255,255,.06) !important;
}
.brand-name {
    font-family:var(--pj) !important;
    letter-spacing:-.02em !important;
    font-weight:800 !important;
}
.brand-sub { letter-spacing:.12em !important; text-transform:uppercase; }
.brand-logo-img { filter:drop-shadow(0 12px 22px rgba(255,201,40,.20)) !important; }

[data-testid="stSidebar"] [role="radiogroup"] > label {
    border-radius:14px !important;
    transition:background .18s ease, transform .18s ease !important;
}
[data-testid="stSidebar"] [role="radiogroup"] > label p {
    font-family:var(--pj) !important;
    font-weight:600 !important;
    letter-spacing:.005em !important;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background:rgba(255,255,255,.06) !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background:linear-gradient(100deg,rgba(255,201,40,.16),rgba(255,255,255,.02)) !important;
    border:1px solid rgba(255,201,40,.24) !important;
    box-shadow:inset 3px 0 0 var(--gold), 0 10px 22px rgba(0,0,0,.20) !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked)::before {
    box-shadow:0 8px 16px rgba(255,201,40,.24) !important;
}
.about-card {
    border:1px solid rgba(255,255,255,.10) !important;
    border-radius:16px !important;
    background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.02)) !important;
}

/* ============================================================
   HERO  —  landing-grade header
   ============================================================ */
.hero-shell {
    border:1px solid var(--hair) !important;
    border-radius:26px !important;
    background:
        radial-gradient(680px 320px at 96% -22%, rgba(255,201,40,.20), transparent 62%),
        radial-gradient(520px 300px at 76% 122%, rgba(20,50,96,.05), transparent 60%),
        linear-gradient(115deg,#FFFFFF 0%,#FFFFFF 52%,#FFFCF3 100%) !important;
    box-shadow:var(--e2) !important;
}
.hero-shell:before { border-color:rgba(255,201,40,.10) !important; }
.hero-eyebrow {
    background:rgba(255,201,40,.12) !important;
    border:1px solid rgba(224,168,15,.34) !important;
    color:#8A6600 !important;
    border-radius:999px !important;
    font-family:var(--pj) !important;
    font-weight:700 !important;
    letter-spacing:.11em !important;
    text-transform:uppercase;
}
.title {
    font-family:var(--pj) !important;
    font-weight:800 !important;
    letter-spacing:-.045em !important;
}
.title span { color:#EDAE00 !important; }
.subtitle { color:var(--ink-3) !important; line-height:1.7 !important; }
.date-pill, .tech-chip {
    border:1px solid var(--hair-2) !important;
    background:rgba(255,255,255,.9) !important;
    color:var(--ink-2) !important;
    border-radius:999px !important;
    box-shadow:var(--e1) !important;
    backdrop-filter:blur(6px);
    font-family:var(--pj) !important;
    font-weight:600 !important;
}
.tech-chip { transition:transform .16s ease, box-shadow .16s ease; }
.tech-chip:hover { transform:translateY(-1px); box-shadow:var(--e2) !important; }

/* ============================================================
   SECTION HEADERS
   ============================================================ */
.section-kicker {
    color:#B07E00 !important;
    font-family:var(--pj) !important;
    letter-spacing:.14em !important;
}
.section-title, .panel-title, .about-flow-title,
.profile-form-title, .profile-visual-title,
.rec-visual-title, .kpi-value, .result-value,
.model-pill-value, .xai-intro-title, .xai-stat-value {
    font-family:var(--pj) !important;
    letter-spacing:-.02em !important;
}
.section-sub, .panel-sub { color:var(--ink-3) !important; }

/* ============================================================
   WORKFLOW STRIP
   ============================================================ */
.workflow-strip {
    border:1px solid var(--hair) !important;
    border-radius:var(--r-md) !important;
    box-shadow:var(--e1) !important;
}
.workflow-number {
    background:var(--gold-grad) !important;
    box-shadow:0 8px 16px rgba(255,192,45,.22) !important;
}
.workflow-title { font-family:var(--pj) !important; }
.workflow-arrow { background:#F3F6FB !important; }

/* ============================================================
   UNIFIED CARD SYSTEM  —  soft elevation, gold hairline accent
   ============================================================ */
/* Large panels */
.st-key-child_card,
.st-key-photo_card,
.st-key-profile_form_card,
.st-key-profile_visual_card,
.about-equal-panel,
.ranking-board,
.section-card {
    border:1px solid var(--hair) !important;
    border-radius:var(--r-lg) !important;
    box-shadow:var(--e2) !important;
}
/* Medium cards */
.kpi-card,
.nutrient-tile,
.model-pill,
.result-card,
.info-card,
.recommend-card-rich,
.rec-hero,
.xai-intro,
.xai-stat,
.flow-node,
.st-key-grad_original,
.st-key-grad_heatmap,
.st-key-grad_overlay {
    border:1px solid var(--hair) !important;
    border-radius:var(--r-md) !important;
    box-shadow:var(--e1) !important;
}
/* Charts & tables share the language */
div[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"] {
    border:1px solid var(--hair) !important;
    border-radius:var(--r-md) !important;
    box-shadow:var(--e1) !important;
}

/* Dark cards keep navy identity, refined depth */
.st-key-action_card,
.section-card-dark {
    border:1px solid rgba(20,46,86,.55) !important;
    border-radius:var(--r-lg) !important;
    background:
        radial-gradient(circle at 88% 10%, rgba(255,255,255,.08), transparent 9rem),
        var(--navy-grad) !important;
    box-shadow:0 22px 50px rgba(9,26,52,.22) !important;
}

/* Subtle premium hover-lift on free-standing cards */
.kpi-card, .nutrient-tile, .model-pill, .result-card, .recommend-card-rich {
    transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease !important;
}
.kpi-card:hover, .nutrient-tile:hover, .model-pill:hover,
.result-card:hover, .recommend-card-rich:hover {
    transform:translateY(-2px);
    box-shadow:var(--e3) !important;
    border-color:#E7D9A6 !important;
}

/* Card headings & numeric emphasis */
.kpi-value, .result-value, .model-pill-value,
.nutrient-title, .rec-hero-title, .info-card-title,
.recommend-card-rich .food { font-family:var(--pj) !important; }
.nutrient-icon, .result-icon {
    background:linear-gradient(180deg,#FFF3C6,#FFE9A0) !important;
    color:#9B7400 !important;
    border-radius:12px !important;
}
.about-cap-icon { background:linear-gradient(180deg,#FFF3C6,#FFE9A0) !important; border-radius:12px !important; }

/* ============================================================
   CHIPS / TAGS / BADGES / PILLS
   ============================================================ */
.tag-chip, .cta-badge, .rec-rank-badge, .grad-card-badge,
.priority, .rec-legend span {
    font-family:var(--pj) !important;
    font-weight:700 !important;
}
.tag-chip {
    background:rgba(255,201,40,.12) !important;
    border:1px solid rgba(224,168,15,.32) !important;
    color:#8A6600 !important;
}

/* ============================================================
   FORM CONTROLS  —  crisp inputs, gold focus ring
   ============================================================ */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    border:1px solid var(--hair-2) !important;
    border-radius:12px !important;
    font-family:var(--body) !important;
    transition:border-color .16s ease, box-shadow .16s ease !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color:#F0C64A !important;
    box-shadow:0 0 0 3px rgba(255,201,40,.20) !important;
    outline:none !important;
}
label[data-testid="stWidgetLabel"] p { font-family:var(--pj) !important; }
[data-testid="stMetric"] {
    background:#F8FAFD !important;
    border:1px solid var(--hair) !important;
    border-radius:14px !important;
}
[data-testid="stMetricValue"] { font-family:var(--pj) !important; }

[data-testid="stFileUploader"] {
    background:linear-gradient(180deg,#FFFDF7,#FFF8E9) !important;
    border:1.5px dashed #EDBE33 !important;
    border-radius:var(--r-md) !important;
}

/* ============================================================
   BUTTONS
   ============================================================ */
.stButton > button {
    font-family:var(--pj) !important;
    border-radius:12px !important;
    transition:transform .16s ease, box-shadow .16s ease, background .16s ease !important;
}
.st-key-action_card .stButton > button {
    background:var(--gold-grad) !important;
    border:1px solid #F2B400 !important;
    color:#15213B !important;
    box-shadow:0 12px 28px rgba(255,193,34,.30) !important;
}
.st-key-action_card .stButton > button:hover:not(:disabled) {
    transform:translateY(-1px);
    box-shadow:0 16px 34px rgba(255,193,34,.40) !important;
}
.st-key-photo_card .stButton > button:hover {
    transform:translateY(-1px);
}

/* ============================================================
   TABS
   ============================================================ */
[data-baseweb="tab-list"] {
    border:1px solid var(--hair) !important;
    border-radius:14px !important;
    background:#F2F5F9 !important;
}
button[data-baseweb="tab"] { font-family:var(--pj) !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    background:#FFFFFF !important;
    box-shadow:var(--e1) !important;
}

/* ============================================================
   INSIGHT / SOFT NOTES
   ============================================================ */
.insight, .xai-explain {
    border:1px solid #EFDD98 !important;
    border-radius:var(--r-md) !important;
    background:linear-gradient(115deg,#FFFDF4,#FFF7D6) !important;
}
.rec-hero { background:linear-gradient(120deg,#FFF9E0,#FFFFFF) !important; }

/* ============================================================
   TENTANG APLIKASI  —  website "About" components
   ============================================================ */
.about-band {
    position:relative;
    overflow:hidden;
    margin:6px 0 20px;
    padding:34px 36px;
    border:1px solid var(--hair);
    border-radius:26px;
    background:
        radial-gradient(560px 260px at 92% -30%, rgba(255,201,40,.16), transparent 62%),
        linear-gradient(118deg,#FFFFFF 0%,#FFFFFF 58%,#FFFCF3 100%);
    box-shadow:var(--e2);
}
.about-band::after {
    content:"";
    position:absolute;
    right:-70px; bottom:-120px;
    width:240px; height:240px;
    border-radius:50%;
    border:24px solid rgba(255,201,40,.10);
}
.about-band-eyebrow {
    display:inline-flex; align-items:center; gap:7px;
    padding:7px 13px; border-radius:999px;
    background:rgba(255,201,40,.12);
    border:1px solid rgba(224,168,15,.34);
    color:#8A6600;
    font-family:var(--pj); font-weight:700;
    font-size:.66rem; letter-spacing:.12em; text-transform:uppercase;
}
.about-band-eyebrow::before { content:"✦"; color:#DBA300; }
.about-band-title {
    max-width:820px;
    margin-top:16px;
    font-family:var(--pj);
    font-size:2.05rem;
    line-height:1.12;
    font-weight:800;
    letter-spacing:-.04em;
    color:var(--ink-1);
}
.about-band-title span { color:#EDAE00; }
.about-band-lead {
    max-width:760px;
    margin-top:14px;
    color:var(--ink-3);
    font-size:.95rem;
    line-height:1.72;
}

.about-section-head { margin:26px 0 14px; }
.about-section-kicker {
    color:#B07E00;
    font-family:var(--pj); font-weight:800;
    font-size:.68rem; letter-spacing:.14em; text-transform:uppercase;
    margin-bottom:5px;
}
.about-section-title {
    font-family:var(--pj);
    font-size:1.3rem; font-weight:800;
    letter-spacing:-.025em; color:var(--ink-1);
}
.about-section-sub { color:var(--ink-3); font-size:.82rem; margin-top:4px; }

/* Values grid */
.about-values {
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:14px;
}
.about-value-card {
    padding:20px 20px 18px;
    border:1px solid var(--hair);
    border-radius:var(--r-md);
    background:linear-gradient(180deg,#FFFFFF,#FBFCFE);
    box-shadow:var(--e1);
    transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
.about-value-card:hover {
    transform:translateY(-2px);
    box-shadow:var(--e3);
    border-color:#E7D9A6;
}
.about-value-icon {
    width:44px; height:44px;
    display:grid; place-items:center;
    border-radius:13px;
    background:linear-gradient(180deg,#FFF3C6,#FFE59A);
    color:#9B7400; font-size:1.15rem;
    margin-bottom:13px;
}
.about-value-title {
    font-family:var(--pj); font-weight:800;
    font-size:1rem; color:var(--ink-1); margin-bottom:6px;
}
.about-value-text { color:var(--ink-3); font-size:.82rem; line-height:1.65; }

/* Tech stack strip */
.about-tech {
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:12px;
}
.about-tech-item {
    padding:16px 16px 15px;
    border:1px solid var(--hair);
    border-radius:var(--r-md);
    background:#FFFFFF;
    box-shadow:var(--e1);
}
.about-tech-label {
    color:var(--ink-3);
    font-family:var(--pj); font-weight:700;
    font-size:.66rem; letter-spacing:.05em; text-transform:uppercase;
    margin-bottom:6px;
}
.about-tech-value {
    font-family:var(--pj); font-weight:800;
    font-size:1rem; color:var(--ink-1); line-height:1.25;
}
.about-tech-note { color:var(--ink-3); font-size:.72rem; margin-top:5px; line-height:1.5; }

/* Closing CTA band (navy signature) */
.about-cta {
    position:relative;
    overflow:hidden;
    margin-top:24px;
    padding:34px 36px;
    border-radius:26px;
    border:1px solid rgba(20,46,86,.55);
    background:
        radial-gradient(circle at 90% 8%, rgba(255,255,255,.08), transparent 10rem),
        var(--navy-grad);
    box-shadow:0 24px 54px rgba(9,26,52,.24);
}
.about-cta::after {
    content:"✦";
    position:absolute; right:30px; top:22px;
    color:var(--gold); font-size:2rem; opacity:.9;
}
.about-cta-title {
    font-family:var(--pj);
    color:#FFFFFF; font-size:1.5rem; font-weight:800;
    letter-spacing:-.03em;
}
.about-cta-text {
    max-width:620px; margin-top:12px;
    color:#B9C4D5; font-size:.9rem; line-height:1.7;
}
.about-cta-actions { display:flex; flex-wrap:wrap; gap:11px; margin-top:20px; }
.about-cta-btn {
    display:inline-flex; align-items:center; gap:8px;
    padding:13px 20px; border-radius:12px;
    font-family:var(--pj); font-weight:800; font-size:.85rem;
    text-decoration:none;
    transition:transform .16s ease, box-shadow .16s ease;
}
.about-cta-btn.primary {
    background:var(--gold-grad); color:#15213B;
    border:1px solid #F2B400;
    box-shadow:0 12px 26px rgba(255,193,34,.28);
}
.about-cta-btn.ghost {
    background:rgba(255,255,255,.06); color:#EAF0F8;
    border:1px solid rgba(255,255,255,.16);
}
.about-cta-btn:hover { transform:translateY(-1px); }

/* ============================================================
   SITE FOOTER  —  the "it's a website" signal
   ============================================================ */
.site-footer {
    margin:40px 0 6px;
    padding:30px 30px 22px;
    border:1px solid var(--hair);
    border-radius:24px;
    background:
        radial-gradient(500px 220px at 96% -40%, rgba(255,201,40,.06), transparent 62%),
        linear-gradient(180deg,#FFFFFF,#FBFCFE);
    box-shadow:var(--e1);
}
.footer-top {
    display:grid;
    grid-template-columns:1.4fr 1fr 1fr;
    gap:26px;
    padding-bottom:22px;
    border-bottom:1px solid var(--hair);
}
.footer-brand-name {
    font-family:var(--pj); font-weight:800;
    font-size:1.15rem; letter-spacing:-.02em; color:var(--ink-1);
}
.footer-brand-name span { color:#EDAE00; }
.footer-brand-text {
    margin-top:9px; max-width:340px;
    color:var(--ink-3); font-size:.8rem; line-height:1.65;
}
.footer-badges { display:flex; flex-wrap:wrap; gap:7px; margin-top:14px; }
.footer-badge {
    padding:6px 11px; border-radius:999px;
    background:rgba(255,201,40,.1);
    border:1px solid rgba(224,168,15,.28);
    color:#8A6600;
    font-family:var(--pj); font-weight:700; font-size:.62rem;
}
.footer-col-title {
    font-family:var(--pj); font-weight:800;
    font-size:.7rem; letter-spacing:.1em; text-transform:uppercase;
    color:var(--ink-2); margin-bottom:12px;
}
.footer-links { display:flex; flex-direction:column; gap:9px; }
.footer-links a {
    color:var(--ink-3); font-size:.82rem; text-decoration:none;
    transition:color .15s ease, transform .15s ease;
    width:fit-content;
}
.footer-links a:hover { color:#B07E00; transform:translateX(2px); }
.footer-note { color:var(--ink-3); font-size:.78rem; line-height:1.6; }
.footer-bottom {
    display:flex; align-items:center; justify-content:space-between;
    flex-wrap:wrap; gap:10px; padding-top:18px;
}
.footer-copy { color:var(--ink-3); font-size:.74rem; }
.footer-disclaimer {
    color:#8A94A6; font-size:.7rem; max-width:560px; text-align:right; line-height:1.55;
}

/* ============================================================
   MOTION & A11Y FLOOR
   ============================================================ */
@media (prefers-reduced-motion: reduce) {
    * { transition:none !important; animation:none !important; }
    .kpi-card:hover, .nutrient-tile:hover, .model-pill:hover,
    .result-card:hover, .recommend-card-rich:hover,
    .about-value-card:hover, .tech-chip:hover { transform:none !important; }
}
a:focus-visible,
button:focus-visible,
[data-testid="stSidebar"] [role="radiogroup"] label:focus-within {
    outline:2px solid rgba(255,201,40,.65) !important;
    outline-offset:2px !important;
}

/* Responsive collapse for the new About + footer blocks */
@media (max-width:900px) {
    .about-values { grid-template-columns:1fr; }
    .about-tech { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .footer-top { grid-template-columns:1fr; gap:20px; }
    .footer-disclaimer { text-align:left; }
    .about-band-title { font-size:1.6rem; }
    .about-band, .about-cta { padding:24px 22px; }
}
@media (max-width:560px) {
    .about-tech { grid-template-columns:1fr; }
    .about-cta-actions { flex-direction:column; }
    .about-cta-btn { justify-content:center; }
}

</style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# LOADERS
# ============================================================

# ============================================================
# NOISYVIT LOADER (robust) — perbaikan pemuatan checkpoint
# ------------------------------------------------------------
# Bobot NoisyViT B/16 tersimpan di dalam key "model_state" (bersama
# metadata: class_names, num_classes, epoch, dst) dengan prefix "vit.".
# Loader ini membuka bungkus itu, mendeteksi jumlah kelas dari head,
# lalu memuat bobot dengan strict=False.
# ============================================================


class _NoisyViTWrapper(nn.Module):
    """Pembungkus ViT-B/16 agar key state_dict = vit.* (sesuai checkpoint)."""

    def __init__(self, num_classes: int, model_name: str = "vit_base_patch16_224"):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.vit(x)


def _looks_like_state_dict(d):
    if not isinstance(d, dict) or not d:
        return False
    return any(hasattr(v, "shape") for v in d.values())


def _noisyvit_extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("model_state", "state_dict", "model_state_dict", "model", "net", "weights"):
        inner = ckpt.get(key)
        if isinstance(inner, dict) and inner:
            return inner
    if _looks_like_state_dict(ckpt):
        return ckpt
    best = None
    for v in ckpt.values():
        if isinstance(v, dict) and _looks_like_state_dict(v):
            if best is None or len(v) > len(best):
                best = v
    return best if best is not None else ckpt


def _load_noisyvit_robust(path, num_classes=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device)
    state = _noisyvit_extract_state_dict(ckpt)
    state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}
    has_vit = any(k.startswith("vit.") for k in state)
    bare_vit = any(k in ("cls_token", "pos_embed") or k.startswith("blocks.") for k in state)
    if not has_vit and bare_vit:
        state = {"vit." + k: v for k, v in state.items()}
    head_w = state.get("vit.head.weight")
    detected = int(head_w.shape[0]) if head_w is not None else (int(num_classes) if num_classes else 1000)
    if num_classes is None:
        num_classes = detected
    model = _NoisyViTWrapper(num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[NoisyViT] key model belum terisi:", missing)
    if unexpected:
        print("[NoisyViT] key checkpoint tak terpakai:", unexpected)
    if not missing and not unexpected:
        print(f"[NoisyViT] OK - bobot dimuat penuh, num_classes={num_classes}")
    model.to(device)
    model.eval()
    return model


def _detect_food_num_classes(convnext_model, noisyvit_model, class_names):
    """Jumlah output klasifikasi makanan yang sebenarnya dari model (mis. 56)."""
    def out_features(m):
        head = getattr(m, "head", None)
        if head is not None and hasattr(head, "fc") and hasattr(head.fc, "out_features"):
            return int(head.fc.out_features)
        vit = getattr(m, "vit", None)
        if vit is not None and hasattr(vit, "head") and hasattr(vit.head, "out_features"):
            return int(vit.head.out_features)
        if head is not None and hasattr(head, "out_features"):
            return int(head.out_features)
        return None
    for m in (noisyvit_model, convnext_model):
        n = out_features(m)
        if n:
            return n
    return len(class_names)


@st.cache_resource(show_spinner=False)
def load_models():
    convnext_path = MODEL_DIR / "convnextv2_tiny_56class_nutrition_babyfood_best.pt"
    noisyvit_path = MODEL_DIR / "noisyvit_b_16_56class_best.pt"
    mal_path = MODEL_DIR / "malnutrition_model.joblib"

    required = [convnext_path, noisyvit_path, mal_path]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Model belum tersedia: " + ", ".join(missing)
        )

    convnext_model = load_convnext_model(
        convnext_path,
        num_classes=None,
    )
    noisyvit_model = _load_noisyvit_robust(
        noisyvit_path,
        num_classes=None,
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
    if len(class_names) < 2:
        raise ValueError(
            f"class_names.json harus berisi daftar kelas yang valid, ditemukan {len(class_names)}."
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
# LABEL TAMPILAN MAKANAN
# ============================================================

FOOD_DISPLAY_NAMES = {
    "fried_rice": "Nasi Goreng",
    "french_fries": "Kentang Goreng",
    "chicken_wings": "Sayap Ayam",
    "chicken_soto": "Soto Ayam",
    "chicken_noodle": "Mi Ayam",
    "chicken_porridge": "Bubur Ayam",
    "chocolate_cake": "Kue Cokelat",
    "cup_cakes": "Cupcake",
    "fish_and_chips": "Ikan dan Kentang Goreng",
    "cheesecake": "Cheesecake",
    "donuts": "Donat",
    "waffles": "Wafel",
    "tempeh": "Tempe",
    "bibimbap": "Bibimbap",
    "churros": "Churros",
}


def food_display_name(value):
    """
    Mengubah label internal model menjadi nama yang
    lebih ramah dan konsisten untuk tampilan dashboard.

    Label internal model tidak diubah.
    """
    if value is None:
        return "-"

    raw = str(value).strip()

    key = (
        raw.lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if key in FOOD_DISPLAY_NAMES:
        return FOOD_DISPLAY_NAMES[key]

    # Fallback agar setidaknya underscore tidak terlihat.
    return raw.replace("_", " ").title()


# ============================================================
# CHARTS
# ============================================================


def build_narrative_id(result):
    """
    Narasi hasil dalam Bahasa Indonesia dengan nama makanan tampilan
    yang konsisten dengan dashboard.
    """
    profile = result.get("child_profile", {})
    status = result.get("nutrition_status", {})
    preds = result.get("food_prediction", [])
    adequacy = result.get("nutrient_contribution", {})
    recs = result.get("recommendations", [])
    meta = result.get("recommendation_meta", {})

    age_months = int(profile.get("age_months", 0))
    age_label = age_text(age_months)

    status_name = html.escape(
        str(status.get("prediction", "-")).title()
    )

    parts = [
        (
            f"Untuk anak usia {html.escape(age_label)} dengan berat "
            f"{float(profile.get('weight_kg', 0)):g} kg, tinggi "
            f"{float(profile.get('height_cm', 0)):g} cm, dan MUAC/LILA "
            f"{float(profile.get('muac_cm', 0)):g} cm, model antropometri "
            f"memberikan hasil skrining <b>{status_name}</b>"
            + (
                f" dengan keyakinan {float(status['confidence']) * 100:.1f}%."
                if status.get("confidence") is not None
                else "."
            )
        )
    ]

    if preds:
        top = preds[0]
        display_food = html.escape(
            food_display_name(top.get("food"))
        )
        parts.append(
            f"Foto makanan paling mungkin dikenali sebagai "
            f"<b>{display_food}</b> dengan probabilitas "
            f"{float(top.get('confidence', 0)) * 100:.1f}%."
        )

    if adequacy:
        ranked = sorted(
            adequacy.items(),
            key=lambda item: float(
                item[1].get("contribution_percent", 0)
            ),
        )
        low = ranked[:3]

        low_text = ", ".join(
            (
                f"{html.escape(DISPLAY_NAMES.get(key, key))} "
                f"({float(info.get('contribution_percent', 0)):.1f}%)"
            )
            for key, info in low
        )

        parts.append(
            "Berdasarkan <b>nilai nutrisi referensi makanan</b>, kontribusi "
            f"terhadap AKG yang paling rendah terlihat pada <b>{low_text}</b>. "
            "Nilai ini menunjukkan kontribusi makanan yang dianalisis terhadap "
            "acuan kelompok umur, bukan komposisi relatif makanan dan bukan bukti "
            "bahwa anak mengalami kekurangan nutrisi."
        )

    if recs:
        names = ", ".join(
            html.escape(str(rec.get("food", "-")))
            for rec in recs[:3]
        )
        group = html.escape(
            str(meta.get("age_group", "-"))
        )
        parts.append(
            f"Sebagai kandidat pelengkap, sistem menempatkan <b>{names}</b> "
            f"pada peringkat teratas menggunakan pemeringkatan berbasis gap "
            f"nutrisi dan kesesuaian kelompok umur {group}."
        )

    parts.append(
        "Hasil ini digunakan untuk skrining dan edukasi. Satu foto makanan "
        "tidak mewakili seluruh asupan harian dan tidak menggantikan "
        "pemeriksaan tenaga kesehatan."
    )

    return "<br><br>".join(parts)

def build_meal_plan(detected_food_display, recommendations, priorities, meal_time="Makan Siang"):
    """Rencana menu harian sederhana (offline, tanpa API)."""
    slots = ["Sarapan", "Makan Siang", "Camilan Sore", "Makan Malam"]
    pool = [str(r.get("food", "-")) for r in (recommendations or []) if r.get("food")]
    reasons = {
        str(r.get("food", "")): str(r.get("reason", ""))
        for r in (recommendations or [])
    }

    plan = []
    idx = 0
    for slot in slots:
        if slot == "Makan Siang" and detected_food_display and detected_food_display != "-":
            food = detected_food_display
            reason = "Menu yang baru saja dianalisis."
        elif idx < len(pool):
            food = pool[idx]
            reason = reasons.get(food, "Membantu melengkapi gap nutrisi prioritas.")
            idx += 1
        else:
            food = "Menu bergizi seimbang"
            reason = "Lengkapi dengan sumber protein, sayur, dan buah."
        plan.append({"waktu": slot, "menu": food, "alasan": reason})
    return plan


def generate_ai_pairings(profile, status_label, priorities, detected_food, recommendations):
    """
    Rekomendasi makanan/minuman PENDAMPING yang cocok dipadukan dengan `detected_food`
    untuk melengkapi nutrisi. Return list [{"nama","jenis","alasan"}] atau None (fallback offline).
    """
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
    except Exception:
        return None

    age_months = int(profile.get("age_months", 0))
    prio_text = ", ".join(priorities) if priorities else "gizi seimbang"
    rec_text = ", ".join(str(r.get("food", "")) for r in (recommendations or [])[:6]) or "-"

    if age_months < 24:
        age_rule = ("anak di bawah 2 tahun: tekstur lunak, tanpa/rendah garam & gula, tanpa madu, "
                    "hindari gorengan dan makanan mudah tersedak.")
    else:
        age_rule = "anak balita: porsi anak, batasi garam/gula/gorengan berlebihan."

    prompt = f"""Kamu ahli gizi anak. Anak (usia {age_months} bulan) sedang makan "{detected_food}".
Sarankan 3 makanan/minuman PENDAMPING yang cocok DISANDINGKAN dengan "{detected_food}" untuk
melengkapi nutrisi yang masih kurang: {prio_text}.

Aturan (WAJIB masuk akal dan realistis untuk sekali makan):
- Pendamping harus WAJAR dan LAZIM dimakan bersama "{detected_food}" dalam satu porsi makan anak.
- Perhatikan bahwa "{detected_food}" sudah menyumbang sebagian gizi. JANGAN menambah sumber yang
  sudah berlebihan. Contoh: jika "{detected_food}" sudah tinggi karbohidrat, JANGAN menyarankan
  sumber karbohidrat lain (nasi, mie, jagung, kentang) sebagai pendamping.
- Utamakan pelengkap ringan yang menutup kekurangan: lauk protein porsi kecil, sayur, buah potong,
  atau segelas minuman (susu/air). Ingat anak usia {age_months} bulan porsinya kecil dan cepat kenyang.
- Hindari menumpuk makanan berat; total "{detected_food}" + pendamping harus realistis untuk sekali makan.
- Sesuai usia: {age_rule}. Hindari gorengan berminyak dan bahan berisiko tersedak untuk balita.
- JANGAN menyarankan "{detected_food}" itu sendiri atau menu berat yang mirip.
- Setiap saran: nama, jenis ("makanan" atau "minuman"), dan alasan singkat (nutrisi apa yang dilengkapi).
- Boleh terinspirasi kandidat sistem: {rec_text}, tetapi utamakan yang benar-benar cocok & realistis dipadukan dengan "{detected_food}".

Selain 3 saran, tulis juga satu kalimat "ringkasan" ramah untuk orang tua, contoh gaya:
"{detected_food} paling baik dipadukan dengan A dan B, terutama untuk anak usia {age_months} bulan, karena ...".

Jawab HANYA JSON valid:
{{"pairings":[{{"nama":"...","jenis":"makanan","alasan":"..."}}],"ringkasan":"..."}}"""

    for model_name in ("gemini-flash-latest", "gemini-flash-lite-latest"):
        for attempt in range(2):
            try:
                client = genai.Client(api_key=api_key)
                resp = client.models.generate_content(model=model_name, contents=prompt)
                text = (resp.text or "").strip().replace("```json", "").replace("```", "").strip()
                data = json.loads(text)
                if data.get("pairings"):
                    return data
            except Exception as e:
                msg = repr(e)
                if any(k in msg for k in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")):
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    break
    return None


def generate_ai_meal_plan(profile, status_label, priorities, detected_food, recommendations, meal_time):
    """
    Rencana menu harian via Gemini (opsional).
    Makanan yang difoto DIKUNCI pada slot `meal_time`; AI menyusun sisa hari
    agar total asupan sehari saling melengkapi & sesuai usia anak.
    Return dict {"meals":[...], "catatan":...} atau None bila gagal -> fallback offline.
    """
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None

    age_months = int(profile.get("age_months", 0))
    prio_text = ", ".join(priorities) if priorities else "gizi seimbang"
    rec_text = ", ".join(
        str(r.get("food", "")) for r in (recommendations or [])[:5]
    ) or "-"

    all_slots = ["Sarapan", "Makan Siang", "Camilan Sore", "Makan Malam"]
    idx_now = all_slots.index(meal_time) if meal_time in all_slots else 0

    # Slot berikutnya dalam hari yang sama.
    after_slots = all_slots[idx_now + 1:]

    if after_slots:
        # Masih ada waktu makan tersisa hari ini + tetap sarankan menu besok pagi.
        plan_slots = after_slots + ["Sarapan (besok)"]
        horizon_text = (
            f'Lengkapi sisa hari ini ({", ".join(after_slots)}) '
            f'lalu tambahkan "Sarapan (besok)" agar transisi ke hari berikutnya tetap bergizi.'
        )
    else:
        # meal_time = Makan Malam -> tidak ada sisa hari ini, rekomendasi untuk BESOK.
        plan_slots = ["Sarapan (besok)", "Makan Siang (besok)", "Camilan Sore (besok)", "Makan Malam (besok)"]
        horizon_text = (
            "Anak sudah menyelesaikan makan malam hari ini. Susun menu untuk BESOK "
            "(sarapan sampai makan malam) yang melengkapi nutrien prioritas."
        )

    other_text = ", ".join(plan_slots)

    if age_months < 6:
        stage = (
            "Bayi di bawah 6 bulan: HANYA ASI/susu formula. JANGAN memberi makanan padat "
            "apa pun. Untuk tiap waktu makan, tulis pemberian ASI/susu, bukan menu makanan."
        )
    elif age_months < 9:
        stage = (
            "Bayi 6-8 bulan (MPASI awal): makanan lumat/puree sangat halus (bubur saring, "
            "puree buah/sayur, hati ayam lumat). TANPA garam, gula, madu, gorengan, makanan "
            "keras, kacang utuh, atau bumbu tajam."
        )
    elif age_months < 12:
        stage = (
            "Bayi 9-11 bulan (MPASI lanjutan): makanan lembik dicincang halus / finger food "
            "lunak (nasi tim lembek, sayur kukus lunak, ikan/ayam cincang tanpa duri). TANPA "
            "garam berlebih, gula, madu, gorengan keras, atau makanan mudah tersedak."
        )
    elif age_months < 24:
        stage = (
            "Anak 12-23 bulan: makanan keluarga bertekstur lunak, potongan kecil, porsi kecil, "
            "rendah garam & gula. Hindari gorengan berminyak, makanan pedas, dan makanan keras "
            "berisiko tersedak."
        )
    else:
        stage = (
            "Anak di atas 2 tahun: makanan keluarga bergizi seimbang porsi anak, tetap batasi "
            "garam, gula, dan gorengan berlebihan."
        )

    prompt = f"""Kamu ahli gizi anak. Susun rencana menu harian (Bahasa Indonesia) yang AMAN
dan SESUAI USIA.

Data anak:
- Usia: {age_months} bulan
- Aturan tahap makan yang WAJIB dipatuhi: {stage}
- Status skrining gizi: {status_label}
- Nutrien prioritas yang masih kurang (harus dilengkapi sepanjang hari): {prio_text}

Konteks waktu makan (SANGAT PENTING):
- Anak SUDAH makan "{detected_food}" pada waktu "{meal_time}". Slot "{meal_time}" TIDAK BOLEH
  diubah - isi persis dengan "{detected_food}".
- {horizon_text}
- Tugasmu menyusun: {other_text}. Susun agar TOTAL asupan saling melengkapi dan memenuhi
  nutrien prioritas (protein, vitamin, energi, dll), tanpa menumpuk nutrien yang sudah cukup.
- Kandidat makanan pendamping dari sistem (boleh dipakai bila cocok usia): {rec_text}

Aturan:
- Menu WAJIB mengikuti aturan tahap makan sesuai usia. Ini prioritas nomor satu.
- Untuk menu "(besok)": susun sebagai rencana makan sehari yang SEHAT, LENGKAP, dan BERVARIASI —
  bukan sekadar menambal gap. Tiap waktu makan sebaiknya punya sumber karbohidrat, protein
  (hewani/nabati bergantian), sayur, dan sesekali buah, sehingga terasa seperti menu harian
  yang menarik dan bergizi seimbang untuk anak.
- LARANGAN KERAS: menu untuk SEMUA waktu makan selain "{meal_time}" HARUS BERBEDA dari "{detected_food}".
  JANGAN pernah menuliskan "{detected_food}" (atau variasi namanya) di slot lain, termasuk di
  "Makan Malam (besok)". Setiap waktu makan harus memakai menu dan bahan utama yang berbeda-beda.
- Utamakan cara masak yang sehat (kukus, rebus, tim, tumis ringan) dan bahan segar; batasi
  gorengan berminyak, makanan tinggi gula/garam.
- Setiap waktu makan: nama menu ringkas + alasan singkat (kaitkan nutrien yang dilengkapi & usia).
- Gunakan bahan umum dan terjangkau di Indonesia.

Keluarkan waktu makan berikut secara berurutan: "{meal_time}" (isi "{detected_food}"),
lalu {other_text}. Gunakan label "waktu" persis seperti nama-nama itu (termasuk "(besok)" bila ada).

Tambahkan juga "ringkasan_gizi": penilaian KUALITATIF jujur (BUKAN angka gram/kalori) tentang
apakah rangkaian menu ini sudah memenuhi gizi sehari - sebutkan nutrien apa yang sudah tercakup
(mis. protein hewani & nabati, sumber kalsium, sayur, buah) dan nutrien apa yang MASIH perlu
diperhatikan. Tulis 2-3 kalimat, ramah untuk orang tua. JANGAN mengarang angka.

Jawab HANYA JSON valid:
{{"meals":[{{"waktu":"Sarapan","menu":"...","alasan":"..."}}],"catatan":"satu kalimat","ringkasan_gizi":"2-3 kalimat"}}"""

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print("=== GEMINI GAGAL (client) ===")
        print(repr(e))
        return None

    models_to_try = ["gemini-flash-latest", "gemini-flash-lite-latest"]
    last_error = None
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                text = (resp.text or "").strip()
                text = text.replace("```json", "").replace("```", "").strip()
                data = json.loads(text)
                meals = data.get("meals", [])

                # Kunci HANYA slot pertama (slot foto hari ini); slot lain milik AI.
                if meals:
                    first_w = str(meals[0].get("waktu", "")).lower()
                    if "besok" not in first_w:
                        meals[0]["menu"] = detected_food

                # Safety net: cegah slot lain mengulang makanan yang sama dengan foto.
                df_low = detected_food.strip().lower()
                alt_pool = [str(r.get("food", "")) for r in (recommendations or []) if r.get("food")]
                alt_used = set()
                for i, mm in enumerate(meals):
                    if i == 0:
                        continue
                    menu_low = str(mm.get("menu", "")).strip().lower()
                    if not menu_low or not df_low:
                        continue
                    if menu_low == df_low or df_low in menu_low or menu_low in df_low:
                        replacement = None
                        for cand in alt_pool:
                            cl = cand.strip().lower()
                            if cl and cl != df_low and cl not in alt_used:
                                replacement = cand
                                break
                        mm["menu"] = replacement or "Menu bergizi seimbang (variasi lain)"
                        mm["alasan"] = "Divariasikan agar tidak mengulang menu yang sama dengan hari ini."
                        alt_used.add(mm["menu"].strip().lower())
                return data
            except Exception as e:
                last_error = e
                msg = repr(e)
                if any(k in msg for k in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")):
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    break

    print("=== GEMINI GAGAL (semua percobaan) ===")
    print(repr(last_error))
    print("====================")
    return None

def top3_chart(predictions):
    df = pd.DataFrame(predictions).copy()

    df["pct"] = df["confidence"] * 100

    df["food_display"] = df["food"].apply(
        food_display_name
    )

    df = df.sort_values("pct")

    fig = go.Figure(
        go.Bar(
            x=df["pct"],
            y=df["food_display"],
            orientation="h",
            marker_color="#FFC51D",
            text=[
                f"{x:.2f}%"
                for x in df["pct"]
            ],
            textposition="outside",
            cliponaxis=False,
        )
    )

    fig.update_layout(
        title=dict(
            text="Top-3 Prediksi Makanan",
            x=.03,
            font=dict(
                size=15,
                color="#272727",
            ),
        ),
        height=355,
        margin=dict(
            l=105,
            r=70,
            t=55,
            b=45,
        ),
        xaxis=dict(
            range=[0, 105],
            title="Probabilitas Prediksi (%)",
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

def nutrient_radar(result):
    """
    Radar KANDUNGAN NUTRISI (bukan %AKG).
    Nilai dinormalisasi 0-100 relatif terhadap nutrien tertinggi agar bentuk
    radar tetap terbaca dan garis kuning tegas. Hover tetap menampilkan angka
    dan satuan asli supaya tidak menyesatkan.
    """
    nutrition = result.get("food_nutrition", {})
    nutrients = nutrition.get("nutrients", {}) if nutrition.get("status") == "ok" else {}

    if not nutrients:
        return go.Figure()

    pretty = {
        "calories": "Energi",
        "protein": "Protein",
        "carbohydrate": "Karbohidrat",
        "fat": "Lemak",
        "iron": "Zat Besi",
        "calcium": "Kalsium",
        "vitamin_a": "Vit A",
        "vitamin_c": "Vit C",
    }

    labels = []
    raw_values = []
    units = []

    for key, value in nutrients.items():
        try:
            amount = float(value)
        except (TypeError, ValueError):
            continue
        labels.append(pretty.get(key, DISPLAY_NAMES.get(key, key)))
        raw_values.append(amount)
        units.append(NUTRIENT_UNITS.get(key, ""))

    if not labels:
        return go.Figure()

    max_val = max(raw_values) or 1.0
    norm_values = [v / max_val * 100 for v in raw_values]

    # Tutup poligon (sambungkan titik terakhir ke titik pertama).
    labels_c = labels + [labels[0]]
    norm_c = norm_values + [norm_values[0]]
    raw_c = raw_values + [raw_values[0]]
    units_c = units + [units[0]]

    customdata = [[r, u] for r, u in zip(raw_c, units_c)]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=norm_c,
            theta=labels_c,
            fill="toself",
            name="Kandungan nutrisi",
            line=dict(color="#F5B800", width=3.4),
            fillcolor="rgba(255,201,40,.28)",
            marker=dict(size=6, color="#F5B800"),
            customdata=customdata,
            hovertemplate="<b>%{theta}</b><br>%{customdata[0]:.1f} %{customdata[1]}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Kandungan Nutrisi Makanan",
            x=.03,
            font=dict(size=15, color="#272727"),
        ),
        height=355,
        margin=dict(l=65, r=65, t=62, b=52),
        polar=dict(
            domain=dict(x=[.14, .86], y=[.14, .90]),
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                tickfont=dict(size=8, color="#B4ADA1"),
                gridcolor="#EAE4D5",
                linecolor="#EAE4D5",
                angle=90,
                showticklabels=False,
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color="#6B655D"),
                gridcolor="#EEE7D8",
                linecolor="#EEE7D8",
            ),
        ),
        showlegend=False,
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

def nutrient_best(adequacy, limit=3):
    """
    Nutrien dengan kontribusi AKG tertinggi.
    Berbeda dengan nutrient_priorities() yang mencari kontribusi terendah.
    """
    if not adequacy:
        return []

    ranked = sorted(
        adequacy.items(),
        key=lambda x: float(x[1].get("contribution_percent", 0)),
        reverse=True,
    )

    results = []

    for key, info in ranked[:limit]:
        contribution = float(
            info.get("contribution_percent", 0)
        )

        results.append(
            {
                "key": key,
                "name": DISPLAY_NAMES.get(key, key),
                "contribution": contribution,
                "gap": max(0.0, 100.0 - contribution),
            }
        )

    return results


def nutrient_largest_gap(adequacy, limit=3):
    """
    Nutrien dengan gap terbesar.
    """
    if not adequacy:
        return []

    ranked = sorted(
        adequacy.items(),
        key=lambda x: float(x[1].get("contribution_percent", 0)),
    )

    results = []

    for key, info in ranked[:limit]:
        contribution = float(
            info.get("contribution_percent", 0)
        )

        results.append(
            {
                "key": key,
                "name": DISPLAY_NAMES.get(key, key),
                "contribution": contribution,
                "gap": max(0.0, 100.0 - contribution),
            }
        )

    return results


def complete_nutrition_dataframe(result, adequacy):
    """
    Menampilkan SEMUA nutrien yang ditemukan pada makanan.

    Jika nutrien mempunyai acuan AKG -> tampilkan kontribusi + gap.
    Jika belum mempunyai acuan AKG -> tetap ditampilkan,
    tetapi diberi status belum dianalisis terhadap AKG.
    """
    nutrition = result.get("food_nutrition", {})

    if nutrition.get("status") != "ok":
        return pd.DataFrame()

    food_nutrients = nutrition.get("nutrients", {})

    rows = []

    for key, raw_value in food_nutrients.items():

        try:
            amount = round(float(raw_value), 2)
        except (TypeError, ValueError):
            amount = raw_value

        info = adequacy.get(key)

        if info is not None:
            contribution = float(
                info.get("contribution_percent", 0)
            )

            gap = info.get("gap_percent")

            if gap is None:
                gap = max(0.0, 100.0 - contribution)

            contribution_display = round(contribution, 1)
            gap_display = round(float(gap), 1)
            analysis_status = "✓ Dianalisis"
        else:
            contribution_display = None
            gap_display = None
            analysis_status = "Belum ada acuan AKG"

        rows.append(
            {
                "Nutrisi": DISPLAY_NAMES.get(key, key),
                "Jumlah": amount,
                "Satuan": NUTRIENT_UNITS.get(key, ""),
                "Kontribusi AKG (%)": contribution_display,
                "Gap (%)": gap_display,
                "Status Analisis": analysis_status,
            }
        )

    return pd.DataFrame(rows)
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
    f"""<div class="hero-shell"><div class="hero-inner"><div class="hero-copy"><div class="hero-eyebrow">AI-POWERED NUTRITION SCREENING</div><div class="title">Nutri<span>Vision</span> AI</div><div class="subtitle">Integrasi data antropometri dan ensemble ConvNeXt V2 Tiny + NoisyViT B/16 untuk screening status gizi, pengenalan makanan, analisis nutrisi, dan rekomendasi yang lebih terarah.</div></div><div class="hero-logo-wrap"><img class="hero-logo-img" src="{LOGO_ICON_URI}" alt="NutriVision AI"></div><div class="hero-meta"><div class="date-pill">▣ {now}</div><div class="tech-row"><span class="tech-chip">◈ Ensemble Deep Learning</span><span class="tech-chip">♜ 56 Food Classes</span><span class="tech-chip">◇ Soft Voting</span></div></div></div></div>""",
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

    input_col, image_col, action_col = st.columns([1.12, .95, .96], gap="small")

    with input_col:
        with st.container(key="child_card"):
            st.markdown('<div class="panel-title">♙ &nbsp;Data Anak</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-sub">Input langsung pada dashboard agar alur analisis lebih ringkas.</div>', unsafe_allow_html=True)
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                age_months = st.number_input("Umur (bulan)", min_value=6, max_value=83, value=int(profile["age_months"]), step=1)
            with r1c2:
                weight_kg = st.number_input("Berat badan (kg)", min_value=3.0, max_value=40.0, value=float(profile["weight_kg"]), step=.1)
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                height_cm = st.number_input("Tinggi badan (cm)", min_value=45.0, max_value=140.0, value=float(profile["height_cm"]), step=.5)
            with r2c2:
                muac_cm = st.number_input("MUAC / LILA (cm)", min_value=5.0, max_value=30.0, value=float(profile["muac_cm"]), step=.1)
            bmi_preview = weight_kg / ((height_cm / 100) ** 2)
            p1, p2, p3 = st.columns(3)
            p1.metric("Usia", age_text(age_months))
            p2.metric("BMI", f"{bmi_preview:.2f}")
            p3.metric("MUAC", f"{muac_cm:.1f} cm")

    with image_col:
        with st.container(key="photo_card"):
            st.markdown('<div class="panel-title">▣ &nbsp;Foto Makanan</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-sub">Gunakan foto dengan satu menu utama yang jelas.</div>', unsafe_allow_html=True)
            image = None
            if st.session_state.food_image_bytes is None:
                source = st.radio("Sumber gambar", ["📁 Upload", "📷 Kamera"], horizontal=True, label_visibility="collapsed", key="food_source_mode")
                if source == "📁 Upload":
                    uploaded = st.file_uploader("Upload foto", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed", key="food_upload_widget")
                    if uploaded is not None:
                        st.session_state.food_image_bytes = uploaded.getvalue()
                        st.session_state.food_image_name = uploaded.name
                        st.rerun()
                else:
                    shot = st.camera_input("Ambil foto makanan", label_visibility="collapsed", key="food_camera_widget")
                    if shot is not None:
                        st.session_state.food_image_bytes = shot.getvalue()
                        st.session_state.food_image_name = "kamera.jpg"
                        st.rerun()
            else:
                try:
                    image = Image.open(io.BytesIO(st.session_state.food_image_bytes)).convert("RGB")
                    preview = fixed_preview(image)
                    st.image(preview, use_container_width=True, caption="Preview makanan")
                    if st.button("↻  Ganti foto makanan", key="change_food_photo", use_container_width=True):
                        st.session_state.food_image_bytes = None
                        st.session_state.food_image_name = None
                        st.session_state.pop("food_upload_widget", None)
                        st.session_state.pop("food_camera_widget", None)
                        st.rerun()
                except Exception:
                    st.session_state.food_image_bytes = None
                    st.session_state.food_image_name = None
                    st.session_state.pop("food_upload_widget", None)
                    st.session_state.pop("food_camera_widget", None)
                    st.rerun()
            st.markdown('<div class="photo-tip"><span class="photo-tip-icon">ⓘ</span><span>Pastikan foto diambil dari atas (top-down) dengan pencahayaan yang baik.</span></div>', unsafe_allow_html=True)

    with action_col:
        with st.container(key="action_card"):
            st.markdown("""<div class="cta-box"><div class="cta-badge">AI ANALYSIS</div><div class="cta-title">Analisis dalam Satu Alur</div><div class="cta-sub">ConvNeXt V2 Tiny dan NoisyViT B/16 digabung dengan weighted soft voting, lalu diteruskan ke nutrition knowledge base dan recommendation engine.</div></div>""", unsafe_allow_html=True)
            analyze = st.button("🚀  Jalankan Analisis", type="primary", use_container_width=True, disabled=image is None)
            st.markdown('<div class="action-note"><span class="note-icon">◇</span><span>Hasil analisis bersifat informatif dan bukan pengganti konsultasi profesional kesehatan.</span></div>', unsafe_allow_html=True)

    # ============================================================
    # INFERENCE — hanya jalan saat tombol diklik, simpan ke session_state
    # ============================================================
    if analyze:
        st.session_state.child_profile.update({
            "age_months": int(age_months),
            "weight_kg": float(weight_kg),
            "height_cm": float(height_cm),
            "muac_cm": float(muac_cm),
        })
        try:
            with st.spinner("NutriVision sedang menganalisis..."):
                convnext_model, noisyvit_model, malnutrition_artifact = load_models()
                class_names, nutricheck_df, indonesia_df, requirements_df = load_data()
                nutricheck_kb, nutricheck_food_col, nutricheck_nutrient_cols = build_nutrition_kb(nutricheck_df)
                indo_kb, indo_food_col, indo_nutrient_cols = build_nutrition_kb(indonesia_df)
                indo_rec_df, indo_rec_cols = build_recommendation_table(indo_kb, indo_food_col, indo_nutrient_cols)
                _inference.FOOD_NUM_CLASSES = _detect_food_num_classes(
                    convnext_model, noisyvit_model, class_names
                )
                result = analyze_child_and_food(
                    age_months=age_months, weight_kg=weight_kg, height_cm=height_cm, muac_cm=muac_cm,
                    image=image, requirements=requirements_df,
                    convnext_model=convnext_model, noisyvit_model=noisyvit_model,
                    class_names=class_names, ensemble_weights=DEFAULT_ENSEMBLE_WEIGHTS,
                    malnutrition_artifact=malnutrition_artifact,
                    nutricheck_kb=nutricheck_kb, nutricheck_food_col=nutricheck_food_col,
                    nutricheck_nutrient_cols=nutricheck_nutrient_cols,
                    indo_rec_df=indo_rec_df, indo_food_col=indo_food_col, indo_rec_cols=indo_rec_cols,
                )
        except Exception as exc:
            st.error("Inference gagal.")
            st.code(str(exc))
            st.stop()
        st.session_state.last_result = result
        st.session_state.last_image = image
        st.session_state.last_convnext = convnext_model
        st.session_state.last_class_names = class_names
        st.session_state.last_analysis_id = datetime.now().isoformat()
        st.session_state.history_saved_id = None

    # ============================================================
    # RENDER — selalu jalan selama ada hasil terakhir (tahan rerun)
    # ============================================================
    if st.session_state.get("last_result") is not None:
        result = st.session_state.last_result
        image = st.session_state.get("last_image")
        convnext_model = st.session_state.get("last_convnext")
        class_names = st.session_state.get("last_class_names")

        status = result["nutrition_status"]
        preds = result["food_prediction"]
        top_food = preds[0]
        top_food_display = food_display_name(top_food["food"])
        certainty = confidence_label(preds)
        bmi = status.get("bmi")
        adequacy = result.get("nutrient_contribution", {})
        recommendations = result.get("recommendations", [])
        recommendation_meta = result.get("recommendation_meta", {})

        pipeline_warning = result.get("warning")
        if pipeline_warning:
            st.warning(pipeline_warning)

        status_conf_raw = status.get("confidence")
        status_conf = float(status_conf_raw) * 100 if status_conf_raw is not None else 0.0
        food_conf = float(top_food.get("confidence", 0)) * 100
        nutrient_score = nutrient_mean(adequacy)

        # Catat riwayat HANYA sekali per analisis (bukan tiap rerun).
        if st.session_state.get("history_saved_id") != st.session_state.get("last_analysis_id"):
            cp = st.session_state.child_profile
            st.session_state.history.insert(0, {
                "waktu": datetime.now().strftime("%d-%m-%Y %H:%M"),
                "umur_bulan": int(cp.get("age_months", 0)),
                "berat_kg": float(cp.get("weight_kg", 0)),
                "tinggi_cm": float(cp.get("height_cm", 0)),
                "muac_cm": float(cp.get("muac_cm", 0)),
                "status_gizi": str(status["prediction"]),
                "makanan": str(top_food["food"]),
                "confidence": round(food_conf, 2),
            })
            st.session_state.history_saved_id = st.session_state.get("last_analysis_id")

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("""
            <div class="section-kicker">Hasil Analisis AI</div>
            <div class="section-title">Ringkasan Hasil Analisis</div>
            <div class="section-sub">Hasil berikut menggabungkan screening antropometri, klasifikasi makanan, dan analisis kontribusi nutrisi.</div>
            """, unsafe_allow_html=True)

        status_css = "green" if str(status["prediction"]).lower() == "normal" else "yellow"
        result_cards_html = (
            f'<div class="results-grid">'
            f'<div class="result-card"><div class="result-icon">🛡️</div><div class="result-label">Status Gizi</div><div class="result-value {status_css}">{str(status["prediction"]).upper()}</div><div class="result-sub">Keyakinan model {status_conf:.1f}%</div></div>'
            f'<div class="result-card"><div class="result-icon">🍴</div><div class="result-label">Makanan Terprediksi</div><div class="result-value yellow">{top_food_display}</div><div class="result-sub">Kelas makanan</div></div>'
            f'<div class="result-card"><div class="result-icon">🎯</div><div class="result-label">Keyakinan Prediksi Makanan</div><div class="ring" style="--v:{food_conf:.1f}"><span>{food_conf:.1f}%</span></div><div class="result-sub">{certainty}</div></div>'
            f'<div class="result-card"><div class="result-icon">⚕</div><div class="result-label">BMI</div><div class="result-value">{bmi:.2f}</div><div class="result-sub">Fitur model • bukan diagnosis</div></div>'
            f'<div class="result-card"><div class="result-icon">❤️</div><div class="result-label">Rata-rata Kontribusi Nutrisi</div><div class="ring" style="--v:{nutrient_score:.1f}"><span>{nutrient_score:.0f}%</span></div><div class="result-sub">Ringkasan makanan</div></div>'
            f'</div>'
        )
        st.markdown(result_cards_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        tab1, tab2, tab4, tab5, tab6 = st.tabs(
            ["Ringkasan", "Nutrisi & Gap", "Grad-CAM", "Detail Model", "Menu Harian"]
        )

        with tab1:
            a, b, c = st.columns([1.02, 1.08, 1.22], gap="large")
            with a:
                st.plotly_chart(nutrient_radar(result), use_container_width=True, config={"displayModeBar": False})
            with b:
                st.plotly_chart(top3_chart(preds), use_container_width=True, config={"displayModeBar": False})
            with c:
                st.markdown("#### 🔎 Rekomendasi Terbaik")
                st.caption(f"Pendamping yang cocok dipadukan dengan {top_food_display}")

                pair_key = f"{st.session_state.get('last_analysis_id')}|{top_food_display}"
                if st.session_state.get("pairings_key") != pair_key:
                    with st.spinner("Menyusun rekomendasi pendamping..."):
                        st.session_state.pairings = generate_ai_pairings(
                            st.session_state.child_profile,
                            str(status["prediction"]),
                            nutrient_priorities(adequacy),
                            top_food_display,
                            recommendations,
                        )
                    st.session_state.pairings_key = pair_key
                pairing_data = st.session_state.get("pairings")
                pairings = (pairing_data or {}).get("pairings") if isinstance(pairing_data, dict) else pairing_data
                pairing_summary = (pairing_data or {}).get("ringkasan") if isinstance(pairing_data, dict) else None

                if pairings:
                    icon_map = {"minuman": "🥤", "makanan": "🍽️"}
                    for pr in pairings[:3]:
                        nama = html.escape(str(pr.get("nama", "-")))
                        jenis = str(pr.get("jenis", "makanan")).strip().lower()
                        alasan = html.escape(str(pr.get("alasan", "")))
                        icon = icon_map.get(jenis, "🍽️")
                        st.markdown(f"""<div class="rec-card"><div class="rec-rank">{icon}</div><div><div class="rec-name">{nama}</div><div class="rec-reason">{alasan}<br><span style="font-size:.64rem;color:#8A95A8;">pendamping {html.escape(jenis)} • cocok dengan {html.escape(top_food_display)}</span></div></div></div>""", unsafe_allow_html=True)
                    st.caption("Disusun oleh Tim Sang Surya 1 — Universitas Muhammadiyah Malang (UMM)")
                elif recommendations:
                    for i, rec in enumerate(recommendations[:3], 1):
                        rec_category = rec.get("category_label", "menu")
                        st.markdown(f"""<div class="rec-card"><div class="rec-rank">{i}</div><div><div class="rec-name">{rec["food"]}</div><div class="rec-reason">{rec.get("reason","")}<br><span style="font-size:.64rem;color:#8A95A8;">{rec_category} • pendamping</span></div></div><div class="rec-score">{rec["score"]:.0f}/100</div></div>""", unsafe_allow_html=True)
                    st.caption("⚙️ Mode offline — aktifkan Gemini untuk saran pendamping spesifik")
                else:
                    st.info(result.get("recommendation_note") or "Belum ada rekomendasi yang memenuhi kriteria.")
            priority_html = "".join(f'<span class="priority">✦ {x}</span>' for x in nutrient_priorities(adequacy))
            # Isi insight = kalimat versi orang tua (pendamping). Narasi panjang lama dihapus.
            if pairing_summary:
                insight_body = html.escape(str(pairing_summary))
            else:
                _prio_join = ", ".join(nutrient_priorities(adequacy)) or "gizi seimbang"
                insight_body = (
                    f"{html.escape(top_food_display)} sebaiknya dilengkapi makanan atau minuman "
                    f"pendamping untuk menutup kekurangan {html.escape(_prio_join)}, dengan porsi "
                    f"wajar sesuai usia anak. Aktifkan mode AI (Gemini) untuk saran pendamping yang lebih spesifik."
                )
            st.markdown(f"""<div class="insight"><b>⭐ Insight & Rekomendasi untuk Anak</b><br><br>{insight_body}<br><br>{priority_html}</div>""", unsafe_allow_html=True)

        with tab2:
            food_nutrients = result.get("food_nutrition", {}).get("nutrients", {})
            best_nutrients = nutrient_best(adequacy, limit=3)
            biggest_gaps = nutrient_largest_gap(adequacy, limit=3)
            analyzed_count = sum(1 for key in food_nutrients if key in adequacy)
            total_nutrients = len(food_nutrients)
            best_name, best_pct = ("-", 0.0)
            if best_nutrients:
                best_name = best_nutrients[0]["name"]; best_pct = best_nutrients[0]["contribution"]
            gap_name, gap_pct = ("-", 0.0)
            if biggest_gaps:
                gap_name = biggest_gaps[0]["name"]; gap_pct = biggest_gaps[0]["gap"]
            st.markdown("""<div class="section-kicker">Analisis Nutrisi</div><div class="section-title">Analisis Profil Nutrisi</div><div class="section-sub">Ringkasan nutrien makanan berdasarkan kandungan nutrisi dan kontribusinya terhadap acuan AKG.</div>""", unsafe_allow_html=True)
            n1, n2, n3 = st.columns(3)
            with n1:
                st.metric("Kontribusi Terbaik", best_name)
                st.caption(f"{best_pct:.1f}% dari acuan AKG" if best_nutrients else "Belum tersedia")
            with n2:
                st.metric("Gap Terbesar", gap_name)
                st.caption(f"{gap_pct:.1f}% masih belum terpenuhi" if biggest_gaps else "Belum tersedia")
            with n3:
                st.metric("Nutrien Dianalisis", f"{analyzed_count}/{total_nutrients}")
                st.caption("Nutrien dengan acuan AKG tersedia")
            if not adequacy:
                st.info("Profil nutrisi belum dapat dihitung untuk kelas makanan ini. Cek hasil pemetaan nama makanan ke knowledge base nutrisi.")
            x, y = st.columns(2, gap="large")
            with x:
                st.plotly_chart(contribution_chart(adequacy), use_container_width=True, config={"displayModeBar": False})
            with y:
                st.plotly_chart(gap_chart(adequacy), use_container_width=True, config={"displayModeBar": False})
            if result.get("food_nutrition", {}).get("status") == "ok":
                st.markdown("#### 📊 Profil Nutrisi Lengkap")
                nutrition_df = complete_nutrition_dataframe(result, adequacy)
                st.dataframe(nutrition_df, hide_index=True, use_container_width=True)
                missing_analysis = [key for key in food_nutrients if key not in adequacy]
                if missing_analysis:
                    missing_names = [DISPLAY_NAMES.get(key, key) for key in missing_analysis]
                    st.info("Nutrien berikut sudah ditemukan pada profil makanan tetapi belum mempunyai analisis kontribusi AKG: " + ", ".join(missing_names) + ".")

        with tab4:
            try:
                heat, idx, conf, layer_info = make_gradcam(image, convnext_model)
                overlay = overlay_gradcam(image, heat)
                colored_heat = colorize_gradcam(heat)
                pred_label_raw = class_names[idx] if (class_names and 0 <= idx < len(class_names)) else f"Kelas {idx}"
                pred_label = food_display_name(pred_label_raw)
                st.markdown(f"""<div class="xai-hero"><div class="xai-intro"><div class="xai-intro-title">Grad-CAM • Explainable AI Workspace</div><div class="xai-intro-text">Visualisasi ini membantu melihat area citra yang relatif lebih berpengaruh pada keputusan <b>branch ConvNeXt V2 Tiny</b>. Ini bukan penjelasan penuh keputusan ensemble ConvNeXt + NoisyViT.</div></div><div class="xai-stat"><div class="xai-stat-label">Prediksi Branch</div><div class="xai-stat-value">{pred_label}</div></div><div class="xai-stat"><div class="xai-stat-label">Keyakinan</div><div class="xai-stat-value">{conf*100:.2f}%</div></div><div class="xai-stat"><div class="xai-stat-label">Explainability</div><div class="xai-stat-value">ConvNeXt</div></div></div>""", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3, gap="large")
                with c1:
                    with st.container(key="grad_original"):
                        st.markdown('<div class="grad-card-head"><div class="grad-card-title">Foto Input</div><div class="grad-card-badge">ORIGINAL</div></div>', unsafe_allow_html=True)
                        st.image(image.resize((224, 224)), use_container_width=True)
                with c2:
                    with st.container(key="grad_heatmap"):
                        st.markdown('<div class="grad-card-head"><div class="grad-card-title">Activation Map</div><div class="grad-card-badge">HEATMAP</div></div>', unsafe_allow_html=True)
                        st.image(colored_heat, use_container_width=True)
                with c3:
                    with st.container(key="grad_overlay"):
                        st.markdown('<div class="grad-card-head"><div class="grad-card-title">Model Attention Overlay</div><div class="grad-card-badge">OVERLAY</div></div>', unsafe_allow_html=True)
                        st.image(overlay, use_container_width=True)
                st.markdown(f"""<div class="xai-explain-grid"><div class="xai-explain"><b>Cara membaca:</b> area dengan aktivasi lebih kuat menunjukkan region yang relatif lebih banyak memengaruhi keluaran branch ConvNeXt ketika memprediksi <b>{pred_label}</b>. Heatmap sebaiknya dibaca bersama foto asli dan overlay, bukan sebagai segmentasi objek atau bukti kausal.</div><div class="xai-layer"><div class="label">TARGET FEATURE LAYER</div><div class="value">{layer_info}</div></div></div>""", unsafe_allow_html=True)
            except Exception as exc:
                st.warning("Grad-CAM belum dapat divisualisasikan.")
                st.code(str(exc))

        with tab5:
            ensemble_info = result.get("ensemble", {})
            st.markdown(f"""<div class="model-pill-grid"><div class="model-pill"><div class="model-pill-label">Backbone 1</div><div class="model-pill-value">ConvNeXt V2 Tiny</div><div class="model-pill-note">Cabang CNN untuk pengenalan citra makanan.</div></div><div class="model-pill"><div class="model-pill-label">Backbone 2</div><div class="model-pill-value">NoisyViT B/16</div><div class="model-pill-note">Cabang transformer untuk visual representation.</div></div><div class="model-pill"><div class="model-pill-label">Metode Fusi</div><div class="model-pill-value">Weighted Soft Voting</div><div class="model-pill-note">Bobot saat ini ConvNeXt {ensemble_info.get('convnext_weight', 0.5):.0%} • NoisyViT {ensemble_info.get('noisyvit_weight', 0.5):.0%}.</div></div></div>""", unsafe_allow_html=True)
            left_model, right_model = st.columns([1.35, .95], gap="large")
            with left_model:
                st.plotly_chart(model_compare_chart(preds), use_container_width=True, config={"displayModeBar": False})
            with right_model:
                st.plotly_chart(ensemble_weight_chart(ensemble_info), use_container_width=True, config={"displayModeBar": False})
            st.dataframe(pd.DataFrame([{"Kelas": food_display_name(p["food"]), "Ensemble (%)": round(p["confidence"] * 100, 2), "ConvNeXt (%)": round(p.get("convnext_confidence", 0) * 100, 2), "NoisyViT (%)": round(p.get("noisyvit_confidence", 0) * 100, 2)} for p in preds]), hide_index=True, use_container_width=True)
            st.markdown('''<div class="flow-grid"><div class="flow-node"><div class="num">01</div><div class="title">Input Antropometri</div><div class="text">Umur, berat, tinggi, dan MUAC/LILA diproses oleh model screening status gizi.</div></div><div class="flow-node"><div class="num">02</div><div class="title">Input Foto Makanan</div><div class="text">Satu foto makanan diproses paralel oleh ConvNeXt V2 Tiny dan NoisyViT B/16.</div></div><div class="flow-node"><div class="num">03</div><div class="title">Ensemble Prediction</div><div class="text">Dua probabilitas keluaran digabung menggunakan weighted soft voting untuk menghasilkan prediksi final 56 kelas.</div></div><div class="flow-node"><div class="num">04</div><div class="title">Nutrition Mapping</div><div class="text">Kelas makanan dipetakan ke knowledge base nutrisi, lalu dibandingkan dengan AKG kelompok umur.</div></div><div class="flow-node"><div class="num">05</div><div class="title">Gap Analysis</div><div class="text">Sistem menghitung kontribusi dan gap nutrisi prioritas dari makanan yang terdeteksi.</div></div><div class="flow-node"><div class="num">06</div><div class="title">Rekomendasi Berbasis Umur & Gap</div><div class="text">Kandidat makanan diranking berdasarkan gap nutrisi, cakupan, kesesuaian kelompok umur, dan kualitas kategori makanan.</div></div></div>''', unsafe_allow_html=True)
            st.caption("Detail model menampilkan keluaran ensemble, kontribusi masing-masing branch, serta alur pemrosesan NutriVision AI secara ringkas.")

        with tab6:
            st.markdown("""<div class="section-kicker">Menu Harian</div><div class="section-title">Rencana Menu Sehari</div><div class="section-sub">Pilih waktu makan dari foto tadi, lalu sistem menyusun waktu makan lainnya agar kebutuhan gizi sehari lebih terpenuhi. Edukatif, bukan resep diet klinis.</div>""", unsafe_allow_html=True)
            meal_time = st.radio("Foto tadi dimakan pada waktu:", ["Sarapan", "Makan Siang", "Camilan Sore", "Makan Malam"], horizontal=True, key="meal_time_slot")
            plan_priorities = nutrient_priorities(adequacy)
            with st.spinner("Menyusun menu harian..."):
                ai_plan = generate_ai_meal_plan(st.session_state.child_profile, str(status["prediction"]), plan_priorities, top_food_display, recommendations, meal_time)
            if ai_plan and ai_plan.get("meals"):
                meals = ai_plan["meals"]; closing = str(ai_plan.get("catatan", "")); source_label = "Disusun oleh Tim Sang Surya 1 — UMM"
            else:
                meals = build_meal_plan(top_food_display, recommendations, plan_priorities, meal_time)
                closing = "Menu disusun otomatis dari recommendation engine. Aktifkan mode AI (Gemini) untuk hasil lebih personal."
                source_label = "⚙️ Disusun otomatis (offline)"
            focus_text = ", ".join(plan_priorities) if plan_priorities else "gizi seimbang umum"
            st.markdown(f'<div class="rec-hero"><div class="rec-hero-title">{source_label}</div><div class="rec-hero-text">Foto dikunci sebagai <b>{html.escape(meal_time)}</b>. Sistem menyusun waktu makan lain untuk melengkapi: <b>{html.escape(focus_text)}</b>.</div></div>', unsafe_allow_html=True)
            for m in meals:
                waktu = html.escape(str(m.get("waktu", "-"))); menu = html.escape(str(m.get("menu", "-"))); alasan = html.escape(str(m.get("alasan", "")))
                is_locked = str(m.get("waktu", "")).strip().lower() == meal_time.strip().lower()
                badge = "📷" if is_locked else "🍽️"
                st.markdown(f'<div class="rec-card"><div class="rec-rank">{badge}</div><div><div class="rec-name">{waktu} — {menu}</div><div class="rec-reason">{alasan}</div></div></div>', unsafe_allow_html=True)

            # Ringkasan gizi sehari (kualitatif, tanpa angka karangan) + fokus nutrien.
            nutri_summary = str(ai_plan.get("ringkasan_gizi", "")) if (ai_plan and isinstance(ai_plan, dict)) else ""
            prio_chip_html = "".join(f'<span class="priority">\u2726 {html.escape(x)}</span>' for x in plan_priorities)
            summary_text = html.escape(nutri_summary) if nutri_summary else "Rangkaian menu di atas disusun untuk saling melengkapi kebutuhan gizi anak sepanjang hari."
            st.markdown(
                f'<div class="insight" style="margin-top:12px;"><b>\U0001F34E Ringkasan Gizi Sehari</b><br><br>'
                f'{summary_text}'
                f'<br><br><span style="font-size:.72rem;color:#8A95A8;">Nutrien yang difokuskan hari ini:</span><br>{prio_chip_html}</div>',
                unsafe_allow_html=True,
            )

            if closing:
                st.markdown(f'<div class="dark-note" style="color:#59677E;background:#FBFCFE;border:1px solid #E8EDF5;margin-top:10px;">{html.escape(closing)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="action-note" style="margin-top:12px;"><span class="note-icon">◇</span><span>Rencana ini edukatif. Untuk anak dengan indikasi gangguan gizi, konsultasikan menu dengan dokter anak atau ahli gizi.</span></div>', unsafe_allow_html=True)

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

    # ---- About: mission band ----
    st.markdown(
        """
        <div class="about-band">
            <div class="about-band-eyebrow">Tentang NutriVision AI</div>
            <div class="about-band-title">
                Teknologi <span>multimodal</span> untuk skrining gizi anak yang lebih terarah.
            </div>
            <div class="about-band-lead">
                NutriVision AI menggabungkan data antropometri anak dengan citra makanan untuk
                membantu skrining status gizi, mengenali makanan, menganalisis kontribusi nutrisi
                terhadap AKG, dan menyusun rekomendasi pendamping. Dirancang sebagai alat edukasi
                dan skrining awal, bukan pengganti diagnosis tenaga kesehatan.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- About: ringkasan sistem (stat cards) ----
    st.markdown(
        """
        <div class="about-section-head">
            <div class="about-section-kicker">Ringkasan Sistem</div>
            <div class="about-section-title">Sekilas kapabilitas utama</div>
            <div class="about-section-sub">Angka-angka penting yang mendefinisikan bagaimana NutriVision bekerja.</div>
        </div>
        <div class="kpi-grid">
            <div class="kpi-card gold">
                <div class="kpi-label">Pendekatan</div>
                <div class="kpi-value" style="font-size:1.08rem;">Multimodal AI</div>
                <div class="kpi-sub">Antropometri + citra makanan.</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Food Classes</div>
                <div class="kpi-value">56</div>
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

    # ---- About: kapabilitas + alur sistem ----
    st.markdown(
        """
        <div class="about-section-head">
            <div class="about-section-kicker">Kapabilitas & Alur</div>
            <div class="about-section-title">Apa yang dilakukan, dan bagaimana urutannya</div>
            <div class="about-section-sub">Empat kapabilitas inti di sisi kiri, satu alur end-to-end di sisi kanan.</div>
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
                    dengan weighted soft voting pada 56 kelas. Grad-CAM menjelaskan branch ConvNeXt.
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

    # ---- About: nilai & prinsip ----
    st.markdown(
        """
        <div class="about-section-head">
            <div class="about-section-kicker">Nilai & Prinsip</div>
            <div class="about-section-title">Prinsip yang menjaga sistem tetap bertanggung jawab</div>
            <div class="about-section-sub">Tiga komitmen yang memandu cara NutriVision menyajikan hasil.</div>
        </div>
        <div class="about-values">
            <div class="about-value-card">
                <div class="about-value-icon">🩺</div>
                <div class="about-value-title">Edukatif, bukan diagnosis</div>
                <div class="about-value-text">
                    Setiap keluaran diposisikan sebagai skrining awal dan bahan edukasi. Keputusan
                    klinis tetap berada di tangan dokter anak atau ahli gizi.
                </div>
            </div>
            <div class="about-value-card">
                <div class="about-value-icon">🔍</div>
                <div class="about-value-title">Transparan & dapat dijelaskan</div>
                <div class="about-value-text">
                    Grad-CAM dan rincian kontribusi nutrisi membantu memahami dasar dari sebuah
                    prediksi, bukan sekadar menampilkan angka akhir.
                </div>
            </div>
            <div class="about-value-card">
                <div class="about-value-icon">👶</div>
                <div class="about-value-title">Sesuai kelompok umur</div>
                <div class="about-value-text">
                    Analisis dan rekomendasi mengacu pada acuan gizi kelompok umur agar saran yang
                    diberikan lebih relevan untuk kebutuhan anak.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- About: arsitektur teknologi ----
    st.markdown(
        """
        <div class="about-section-head">
            <div class="about-section-kicker">Arsitektur Teknologi</div>
            <div class="about-section-title">Komponen di balik layar</div>
            <div class="about-section-sub">Susunan model dan mesin analisis yang menggerakkan NutriVision.</div>
        </div>
        <div class="about-tech">
            <div class="about-tech-item">
                <div class="about-tech-label">Backbone 1</div>
                <div class="about-tech-value">ConvNeXt V2 Tiny</div>
                <div class="about-tech-note">Cabang CNN untuk pengenalan citra makanan.</div>
            </div>
            <div class="about-tech-item">
                <div class="about-tech-label">Backbone 2</div>
                <div class="about-tech-value">NoisyViT B/16</div>
                <div class="about-tech-note">Cabang transformer untuk representasi visual.</div>
            </div>
            <div class="about-tech-item">
                <div class="about-tech-label">Fusi</div>
                <div class="about-tech-value">Weighted Soft Voting</div>
                <div class="about-tech-note">Menggabungkan dua probabilitas menjadi prediksi final.</div>
            </div>
            <div class="about-tech-item">
                <div class="about-tech-label">Mesin Analisis</div>
                <div class="about-tech-value">KB Gizi + Rekomendasi</div>
                <div class="about-tech-note">Pemetaan AKG, analisis gap, dan ranking makanan.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- About: closing CTA ----
    st.markdown(
        """
        <div class="about-cta">
            <div class="about-cta-title">Siap mencoba analisis pertama?</div>
            <div class="about-cta-text">
                Masukkan data anak dan unggah satu foto makanan untuk melihat skrining status gizi,
                pengenalan makanan, dan rekomendasi pendamping dalam satu alur.
            </div>
            <div class="about-cta-actions">
                <a class="about-cta-btn primary" href="?nav=dashboard" target="_self">Mulai Analisis →</a>
                <a class="about-cta-btn ghost" href="?nav=gizi" target="_self">Pelajari Nutrisi</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SITE FOOTER  (tampil di semua halaman)
# ============================================================

_year = datetime.now().strftime("%Y")

st.markdown(
    f"""
    <div class="site-footer">
        <div class="footer-top">
            <div>
                <div class="footer-brand-name">Nutri<span>Vision</span> AI</div>
                <div class="footer-brand-text">
                    Skrining gizi anak berbasis AI multimodal — menggabungkan antropometri dan
                    citra makanan untuk analisis nutrisi dan rekomendasi yang lebih terarah.
                </div>
                <div class="footer-badges">
                    <span class="footer-badge">Ensemble Deep Learning</span>
                    <span class="footer-badge">56 Food Classes</span>
                    <span class="footer-badge">Explainable AI</span>
                </div>
            </div>
            <div>
                <div class="footer-col-title">Navigasi</div>
                <div class="footer-links">
                    <a href="?nav=dashboard" target="_self">Dashboard</a>
                    <a href="?nav=riwayat" target="_self">Riwayat Analisis</a>
                    <a href="?nav=profil" target="_self">Profil Anak</a>
                    <a href="?nav=gizi" target="_self">Pengetahuan Gizi</a>
                    <a href="?nav=tentang" target="_self">Tentang Aplikasi</a>
                </div>
            </div>
            <div>
                <div class="footer-col-title">Catatan Penggunaan</div>
                <div class="footer-note">
                    NutriVision AI adalah alat edukasi dan skrining awal. Hasil bukan diagnosis
                    klinis. Untuk anak dengan indikasi gangguan gizi, konsultasikan dengan dokter
                    anak atau ahli gizi.
                </div>
            </div>
        </div>
        <div class="footer-bottom">
            <div class="footer-copy">© {_year} Tim Sang Surya 1 • Universitas Muhammadiyah Malang (UMM)</div>
            <div class="footer-disclaimer">
                Dibuat untuk tujuan edukasi. Satu foto tidak mewakili asupan 24 jam, dan
                rekomendasi bukan pengganti tenaga kesehatan profesional.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)