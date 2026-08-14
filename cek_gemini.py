import os
import streamlit as st

st.title("Diagnostik Gemini")

# ============================================================
# 1) Cek apakah API key kebaca
# ============================================================

key_secrets = None
try:
    key_secrets = st.secrets.get("GEMINI_API_KEY")
except Exception as e:
    st.write("st.secrets error:", repr(e))

key_env = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

st.write("Key dari secrets.toml:", "ADA" if key_secrets else "TIDAK ADA")
st.write("Key dari environment :", "ADA" if key_env else "TIDAK ADA")

api_key = key_secrets or key_env

if api_key:
    st.write("4 huruf awal key:", str(api_key)[:4])
else:
    st.error("Key tidak ditemukan sama sekali. Berhenti di sini.")
    st.stop()

# ============================================================
# 2) Cek paket google-genai
# ============================================================

try:
    from google import genai
    from google.genai import types
    st.write("Paket google-genai: TERPASANG")
except Exception as e:
    st.error("Paket google-genai GAGAL IMPORT")
    st.exception(e)
    st.info("Jalankan: python -m pip install google-genai")
    st.stop()

# Buat client sekali, dipakai ulang di bawah.
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error("Gagal membuat client Gemini")
    st.exception(e)
    st.stop()

# ============================================================
# 3) Daftar model yang tersedia untuk key ini
# ============================================================

st.subheader("Model yang tersedia untuk key ini")

usable_models = []

try:
    for m in client.models.list():
        actions = (
            getattr(m, "supported_actions", None)
            or getattr(m, "supported_generation_methods", None)
            or []
        )
        name = getattr(m, "name", str(m))
        st.write(name, "→", actions)

        # Kumpulkan yang mendukung generateContent
        joined = " ".join(str(a).lower() for a in actions)
        if "generatecontent" in joined or "generate_content" in joined:
            usable_models.append(name)
except Exception as e:
    st.error("Gagal mengambil daftar model")
    st.exception(e)

if usable_models:
    st.success("Model yang mendukung generate_content:")
    for name in usable_models:
        st.write("•", name)
else:
    st.warning("Tidak ada model generate_content yang terdeteksi dari daftar.")

# ============================================================
# 4) Tes panggilan sungguhan
# ============================================================

st.subheader("Tes panggilan model")

# Urutan kandidat yang dicoba otomatis (dari yang paling mungkin tersedia).
candidates = [
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash-001",
]

# Sisipkan model dari daftar API di depan, tanpa prefix "models/".
for name in usable_models:
    short = name.replace("models/", "")
    if short not in candidates:
        candidates.insert(0, short)

berhasil = False

for model_name in candidates:
    st.write(f"Mencoba: `{model_name}` ...")
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents="Balas satu kata: OK",
        )
        st.success(f"BERHASIL dengan model: {model_name}")
        st.write("Balasan:", resp.text)
        st.info(
            f"Pakai nama model ini di app.py "
            f"(fungsi generate_ai_meal_plan):  model=\"{model_name}\""
        )
        berhasil = True
        break
    except Exception as e:
        st.write("→ gagal:", repr(e))

if not berhasil:
    st.error(
        "Semua kandidat model gagal. Lihat daftar 'Model yang tersedia' di atas, "
        "salin salah satu yang mendukung generate_content, lalu pakai namanya."
    )
    
st.subheader("Tes panggilan model")

candidates = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
]

berhasil = None
for model_name in candidates:
    st.write(f"Mencoba: `{model_name}` ...")
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents="Balas satu kata: OK",
        )
        st.success(f"BERHASIL: {model_name} → {resp.text}")
        berhasil = model_name
        break
    except Exception as e:
        st.write("→ gagal:", repr(e))

if berhasil:
    st.info(f'Pakai di app.py:  model="{berhasil}"')
else:
    st.error("Ketiga kandidat gagal. Kirim pesan errornya ke sini.")