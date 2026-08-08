"""
JALANKAN BAGIAN INI DI COLAB SETELAH TRAINING SELESAI.

Tujuan:
- menyimpan model EfficientNet terbaik;
- memastikan artifact malnutrition tersedia;
- menyimpan class_names;
- menyimpan 2 tabel nutrisi yang sudah dipilih notebook;
- menyimpan tabel AKG;
- membungkus semuanya menjadi ZIP untuk dipindah ke project Streamlit.
"""

from pathlib import Path
import json
import shutil
import zipfile

EXPORT_DIR = Path("/content/nutrivision_deployment")
MODELS = EXPORT_DIR / "models"
DATA = EXPORT_DIR / "data"

MODELS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# 1. MODEL CITRA
source_food = MODEL_DIR / "food_classifier_best.keras"
if not source_food.exists():
    raise FileNotFoundError(f"Model citra tidak ditemukan: {source_food}")

shutil.copy2(
    source_food,
    MODELS / "food_classifier_best.keras"
)

# 2. MODEL MALNUTRITION
source_mal = MODEL_DIR / "malnutrition_model.joblib"
if not source_mal.exists():
    raise FileNotFoundError(f"Model malnutrition tidak ditemukan: {source_mal}")

shutil.copy2(
    source_mal,
    MODELS / "malnutrition_model.joblib"
)

# 3. CLASS NAMES
(MODELS / "class_names.json").write_text(
    json.dumps(list(class_names), ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# 4. NUTRICHECK NUTRITION TABLE
nutricheck_nut_df.to_csv(
    DATA / "nutricheck_nutrition.csv",
    index=False
)

# 5. INDONESIAN FOOD NUTRITION TABLE
indo_nut_df.to_csv(
    DATA / "indonesia_nutrition.csv",
    index=False
)

# 6. AKG
requirements_df.to_csv(
    DATA / "nutrition_requirements.csv",
    index=False
)

# 7. ZIP
zip_path = Path("/content/nutrivision_deployment.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for file in EXPORT_DIR.rglob("*"):
        if file.is_file():
            zf.write(file, file.relative_to(EXPORT_DIR))

print("✅ Artifact deployment siap:", zip_path)
print("Isi:")
for file in EXPORT_DIR.rglob("*"):
    if file.is_file():
        print(" -", file.relative_to(EXPORT_DIR))

# Optional download:
# from google.colab import files
# files.download(str(zip_path))