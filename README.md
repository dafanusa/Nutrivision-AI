# NutriVision AI — Streamlit Dashboard

Dashboard ini dipisahkan dari proses training. **Colab dipakai untuk training/evaluasi**, sedangkan project ini dipakai untuk **inference/dashboard**.

## Struktur

```text
NutriVision_Streamlit/
├── app.py
├── EXPORT_DARI_COLAB.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── models/
│   ├── food_classifier_best.keras        ← isi dari Colab
│   ├── malnutrition_model.joblib         ← isi dari Colab
│   └── class_names.json                  ← isi dari Colab
├── data/
│   ├── nutricheck_nutrition.csv           ← isi dari Colab
│   ├── indonesia_nutrition.csv            ← isi dari Colab
│   └── nutrition_requirements.csv
└── src/
    ├── inference.py
    ├── nutrition.py
    ├── recommendation.py
    └── ui.py
```

## 1. Export model dari notebook Colab

Setelah semua training selesai, buka file `EXPORT_DARI_COLAB.py`, copy isi script ke **cell paling akhir notebook Colab**, lalu jalankan.

Cell akan menghasilkan:

```text
/content/nutrivision_deployment.zip
```

Download ZIP tersebut.

## 2. Masukkan artifact ke project

Extract `nutrivision_deployment.zip`, kemudian copy:

- isi `models/` → folder `models/` project ini;
- isi `data/` → folder `data/` project ini.

Jangan rename file.

## 3. Jalankan di VS Code

Buka terminal di root project:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Jalankan:

```bash
streamlit run app.py
```

## Alur dashboard

1. User mengisi umur, BB, TB, MUAC/LILA.
2. User upload foto makanan.
3. Model XGBoost/artefak terbaik melakukan screening status gizi.
4. EfficientNetB0 memprediksi kelas makanan.
5. Label makanan dipetakan ke tabel nutrisi NutriCheck.
6. Profil nutrisi dibandingkan dengan AKG kelompok umur.
7. Dashboard menampilkan nutrient contribution dan gap.
8. Dataset Indonesian Food Nutrition digunakan sebagai knowledge base recommendation engine.
9. User mendapat kandidat makanan dan interpretasi edukatif.

## Catatan penting

- Dashboard **tidak melakukan training ulang**.
- `food_classifier_best.keras` berasal dari checkpoint EfficientNetB0 notebook.
- `malnutrition_model.joblib` memuat model + urutan fitur + label encoder.
- Notebook Anda memilih XGBoost sebagai model malnutrition terbaik berdasarkan Macro-F1.
- Dataset Indonesian Food Nutrition yang digunakan pada notebook hanya memiliki energi, protein, lemak, dan karbohidrat; maka recommendation engine hanya dapat meranking berdasarkan nutrien yang benar-benar tersedia di dataset tersebut.
- Basis porsi nutrisi NutriCheck harus diverifikasi sebelum klaim final GEMASTIK.
- Output adalah screening/edukasi, bukan diagnosis.