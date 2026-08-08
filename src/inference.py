from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .nutrition import (
    analyze_nutrient_contribution,
    get_requirement,
    nutrient_lookup,
)
from .recommendation import recommend_foods


def load_malnutrition_artifact(path: str | Path):
    try:
        artifact = joblib.load(path)
    except (AttributeError, ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "Gagal memuat models/malnutrition_model.joblib. "
            "File model ini harus berupa artifact dictionary dari EXPORT_DARI_COLAB.py "
            "dan dijalankan dengan environment yang memiliki versi dependency sesuai "
            "requirements.txt. Jika model dibuat dari kode lama yang menyimpan class "
            "custom, export ulang model dari Colab lalu salin ulang ke folder models/."
        ) from exc

    if not isinstance(artifact, dict):
        raise ValueError(
            "Artifact malnutrition harus berupa dictionary berisi model, features, "
            "dan label_encoder. Export ulang model menggunakan EXPORT_DARI_COLAB.py."
        )

    required = {"model", "features", "label_encoder"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Artifact malnutrition tidak lengkap: {sorted(missing)}")
    return artifact


def load_food_model(path: str | Path):
    import tensorflow as tf

    path = Path(path)

    try:
        return tf.keras.models.load_model(path, compile=False)
    except TypeError as exc:
        if "quantization_config" not in str(exc):
            raise

        sanitized_path = _create_keras_compat_copy(path)
        try:
            return tf.keras.models.load_model(sanitized_path, compile=False)
        finally:
            sanitized_path.unlink(missing_ok=True)


def _create_keras_compat_copy(path: Path) -> Path:
    """Strip newer Keras config keys that older local loaders reject."""

    def strip_unsupported_keys(value):
        if isinstance(value, dict):
            value.pop("quantization_config", None)
            for item in value.values():
                strip_unsupported_keys(item)
        elif isinstance(value, list):
            for item in value:
                strip_unsupported_keys(item)

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        sanitized_path = Path(tmp.name)

    with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(sanitized_path, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "config.json":
                config = json.loads(data)
                strip_unsupported_keys(config)
                data = json.dumps(config).encode("utf-8")
            target.writestr(info, data)

    return sanitized_path


def predict_food(image, model, class_names, img_size=(224, 224), top_k=3):
    import tensorflow as tf

    image = image.convert("RGB").resize(img_size)
    arr = tf.keras.utils.img_to_array(image)
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    idxs = np.argsort(probs)[::-1][:top_k]

    return [
        {"food": class_names[i], "confidence": float(probs[i])}
        for i in idxs
    ]


def predict_malnutrition(
    age_months,
    weight_kg,
    height_cm,
    muac_cm,
    artifact,
):
    model = artifact["model"]
    features = artifact["features"]
    label_encoder = artifact["label_encoder"]

    bmi = weight_kg / ((height_cm / 100) ** 2) if height_cm else np.nan

    canonical = {
        "age_months": age_months,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "muac_cm": muac_cm,
        "bmi": bmi,
    }

    row = {}
    for feature in features:
        # Notebook Anda menggunakan nama canonical ini.
        row[feature] = canonical.get(feature, np.nan)

    X = pd.DataFrame([row], columns=features)
    pred_idx = int(np.asarray(model.predict(X)).reshape(-1)[0])
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        confidence = float(np.max(probs))

    return {
        "prediction": str(pred_label),
        "confidence": confidence,
        "bmi": float(bmi) if not np.isnan(bmi) else None,
    }


def analyze_child_and_food(
    *,
    age_months,
    weight_kg,
    height_cm,
    muac_cm,
    image,
    requirements,
    food_model,
    class_names,
    malnutrition_artifact,
    nutricheck_kb,
    nutricheck_food_col,
    nutricheck_nutrient_cols,
    indo_rec_df,
    indo_food_col,
    indo_rec_cols,
):
    screening = predict_malnutrition(
        age_months,
        weight_kg,
        height_cm,
        muac_cm,
        malnutrition_artifact,
    )

    food_predictions = predict_food(
        image=image,
        model=food_model,
        class_names=class_names,
        top_k=3,
    )

    best_food = food_predictions[0]["food"]

    food_nutrition = nutrient_lookup(
        best_food,
        nutricheck_kb,
        nutricheck_food_col,
        nutricheck_nutrient_cols,
    )

    result = {
        "child_profile": {
            "age_months": int(age_months),
            "weight_kg": float(weight_kg),
            "height_cm": float(height_cm),
            "muac_cm": float(muac_cm),
        },
        "nutrition_status": screening,
        "food_prediction": food_predictions,
        "food_nutrition": food_nutrition,
        "nutrient_contribution": {},
        "recommendations": [],
    }

    if food_nutrition.get("status") != "ok":
        result["warning"] = (
            "Label makanan berhasil diprediksi tetapi tidak menemukan mapping nutrisi yang cukup mirip."
        )
        return result

    requirement = get_requirement(age_months, requirements)
    adequacy = analyze_nutrient_contribution(
        food_nutrition["nutrients"],
        requirement,
    )

    recommendations = recommend_foods(
        adequacy,
        indo_rec_df,
        indo_food_col,
        indo_rec_cols,
        top_n=5,
    )

    result["requirement"] = requirement
    result["nutrient_contribution"] = adequacy
    result["recommendations"] = recommendations
    return result
