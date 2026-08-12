from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .nutrition import (
    analyze_nutrient_contribution,
    get_requirement,
    nutrient_lookup,
)
from .recommendation import recommend_foods


CONVNEXT_MODEL_NAME = "convnextv2_tiny"
NOISYVIT_MODEL_NAME = "vit_base_patch16_224"
FOOD_NUM_CLASSES = 53

# Baseline ensemble. Optimalkan bobot ini menggunakan validation set.
DEFAULT_ENSEMBLE_WEIGHTS = (0.50, 0.50)


class _NoisyViTB16Inference(torch.nn.Module):
    """
    Wrapper inference yang sengaja mempertahankan prefix `vit.*`.

    Checkpoint NoisyViT yang dipakai NutriVision berisi key:
    vit.cls_token, vit.pos_embed, ..., vit.head.weight, vit.head.bias.

    Tidak ada parameter tambahan di luar `vit.*`, sehingga pada inference
    backbone ViT dapat direkonstruksi di dalam wrapper ini dan state_dict
    checkpoint dapat dimuat secara strict.
    """

    def __init__(self, vit: torch.nn.Module):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit(x)


def _device_from_model(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Fallback untuk PyTorch lama.
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint):
    """Menerima state_dict langsung atau checkpoint dictionary umum."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint tidak berisi state_dict yang valid.")

    cleaned = {}
    for key, value in checkpoint.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def load_malnutrition_artifact(path: str | Path):
    """Load artifact model antropometri NutriVision."""
    try:
        artifact = joblib.load(path)
    except (AttributeError, ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "Gagal memuat models/malnutrition_model.joblib. "
            "Pastikan artifact dan versi dependency kompatibel."
        ) from exc

    if not isinstance(artifact, dict):
        raise ValueError(
            "Artifact malnutrition harus berupa dictionary berisi "
            "model, features, dan label_encoder."
        )

    required = {"model", "features", "label_encoder"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(
            f"Artifact malnutrition tidak lengkap: {sorted(missing)}"
        )

    return artifact


def _attach_transform(model: torch.nn.Module, backbone: torch.nn.Module):
    """
    Simpan transform inference timm pada model.

    Masing-masing backbone memiliki transform sendiri sehingga satu gambar
    dapat diproses sesuai konfigurasi model ConvNeXt dan ViT masing-masing.
    """
    import timm

    data_config = timm.data.resolve_model_data_config(backbone)
    transform = timm.data.create_transform(
        **data_config,
        is_training=False,
    )
    model._nutrivision_transform = transform
    model._nutrivision_data_config = data_config


def load_convnext_model(
    path: str | Path,
    num_classes: int = FOOD_NUM_CLASSES,
    device: str | torch.device | None = None,
):
    """Rekonstruksi ConvNeXt V2 Tiny dan muat checkpoint 53 kelas."""
    import timm

    device = _resolve_device(device)
    model = timm.create_model(
        CONVNEXT_MODEL_NAME,
        pretrained=False,
        num_classes=num_classes,
    )

    state_dict = _extract_state_dict(_load_checkpoint(path))

    head_weight = state_dict.get("head.fc.weight")
    if head_weight is None:
        raise ValueError(
            "Checkpoint ConvNeXt tidak memiliki `head.fc.weight`."
        )
    if int(head_weight.shape[0]) != int(num_classes):
        raise ValueError(
            "Jumlah kelas ConvNeXt tidak cocok: "
            f"checkpoint={int(head_weight.shape[0])}, config={num_classes}."
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    _attach_transform(model, model)
    return model


def load_noisyvit_model(
    path: str | Path,
    num_classes: int = FOOD_NUM_CLASSES,
    device: str | torch.device | None = None,
):
    """
    Rekonstruksi NoisyViT B/16 untuk inference.

    Checkpoint terverifikasi memiliki:
    - vit.patch_embed.proj.weight = [768, 3, 16, 16]
    - vit.pos_embed = [1, 197, 768]
    - vit.head.weight = [53, 768]

    Struktur tersebut sesuai ViT Base patch16 dengan 53 output.
    """
    import timm

    device = _resolve_device(device)

    vit = timm.create_model(
        NOISYVIT_MODEL_NAME,
        pretrained=False,
        num_classes=num_classes,
    )
    model = _NoisyViTB16Inference(vit)

    state_dict = _extract_state_dict(_load_checkpoint(path))

    head_weight = state_dict.get("vit.head.weight")
    patch_weight = state_dict.get("vit.patch_embed.proj.weight")
    pos_embed = state_dict.get("vit.pos_embed")

    if head_weight is None:
        raise ValueError(
            "Checkpoint NoisyViT tidak memiliki `vit.head.weight`."
        )
    if int(head_weight.shape[0]) != int(num_classes):
        raise ValueError(
            "Jumlah kelas NoisyViT tidak cocok: "
            f"checkpoint={int(head_weight.shape[0])}, config={num_classes}."
        )
    if patch_weight is not None and tuple(patch_weight.shape[-2:]) != (16, 16):
        raise ValueError(
            "Checkpoint NoisyViT bukan patch-size 16 seperti yang diharapkan."
        )
    if pos_embed is not None and tuple(pos_embed.shape[1:]) != (197, 768):
        raise ValueError(
            "Shape positional embedding NoisyViT tidak sesuai ViT-B/16 224."
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    _attach_transform(model, model.vit)
    return model


# Alias lama agar kode lain yang masih memanggil load_food_model tidak langsung rusak.
def load_food_model(
    path: str | Path,
    num_classes: int = FOOD_NUM_CLASSES,
    device: str | torch.device | None = None,
):
    return load_convnext_model(
        path,
        num_classes=num_classes,
        device=device,
    )


def _food_transform(model: torch.nn.Module):
    transform = getattr(model, "_nutrivision_transform", None)
    if transform is not None:
        return transform

    import timm

    backbone = model.vit if hasattr(model, "vit") else model
    data_config = timm.data.resolve_model_data_config(backbone)
    transform = timm.data.create_transform(
        **data_config,
        is_training=False,
    )
    model._nutrivision_transform = transform
    model._nutrivision_data_config = data_config
    return transform


def _output_features(model: torch.nn.Module) -> int:
    if hasattr(model, "head") and hasattr(model.head, "fc"):
        return int(model.head.fc.out_features)

    if hasattr(model, "vit") and hasattr(model.vit, "head"):
        return int(model.vit.head.out_features)

    raise ValueError("Classifier head model tidak dikenali.")


def _predict_probabilities(
    image: Image.Image,
    model: torch.nn.Module,
) -> torch.Tensor:
    transform = _food_transform(model)
    device = _device_from_model(model)

    x = transform(image.convert("RGB")).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    return probs.detach().cpu()


def _normalize_ensemble_weights(
    weights: Sequence[float],
) -> tuple[float, float]:
    if len(weights) != 2:
        raise ValueError("Ensemble membutuhkan tepat dua bobot.")

    conv_w = float(weights[0])
    vit_w = float(weights[1])

    if conv_w < 0 or vit_w < 0:
        raise ValueError("Bobot ensemble tidak boleh negatif.")

    total = conv_w + vit_w
    if total <= 0:
        raise ValueError("Total bobot ensemble harus lebih dari 0.")

    return conv_w / total, vit_w / total


def predict_food_ensemble(
    image,
    convnext_model,
    noisyvit_model,
    class_names,
    top_k: int = 3,
    weights: Sequence[float] = DEFAULT_ENSEMBLE_WEIGHTS,
):
    """
    Weighted soft-voting ConvNeXt V2 Tiny + NoisyViT B/16.

    Catatan metodologis:
    `class_names` HARUS menggunakan indeks kelas yang sama pada kedua model.
    Checkpoint hanya membuktikan bahwa keduanya memiliki 53 output; checkpoint
    tidak menyimpan nama/urutan label.
    """
    if not class_names:
        raise ValueError("class_names kosong.")

    conv_outputs = _output_features(convnext_model)
    vit_outputs = _output_features(noisyvit_model)

    if conv_outputs != FOOD_NUM_CLASSES or vit_outputs != FOOD_NUM_CLASSES:
        raise ValueError(
            f"Output model tidak konsisten: ConvNeXt={conv_outputs}, "
            f"NoisyViT={vit_outputs}, expected={FOOD_NUM_CLASSES}."
        )

    if len(class_names) != conv_outputs:
        raise ValueError(
            f"class_names berisi {len(class_names)} kelas, sedangkan ensemble "
            f"memiliki {conv_outputs} output."
        )

    conv_w, vit_w = _normalize_ensemble_weights(weights)

    conv_probs = _predict_probabilities(image, convnext_model)
    vit_probs = _predict_probabilities(image, noisyvit_model)

    if conv_probs.shape != vit_probs.shape:
        raise ValueError(
            "Shape probability ConvNeXt dan NoisyViT berbeda: "
            f"{tuple(conv_probs.shape)} vs {tuple(vit_probs.shape)}"
        )

    ensemble_probs = (
        conv_w * conv_probs
        + vit_w * vit_probs
    )

    k = min(int(top_k), len(class_names))
    top_probs, top_indices = torch.topk(ensemble_probs, k=k)

    output = []
    for final_prob, idx in zip(
        top_probs.tolist(),
        top_indices.tolist(),
    ):
        output.append(
            {
                "food": str(class_names[idx]),
                "confidence": float(final_prob),
                "class_index": int(idx),
                "convnext_confidence": float(conv_probs[idx]),
                "noisyvit_confidence": float(vit_probs[idx]),
                "ensemble_weight_convnext": float(conv_w),
                "ensemble_weight_noisyvit": float(vit_w),
            }
        )

    return output


# Alias agar nama `predict_food` tetap tersedia untuk pemakaian lama.
def predict_food(
    image,
    model,
    class_names,
    top_k=3,
):
    """Prediksi ConvNeXt tunggal untuk backward compatibility."""
    probs = _predict_probabilities(image, model)
    k = min(int(top_k), len(class_names))
    top_probs, top_indices = torch.topk(probs, k=k)

    return [
        {
            "food": str(class_names[idx]),
            "confidence": float(prob),
            "class_index": int(idx),
        }
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist())
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

    bmi = (
        weight_kg / ((height_cm / 100) ** 2)
        if height_cm
        else np.nan
    )

    canonical = {
        "age_months": age_months,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "muac_cm": muac_cm,
        "bmi": bmi,
    }

    row = {}
    for feature in features:
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
    convnext_model,
    noisyvit_model,
    class_names,
    malnutrition_artifact,
    nutricheck_kb,
    nutricheck_food_col,
    nutricheck_nutrient_cols,
    indo_rec_df,
    indo_food_col,
    indo_rec_cols,
    ensemble_weights: Sequence[float] = DEFAULT_ENSEMBLE_WEIGHTS,
):
    """Decision layer utama NutriVision AI versi ensemble."""
    screening = predict_malnutrition(
        age_months,
        weight_kg,
        height_cm,
        muac_cm,
        malnutrition_artifact,
    )

    food_predictions = predict_food_ensemble(
        image=image,
        convnext_model=convnext_model,
        noisyvit_model=noisyvit_model,
        class_names=class_names,
        top_k=3,
        weights=ensemble_weights,
    )

    best_food = food_predictions[0]["food"]

    food_nutrition = nutrient_lookup(
        best_food,
        nutricheck_kb,
        nutricheck_food_col,
        nutricheck_nutrient_cols,
    )

    conv_w, vit_w = _normalize_ensemble_weights(ensemble_weights)

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
        "ensemble": {
            "method": "weighted_soft_voting",
            "convnext_model": CONVNEXT_MODEL_NAME,
            "noisyvit_model": NOISYVIT_MODEL_NAME,
            "convnext_weight": conv_w,
            "noisyvit_weight": vit_w,
            "num_classes": FOOD_NUM_CLASSES,
        },
    }

    if food_nutrition.get("status") != "ok":
        result["warning"] = (
            "Label makanan berhasil diprediksi tetapi tidak menemukan mapping "
            "nutrisi yang cukup mirip."
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


def make_gradcam(image, model):
    """
    Grad-CAM khusus branch ConvNeXt V2 Tiny.

    Soft-voting ensemble sendiri tidak mempunyai satu convolutional feature map.
    Karena itu visualisasi ini menjelaskan branch ConvNeXt, bukan keseluruhan
    keputusan ensemble secara penuh.
    """
    device = _device_from_model(model)
    transform = _food_transform(model)

    try:
        target_layer = model.stages[-1].blocks[-1].conv_dw
        layer_name = "ConvNeXt: stages.3.blocks.2.conv_dw"
    except Exception as exc:
        raise ValueError(
            "Layer Grad-CAM ConvNeXt V2 tidak ditemukan."
        ) from exc

    activation_holder = {}
    gradient_holder = {}

    def forward_hook(_module, _inputs, output):
        activation_holder["value"] = output
        output.register_hook(
            lambda grad: gradient_holder.__setitem__(
                "value",
                grad,
            )
        )

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        x = transform(image.convert("RGB")).unsqueeze(0).to(device)
        model.eval()
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = model(x)
            class_idx = int(torch.argmax(logits[0]).item())
            confidence = float(
                torch.softmax(logits, dim=1)[0, class_idx].item()
            )
            logits[0, class_idx].backward()

        activations = activation_holder.get("value")
        gradients = gradient_holder.get("value")
        if activations is None or gradients is None:
            raise RuntimeError(
                "Aktivasi/gradient Grad-CAM tidak tersedia."
            )

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(
            dim=1,
            keepdim=True,
        )
        cam = torch.relu(cam)
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam_min = cam.min()
        cam_max = cam.max()
        cam = (
            (cam - cam_min)
            / (cam_max - cam_min + 1e-8)
        )

        return (
            cam.detach().cpu().numpy(),
            class_idx,
            confidence,
            layer_name,
        )
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def overlay_gradcam(image, heatmap):
    """Overlay heatmap Grad-CAM tanpa TensorFlow."""
    base = np.asarray(
        image.convert("RGB").resize(
            (224, 224),
            Image.Resampling.LANCZOS,
        )
    ).astype(np.float32)

    heat_img = Image.fromarray(
        np.uint8(
            np.clip(
                np.asarray(heatmap) * 255.0,
                0,
                255,
            )
        )
    ).resize(
        (224, 224),
        Image.Resampling.BILINEAR,
    )

    heat = (
        np.asarray(heat_img).astype(np.float32)
        / 255.0
    )

    color = np.zeros(
        (224, 224, 3),
        dtype=np.float32,
    )
    color[..., 0] = 255
    color[..., 1] = 210 * (1 - 0.8 * heat)
    color[..., 2] = 25 * (1 - heat)

    opacity = heat[..., None] * 0.48
    out = base * (1 - opacity) + color * opacity

    return Image.fromarray(
        np.clip(out, 0, 255).astype(np.uint8)
    )
