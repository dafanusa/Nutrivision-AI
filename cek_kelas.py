"""
cek_class.py - Cek total class dan daftar namanya untuk model NutriVision.

Jalankan dari root project:
    python cek_class.py

Skrip ini mencoba beberapa sumber, berurutan:
  1. Checkpoint .pt di folder models/  (biasanya menyimpan 'class_names' & 'num_classes')
  2. File JSON class di models/ atau data/ (mis. class_names.json, classes.json)
Tidak butuh streamlit. Butuh torch hanya jika membaca file .pt.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


def _print_classes(names, sumber):
    names = list(names)
    print("=" * 60)
    print(f"SUMBER : {sumber}")
    print(f"TOTAL CLASS : {len(names)}")
    print("-" * 60)
    for i, nama in enumerate(names):
        print(f"  [{i:>2}] {nama}")
    print("=" * 60)
    print()


def cek_dari_checkpoint():
    """Baca class_names dari file checkpoint .pt / .pth."""
    try:
        import torch
    except Exception:
        print("(!) torch tidak terpasang, lewati pengecekan file .pt")
        return False

    if not MODEL_DIR.exists():
        print(f"(!) Folder tidak ada: {MODEL_DIR}")
        return False

    ckpt_files = sorted(list(MODEL_DIR.glob("*.pt")) + list(MODEL_DIR.glob("*.pth")))
    if not ckpt_files:
        print(f"(!) Tidak ada file .pt/.pth di {MODEL_DIR}")
        return False

    ketemu = False
    for ckpt_path in ckpt_files:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            print(f"(!) Gagal membaca {ckpt_path.name}: {e!r}")
            continue

        if not isinstance(ckpt, dict):
            continue

        class_names = ckpt.get("class_names")
        num_classes = ckpt.get("num_classes")

        if class_names:
            _print_classes(class_names, f"checkpoint: {ckpt_path.name}")
            if num_classes is not None and int(num_classes) != len(class_names):
                print(f"(!) CATATAN: num_classes metadata = {num_classes}, "
                      f"tetapi jumlah class_names = {len(class_names)} (tidak cocok).\n")
            ketemu = True
        elif num_classes is not None:
            print(f"[{ckpt_path.name}] num_classes = {num_classes}, "
                  f"tetapi tidak menyimpan 'class_names'.\n")

    return ketemu


def cek_dari_json():
    """Cari file JSON yang berisi daftar class."""
    kandidat = []
    for folder in (MODEL_DIR, DATA_DIR):
        if folder.exists():
            kandidat += list(folder.glob("*.json"))

    ketemu = False
    for jpath in sorted(kandidat):
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception:
            continue

        names = None
        if isinstance(data, list):
            names = data
        elif isinstance(data, dict):
            # Coba key umum, atau mapping index->nama
            for key in ("class_names", "classes", "labels", "names"):
                if key in data and isinstance(data[key], list):
                    names = data[key]
                    break
            if names is None and data and all(
                isinstance(v, str) for v in data.values()
            ):
                # dict {index: nama} -> urutkan berdasarkan key
                try:
                    names = [data[k] for k in sorted(data, key=lambda x: int(x))]
                except Exception:
                    names = list(data.values())

        if names and all(isinstance(x, str) for x in names):
            _print_classes(names, f"JSON: {jpath.relative_to(BASE_DIR)}")
            ketemu = True

    return ketemu


if __name__ == "__main__":
    print("\nMengecek total & nama class NutriVision...\n")
    ada_ckpt = cek_dari_checkpoint()
    ada_json = cek_dari_json()

    if not ada_ckpt and not ada_json:
        print("Tidak menemukan daftar class di checkpoint maupun JSON.")
        print("Pastikan file model ada di folder 'models/' atau daftar class di 'data/'.")