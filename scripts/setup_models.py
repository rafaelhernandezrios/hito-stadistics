"""Descarga los modelos que el pipeline necesita (una sola vez por computadora).

    python scripts/setup_models.py

- models/yolo11n.pt                         detector de personas (Ultralytics)
- ~/.insightface/models/buffalo_s/          rostro (det_500m), edad/género (genderage), re-id (w600k_mbf)
- ~/.hsemotion/enet_b0_8_best_vgaf.onnx     expresión facial

Si un modelo ya existe, se omite.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def yolo() -> None:
    dest = ROOT / "models" / "yolo11n.pt"
    if dest.exists():
        print(f"OK  {dest.relative_to(ROOT)} (ya existe)")
        return
    from ultralytics.utils.downloads import attempt_download_asset

    dest.parent.mkdir(parents=True, exist_ok=True)
    path = Path(attempt_download_asset("yolo11n.pt"))
    shutil.move(str(path), dest)
    print(f"OK  {dest.relative_to(ROOT)}")


def insightface_buffalo_s() -> None:
    from insightface.utils.storage import ensure_available

    path = ensure_available("models", "buffalo_s", root="~/.insightface")
    print(f"OK  {path}")


def hsemotion() -> None:
    from hsemotion_onnx.facial_emotions import get_model_path

    print(f"OK  {get_model_path('enet_b0_8_best_vgaf')}")


if __name__ == "__main__":
    yolo()
    insightface_buffalo_s()
    hsemotion()
    print("Modelos listos.")
