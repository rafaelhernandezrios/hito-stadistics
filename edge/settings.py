"""Carga de config.yaml y utilidades compartidas (rutas, hora local)."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def camera_backend() -> int:
    """Backend de OpenCV para abrir cámaras según el sistema operativo."""
    import cv2

    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform == "win32":
        return cv2.CAP_DSHOW  # abre más rápido que MSMF y respeta la resolución pedida
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def camera_configs(cfg: dict[str, Any], sources: list[str] | None = None,
                   resolution: str | None = None) -> list[dict[str, Any]]:
    """Lista de cámaras a procesar. Cada una hereda `camera` y `zones` de arriba y puede sobrescribirlos.

    - Con `cameras:` en config se usa esa lista (y `sources`, si se pasa, reemplaza sus fuentes en orden).
    - Sin lista: una cámara con `device_id`; con varias `sources` se crean `<device_id>-cam1`, `-cam2`…
    """
    base = {"device_id": cfg["device_id"], **cfg["camera"], "zones": cfg.get("zones") or {}}
    cams = [{**base, **c} for c in (cfg.get("cameras") or [])]
    if sources:
        if cams and len(cams) == len(sources):
            for c, s in zip(cams, sources):
                c["source"] = s
        elif len(sources) == 1:
            cams = [{**(cams[0] if cams else base), "source": sources[0]}]
        else:
            cams = [{**base, "device_id": f"{cfg['device_id']}-cam{i + 1}", "source": s}
                    for i, s in enumerate(sources)]
    if not cams:
        cams = [base]
    if resolution:
        w, h = (int(x) for x in resolution.lower().split("x"))
        for c in cams:
            c["width"], c["height"] = w, h
    ids = [c["device_id"] for c in cams]
    if len(set(ids)) != len(ids):
        raise ValueError(f"device_id repetido entre cámaras: {ids}")
    return cams


def resolve_path(p: str) -> Path:
    """Expande ~ y resuelve rutas relativas desde la raíz del proyecto."""
    path = Path(os.path.expanduser(p))
    return path if path.is_absolute() else PROJECT_ROOT / path


class Clock:
    """Hora local con zona horaria fija, en ISO 8601 (+09:00 para Japón)."""

    def __init__(self, tz_name: str) -> None:
        self.tz = ZoneInfo(tz_name)

    def iso(self, ts: float) -> str:
        return datetime.fromtimestamp(ts, self.tz).isoformat(timespec="seconds")
