"""Detección y tracking de personas.

La interfaz `PersonTracker` es común a todos los backends de inferencia.
Hoy existe `UltralyticsTracker` (YOLO + ByteTrack, MPS/CUDA/CPU); en la Pi se añadirá
un backend Hailo con la misma interfaz.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from settings import resolve_path

log = logging.getLogger(__name__)

PERSON_CLASS = 0


@dataclass
class PersonDetection:
    track_id: int            # ID del tracker; solo válido mientras la persona está en escena
    bbox: np.ndarray         # x1, y1, x2, y2 en píxeles
    conf: float


class PersonTracker(Protocol):
    def update(self, frame: np.ndarray) -> list[PersonDetection]: ...


def pick_device(device: str) -> str:
    """"auto" elige mps (Mac Apple Silicon), cuda (GPU NVIDIA) o cpu, en ese orden."""
    if device != "auto":
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class UltralyticsTracker:
    def __init__(self, cfg: dict[str, Any]) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(resolve_path(cfg["model"])))
        self.device = pick_device(cfg.get("device", "auto"))
        self.conf = float(cfg.get("conf", 0.1))
        self.imgsz = int(cfg.get("imgsz", 640))
        self.tracker = str(resolve_path(cfg["tracker"])) if cfg.get("tracker") else "bytetrack.yaml"
        log.info("YOLO cargado (%s) en device=%s", cfg["model"], self.device)

    def update(self, frame: np.ndarray) -> list[PersonDetection]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            classes=[PERSON_CLASS],
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        return [
            PersonDetection(track_id=int(i), bbox=b, conf=float(c))
            for b, i, c in zip(xyxy, ids, confs)
        ]


def make_tracker(cfg: dict[str, Any]) -> PersonTracker:
    backend = cfg.get("backend", "ultralytics")
    if backend == "ultralytics":
        return UltralyticsTracker(cfg)
    raise ValueError(f"Backend de detector no soportado: {backend}")
