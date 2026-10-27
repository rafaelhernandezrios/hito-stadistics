"""Detección de rostro (SCRFD de InsightFace) dentro de la caja de cada persona y cálculo de calidad."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from settings import resolve_path

log = logging.getLogger(__name__)


@dataclass
class Face:
    bbox: np.ndarray      # x1, y1, x2, y2 en píxeles del frame completo
    kps: np.ndarray       # 5 puntos (ojo izq, ojo der, nariz, boca izq, boca der)
    det_score: float
    size_px: int
    frontal: float        # 0 = perfil, 1 = frontal
    sharpness: float      # varianza del Laplaciano
    quality: float        # 0–1, combina tamaño, pose y nitidez
    reject: str | None = None  # None = utilizable; si no: "pequeño", "perfil" o "borroso"

    @property
    def usable(self) -> bool:
        return self.reject is None


class FaceDetector:
    def __init__(self, cfg: dict[str, Any]) -> None:
        from insightface.model_zoo import get_model

        path = resolve_path(cfg["models_dir"]) / cfg["det_model"]
        self.det = get_model(str(path), providers=["CPUExecutionProvider"])
        size = int(cfg.get("det_input", 320))
        self.det.prepare(ctx_id=-1, input_size=(size, size), det_thresh=float(cfg.get("det_thresh", 0.5)))
        self.head_fraction = float(cfg.get("head_fraction", 0.6))
        self.min_face_px = int(cfg.get("min_face_px", 60))
        self.min_frontal = float(cfg.get("min_frontal", 0.35))
        self.min_sharpness = float(cfg.get("min_sharpness", 40.0))
        log.info("Detector de rostro cargado: %s", path.name)

    def detect(self, frame: np.ndarray, person_bbox: np.ndarray) -> Face | None:
        """Devuelve el rostro principal de la persona, o None si no se encuentra.

        El rostro se devuelve aunque no pase los umbrales (con `reject` indicando el motivo),
        para poder medir el montaje; solo los `usable` se usan para atributos.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = person_bbox.astype(int)
        y2 = y1 + int((y2 - y1) * self.head_fraction)
        pad = int(0.1 * (x2 - x1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None

        crop = frame[y1:y2, x1:x2]
        bboxes, kpss = self.det.detect(crop, max_num=1)
        if bboxes is None or len(bboxes) == 0:
            return None

        offset = np.array([x1, y1], dtype=np.float32)
        fb = bboxes[0][:4].copy()
        fb[:2] += offset
        fb[2:] += offset
        kps = kpss[0] + offset if kpss is not None else None
        if kps is None:
            return None

        size = int(min(fb[2] - fb[0], fb[3] - fb[1]))
        frontal = _frontal_score(kps)
        sharp = _sharpness(frame, fb)
        reject = None
        if size < self.min_face_px:
            reject = "pequeño"
        elif frontal < self.min_frontal:
            reject = "perfil"
        elif sharp < self.min_sharpness:
            reject = "borroso"

        quality = min(1.0, size / 120.0) * frontal * min(1.0, sharp / 200.0)
        return Face(
            bbox=fb, kps=kps, det_score=float(bboxes[0][4]),
            size_px=size, frontal=frontal, sharpness=sharp, quality=quality, reject=reject,
        )


def _frontal_score(kps: np.ndarray) -> float:
    """Aproxima el giro (yaw) por la posición de la nariz respecto al centro de los ojos."""
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < 1e-3:
        return 0.0
    offset = abs(nose[0] - (left_eye[0] + right_eye[0]) / 2) / eye_dist
    return float(max(0.0, 1.0 - offset / 0.5))


def _sharpness(frame: np.ndarray, bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox.astype(int)
    roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(cv2.resize(roi, (112, 112)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
