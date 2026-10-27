"""Edad y género (InsightFace genderage) y expresión facial (hsemotion-onnx).

Solo se llama sobre rostros que ya pasaron el filtro de calidad. Devuelve números;
no guarda ni devuelve imágenes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from faces import Face
from settings import resolve_path

log = logging.getLogger(__name__)

# Nombres de hsemotion -> nombres en el evento
_EXPR_NAMES = {
    "Anger": "anger", "Contempt": "contempt", "Disgust": "disgust", "Fear": "fear",
    "Happiness": "happy", "Neutral": "neutral", "Sadness": "sad", "Surprise": "surprise",
}


@dataclass
class FaceAttributes:
    age: float
    p_male: float                     # probabilidad de género masculino estimado
    expression: dict[str, float]      # distribución completa (8 clases)


class AttributeEstimator:
    def __init__(self, cfg: dict[str, Any], faces_cfg: dict[str, Any]) -> None:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        from insightface.model_zoo import get_model

        path = resolve_path(faces_cfg["models_dir"]) / cfg["genderage_model"]
        self.ga = get_model(str(path), providers=["CPUExecutionProvider"])
        self.ga.prepare(ctx_id=-1)
        self.expr = HSEmotionRecognizer(model_name=cfg["expression_model"])
        log.info("Modelos de atributos cargados: %s, %s", path.name, cfg["expression_model"])

    def estimate(self, frame: np.ndarray, face: Face) -> FaceAttributes:
        age, p_male = self._gender_age(frame, face.bbox)
        expression = self._expression(frame, face.bbox)
        return FaceAttributes(age=age, p_male=p_male, expression=expression)

    def _gender_age(self, frame: np.ndarray, bbox: np.ndarray) -> tuple[float, float]:
        # Mismo preprocesado que insightface Attribute.get, pero conservando las probabilidades.
        from insightface.utils import face_align

        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = ((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2)
        size = self.ga.input_size[0]
        scale = size / (max(w, h) * 1.5)
        aimg, _ = face_align.transform(frame, center, size, scale, 0)
        blob = cv2.dnn.blobFromImage(
            aimg, 1.0 / self.ga.input_std, (size, size),
            (self.ga.input_mean,) * 3, swapRB=True,
        )
        pred = self.ga.session.run(self.ga.output_names, {self.ga.input_name: blob})[0][0]
        g = pred[:2] - pred[:2].max()
        probs = np.exp(g) / np.exp(g).sum()
        return float(pred[2] * 100.0), float(probs[1])  # índice 1 = masculino en InsightFace

    def _expression(self, frame: np.ndarray, bbox: np.ndarray) -> dict[str, float]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        mx, my = int(0.1 * (x2 - x1)), int(0.1 * (y2 - y1))
        roi = frame[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        _, scores = self.expr.predict_emotions(rgb, logits=False)
        return {_EXPR_NAMES[self.expr.idx_to_class[i]]: float(s) for i, s in enumerate(scores)}
