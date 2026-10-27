"""Estadísticas de montaje: ¿los rostros salen con tamaño suficiente desde esta posición de cámara?

Se acumulan durante la sesión (solo números) y se imprime un resumen al terminar. También estima
el tamaño equivalente en la cámara de despliegue, que suele tener un campo de visión más amplio
y por tanto muestra rostros más pequeños a la misma distancia.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from faces import Face


class MountStats:
    def __init__(self, cfg: dict[str, Any], min_face_px: int) -> None:
        mc = cfg.get("mount_check") or {}
        self.src_hfov = float(mc.get("source_hfov_deg", 69))
        self.tgt_hfov = float(mc.get("target_hfov_deg", 102))
        self.tgt_width = int(mc.get("target_width", 1920))
        self.min_face_px = min_face_px
        self.checks = 0
        self.outcomes: Counter = Counter()
        self.sizes: list[int] = []
        self.frame_width = 0

    def record(self, face: Face | None, frame_width: int) -> None:
        self.frame_width = frame_width
        self.checks += 1
        if face is None:
            self.outcomes["sin rostro"] += 1
            return
        self.sizes.append(face.size_px)
        self.outcomes[face.reject or "utilizable"] += 1

    def equivalent_scale(self) -> float:
        """Factor para pasar tamaño de rostro de la cámara de prueba a la de despliegue."""
        if not self.frame_width:
            return 1.0
        fov = math.tan(math.radians(self.src_hfov / 2)) / math.tan(math.radians(self.tgt_hfov / 2))
        return fov * self.tgt_width / self.frame_width

    def summary_lines(self) -> list[str]:
        if not self.checks:
            return ["Montaje: no se comprobó ningún rostro (¿hubo personas en escena?)"]
        pct = lambda n: f"{100 * n / self.checks:.0f}%"  # noqa: E731
        lines = [
            f"Montaje: {self.checks} comprobaciones de rostro en personas detectadas",
            "  " + "  ".join(f"{k}: {pct(v)}" for k, v in self.outcomes.most_common()),
        ]
        if self.sizes:
            p10, p50, p90 = (int(np.percentile(self.sizes, q)) for q in (10, 50, 90))
            scale = self.equivalent_scale()
            ok_here = sum(s >= self.min_face_px for s in self.sizes) / len(self.sizes)
            ok_target = sum(s * scale >= self.min_face_px for s in self.sizes) / len(self.sizes)
            lines += [
                f"  Tamaño de rostro (px, ancho de frame {self.frame_width}): p10={p10}  mediana={p50}  p90={p90}"
                f"  · ≥{self.min_face_px}px: {100 * ok_here:.0f}%",
                f"  Equivalente aprox. en la cámara de despliegue ({self.tgt_hfov:.0f}°, {self.tgt_width}px de ancho):"
                f" mediana≈{int(p50 * scale)}px · ≥{self.min_face_px}px: {100 * ok_target:.0f}%  (factor ×{scale:.2f})",
            ]
        return lines
