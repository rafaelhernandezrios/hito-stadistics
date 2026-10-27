"""Re-identificación de visitantes. SOLO para el laboratorio y con consentimiento de quienes aparecen.

Se activa con `--reid` (o `reid.enabled: true`). Sin eso, el sistema es totalmente anónimo.

Un único `VisitorRegistry` lo comparten todas las cámaras del proceso, así que un visitante
("v003") es la misma persona en cualquier cámara, y lo sigue siendo aunque salga de todas y vuelva
ese mismo día.

Reglas de este modo:
- Las huellas faciales (embeddings) viven solo en memoria: nunca se escriben a disco ni se transmiten.
- El registro se borra cada día a medianoche (hora local) y al detener el pipeline.
- Los eventos solo llevan el ID de visitante del día; nada del rostro.

Bajo la APPI un embedding facial es un 個人識別符号 (dato personal identificable), también en un laboratorio.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from faces import Face
from settings import Clock, resolve_path

log = logging.getLogger(__name__)


class FaceEmbedder:
    def __init__(self, reid_cfg: dict[str, Any], faces_cfg: dict[str, Any]) -> None:
        from insightface.model_zoo import get_model

        path = resolve_path(faces_cfg["models_dir"]) / reid_cfg.get("model", "w600k_mbf.onnx")
        self.rec = get_model(str(path), providers=["CPUExecutionProvider"])
        self.rec.prepare(ctx_id=-1)
        log.info("Modelo de reconocimiento cargado: %s", path.name)

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray:
        from insightface.utils import face_align

        aimg = face_align.norm_crop(frame, landmark=face.kps, image_size=self.rec.input_size[0])
        v = self.rec.get_feat(aimg).flatten().astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


def _mean_unit(embs: list[np.ndarray]) -> np.ndarray:
    m = np.mean(embs, axis=0)
    return m / (np.linalg.norm(m) + 1e-9)


@dataclass
class _Visitor:
    visitor_id: str
    centroid: np.ndarray
    n_closed: int = 0                 # tracks cerrados asignados a este visitante (todas las cámaras)
    refs: int = 0                     # tracks activos que lo tienen asignado ahora mismo
    devices: set[str] = field(default_factory=set)


class VisitorRegistry:
    """Registro de visitantes del día, compartido entre cámaras y seguro entre hilos.

    - `identify()` se llama en vivo cuando un track reúne `min_faces` rostros: le da un ID ya.
    - `close()` se llama al cerrar el track: confirma (o corrige) el ID con todos sus rostros.
    - `release()` se llama si el track se descarta sin evento.
    """

    def __init__(self, reid_cfg: dict[str, Any], clock: Clock) -> None:
        self.threshold = float(reid_cfg.get("match_threshold", 0.35))
        self.min_faces = int(reid_cfg.get("min_faces", 2))
        self.switch_margin = float(reid_cfg.get("switch_margin", 0.05))
        self.clock = clock
        self._lock = threading.Lock()
        self._day: str | None = None
        self._visitors: dict[str, _Visitor] = {}
        self._next_n = 1

    # ---------- API ----------

    def identify(self, embeddings: list[np.ndarray], now: float, exclude: set[str]) -> tuple[str | None, float | None]:
        """ID en vivo. `exclude` = IDs que tienen otros tracks activos de la MISMA cámara
        (dos personas en el mismo frame no pueden ser la misma). Entre cámaras sí se permite."""
        if len(embeddings) < self.min_faces:
            return None, None
        emb = _mean_unit(embeddings)
        with self._lock:
            self._roll_day(now)
            best, sim = self._best(emb, exclude)
            if best is not None and sim >= self.threshold:
                best.refs += 1
                return best.visitor_id, round(sim, 2)
            v = self._new(emb)
            v.refs = 1
            return v.visitor_id, (round(sim, 2) if best is not None else None)

    def close(self, live_id: str | None, embeddings: list[np.ndarray], now: float, device_id: str,
              exclude: set[str]) -> tuple[str | None, int | None, float | None]:
        """Confirma el visitante de un track que se cierra. Devuelve (id, nº de track del visitante, similitud)."""
        with self._lock:
            if self._roll_day(now):
                live_id = None
            live = self._visitors.get(live_id) if live_id else None
            if live is not None:
                live.refs -= 1
            if len(embeddings) < self.min_faces:
                self._cleanup()
                return None, None, None

            emb = _mean_unit(embeddings)
            best, best_sim = self._best(emb, exclude)
            chosen, sim = None, None
            if live is not None:
                own_sim = float(np.dot(live.centroid, emb))
                # Con todos los rostros, si otro visitante encaja claramente mejor, se corrige.
                if best is not None and best is not live and best_sim >= self.threshold \
                        and best_sim > own_sim + self.switch_margin:
                    chosen, sim = best, best_sim
                    log.info("Re-ID: %s corregido a %s (%.2f > %.2f)", live.visitor_id, best.visitor_id, best_sim, own_sim)
                else:
                    chosen, sim = live, own_sim
            elif best is not None and best_sim >= self.threshold:
                chosen, sim = best, best_sim
            else:
                chosen, sim = self._new(emb), (best_sim if best is not None else None)

            # media del centroide ponderada por tracks, para adaptarse a luz y pose de cada cámara
            c = chosen.centroid * max(chosen.n_closed, 1) + emb
            chosen.centroid = c / (np.linalg.norm(c) + 1e-9)
            chosen.n_closed += 1
            chosen.devices.add(device_id)
            self._cleanup()
            return chosen.visitor_id, chosen.n_closed, (round(sim, 2) if sim is not None else None)

    def release(self, live_id: str | None) -> None:
        if not live_id:
            return
        with self._lock:
            v = self._visitors.get(live_id)
            if v is not None:
                v.refs -= 1
            self._cleanup()

    def unique_today(self) -> int:
        with self._lock:
            return sum(1 for v in self._visitors.values() if v.n_closed > 0 or v.refs > 0)

    def cross_camera_today(self) -> int:
        with self._lock:
            return sum(1 for v in self._visitors.values() if len(v.devices) > 1)

    def forget(self) -> None:
        with self._lock:
            self._visitors.clear()

    # ---------- internos (con el lock tomado) ----------

    def _roll_day(self, now: float) -> bool:
        day = self.clock.iso(now)[:10]
        if day == self._day:
            return False
        if self._visitors:
            log.info("Nuevo día: se borran %d visitantes del registro", len(self._visitors))
        self._visitors.clear()
        self._next_n = 1
        changed = self._day is not None
        self._day = day
        return changed

    def _best(self, emb: np.ndarray, exclude: set[str]) -> tuple[_Visitor | None, float]:
        best, best_sim = None, -1.0
        for v in self._visitors.values():
            if v.visitor_id in exclude:
                continue
            sim = float(np.dot(v.centroid, emb))
            if sim > best_sim:
                best, best_sim = v, sim
        return best, best_sim

    def _new(self, emb: np.ndarray) -> _Visitor:
        v = _Visitor(visitor_id=f"v{self._next_n:03d}", centroid=emb)
        self._next_n += 1
        self._visitors[v.visitor_id] = v
        return v

    def _cleanup(self) -> None:
        """Quita visitantes creados en vivo que ningún track terminó usando."""
        for vid in [k for k, v in self._visitors.items() if v.n_closed == 0 and v.refs <= 0]:
            del self._visitors[vid]
