"""Estado por track, agregación de atributos y emisión de eventos y snapshots de ocupación.

Por privacidad, de cada rostro bueno solo se guarda en memoria su calidad y los atributos
estimados (números), nunca el recorte. Al cerrar el track se descarta todo.
"""
from __future__ import annotations

import logging
import secrets
import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from attributes import FaceAttributes
from detector import PersonDetection
from settings import Clock

log = logging.getLogger(__name__)


@dataclass
class FaceSample:
    quality: float
    attrs: FaceAttributes
    embedding: np.ndarray | None = None  # solo en modo re-ID; nunca sale de memoria


@dataclass
class TrackState:
    uid: str                      # ID corto aleatorio; no se relaciona con nada fuera de esta sesión
    tracker_id: int
    t_enter: float
    last_seen: float
    bbox: np.ndarray
    zone: str | None = None
    zone_frames: Counter = field(default_factory=Counter)
    samples: list[FaceSample] = field(default_factory=list)
    frames_seen: int = 0
    last_attr_time: float = 0.0
    at_exit: bool = False         # estaba en un borde lateral y moviéndose hacia él (salió)
    relinks: int = 0
    visitor_id: str | None = None  # modo re-ID: visitante asignado en vivo
    visitor_sim: float | None = None
    cx_hist: deque = field(default_factory=lambda: deque(maxlen=30))  # (t, centro x) recientes


class Aggregator:
    def __init__(self, cfg: dict[str, Any], device_id: str, clock: Clock) -> None:
        agg = cfg["aggregator"]
        self.top_n = int(agg["top_n_faces"])
        self.close_after_s = float(agg["close_after_s"])
        self.relink_window_s = float(agg.get("relink_window_s", self.close_after_s))
        self.relink_max_dist = float(agg.get("relink_max_dist", 0.6))
        self.exit_margin = float(agg.get("exit_margin", 0.03))
        self.min_dwell_s = float(agg["min_dwell_s"])
        self.age_bands: list[str] = list(agg["age_bands"])
        self.expr_classes: list[str] = list(cfg["attributes"]["expression_classes"])
        self.attr_interval_s = float(cfg["attributes"]["min_interval_s"])
        self.present_window_s = float(cfg["occupancy"]["present_window_s"])
        self.zones = {
            name: np.array(poly, dtype=np.float32) for name, poly in (cfg.get("zones") or {}).items()
        }
        self.device_id = device_id
        self.clock = clock
        self.tracks: dict[int, TrackState] = {}
        self.stats: Counter = Counter()  # abiertos, re-enlazados, descartados, emitidos
        self.registry: Any = None        # VisitorRegistry si el modo re-ID está activo

    # ---------- actualización por frame ----------

    def observe(self, dets: list[PersonDetection], frame_shape: tuple[int, ...], now: float) -> list[TrackState]:
        h, w = frame_shape[:2]
        active: list[TrackState] = []
        new_dets: list[PersonDetection] = []
        for d in dets:
            st = self.tracks.get(d.track_id)
            if st is None:
                new_dets.append(d)
                continue
            self._update(st, d, now, w, h)
            active.append(st)

        # IDs nuevos: primero se intenta re-enlazar con un track perdido cerca de esa posición.
        for d in new_dets:
            st = self._find_relink(d, now, active)
            if st is not None:
                del self.tracks[st.tracker_id]
                st.tracker_id = d.track_id
                st.relinks += 1
                self.stats["relinked"] += 1
                log.debug("Track %s re-enlazado (tracker %d, perdido %.1fs)", st.uid, d.track_id, now - st.last_seen)
            else:
                st = TrackState(uid=secrets.token_hex(2), tracker_id=d.track_id,
                                t_enter=now, last_seen=now, bbox=d.bbox)
                self.stats["opened"] += 1
                log.debug("Track abierto %s (tracker %d)", st.uid, d.track_id)
            self.tracks[d.track_id] = st
            self._update(st, d, now, w, h)
            active.append(st)
        return active

    def _update(self, st: TrackState, d: PersonDetection, now: float, w: int, h: int) -> None:
        st.last_seen = now
        st.bbox = d.bbox
        st.frames_seen += 1
        # Salida = en un borde lateral y desplazándose hacia él en el último ~1.5 s.
        # Alguien sentado y cortado por el borde no cuenta como salida.
        cx = float(d.bbox[0] + d.bbox[2]) / 2
        st.cx_hist.append((now, cx))
        while st.cx_hist and now - st.cx_hist[0][0] > 1.5:
            st.cx_hist.popleft()
        shift = cx - st.cx_hist[0][1]
        margin = self.exit_margin * w
        min_shift = 0.03 * w
        st.at_exit = bool((d.bbox[0] <= margin and shift < -min_shift)
                          or (d.bbox[2] >= w - margin and shift > min_shift))
        st.zone = self._zone_of(d.bbox, w, h)
        if st.zone:
            st.zone_frames[st.zone] += 1

    def _find_relink(self, d: PersonDetection, now: float, active: list[TrackState]) -> TrackState | None:
        """Track perdido en medio de la escena, reciente, cerca y de tamaño parecido."""
        cx, cy = (d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2
        new_h = d.bbox[3] - d.bbox[1]
        best, best_dist = None, self.relink_max_dist
        taken = {id(a) for a in active}  # por identidad: TrackState contiene arrays
        for st in self.tracks.values():
            if id(st) in taken or st.at_exit or now - st.last_seen > self.relink_window_s:
                continue
            old_h = max(1.0, float(st.bbox[3] - st.bbox[1]))
            if not 0.5 <= new_h / old_h <= 2.0:
                continue
            ox, oy = (st.bbox[0] + st.bbox[2]) / 2, (st.bbox[1] + st.bbox[3]) / 2
            dist = float(np.hypot(cx - ox, cy - oy)) / old_h
            if dist <= best_dist:
                best, best_dist = st, dist
        return best

    def wants_attributes(self, st: TrackState, quality: float, now: float) -> bool:
        """Solo se estiman atributos si el rostro entra en el top-N del track."""
        if now - st.last_attr_time < self.attr_interval_s:
            return False
        if len(st.samples) < self.top_n:
            return True
        return quality > st.samples[-1].quality

    def add_sample(self, st: TrackState, quality: float, attrs: FaceAttributes, now: float,
                   embedding: np.ndarray | None = None) -> None:
        st.last_attr_time = now
        st.samples.append(FaceSample(quality, attrs, embedding))
        st.samples.sort(key=lambda s: s.quality, reverse=True)
        del st.samples[self.top_n:]
        if self.registry is not None and st.visitor_id is None:
            embs = [s.embedding for s in st.samples if s.embedding is not None]
            st.visitor_id, st.visitor_sim = self.registry.identify(embs, now, self._ids_in_use(st))
            if st.visitor_id:
                log.info("[%s] track %s identificado en vivo como %s (similitud %s)",
                         self.device_id, st.uid, st.visitor_id, st.visitor_sim)

    def _ids_in_use(self, exclude_track: TrackState | None) -> set[str]:
        """IDs de visitante de los demás tracks activos de esta cámara."""
        return {t.visitor_id for t in self.tracks.values() if t is not exclude_track and t.visitor_id}

    # ---------- cierre de tracks ----------

    def close_stale(self, now: float, force: bool = False) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for tid in list(self.tracks):
            st = self.tracks[tid]
            timeout = self.close_after_s if st.at_exit else self.relink_window_s
            if force or now - st.last_seen > timeout:
                del self.tracks[tid]
                dwell = st.last_seen - st.t_enter
                if dwell < self.min_dwell_s:
                    self.stats["discarded"] += 1
                    if self.registry is not None:
                        self.registry.release(st.visitor_id)
                    log.debug("Track %s descartado (%.1fs)", st.uid, dwell)
                    continue
                self.stats["emitted"] += 1
                events.append(self._build_event(st))
        return events

    def _build_event(self, st: TrackState) -> dict[str, Any]:
        ev: dict[str, Any] = {
            "device_id": self.device_id,
            "track_id": st.uid,
            "t_enter": self.clock.iso(st.t_enter),
            "t_exit": self.clock.iso(st.last_seen),
            "dwell_s": int(round(st.last_seen - st.t_enter)),
            "age_est": None, "age_range": None, "age_conf": None,
            "gender_est": None, "gender_conf": None,
            "expression_dist": None,
            "face_frames_used": len(st.samples),
            "zone": st.zone_frames.most_common(1)[0][0] if st.zone_frames else None,
        }
        if st.samples:
            ages = [s.attrs.age for s in st.samples]
            age = statistics.median(ages)
            band = self._age_band(age)
            # confianza de edad = fracción de estimaciones que caen en la misma banda
            ev["age_est"] = int(round(age))
            ev["age_range"] = band
            ev["age_conf"] = round(sum(self._age_band(a) == band for a in ages) / len(ages), 2)

            votes_m = sum(s.attrs.p_male >= 0.5 for s in st.samples)
            is_m = votes_m * 2 >= len(st.samples)
            p_win = [s.attrs.p_male if is_m else 1 - s.attrs.p_male for s in st.samples]
            ev["gender_est"] = "M" if is_m else "F"
            ev["gender_conf"] = round(float(np.mean(p_win)), 2)

            ev["expression_dist"] = self._expression_dist(st.samples)

        if self.registry is not None:
            embs = [s.embedding for s in st.samples if s.embedding is not None]
            vid, visit_n, sim = self.registry.close(st.visitor_id, embs, st.last_seen, self.device_id,
                                                    self._ids_in_use(None))
            ev["visitor_id"], ev["visit_n"], ev["reid_sim"] = vid, visit_n, sim
            if vid:
                log.info("[%s] track %s cerrado -> visitante %s (track %d del visitante hoy, similitud %s)",
                         self.device_id, st.uid, vid, visit_n, sim)
        return ev

    def _expression_dist(self, samples: list[FaceSample]) -> dict[str, float]:
        keys = samples[0].attrs.expression.keys()
        mean = {k: float(np.mean([s.attrs.expression[k] for s in samples])) for k in keys}
        dist = {k: round(mean.get(k, 0.0), 2) for k in self.expr_classes}
        dist["other"] = round(max(0.0, 1.0 - sum(mean.get(k, 0.0) for k in self.expr_classes)), 2)
        return dist

    def _age_band(self, age: float) -> str:
        for band in self.age_bands:
            if band.startswith("<"):
                if age < float(band[1:]):
                    return band
            elif band.endswith("+"):
                if age >= float(band[:-1]):
                    return band
            else:
                lo, hi = (float(x) for x in band.split("-"))
                if lo <= age < hi + 1:
                    return band
        return self.age_bands[-1]

    # ---------- ocupación ----------

    def occupancy(self, now: float) -> dict[str, Any]:
        zones = {name: 0 for name in self.zones}
        total = 0
        for st in self.tracks.values():
            if now - st.last_seen <= self.present_window_s:
                total += 1
                if st.zone:
                    zones[st.zone] += 1
        return {"device_id": self.device_id, "ts": self.clock.iso(now), "total": total, "zones": zones}

    def _zone_of(self, bbox: np.ndarray, w: int, h: int) -> str | None:
        foot = (float((bbox[0] + bbox[2]) / 2 / w), float(min(bbox[3] / h, 0.999)))
        for name, poly in self.zones.items():
            if cv2.pointPolygonTest(poly, foot, False) >= 0:
                return name
        return None
