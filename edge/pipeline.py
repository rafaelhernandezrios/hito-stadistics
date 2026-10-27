"""Lectura de cámaras y pipeline por cámara.

Cada cámara en vivo tiene un hilo que solo captura y guarda el último frame (así ninguna se queda
atrasada). El procesamiento se hace por turnos en el hilo principal, compartiendo los modelos de
rostro/atributos y, en modo re-ID, el registro de visitantes.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

import cv2
import numpy as np

from aggregator import Aggregator, TrackState
from attributes import AttributeEstimator
from detector import make_tracker
from faces import Face, FaceDetector
from mount import MountStats
from publisher import Publisher
from settings import Clock, camera_backend

log = logging.getLogger("edge")


def is_camera_index(source: Any) -> bool:
    return isinstance(source, int) or (isinstance(source, str) and source.isdigit())


class LiveReader:
    """Captura en un hilo y entrega solo el frame más reciente."""

    is_file = False

    def __init__(self, source: int, width: int, height: int, name: str) -> None:
        self.name = name
        self.cap = self._open(int(source), width, height)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self._seq = 0
        self._last_seq = 0
        self._running = self.cap.isOpened()
        self.finished = not self._running
        if self._running:
            self._thread = threading.Thread(target=self._loop, name=f"cap-{name}", daemon=True)
            self._thread.start()

    @staticmethod
    def _open(index: int, width: int, height: int) -> cv2.VideoCapture:
        # La primera vez el sistema (macOS/Windows) puede pedir permiso de cámara y la apertura falla; se reintenta ~30 s.
        cap = cv2.VideoCapture()
        for attempt in range(15):
            cap = cv2.VideoCapture(index, camera_backend())
            if cap.isOpened():
                break
            cap.release()
            if attempt == 0:
                log.warning("Cámara %d no disponible; si el sistema pide permiso, acéptalo. Reintentando…", index)
            time.sleep(2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    def _loop(self) -> None:
        fails = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails % 50 == 1:
                    log.warning("[%s] frame no disponible; reintentando", self.name)
                time.sleep(0.02)
                continue
            fails = 0
            with self._lock:
                self._frame, self._ts, self._seq = frame, time.time(), self._seq + 1

    def read(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            if self._seq == self._last_seq or self._frame is None:
                return None
            self._last_seq = self._seq
            return self._frame, self._ts

    def size(self) -> tuple[int, int]:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def stop(self) -> None:
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=2)
        self.cap.release()


class FileReader:
    """Archivo de video: el tiempo sale del propio video y se saltan frames para simular target_fps."""

    is_file = True

    def __init__(self, path: str, target_fps: float, start: float, name: str) -> None:
        self.name = name
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.step = max(1, round(self.fps / target_fps))
        self.start = start
        self.idx = 0
        self.finished = not self.cap.isOpened()

    def read(self) -> tuple[np.ndarray, float] | None:
        while not self.finished:
            ok, frame = self.cap.read()
            if not ok:
                self.finished = True
                return None
            self.idx += 1
            if (self.idx - 1) % self.step == 0:
                return frame, self.now()
        return None

    def now(self) -> float:
        return self.start + self.idx / self.fps

    def size(self) -> tuple[int, int]:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def stop(self) -> None:
        self.cap.release()


class SharedModels:
    """Modelos sin estado por cámara: se cargan una vez y los usan todas."""

    def __init__(self, cfg: dict[str, Any], reid_enabled: bool, clock: Clock) -> None:
        self.faces = FaceDetector(cfg["faces"])
        self.attrs = AttributeEstimator(cfg["attributes"], cfg["faces"])
        self.embedder = None
        self.registry = None
        if reid_enabled:
            from reid import FaceEmbedder, VisitorRegistry

            reid_cfg = cfg.get("reid") or {}
            self.embedder = FaceEmbedder(reid_cfg, cfg["faces"])
            self.registry = VisitorRegistry(reid_cfg, clock)


class CameraPipeline:
    def __init__(self, cam: dict[str, Any], cfg: dict[str, Any], shared: SharedModels,
                 publisher: Publisher, clock: Clock, start: float) -> None:
        self.device_id: str = cam["device_id"]
        self.shared = shared
        self.pub = publisher
        self.tracker = make_tracker(cfg["detector"])  # cada cámara con su propio tracker
        self.agg = Aggregator({**cfg, "zones": cam.get("zones") or {}}, self.device_id, clock)
        self.agg.registry = shared.registry
        self.mount = MountStats(cfg, shared.faces.min_face_px)
        self.face_every = max(1, int(cfg["faces"]["every_n_frames"]))
        self.period = 1.0 / float(cam["target_fps"])
        self.occ_interval = float(cfg["occupancy"]["interval_s"])
        self.next_occ = math.ceil(start / self.occ_interval) * self.occ_interval
        self.next_due = 0.0
        self.proc_idx = 0
        self.fps_ema = 0.0
        self.last_now = start

        src = cam["source"]
        if is_camera_index(src):
            self.reader: LiveReader | FileReader = LiveReader(int(src), cam["width"], cam["height"], self.device_id)
        else:
            self.reader = FileReader(src, float(cam["target_fps"]), start, self.device_id)
        if self.reader.finished:
            log.error("[%s] no se pudo abrir la fuente %r. Si es una cámara, revisa en Ajustes del Sistema > "
                      "Privacidad y seguridad > Cámara que la app desde la que corres esto tenga permiso.",
                      self.device_id, src)
        else:
            w, h = self.reader.size()
            log.info("[%s] fuente=%r %dx%d, %.0f fps objetivo", self.device_id, src, w, h, cam["target_fps"])

    @property
    def done(self) -> bool:
        return self.reader.finished

    def poll(self, wall_now: float) -> tuple[np.ndarray, list[TrackState], dict[str, Face]] | None:
        """Procesa un frame si toca. Devuelve (frame, tracks activos, rostros) o None."""
        if not self.reader.is_file and wall_now < self.next_due:
            return None
        item = self.reader.read()
        if item is None:
            return None
        frame, now = item
        self.next_due = max(self.next_due + self.period, wall_now)
        t0 = time.time()
        active, shown = self._process(frame, now)
        self.last_now = now
        self.proc_idx += 1
        dt = max(time.time() - t0, 1e-3)
        self.fps_ema = 0.9 * self.fps_ema + 0.1 / dt if self.proc_idx > 1 else 1.0 / dt
        if self.proc_idx % 100 == 0:
            log.info("[%s] frames=%d  tracks activos=%d  capacidad≈%.1f fps",
                     self.device_id, self.proc_idx, len(active), self.fps_ema)
        return frame, active, shown

    def _process(self, frame: np.ndarray, now: float) -> tuple[list[TrackState], dict[str, Face]]:
        sh = self.shared
        dets = self.tracker.update(frame)
        active = self.agg.observe(dets, frame.shape, now)
        shown: dict[str, Face] = {}
        for st in active:
            if (st.frames_seen - 1) % self.face_every:
                continue
            face = sh.faces.detect(frame, st.bbox)
            self.mount.record(face, frame.shape[1])
            if face is None:
                continue
            shown[st.uid] = face
            if face.usable and self.agg.wants_attributes(st, face.quality, now):
                emb = sh.embedder.embed(frame, face) if sh.embedder else None
                self.agg.add_sample(st, face.quality, sh.attrs.estimate(frame, face), now, emb)

        for ev in self.agg.close_stale(now):
            self.pub.save_event(ev)
        if now >= self.next_occ:
            self.pub.save_occupancy(self.agg.occupancy(now))
            self.next_occ += self.occ_interval
        return active, shown

    def finish(self) -> None:
        end = self.reader.now() if self.reader.is_file else time.time()
        for ev in self.agg.close_stale(end, force=True):
            self.pub.save_event(ev)
        self.reader.stop()
        s = self.agg.stats
        log.info("[%s] detenido (%d frames). Tracks: %d abiertos, %d re-enlazados, %d descartados (<%.0fs), "
                 "%d eventos emitidos", self.device_id, self.proc_idx, s["opened"], s["relinked"],
                 s["discarded"], self.agg.min_dwell_s, s["emitted"])
        for line in self.mount.summary_lines():
            log.info("[%s] %s", self.device_id, line)


def draw_preview(frame: np.ndarray, active: list[TrackState], shown_faces: dict[str, Face],
                 agg: Aggregator, fps: float) -> None:
    h, w = frame.shape[:2]
    for name, poly in agg.zones.items():
        pts = (poly * np.array([w, h])).astype(np.int32)
        cv2.polylines(frame, [pts], True, (200, 200, 0), 1)
        cv2.putText(frame, name, tuple(pts[0] + 6) if pts[0][1] < h - 20 else tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    for st in active:
        x1, y1, x2, y2 = st.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"{st.uid} {int(st.last_seen - st.t_enter)}s"
        if st.samples:
            ages = sorted(s.attrs.age for s in st.samples)
            pm = float(np.mean([s.attrs.p_male for s in st.samples]))
            expr = max(st.samples[0].attrs.expression.items(), key=lambda kv: kv[1])[0]
            label += f" {agg._age_band(ages[len(ages) // 2])} {'M' if pm >= 0.5 else 'F'} {expr} n={len(st.samples)}"
        cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if st.visitor_id:  # modo re-ID: el mismo ID en todas las cámaras
            cv2.putText(frame, st.visitor_id, (x1 + 6, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        face = shown_faces.get(st.uid)
        if face is not None:
            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            color = (0, 140, 255) if face.usable else (0, 0, 230)  # naranja = utilizable, rojo = no
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
            text = f"{face.size_px}px" + (f" q={face.quality:.2f}" if face.usable else f" {face.reject}")
            cv2.putText(frame, text, (fx1, fy2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"{agg.device_id}  {fps:.1f} fps  personas={len(active)}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if agg.registry is not None:
        cv2.putText(frame, f"RE-ID ACTIVO (laboratorio) - visitantes hoy: {agg.registry.unique_today()}",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
