"""Guarda eventos y snapshots en SQLite y, si MQTT está activo, los reenvía (QoS 1).

SQLite funciona como buffer offline: cada fila queda con sent=0 hasta que el broker
confirma la publicación. La clave (device_id, track_id, t_enter) evita duplicados.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any

from settings import resolve_path

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS person_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    track_id TEXT NOT NULL,
    t_enter TEXT NOT NULL,
    payload TEXT NOT NULL,
    sent INTEGER NOT NULL DEFAULT 0,
    UNIQUE (device_id, track_id, t_enter)
);
CREATE TABLE IF NOT EXISTS occupancy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    payload TEXT NOT NULL,
    sent INTEGER NOT NULL DEFAULT 0,
    UNIQUE (device_id, ts)
);
"""


class Publisher:
    def __init__(self, cfg: dict[str, Any], device_id: str) -> None:
        path = resolve_path(cfg["storage"]["sqlite_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")  # permite que el dashboard lea mientras se escribe
        self.db.executescript(SCHEMA)
        self.device_id = device_id
        self.mqtt_cfg = cfg.get("mqtt") or {}
        self.client = None
        self._pending: dict[int, tuple[str, int]] = {}  # mid -> (tabla, id)
        self._acked: list[tuple[str, int]] = []          # confirmados por el broker, por marcar
        self._last_flush = 0.0
        log.info("SQLite en %s", path)
        if self.mqtt_cfg.get("enabled"):
            self._connect_mqtt()

    # ---------- guardado local ----------

    def save_event(self, ev: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO person_events (device_id, track_id, t_enter, payload) VALUES (?,?,?,?)",
            (ev["device_id"], ev["track_id"], ev["t_enter"], json.dumps(ev, ensure_ascii=False)),
        )
        self.db.commit()
        log.info("Evento: %s", json.dumps(ev, ensure_ascii=False))

    def save_occupancy(self, snap: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO occupancy (device_id, ts, payload) VALUES (?,?,?)",
            (snap["device_id"], snap["ts"], json.dumps(snap, ensure_ascii=False)),
        )
        self.db.commit()
        log.info("Ocupación: %s", json.dumps(snap, ensure_ascii=False))

    # ---------- MQTT ----------

    def _connect_mqtt(self) -> None:
        import paho.mqtt.client as mqtt

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.device_id)
        self.client.on_publish = self._on_publish
        self.client.connect_async(self.mqtt_cfg["host"], int(self.mqtt_cfg["port"]))
        self.client.loop_start()
        log.info("MQTT -> %s:%s", self.mqtt_cfg["host"], self.mqtt_cfg["port"])

    def _on_publish(self, client: Any, userdata: Any, mid: int, reason_code: Any, properties: Any) -> None:
        item = self._pending.pop(mid, None)
        if item:
            table, row_id = item
            # Se marca en flush() desde el hilo principal para no compartir la conexión SQLite.
            self._acked.append((table, row_id))

    def flush(self, now: float | None = None, force: bool = False) -> None:
        """Reenvía filas no enviadas. Se llama periódicamente desde el loop principal."""
        if self.client is None:
            return
        now = now or time.time()
        for table, row_id in self._acked:
            self.db.execute(f"UPDATE {table} SET sent=1 WHERE id=?", (row_id,))
        self._acked.clear()
        self.db.commit()
        if not force and now - self._last_flush < float(self.mqtt_cfg.get("flush_interval_s", 5)):
            return
        self._last_flush = now
        if not self.client.is_connected():
            return
        qos = int(self.mqtt_cfg.get("qos", 1))
        in_flight = {v for v in self._pending.values()}
        for table, topic in (("person_events", "events"), ("occupancy", "occupancy")):
            rows = self.db.execute(f"SELECT id, device_id, payload FROM {table} WHERE sent=0 LIMIT 200").fetchall()
            for row_id, dev, payload in rows:
                if (table, row_id) in in_flight:
                    continue
                info = self.client.publish(f"rooms/{dev}/{topic}", payload, qos=qos)
                self._pending[info.mid] = (table, row_id)

    def close(self) -> None:
        if self.client is not None:
            self.flush(force=True)
            self.client.loop_stop()
            self.client.disconnect()
        self.db.close()
