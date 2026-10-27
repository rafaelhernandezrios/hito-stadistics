"""Dashboard local del prototipo: lee la SQLite del edge y sirve una página con gráficas.

Uso:
    python dashboard/server.py            # http://localhost:8000
    python dashboard/server.py --port 8080

Solo lectura y solo localhost. Es un puente hasta la Fase 3 (TimescaleDB + Grafana).
Solo usa la librería estándar.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "edge"))
from settings import load_config, resolve_path  # noqa: E402

log = logging.getLogger("dashboard")
STATIC = Path(__file__).resolve().parent / "static"
DWELL_BUCKETS = [("<30 s", 0, 30), ("30 s–2 min", 30, 120), ("2–10 min", 120, 600),
                 ("10–30 min", 600, 1800), ("30 min+", 1800, float("inf"))]


class Store:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.path = resolve_path(cfg["storage"]["sqlite_path"])
        self.tz = ZoneInfo(cfg.get("timezone", "Asia/Tokyo"))
        self.age_bands: list[str] = list(cfg["aggregator"]["age_bands"])
        self.zones: list[str] = list((cfg.get("zones") or {}).keys())
        self.device_id: str = cfg["device_id"]

    def _rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            return [json.loads(r[0]) for r in db.execute(sql, params)]
        finally:
            db.close()

    def dates(self) -> list[str]:
        if not self.path.exists():
            return []
        db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            rows = db.execute(
                "SELECT DISTINCT substr(t_enter,1,10) FROM person_events "
                "UNION SELECT DISTINCT substr(ts,1,10) FROM occupancy ORDER BY 1 DESC"
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            db.close()

    def summary(self, date: str, device: str | None = None) -> dict[str, Any]:
        like = f"{date}%"
        all_events = self._rows("SELECT payload FROM person_events WHERE t_enter LIKE ? ORDER BY t_enter", (like,))
        all_occ = self._rows("SELECT payload FROM occupancy WHERE ts LIKE ? ORDER BY ts", (like,))
        devices = sorted({e["device_id"] for e in all_events} | {o["device_id"] for o in all_occ})
        if device:
            events = [e for e in all_events if e["device_id"] == device]
            occ = [o for o in all_occ if o["device_id"] == device]
        else:
            events, occ = all_events, all_occ
        shown_devices = [device] if device else devices
        faces = [e for e in events if e.get("face_frames_used")]
        dwell = [e["dwell_s"] for e in events]

        def hour(ts: str) -> int:
            return int(ts[11:13])

        hours_present = [hour(e["t_enter"]) for e in events] + [hour(o["ts"]) for o in occ]
        h0, h1 = (min(hours_present), max(hours_present)) if hours_present else (9, 18)
        hours = list(range(h0, h1 + 1))

        # Ocupación por cámara: no se suma entre cámaras porque pueden ver a la misma persona.
        occ_by_dev_hour: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for o in occ:
            occ_by_dev_hour[o["device_id"]][hour(o["ts"])].append(o["total"])
        timeline_ts = sorted({o["ts"] for o in occ})
        by_dev_ts = {(o["device_id"], o["ts"]): o["total"] for o in occ}
        now = datetime.now(self.tz)
        occ_now: dict[str, Any] = {}
        for dev in shown_devices:
            dev_occ = [o for o in occ if o["device_id"] == dev]
            if dev_occ:
                last = dev_occ[-1]
                occ_now[dev] = {"total": last["total"], "ts": last["ts"],
                                "age_s": round((now - datetime.fromisoformat(last["ts"])).total_seconds()),
                                "zones": last["zones"]}
        zone_names = sorted({z for o in occ for z in o["zones"]} | {e["zone"] for e in events if e["zone"]}) \
            or self.zones
        entries_by_hour = Counter(hour(e["t_enter"]) for e in events)

        expr_keys: list[str] = list(faces[0]["expression_dist"].keys()) if faces else []
        expr_by_hour: dict[int, list[dict[str, float]]] = defaultdict(list)
        for e in faces:
            expr_by_hour[hour(e["t_enter"])].append(e["expression_dist"])

        def mean_dist(dists: list[dict[str, float]]) -> dict[str, float] | None:
            if not dists:
                return None
            return {k: round(statistics.mean(d[k] for d in dists), 3) for k in expr_keys}

        reid_events = [e for e in events if "visitor_id" in e]
        visitors = {e["visitor_id"] for e in reid_events if e["visitor_id"]}
        # visitantes vistos en más de una cámara (se mira el día completo, no solo el filtro)
        vis_devices: dict[str, set[str]] = defaultdict(set)
        for e in all_events:
            if e.get("visitor_id"):
                vis_devices[e["visitor_id"]].add(e["device_id"])
        cross = sum(1 for v in visitors if len(vis_devices[v]) > 1)

        return {
            "device_id": device or "todas",
            "devices": devices,
            "shown_devices": shown_devices,
            "date": date,
            "generated_at": now.isoformat(timespec="seconds"),
            "kpis": {
                "visits": len(events),
                "dwell_mean_s": round(statistics.mean(dwell)) if dwell else None,
                "dwell_median_s": round(statistics.median(dwell)) if dwell else None,
                "face_pct": round(100 * len(faces) / len(events)) if events else None,
                "occ_now": occ_now,
                "occ_max": max((o["total"] for o in occ), default=None),
                # solo si hubo sesiones con re-ID (laboratorio)
                "reid_visits": len(reid_events),
                "unique_visitors": len(visitors) if reid_events else None,
                "unidentified": sum(1 for e in reid_events if not e["visitor_id"]),
                "cross_camera": cross if reid_events else None,
            },
            "hours": hours,
            "occ_by_hour": {
                dev: {
                    "avg": [round(statistics.mean(occ_by_dev_hour[dev][h]), 2) if occ_by_dev_hour[dev][h] else None
                            for h in hours],
                    "max": [max(occ_by_dev_hour[dev][h]) if occ_by_dev_hour[dev][h] else None for h in hours],
                } for dev in shown_devices
            },
            "entries_by_hour": [entries_by_hour.get(h, 0) for h in hours],
            "occ_timeline": {
                "ts": timeline_ts,
                "series": {dev: [by_dev_ts.get((dev, ts)) for ts in timeline_ts] for dev in shown_devices},
            },
            "dwell_buckets": {
                "labels": [b[0] for b in DWELL_BUCKETS],
                "counts": [sum(lo <= d < hi for d in dwell) for _, lo, hi in DWELL_BUCKETS],
            },
            "age": {
                "labels": self.age_bands,
                "counts": [sum(e["age_range"] == b for e in faces) for b in self.age_bands],
            },
            "gender": dict(Counter(e["gender_est"] for e in faces)),
            "expression": {
                "keys": expr_keys,
                "overall": mean_dist([e["expression_dist"] for e in faces]),
                "by_hour": [mean_dist(expr_by_hour[h]) for h in hours],
                "n_by_hour": [len(expr_by_hour[h]) for h in hours],
            },
            "zones": {
                "labels": zone_names,
                "visits": [sum(e["zone"] == z for e in events) for z in zone_names],
                "now": [sum(o["zones"].get(z, 0) for o in occ_now.values()) for z in zone_names] if occ_now else None,
            },
            "recent": list(reversed(events[-25:])),
        }


class Handler(BaseHTTPRequestHandler):
    store: Store

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                self._file(STATIC / "index.html", "text/html; charset=utf-8")
            elif url.path == "/chart.umd.min.js":
                self._file(STATIC / "chart.umd.min.js", "application/javascript")
            elif url.path == "/api/dates":
                self._json(self.store.dates())
            elif url.path == "/api/summary":
                q = parse_qs(url.query)
                date = q.get("date", [datetime.now(self.store.tz).date().isoformat()])[0]
                device = q.get("device", [""])[0] or None
                self._json(self.store.summary(date, device))
            else:
                self.send_error(404)
        except Exception:  # noqa: BLE001
            log.exception("Error en %s", self.path)
            self.send_error(500)

    def _file(self, path: Path, ctype: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug(fmt, *args)


def main() -> None:
    p = argparse.ArgumentParser(description="Dashboard local del prototipo")
    p.add_argument("--config", default=None)
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    Handler.store = Store(load_config(args.config))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    log.info("Dashboard en http://localhost:%d  (datos: %s)", args.port, Handler.store.path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
