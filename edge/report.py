"""Resumen rápido de lo guardado en SQLite (útil mientras no existe Grafana).

Uso: python edge/report.py [--date 2026-10-21]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import Counter

from settings import load_config, resolve_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="AAAA-MM-DD (por defecto, todo)")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    db = sqlite3.connect(resolve_path(cfg["storage"]["sqlite_path"]))
    like = f"{args.date}%" if args.date else "%"
    events = [json.loads(r[0]) for r in db.execute(
        "SELECT payload FROM person_events WHERE t_enter LIKE ? ORDER BY t_enter", (like,))]
    occ = [json.loads(r[0]) for r in db.execute(
        "SELECT payload FROM occupancy WHERE ts LIKE ? ORDER BY ts", (like,))]

    print(f"Personas (tracks cerrados): {len(events)}")
    if events:
        dwell = [e["dwell_s"] for e in events]
        with_face = [e for e in events if e["face_frames_used"]]
        print(f"Permanencia: media {statistics.mean(dwell):.0f}s, mediana {statistics.median(dwell):.0f}s")
        print(f"Con rostro utilizable: {len(with_face)}/{len(events)} ({100 * len(with_face) / len(events):.0f}%)")
        print("Rangos de edad:", dict(Counter(e["age_range"] for e in with_face)))
        print("Género estimado:", dict(Counter(e["gender_est"] for e in with_face)))
        print("Zonas:", dict(Counter(e["zone"] for e in events)))
        devices = Counter(e["device_id"] for e in events)
        if len(devices) > 1:
            print("Visitas por cámara:", dict(devices))
        reid = [e for e in events if "visitor_id" in e]
        if reid:
            ids = {e["visitor_id"] for e in reid if e["visitor_id"]}
            cams: dict[str, set[str]] = {}
            for e in reid:
                if e["visitor_id"]:
                    cams.setdefault(e["visitor_id"], set()).add(e["device_id"])
            cross = sum(1 for v in cams.values() if len(v) > 1)
            print(f"Re-ID (laboratorio): {len(ids)} personas distintas en {len(reid)} visitas "
                  f"({cross} vistas en varias cámaras, {sum(not e['visitor_id'] for e in reid)} sin identificar)")
        if with_face:
            keys = with_face[0]["expression_dist"].keys()
            mean = {k: round(statistics.mean(e["expression_dist"][k] for e in with_face), 2) for k in keys}
            print("Expresión promedio:", mean)
    print(f"Snapshots de ocupación: {len(occ)}")
    if occ:
        print(f"Ocupación máxima: {max(o['total'] for o in occ)}  promedio: {statistics.mean(o['total'] for o in occ):.1f}")


if __name__ == "__main__":
    main()
