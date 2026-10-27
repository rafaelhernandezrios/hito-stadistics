"""Loop principal: captura -> personas + tracking -> rostro -> atributos -> agregación -> SQLite/MQTT.

Uso:
    python edge/main.py                          # cámara(s) de config.yaml (0 = cámara de la Mac)
    python edge/main.py --source 1               # otra cámara (p. ej. iPhone con Continuity Camera)
    python edge/main.py --sources 0,1            # dos cámaras en paralelo
    python edge/main.py --sources 0,1 --reid     # dos cámaras con IDs de visitante compartidos (laboratorio)
    python edge/main.py --source video.mp4       # archivo de video (pruebas y validación)
    python edge/main.py --no-preview --duration 600

Ningún frame se guarda en disco: la vista previa solo existe en memoria.
"""
from __future__ import annotations

import argparse
import logging
import time

import cv2

from pipeline import CameraPipeline, SharedModels, draw_preview
from publisher import Publisher
from settings import Clock, camera_configs, load_config

log = logging.getLogger("edge")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline edge de analítica de sala")
    p.add_argument("--config", default=None, help="ruta a config.yaml")
    p.add_argument("--source", default=None, help="índice de cámara o ruta de video (una cámara)")
    p.add_argument("--sources", default=None, help="varias fuentes separadas por coma, p. ej. 0,1")
    p.add_argument("--resolution", default=None, help="resolución de captura, p. ej. 1920x1080")
    p.add_argument("--reid", action="store_true",
                   help="activa la re-identificación (solo laboratorio, con consentimiento)")
    p.add_argument("--no-preview", action="store_true", help="sin ventana de vista previa")
    p.add_argument("--duration", type=float, default=0, help="segundos a ejecutar (0 = sin límite)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    clock = Clock(cfg.get("timezone", "Asia/Tokyo"))
    preview = cfg["preview"]["enabled"] and not args.no_preview
    reid_enabled = bool((cfg.get("reid") or {}).get("enabled") or args.reid)

    sources = args.sources.split(",") if args.sources else ([args.source] if args.source is not None else None)
    cams = camera_configs(cfg, [s.strip() for s in sources] if sources else None, args.resolution)

    shared = SharedModels(cfg, reid_enabled, clock)
    if reid_enabled:
        log.warning("=" * 70)
        log.warning("MODO RE-IDENTIFICACIÓN ACTIVO: solo laboratorio y con consentimiento.")
        log.warning("IDs de visitante compartidos entre %d cámara(s). Embeddings solo en memoria;", len(cams))
        log.warning("se borran cada día y al detener. En lugar público: sin --reid y reid.enabled: false")
        log.warning("=" * 70)

    pub = Publisher(cfg, cams[0]["device_id"])
    start = time.time()
    pipes = [CameraPipeline(cam, cfg, shared, pub, clock, start) for cam in cams]
    if all(p.done for p in pipes):
        pub.close()
        return
    log.info("Pipeline iniciado con %d cámara(s): %s · preview=%s",
             len(pipes), ", ".join(p.device_id for p in pipes), preview)

    try:
        while any(not p.done for p in pipes):
            wall = time.time()
            if args.duration and max(p.last_now for p in pipes) - start >= args.duration:
                break
            worked = False
            for p in pipes:
                if p.done:
                    continue
                out = p.poll(wall)
                if out is None:
                    continue
                worked = True
                if preview:
                    frame, active, shown = out
                    draw_preview(frame, active, shown, p.agg, p.fps_ema)
                    cv2.imshow(f"{p.device_id} (q para salir)", frame)
            pub.flush(wall)
            if preview and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
            if not worked:
                time.sleep(0.003)
    except KeyboardInterrupt:
        log.info("Interrumpido por el usuario")
    finally:
        for p in pipes:
            p.finish()
        pub.close()
        if preview:
            cv2.destroyAllWindows()
        if shared.registry is not None:
            log.info("Re-ID: %d visitantes distintos hoy (%d vistos en más de una cámara). "
                     "Registro de embeddings borrado.",
                     shared.registry.unique_today(), shared.registry.cross_camera_today())
            shared.registry.forget()


if __name__ == "__main__":
    main()
