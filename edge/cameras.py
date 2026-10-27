"""Identifica qué índice de OpenCV corresponde a cada cámara (Mac, iPhone…).

OpenCV no da nombres de cámara, así que este script:
  1. muestra los nombres que ve macOS;
  2. abre cada índice y enseña un frame en vivo con el número, para que reconozcas cuál es.

Uso: python edge/cameras.py   (cualquier tecla pasa a la siguiente cámara; nada se guarda)
"""
from __future__ import annotations

import subprocess
import time

import cv2


def macos_camera_names() -> list[str]:
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    names = []
    for line in out.splitlines():
        s = line.strip()
        if s.endswith(":") and line.startswith("    ") and not line.startswith("      "):
            names.append(s[:-1])
    return names


def main() -> None:
    names = macos_camera_names()
    print("Cámaras que ve macOS:", ", ".join(names) if names else "(ninguna)")
    if not any("iphone" in n.lower() for n in names):
        print("  El iPhone no aparece: acércalo a la Mac, bloquéalo y ponlo en horizontal (ver README).")
    print()

    found = 0
    for idx in range(6):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        frame = None
        t0 = time.time()
        while time.time() - t0 < 4:  # el iPhone tarda un par de segundos en entregar imagen
            ok, f = cap.read()
            if ok and f is not None and f.mean() > 1:
                frame = f
                break
        if frame is None:
            print(f"Índice {idx}: abre pero no entrega imagen")
            cap.release()
            continue
        found += 1
        h, w = frame.shape[:2]
        print(f"Índice {idx}: {w}x{h}  -> mira la ventana y pulsa una tecla para seguir")
        title = f"Indice {idx} ({w}x{h}) - pulsa una tecla"
        while True:
            ok, f = cap.read()
            if ok:
                view = cv2.resize(f, (960, int(960 * f.shape[0] / f.shape[1])))
                cv2.putText(view, f"INDICE {idx}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                cv2.imshow(title, view)
            if cv2.waitKey(30) != -1:
                break
        cv2.destroyWindow(title)
        cv2.waitKey(1)
        cap.release()

    print()
    print(f"{found} cámara(s) con imagen. Usa el índice del iPhone así:")
    print("  python edge/main.py --source <indice> --resolution 1920x1080")


if __name__ == "__main__":
    main()
