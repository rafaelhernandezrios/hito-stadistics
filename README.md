# camera-room-analytics

Analítica de personas con cámara para la sala común. Fase 1: prototipo en Mac M1 Pro.
El video se procesa en el dispositivo y solo se guardan eventos JSON; **ningún frame, recorte
de rostro ni embedding se guarda o transmite**.

## Requisitos

- **Python 3.11** (probado con 3.11; 3.12 debería funcionar, 3.13 aún no para todas las dependencias).
- Una cámara (webcam, cámara de la Mac, iPhone con Continuity Camera) o un archivo de video.
- ~2 GB libres para el entorno virtual (PyTorch es grande).
- Funciona en **macOS (Apple Silicon)**, **Linux** y **Windows**. La GPU se elige sola
  (`detector.device: auto` → `mps` en Mac, `cuda` con NVIDIA, si no `cpu`).

## Instalación (una vez por computadora)

```bash
git clone https://github.com/rafaelhernandezrios/hito-stadistics.git
cd hito-stadistics
```

**macOS / Linux**

```bash
# macOS: brew install python@3.11   ·   Ubuntu: sudo apt install python3.11 python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/setup_models.py
```

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python scripts/setup_models.py
```

> En Windows, `insightface` necesita compilarse: si `pip install` falla, instala
> [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> ("Desarrollo para el escritorio con C++") y repite.
>
> En Linux con GPU NVIDIA, `pip` instala PyTorch con CUDA (descarga grande). Para solo CPU:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` antes del `requirements.txt`.

`scripts/setup_models.py` descarga los modelos (no se suben al repo):
- `models/yolo11n.pt` (detector de personas, Ultralytics)
- `~/.insightface/models/buffalo_s/` (detección de rostro `det_500m`, `genderage` y `w600k_mbf` para re-id)
- `~/.hsemotion/enet_b0_8_best_vgaf.onnx` (expresión)

### Qué no se sube al repo

El `.gitignore` excluye el entorno virtual, los modelos, la base de datos `data/edge.db` (se crea sola al
correr el pipeline), cualquier video o imagen, y archivos `.env`. Para cambiar la configuración solo en tu
computadora sin tocar la del repo, copia `edge/config.yaml` a `edge/config.local.yaml` (ignorado) y úsalo con
`--config edge/config.local.yaml`.

## Uso

```bash
source .venv/bin/activate           # Windows: .venv\Scripts\Activate.ps1
python edge/main.py                    # cámara por defecto (la de la Mac / webcam), con vista previa (q para salir)
python edge/main.py --source 1         # iPhone vía Continuity Camera (el índice puede variar)
python edge/main.py --source video.mp4 # archivo de video
caffeinate -i python edge/main.py --no-preview   # sesión larga sin que la Mac se duerma
python edge/report.py --date 2026-10-21          # resumen de lo guardado
```

### Prueba de montaje con iPhone (Continuity Camera)

Requisitos: iPhone XR o posterior con iOS 16+, misma cuenta de Apple que la Mac, Wi-Fi y Bluetooth
activados y Continuity Camera activa en el iPhone (Ajustes > General > AirPlay y Continuidad).

1. Monta el iPhone **en horizontal, bloqueado y quieto**, a 2–2.5 m de altura con leve inclinación hacia abajo,
   apuntando hacia donde la gente mira de forma natural (ver la guía).
2. Encuentra su índice: `python edge/cameras.py` (muestra cada cámara con su número).
3. Corre la sesión a la resolución prevista para la Pi:
   `python edge/main.py --source <indice> --resolution 1920x1080`
4. En macOS, desactiva **Encuadre centrado** (Center Stage) en Centro de control > Efectos de vídeo:
   recorta y hace zoom solo, lo que mueve las zonas y falsea el tamaño de los rostros.
5. Al salir (`q`), el log imprime el **resumen de montaje**: porcentaje de rostros utilizables, motivos de
   rechazo (pequeño, perfil, borroso) y tamaño de rostro, más una estimación del tamaño en la Camera
   Module 3 Wide (102°), que ve los rostros más pequeños que el iPhone a la misma distancia.
   El criterio de la guía es rostro ≥ 60 px; los campos de visión en `mount_check` (config) son aproximados.

En la vista previa, el rostro se marca en naranja si es utilizable y en rojo si no, con su tamaño en px.

### Modo laboratorio: re-identificación (reconocer a quien vuelve)

```bash
python edge/main.py --reid
```

Solo para el laboratorio y **con consentimiento de quienes aparecen**: bajo la APPI un embedding
facial es un dato personal identificable (個人識別符号) en cualquier lugar. Sin `--reid` (y con
`reid.enabled: false` en config) el sistema es totalmente anónimo; no hay que cambiar código para volver.

- Cada track cerrado con ≥ 2 rostros utilizables se compara (ArcFace `w600k_mbf` de buffalo_s) con los
  visitantes del día; similitud coseno ≥ `match_threshold` (0.35) = misma persona.
- Los eventos añaden `visitor_id` ("v003", válido solo ese día), `visit_n` y `reid_sim`.
  El dashboard y `report.py` muestran entonces "personas distintas".
- Los embeddings **solo viven en memoria**: nunca van a SQLite, MQTT ni disco; el registro se borra a
  medianoche y al detener el pipeline (reiniciar el pipeline = empezar de cero ese día).
- Un track sin rostro utilizable queda "sin identificar" y sigue contando como visita.
- El umbral se validó con fotos (distintas personas ≤ 0.17, misma persona ≥ 0.94); con cámara real la
  misma persona dará similitudes más bajas: si ves a alguien partido en dos visitantes, baja el umbral;
  si dos personas se fusionan, súbelo.

### Varias cámaras en paralelo

```bash
python edge/main.py --sources 0,1                # Mac + iPhone, cada una con sus estadísticas
python edge/main.py --sources 0,1 --reid         # + IDs de visitante compartidos (laboratorio)
```

- Un solo proceso: cada cámara tiene su hilo de captura, su propio YOLO + ByteTrack y su agregador;
  el procesamiento va por turnos en el hilo principal (GPU MPS y ventanas de macOS no admiten varios hilos).
  Los modelos de rostro y atributos se comparten. Cada cámara abre su propia ventana de vista previa.
- `device_id` por cámara: con `--sources` se crean `<device_id>-cam1`, `-cam2`…; para nombres y zonas
  propios por cámara, usa la lista `cameras:` en config.yaml (ejemplo comentado allí).
- **Sin `--reid`** las cámaras son independientes: no hay forma anónima de saber que la persona de una
  es la misma que la de otra. La ocupación no se suma entre cámaras (pueden ver a la misma persona).
- **Con `--reid`** todas comparten un registro de visitantes en memoria: `v003` es la misma persona en
  cualquier cámara y lo sigue siendo aunque salga de todas y vuelva ese día.
  - El ID se asigna **en vivo** en cuanto un track reúne 2 rostros buenos (se ve en las ventanas) y se
    confirma al cerrar el track con todos sus rostros; si con más rostros encaja claramente mejor con
    otro visitante, se corrige.
  - Dos personas en el mismo frame de una cámara nunca reciben el mismo ID; la misma persona sí puede
    estar a la vez en dos cámaras con el mismo ID.
  - Quien no muestra un rostro utilizable en una cámara (de espaldas, cámara cenital) no se puede enlazar
    desde esa cámara.
- Probado con dos videos simulando dos cámaras: 5 de 5 personas identificables con el mismo ID en ambas,
  tanto apareciendo a la vez como en momentos distintos.
- Para cámaras en dispositivos separados (dos Pi), los embeddings tendrían que viajar a un registro central
  por la red; eso choca con "nada biométrico sale del dispositivo" y queda fuera de este prototipo.

### Dashboard local

En otra pestaña de Terminal, mientras corre el pipeline:

```bash
source .venv/bin/activate
python dashboard/server.py        # abrir http://localhost:8000
```

Lee la misma SQLite en modo solo lectura y se actualiza cada 10 s. Usa solo la librería estándar
y Chart.js copiado en `dashboard/static/` (funciona sin internet). Es un puente hasta Grafana (Fase 3).

La primera vez, macOS pide permiso de cámara para la app desde la que se ejecuta (Terminal, VS Code…).

## Estructura

```
edge/
  main.py        punto de entrada: una o varias cámaras
  pipeline.py    lectura de cámaras (hilo por cámara) y pipeline por cámara
  detector.py    YOLO11n + ByteTrack, detrás de la interfaz PersonTracker (Hailo irá aquí en la Pi)
  faces.py       rostro (SCRFD) dentro de cada persona + calidad (tamaño ≥ 60 px, pose frontal, nitidez)
  attributes.py  edad y género (InsightFace genderage), expresión (hsemotion-onnx)
  aggregator.py  estado por track, mejores N rostros, eventos y snapshots de ocupación
  publisher.py   SQLite (buffer local) + MQTT QoS 1 opcional
  report.py      resumen rápido de SQLite
  cameras.py     identifica el índice de cada cámara (Mac, iPhone)
  mount.py       estadísticas de montaje (tamaño de rostro, equivalente en la Pi)
  reid.py        re-identificación del modo laboratorio (embeddings solo en memoria)
  bytetrack.yaml parámetros del tracker
  settings.py    carga de config y hora local (+09:00)
  config.yaml    device_id, zonas, umbrales
dashboard/
  server.py      API JSON sobre la SQLite + sirve la página
  static/        index.html y Chart.js
scripts/
  setup_models.py  descarga de modelos (una vez por computadora)
models/          pesos de YOLO (descargados, no versionados)
data/            SQLite de eventos (generada, no versionada)
backend/         Fase 3 (Mosquitto, TimescaleDB, Grafana)
```

## Decisiones de implementación

- **Mejores frames sin guardar recortes:** cuando un rostro entra en el top-N de calidad de su track,
  se calculan sus atributos en ese momento y solo se guardan los números. Equivale a guardar los N
  mejores recortes, pero sin tener imágenes en memoria.
- **Zonas** en coordenadas normalizadas (0–1) para no depender de la resolución. La zona de una persona
  es donde están sus pies (centro inferior de la caja); en el evento va la zona donde pasó más tiempo.
- **Bandas de edad:** se añadió `<18` a las bandas del documento.
- **`age_conf`:** fracción de las estimaciones del track que caen en la banda reportada.
  **`gender_conf`:** probabilidad media de la clase ganadora.
- **Expresión:** se reportan `neutral, happy, sad, surprise` y el resto suma en `other` (configurable).
- Tracks de menos de `min_dwell_s` (2 s) se descartan como ruido.
- **Tracking sin fragmentar** (para que una persona no cuente como varias visitas):
  - El detector pasa detecciones desde 0.1 y `edge/bytetrack.yaml` decide qué es persona; así funciona
    la segunda etapa de ByteTrack, que recupera a alguien girado o tapado a medias. ByteTrack recuerda
    un track perdido 60 frames (~5 s).
  - Si aun así aparece un ID nuevo, el agregador lo **re-enlaza** con un track perdido hace menos de
    `relink_window_s` (12 s) si está cerca (≤ 0.6 alturas de caja) y tiene tamaño parecido. Solo usa
    posición y tamaño, nada del rostro, así que no es re-identificación.
  - Un track se cierra a los 3 s solo si salió por un borde lateral **moviéndose hacia él**. Si no, se
    espera la ventana de re-enlace. La permanencia usa la última vez que se vio, así que esa espera no
    la infla; solo retrasa la emisión del evento.
  - Limitación: si una persona se va por el centro del encuadre y otra ocupa el mismo sitio en menos de
    12 s, se cuentan como una sola.
- Al terminar, el log muestra cuántos tracks se abrieron, re-enlazaron, descartaron y emitieron.
- Con un archivo de video, el tiempo de los eventos se calcula a partir del propio video.
