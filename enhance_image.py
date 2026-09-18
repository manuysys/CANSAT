"""
CanSat La Base — Mejora de calidad de imagen con IA (super-resolución). DEFINITIVO.
Requisito DPD: "imágenes mejoradas en su calidad con ayuda de IA".

Uso:
    python enhance_image.py --image dataset/pruebas/PERSONAS_AUTOS.png
    python enhance_image.py --image dataset/pruebas/PERSONAS_AUTOS.png --model edsr_x2
"""

import argparse
import time
import urllib.request
from pathlib import Path

import cv2

# NOTA: los nombres de algoritmo van en MINÚSCULA (OpenCV lo exige)
MODELS = {
    "espcn_x2": ("espcn", "models/ESPCN_x2.pb",
                 "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x2.pb"),
    "fsrcnn_x2": ("fsrcnn", "models/FSRCNN_x2.pb",
                  "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x2.pb"),
    "lapsrn_x2": ("lapsrn", "models/LapSRN_x2.pb",
                  "https://raw.githubusercontent.com/fannymonori/TF-LapSRN/master/export/LapSRN_x2.pb"),
    "edsr_x2": ("edsr", "models/EDSR_x2.pb",
                "https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x2.pb"),
}


# Checksums SHA-256 de los modelos de super-resolución.
# ⚠ La versión anterior hacía urlretrieve() SIN checksum ni timeout: cualquier
#   respuesta (un 404 con HTML, un proxy cautivo, un archivo truncado) se
#   guardaba como si fuera un modelo, y recién fallaba al readModel().
# None = "descargalo y anotá el hash que imprime el script". No hay ningún hash
# inventado acá: un hash falso haría fallar la validación SIEMPRE, que es peor
# que no validar. Con --allow-download el script imprime el SHA-256 real; hay
# que pegarlo acá una vez y a partir de ahí queda verificado.
SHA256 = {
    "models/ESPCN_x2.pb":  None,
    "models/FSRCNN_x2.pb": None,
    "models/LapSRN_x2.pb": None,
    "models/EDSR_x2.pb":   None,
}


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(key, allow_download: bool = False):
    """
    Devuelve ``(algoritmo, ruta)`` del modelo de super-resolución.

    ⚠ CAMBIO DE COMPORTAMIENTO: **ya no descarga en runtime por defecto**.

    Antes hacía ``urllib.request.urlretrieve(url, path)`` contra
    ``raw.githubusercontent.com`` sin timeout, sin reintento y sin checksum.
    Eso está bien en una PC con wifi y pésimo el día del vuelo: sin red, el
    post-vuelo falla en el paso de EDSR, que es justo el que genera las
    imágenes "mejoradas con IA" que pide el DPD.

    Los ``.pb`` tienen que estar en ``models/`` antes de viajar. Para
    descargarlos en una PC con red:  ``--allow-download``.
    """
    algo, rel, url = MODELS[key]
    path = Path(rel)

    if path.exists():
        want = SHA256.get(rel)
        if want:
            got = _sha256(path)
            if got != want:
                raise RuntimeError(
                    f"{path}: checksum SHA-256 incorrecto.\n"
                    f"  esperado {want}\n  obtenido {got}\n"
                    f"  El archivo está corrupto o fue reemplazado. Borrá y volvé a bajar.")
        return algo, str(path)

    if not allow_download:
        raise FileNotFoundError(
            f"Falta el modelo de super-resolución: {path}\n"
            f"  No se descarga en runtime (en vuelo no hay red).\n"
            f"  · En una PC con red:  python enhance_image.py --image X --model {key} "
            f"--allow-download\n"
            f"  · URL oficial: {url}\n"
            f"  · Guardalo en {path} y verificá el SHA-256.\n"
            f"  NOTA: hace falta opencv-CONTRIB (cv2.dnn_superres). Con\n"
            f"        opencv-python-headless común esta funcionalidad no existe.")

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [DL] Bajando {path.name} desde {url}")
    import socket
    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(60)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CanSatLB135/1.0"})
        with urllib.request.urlopen(req) as r, path.open("wb"):
            data = r.read()
        if not data or data[:5] == b"<?xml" or data[:6] == b"<html>":
            raise RuntimeError(f"la respuesta no es un modelo (.pb), es "
                               f"{len(data)} bytes de texto/HTML. ¿URL caída?")
        path.write_bytes(data)
    finally:
        socket.setdefaulttimeout(old)
    print(f"  [DL] {len(data) / 1e6:.1f} MB · SHA-256 {_sha256(path)}")
    print("       Anotá ese hash en SHA256 (enhance_image.py) para validarlo la próxima.")
    return algo, str(path)


def make_compare(a, b, size=420):
    """Recorte central lado a lado: izquierda bicúbico, derecha IA."""
    h, w = a.shape[:2]
    # Antes asumía h,w >= size: con una imagen chica el slice quedaba vacío o
    # invertido y hconcat reventaba.
    if h < size or w < size:
        size = max(16, min(h, w))
    y, x = (h - size) // 2, (w - size) // 2
    comp = cv2.hconcat([a[y:y+size, x:x+size], b[y:y+size, x:x+size]])
    cv2.putText(comp, "BICUBIC", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(comp, "IA (SR)", (size + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return comp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Super-resolución CanSat")
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="espcn_x2", choices=list(MODELS))
    ap.add_argument("--allow-download", action="store_true",
                    help="permitir descargar el .pb si falta (sólo con red; NUNCA en vuelo)")
    args = ap.parse_args(argv)

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] No se pudo cargar {args.image}")
        return 1

    try:
        algo, model_path = ensure_model(args.model, allow_download=args.allow_download)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        return 1

    if not hasattr(cv2, "dnn_superres"):
        print("[ERROR] Este OpenCV no tiene cv2.dnn_superres.\n"
              "        Hace falta el flavour CONTRIB:\n"
              "          pip install opencv-contrib-python          (PC)\n"
              "          pip install opencv-contrib-python-headless (Raspberry)\n"
              "        requirements.txt no lo declaraba.")
        return 1
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(algo.lower(), 2)   # minúscula sí o sí

    print(f"  Entrada : {img.shape[1]}x{img.shape[0]}")
    t0 = time.time()
    up = sr.upsample(img)
    dt = time.time() - t0
    print(f"  Salida  : {up.shape[1]}x{up.shape[0]}  ({dt:.2f} s)")

    naive = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # La versión anterior hacía Path(args.image).with_name(Path(args.image).stem),
    # que es un no-op confuso (with_name(al mismo nombre) devuelve la misma ruta).
    base = Path(args.image)
    for suf, data in (("_sr.jpg", up), ("_bicubic.jpg", naive),
                      ("_compare.jpg", make_compare(naive, up))):
        cv2.imwrite(str(base.with_name(base.stem + suf)), data,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Guardado: {base.stem}_sr.jpg / _bicubic.jpg / _compare.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
