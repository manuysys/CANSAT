"""
CanSat La Base — Post-vuelo (estación terrena de referencia).

Sobre la carpeta de frames recuperada genera ``entrega/``:
  · telemetría del pipeline de misión
  · ``corridor_map.jpg``
  · ``enhanced/``  EDSR sobre los frames más nítidos
  · ``ens_seg/``  overlays de la segunda pasada de alta calidad (siempre esta
    carpeta; ``--ensemble`` elige la MEZCLA de modelos, no el directorio: antes
    sin ``--ensemble`` escribía en ``b5_seg/`` y la estación no lo veía porque
    su bucket mira sólo ``ens_seg/``)
  · ``masks/``  máscaras de clase por frame (PNG gris, consulta terrestre)
  · ``summary.json``  ← CONTRATO con la app de visualización

El schema de ``summary.json`` está definido en ``cansat/summary.py`` y es el
MISMO que usa ``tools/make_demo_mission.py`` de la estación terrena. Antes cada
productor escribía un formato distinto (``alertas`` era ``list[str]`` acá y
``list[dict]`` allá; ``archivos`` eran rutas acá y conteos allá) y el frontend
tuvo que tipar el campo como ``unknown[]``.

Uso:
    python post_flight.py --frames dataset/loveda_raw/Test/Urban/images_png --n 12
    python post_flight.py --frames ... --b5 --ensemble     # segunda pasada B5
    python post_flight.py --frames ... --no-edsr           # iterar rápido

────────────────────────────────────────────────────────────────────────────
ARREGLOS DE LA AUDITORÍA (2026-09-16)
────────────────────────────────────────────────────────────────────────────
Este script **no podía funcionar**. Tenía cinco defectos independientes:

a) ``B5_ONNX`` apuntaba a ``cansat_seg_terrain_segformer_b5_640.onnx``, que NO
   existe en el repo (hay ``_320`` y ``_512``). El ``if args.b5 and
   Path(B5_ONNX).exists()`` era siempre falso, así que **todo el bloque
   ensemble/CRF se salteaba en silencio**, sin un solo warning, y
   ``entrega/ens_seg/`` y ``terrain_b5_por_frame`` quedaban vacíos.
b) ``sorted(rows, key=lambda q: int(q["src"]))`` → ``ValueError`` en cuanto el
   ``src`` no es numérico. En vuelo real ``--camera`` produce ``cam_001``.
c) ``sliding_logits()`` con ``H < win`` metía un índice negativo en ``ys``, y el
   acumulado quedaba desalineado (tile de tamaño incorrecto).
d) ``prep_tile(size=512)`` forzaba 512×512 aunque el modelo se llamara ``_640``:
   ONNX Runtime habría rechazado el tensor.
e) ``--ensemble`` usaba ``sess_t.run(...)`` sin verificar ``sess_t is not None``.

Además: el tamaño de entrada ahora se lee del propio ONNX, el ``summary.json``
sale del schema canónico, y el daño se calcula con el MISMO consenso de modelos
que el pipeline de vuelo (antes usaba un solo modelo mientras la UI anunciaba
"consenso de 3 modelos").
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
from cansat import indices as IDX
from cansat import masks as MK
from cansat import nodata as ND
from cansat import onnxio
from cansat import preprocess as PP
from cansat import summary as SUM
from cansat.crf import dense_crf
from enhance_image import ensure_model

# ── Artefactos de entrada ─────────────────────────────────────────────── #
DAMAGE = "outputs/cansat_damage3_mobilenetv2.onnx"
# F2 (2026-09-18): el two-stage de vuelo es el adaptado a UAV con RescueNet.
# El xBD puro (cansat_damage_v3.onnx) queda como alternativa cross-event.
DAMAGE2 = "outputs/cansat_damage_v3_bal.onnx"
# El siamés está DESACTIVADO por defecto: los pesos del repo están marcados
# ROTO en MODELS.yaml (alucinaban daño sin cambio). Se activa con --siamese-onnx.
SIAMESE = ""
FLOOD_ONNX = "outputs/cansat_flood_specialist_224.onnx"   # F3: 224 px, IoU 0.489
FIRE_ONNX = "outputs/cansat_fire_smoke.onnx"              # F3: fuego/humo
SEVERITY_ONNX = "outputs/cansat_severity.onnx"            # F2b: colapso medido
TERRAIN_V2 = "outputs/cansat_seg_terrain_v2.onnx"
# Vías para la consulta terrestre (NO vuela: sólo pase post-vuelo). Gate medido:
# IoU 0.515 en FloodNet val (dominio de entrenamiento); cross-domain LoveDA 0.169.
VIAS_ONNX = "outputs/cansat_vias_floodnet.onnx"

# Candidatos de SegFormer B5 en orden de preferencia. El anterior hardcodeaba
# uno que NO existe en el repo y se salteaba todo en silencio.
B5_CANDIDATES = (
    "outputs/cansat_seg_terrain_segformer_b5_512.onnx",
    "outputs/cansat_seg_terrain_segformer_b5_320.onnx",
    "outputs/cansat_seg_terrain_segformer_b5_640.onnx",
    "outputs/cansat_seg_terrain_segformer.onnx",
)

TELEMETRY_CSV = "outputs/mission/telemetry.csv"
CORRIDOR_JPG = "outputs/corridor_map.jpg"

# Paleta BGR — FUENTE ÚNICA. Estaba duplicada en inference.py y acá, y aunque
# los valores coincidían, dos copias de una paleta es una invitación a que
# dejen de coincidir. analyze_stress.py hace el camino inverso (color → clase).
COLORS = [
    (0, 200, 0),      # vegetation
    (180, 100, 0),    # building
    (255, 100, 0),    # water
    (0, 200, 255),    # bare_ground
    (128, 128, 128),  # other
]


# ══════════════════════════════════════════════════════════════════════ #
#  Utilidades
# ══════════════════════════════════════════════════════════════════════ #
def color_seg(seg, img):
    overlay = np.zeros((*seg.shape, 3), np.uint8)
    for i, c in enumerate(COLORS):
        overlay[seg == i] = c
    full = cv2.resize(overlay, (img.shape[1], img.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(img, 0.45, full, 0.55, 0)


def sliding_logits(sess: onnxio.OnnxModel, img: np.ndarray,
                   win: int = 512, stride: int = 256) -> np.ndarray:
    """
    Barre el frame nativo con ventanas solapadas y cose con pesos de Hann.

    ⚠ ARREGLADO: la versión anterior hacía ``ys.append(H - win)`` sin clampear.
      Con ``H < win`` eso agregaba un **índice negativo**, el tile salía de
      tamaño incorrecto, ``prep_tile`` lo reescalaba en silencio y el acumulado
      ``acc[:, y:y+win, ...]`` quedaba desalineado. Ahora las posiciones se
      clampean a ``[0, max(H-win, 0)]`` y se deduplican.
    """
    H, W = img.shape[:2]
    n_cls = sess.n_classes or 5
    acc = np.zeros((n_cls, H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)

    ramp = np.hanning(max(2, win)).astype(np.float32) + 1e-4
    w2 = np.outer(ramp, ramp)

    def positions(total: int) -> list[int]:
        last = max(total - win, 0)
        pos = list(range(0, last + 1, stride))
        if not pos or pos[-1] != last:
            pos.append(last)
        # Deduplicar y clampear: nunca negativos, nunca fuera de rango.
        return sorted({max(0, min(p, last)) for p in pos})

    for y in positions(H):
        for x in positions(W):
            tile = img[y:y + win, x:x + win]
            th, tw = tile.shape[:2]
            lg = sess.run({"input": PP.preprocess_bgr(tile, win)})[0]
            # El modelo devuelve win×win; recortamos lo que sobresale del frame.
            lg = lg[:, :th, :tw]
            ww = w2[:th, :tw]
            acc[:, y:y + th, x:x + tw] += lg * ww
            wsum[y:y + th, x:x + tw] += ww
    return acc / np.maximum(wsum, 1e-6)[None]


def read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No existe {csv_path}.\n"
            f"  → Corré primero mission_pipeline.py (este script la invoca, así\n"
            f"     que si llegaste acá es porque el pipeline no escribió nada)."
        )
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def frame_sort_key(row: dict, idx: int) -> tuple:
    """
    Orden estable de frames.

    ⚠ ARREGLADO: antes era ``int(q["src"])``, que lanza ``ValueError`` con los
      ``cam_001`` que produce ``mission_pipeline.py --camera`` — o sea, en el
      vuelo real. Ahora se usa ``t_s`` cuando existe y el ``src`` como texto.
    """
    try:
        t = float(row.get("t_s") or "")
    except (TypeError, ValueError):
        t = float("inf")
    return (t, str(row.get("src") or ""), idx)


def _guardar_mascaras(
    destino: Path,
    src: str,
    shape_nativa: tuple[int, int],
    mascaras: dict[str, tuple[np.ndarray | None, np.ndarray | None]],
) -> int:
    """
    Persiste máscaras de clase a resolución NATIVA (nearest) para la consulta.

    ``mascaras``: nombre → (mapa de índices a resolución de modelo, máscara
    booleana de válidos a la misma resolución). Los píxeles inválidos van a
    ``MK.NODATA`` para que el motor de consultas no los cuente como clase.
    """
    h, w = shape_nativa
    n = 0
    for nombre, (m, val) in mascaras.items():
        if m is None:
            continue
        out = np.asarray(m).astype(np.uint8)
        if val is not None:
            v = np.asarray(val).astype(bool)
            if v.shape != out.shape:
                v = cv2.resize(v.astype(np.uint8), (out.shape[1], out.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            out = out.copy()
            out[~v] = MK.NODATA
        if out.shape != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
        MK.save_mask(destino / f"{src}_{nombre}.png", out)
        n += 1
    return n


def frame_path(frames_dir: Path, src: str) -> Path | None:
    """Busca el frame original por ``src`` con cualquier extensión de imagen."""
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
        cand = frames_dir / f"{src}{ext}"
        if cand.is_file():
            return cand
    hits = sorted(frames_dir.glob(f"{src}.*"))
    return hits[0] if hits else None


def resolve_b5(explicit: str | None) -> Path | None:
    """Elige el primer SegFormer B5 disponible, avisando si el pedido no está."""
    cands = (explicit,) if explicit else B5_CANDIDATES
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    print(f"  [WARN] No encontré ningún SegFormer B5 entre: "
          f"{', '.join(x for x in cands if x)}")
    print("         La segunda pasada de alta calidad queda desactivada.")
    print("         Generalo con: python export_onnx.py --arch segformer   (ver MODELS.yaml)")
    return None


# ══════════════════════════════════════════════════════════════════════ #
#  Main
# ══════════════════════════════════════════════════════════════════════ #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Post-vuelo CanSat LB135",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--frames", required=True, help="carpeta con los frames del vuelo")
    ap.add_argument("--n", type=int, default=12, help="cantidad de frames a procesar")
    ap.add_argument("--telemetry", default=TELEMETRY_CSV,
                    help="CSV del pipeline (default: el del contrato)")
    ap.add_argument("--b5", action="store_true",
                    help="2ª pasada de alta calidad con SegFormer B5")
    ap.add_argument("--b5-onnx", default=None, help="forzar un ONNX de B5 concreto")
    ap.add_argument("--ensemble", action="store_true",
                    help="ensemble B5 + destilado + flood specialist")
    ap.add_argument("--sliding", action="store_true",
                    help="sliding-window a resolución nativa (default ON con --b5)")
    ap.add_argument("--no-sliding", action="store_true",
                    help="usar la pasada global sin sliding-window")
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--temporal-alpha", type=float, default=0.6,
                    help="suavizado temporal (1 = solo frame actual)")
    ap.add_argument("--crf-iters", type=int, default=5)
    ap.add_argument("--edsr-frames", type=int, default=3,
                    help="cuántos frames pasar por EDSR (es lento: minutos c/u)")
    ap.add_argument("--no-edsr", action="store_true", help="saltear EDSR (iterar rápido)")
    ap.add_argument("--no-masks", action="store_true",
                    help="no persistir las máscaras de clase por frame "
                         "(entrega/masks/, consulta terrestre)")
    ap.add_argument("--no-enhance", action="store_true",
                    help="no aplicar denoise+unsharp en el pipeline de misión "
                         "(el --enhance estaba hardcodeado y no se podía apagar)")
    ap.add_argument("--no-pipeline", action="store_true",
                    help="no re-correr mission_pipeline.py (usar el CSV que ya existe)")
    ap.add_argument("--siamese-onnx", default=None,
                    help="siamés de cambio pre/post. Desactivado por defecto: los "
                         "pesos del repo están ROTO (ver MODELS.yaml).")
    ap.add_argument("--vias-onnx", default=None,
                    help="especialista de vías para la consulta terrestre "
                         f"(default: {VIAS_ONNX} si existe; sólo post-vuelo)")
    ap.add_argument("--pop-density", type=float, default=1500.0,
                    help="hab/km² para la estimación de pérdidas humanas")
    ap.add_argument("--occupancy", type=float, default=0.6)
    ap.add_argument("--collapse-frac", type=float, default=0.3)
    ap.add_argument("--fatality-ratio", type=float, default=0.1)
    args = ap.parse_args(argv)

    frames_dir = Path(args.frames)
    if not frames_dir.is_dir():
        print(f"[ERROR] No existe la carpeta de frames: {frames_dir}")
        return 1

    entrega = Path("entrega")
    enh_dir = entrega / "enhanced"
    enh_dir.mkdir(parents=True, exist_ok=True)
    # Máscaras de clase (consulta terrestre). Se limpia el directorio: si el
    # post-vuelo se re-corre con menos frames, no deben quedar máscaras viejas
    # sin telemetría que las respalde.
    masks_dir = entrega / "masks"
    if not args.no_masks:
        shutil.rmtree(masks_dir, ignore_errors=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

    # ── [1/5] Pipeline de misión ────────────────────────────────────────
    if args.no_pipeline:
        print("[1/5] Usando la telemetría existente (--no-pipeline).")
    else:
        print("[1/5] Pipeline de misión...")
        # ⚠ --enhance ya no está hardcodeado (--no-enhance lo apaga) y
        # --temporal-alpha se propaga al pipeline. Antes el comentario decía
        # que los flags se propagaban, pero el comando los ignoraba.
        cmd = [sys.executable, "mission_pipeline.py",
               "--folder", str(frames_dir), "--frames", str(args.n),
               "--shuffle", "--overwrite",
               "--temporal-alpha", str(args.temporal_alpha)]
        if not args.no_enhance:
            cmd.append("--enhance")
        subprocess.run(cmd, check=True)

    rows = read_rows(Path(args.telemetry))
    if not rows:
        print(f"[ERROR] {args.telemetry} existe pero no tiene filas.")
        return 1
    print(f"      {len(rows)} frames de telemetría.")

    # ── [2/5] Mapa de corredor ──────────────────────────────────────────
    print("[2/5] Mapa de corredor...")
    subprocess.run([sys.executable, "corridor_map.py", "--csv", args.telemetry],
                   check=False)
    rutas: dict[str, str | None] = {}
    if Path(CORRIDOR_JPG).is_file():
        shutil.copy(CORRIDOR_JPG, entrega / "corridor_map.jpg")
        rutas["corridor_map"] = "entrega/corridor_map.jpg"

    # ── [3/5] EDSR sobre los frames más nítidos ─────────────────────────
    enhanced: list[str] = []
    if args.no_edsr:
        print("[3/5] EDSR salteado (--no-edsr).")
    elif not (hasattr(cv2, "dnn_superres")
              and hasattr(cv2.dnn_superres, "DnnSuperResImpl_create")):
        # ⚠ El wheel no-contrib expone cv2.dnn_superres VACÍO: mirar solo el
        # módulo dejaba pasar el guard y reventaba con AttributeError.
        print("[3/5] [WARN] Este OpenCV no tiene cv2.dnn_superres (falta el flavour "
              "contrib). EDSR salteado; el resto del post-vuelo sigue.")
    else:
        algo = model_path = None
        try:
            algo, model_path = ensure_model("edsr_x2")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[3/5] [WARN] EDSR desactivado: {e}")
        if algo is None:
            pass
        else:
            print(f"[3/5] EDSR sobre los {args.edsr_frames} frames más nítidos "
                  f"(tarda varios minutos)...")
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(model_path)
            sr.setModel(algo.lower(), 2)
            sharpest = sorted(
                enumerate(rows),
                key=lambda ir: -float(ir[1].get("sharp") or 0.0))[:args.edsr_frames]
            for _i, r in sharpest:
                fp = frame_path(frames_dir, str(r.get("src")))
                if fp is None:
                    continue
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                t0 = time.time()
                up = sr.upsample(img)
                dst = enh_dir / f"{r['src']}_edsr.jpg"
                cv2.imwrite(str(dst), up, [cv2.IMWRITE_JPEG_QUALITY, 92])
                enhanced.append(dst.as_posix())
                print(f"      {r['src']} ({time.time() - t0:.0f} s)")

    # ── [4/5] Daño por frame — MISMO consenso que el vuelo ──────────────
    # ⚠ ARREGLADO: antes se calculaba con UN solo modelo (sess_d) mientras el
    # frontend anunciaba "consenso de 3 modelos". Ahora corren los tres (más el
    # flood si está) y se aplica cansat.indices.diagnose, idéntico al pipeline.
    #
    # ⚠ También se unificó la SEMÁNTICA de los porcentajes con el vuelo:
    #   · se dividen por píxeles VÁLIDOS (borde negro de tiles fuera), y
    #   · el two-stage se enmascara con los edificios que predice el modelo de
    #     terreno (antes se dividía por el frame completo: daba ~0 siempre y
    #     no coincidía con lo que el vuelo guardaba en el CSV).
    print("[4/5] Consenso de daño por frame...")
    sess_d = onnxio.load_required(DAMAGE, "daño principal")
    sess_d2 = onnxio.load_optional(DAMAGE2, "daño two-stage")
    sess_siam = onnxio.load_optional(args.siamese_onnx, "siamés") if args.siamese_onnx else None
    sess_f = onnxio.load_optional(FLOOD_ONNX, "flood specialist")
    sess_fire = onnxio.load_optional(FIRE_ONNX, "fuego/humo")
    sess_sev = onnxio.load_optional(SEVERITY_ONNX, "severidad")
    sess_terr = onnxio.load_optional(TERRAIN_V2, "terreno (para máscaras)")
    sess_vias = onnxio.load_optional(args.vias_onnx or VIAS_ONNX,
                                     "vías (consulta terrestre)")
    if not sess_terr:
        print("  [WARN] sin el modelo de terreno: el daño se calcula sin máscara "
              "de edificios ni de píxeles válidos (no comparable con el vuelo).")

    base_tensor = None
    if sess_siam and Path("outputs/baseline.png").is_file():
        b0 = cv2.imread("outputs/baseline.png")
        if b0 is not None:
            base_tensor = PP.preprocess_bgr(b0, 320)
        else:
            sess_siam = None
    else:
        sess_siam = None

    dan: dict[str, float] = {}
    diag_por_frame: dict[str, str] = {}
    fuego: dict[str, float] = {}
    humo: dict[str, float] = {}
    colapso: dict[str, float] = {}
    for r in rows:
        src = str(r.get("src"))
        fp = frame_path(frames_dir, src)
        if fp is None:
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        tensor = PP.preprocess_bgr(img, 320)

        # Máscaras coherentes con el vuelo: válidos (borde negro) y edificios.
        if sess_terr:
            seg_t = np.argmax(sess_terr.run({"input": tensor})[0], axis=0)
            valid = ND.valid_at_size(ND.border_mask(img, 15), 320)
            bmask = (seg_t == 1) & valid
        else:
            valid = np.ones((320, 320), dtype=bool)
            bmask = np.zeros((320, 320), dtype=bool)
        n_valid = max(1, int(valid.sum()))

        pd1 = np.argmax(sess_d.run({"input": tensor})[0], axis=0)
        pct_dan = float(((pd1 == 2) & valid).sum()) / n_valid * 100.0

        pd2 = pf = pfr = psv = pvias = None
        pct_dan2 = pct_siam = 0.0
        if sess_d2:
            pd2 = np.argmax(sess_d2.run({"input": tensor})[0], axis=0)
            pct_dan2 = float(((pd2 == 2) & bmask).sum()) / n_valid * 100.0
        if sess_siam and base_tensor is not None:
            ps = np.argmax(sess_siam.run({"pre": base_tensor, "post": tensor})[0], axis=0)
            pct_siam = float(((ps == 2) & valid).sum()) / n_valid * 100.0

        pct_flood = pct_fw = None
        if sess_f:
            pf = np.argmax(sess_f.run({"input": tensor})[0], axis=0)
            pct_flood = float(((pf == 1) & valid).sum()) / n_valid * 100.0
            pct_fw = float(((pf == 2) & valid).sum()) / n_valid * 100.0

        # F3/F2b: fuego/humo y severidad (cada modelo con su propio tamaño).
        pct_fire = pct_smoke = colapso_pct = None
        if sess_fire:
            sz = sess_fire.size_px or 320
            t_f = tensor if sz == 320 else PP.preprocess_bgr(img, sz)
            pfr = np.argmax(sess_fire.run({"input": t_f})[0], axis=0)
            pct_fire = float((pfr == 1).mean() * 100.0)
            pct_smoke = float((pfr == 2).mean() * 100.0)
        if sess_sev:
            sz = sess_sev.size_px or 320
            t_s = tensor if sz == 320 else PP.preprocess_bgr(img, sz)
            psv = np.argmax(sess_sev.run({"input": t_s})[0], axis=0)
            n_ed = int((psv >= 1).sum())
            if n_ed:
                colapso_pct = float((psv >= 3).sum()) / n_ed * 100.0
        if sess_vias:
            sz = sess_vias.size_px or 320
            t_v = tensor if sz == 320 else PP.preprocess_bgr(img, sz)
            pvias = np.argmax(sess_vias.run({"input": t_v})[0], axis=0)
        if pct_fire is not None:
            fuego[src] = round(pct_fire, 1)
            humo[src] = round(pct_smoke, 1)
        if colapso_pct is not None:
            colapso[src] = round(colapso_pct, 1)

        if not args.no_masks:
            _guardar_mascaras(masks_dir, src, img.shape[:2], {
                "terreno": (seg_t if sess_terr else None, valid),
                "dano": (pd1, valid),
                "dano2": (pd2, valid),
                "flood": (pf, valid),
                "fuego": (pfr, valid),
                "sev": (psv, valid),
                "vias": (pvias, valid),
            })

        dan[src] = round(max(pct_dan, pct_dan2, pct_siam), 1)
        pcts = [float(r.get(k) or 0.0) for k in ("veg", "bui", "wat", "bare", "oth")]
        d, _a = IDX.diagnose(pcts, pct_dan, pct_dan2, pct_siam,
                             pct_flood=pct_flood, pct_flood_water=pct_fw,
                             flood_available=bool(sess_f),
                             pct_fire=pct_fire, pct_smoke=pct_smoke)
        diag_por_frame[src] = d
    print(f"      {len(dan)} frames · daño medio "
          f"{(sum(dan.values()) / len(dan)) if dan else 0.0:.1f}%")
    if fuego:
        print(f"      fuego medio {sum(fuego.values()) / len(fuego):.2f}% · "
              f"humo medio {sum(humo.values()) / len(humo):.2f}%")
    if colapso:
        print(f"      colapso medido medio {sum(colapso.values()) / len(colapso):.1f}%")

    # ── [5/5] Segunda pasada SegFormer B5 ───────────────────────────────
    b5_pct: dict[str, dict[str, float]] = {}
    b5_dir_rel: str | None = None
    if args.b5:
        b5_path = resolve_b5(args.b5_onnx)
        if b5_path is None:
            pass                      # ya avisó resolve_b5()
        else:
            sess_b5 = onnxio.load_required(b5_path, "SegFormer B5")
            # ⚠ ARREGLADO: el tamaño se lee del modelo, no se hardcodea 512.
            b5_size = sess_b5.size_px or 512
            use_sliding = args.sliding or not args.no_sliding
            modo = "ENSEMBLE B5+destilado+flood" if args.ensemble else "SegFormer B5"
            print(f"[5/5] {modo} · {b5_path.name} @{b5_size}px"
                  f"{' · sliding ' + str(args.stride) if use_sliding else ''}")

            sess_t = sess_f2 = None
            if args.ensemble:
                sess_t = onnxio.load_optional(TERRAIN_V2, "destilado")
                if not sess_t:
                    print("  [WARN] --ensemble sin el modelo destilado: "
                          "el ensemble queda en B5 + flood.")
                sess_f2 = sess_f

            # Siempre entrega/ens_seg: si se escribía en b5_seg/ la estación
            # no encontraba los overlays (su bucket sólo mira ens_seg/).
            b5_dir = entrega / "ens_seg"
            b5_dir.mkdir(parents=True, exist_ok=True)
            b5_dir_rel = b5_dir.as_posix()

            prev = None
            ordered = sorted(enumerate(rows), key=lambda ir: frame_sort_key(ir[1], ir[0]))
            for _i, r in ordered:
                src = str(r.get("src"))
                fp = frame_path(frames_dir, src)
                if fp is None:
                    continue
                img = cv2.imread(str(fp))
                if img is None:
                    continue

                lg_global = sess_b5.run({"input": PP.preprocess_bgr(img, b5_size)})[0]
                HW = (img.shape[1], img.shape[0])
                lg_up = np.stack([cv2.resize(m, HW) for m in lg_global])

                if use_sliding and min(img.shape[:2]) >= b5_size:
                    logits = 0.5 * sliding_logits(sess_b5, img, b5_size, args.stride) \
                        + 0.5 * lg_up
                else:
                    logits = lg_up

                if args.temporal_alpha < 1.0 and prev is not None \
                        and prev.shape == logits.shape:
                    logits = (args.temporal_alpha * logits
                              + (1 - args.temporal_alpha) * prev)
                prev = logits

                Pm = PP_softmax(logits)
                H, W = Pm.shape[1], Pm.shape[2]
                if args.ensemble:
                    if sess_t:
                        pt = PP_softmax(sess_t.run(
                            {"input": PP.preprocess_bgr(img, 320)})[0])
                        Pm = 0.55 * Pm + 0.25 * np.stack(
                            [cv2.resize(m, (W, H)) for m in pt])
                    if sess_f2:
                        pf = PP_softmax(sess_f2.run(
                            {"input": PP.preprocess_bgr(img, 320)})[0])
                        pf = np.stack([cv2.resize(m, (W, H)) for m in pf])
                        Pm[2] += 0.20 * (pf[1] + pf[2])
                Pn = Pm / np.maximum(Pm.sum(axis=0, keepdims=True), 1e-9)

                # CRF guiado por la imagen de verdad (ver cansat/crf.py).
                if args.crf_iters > 0:
                    Pn = dense_crf(Pn, img, iters=args.crf_iters)

                seg = Pn.argmax(axis=0)
                # Agua de baja confianza: dejar que gane la segunda clase.
                baja = (seg == 2) & (Pn[2] < 0.6)
                if baja.any():
                    alt = Pn.copy()
                    alt[2] = -1.0
                    seg[baja] = alt.argmax(axis=0)[baja]

                # ⚠ Los % de terreno del post-vuelo se calculan sobre píxeles
                #   VÁLIDOS, igual que el CSV del vuelo. Antes se dividían por
                #   el frame completo (con bordes negros de tile incluidos), así
                #   que la estación mostraba números distintos de los del vuelo
                #   para el mismo frame.
                valid_nat = cv2.resize(
                    ND.valid_at_size(ND.border_mask(img, 15), b5_size).astype(np.uint8),
                    (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                n_valid = max(1, int(valid_nat.sum()))
                b5_pct[src] = {
                    k: round(float(((seg == i) & valid_nat).sum()) / n_valid * 100, 1)
                    for i, k in enumerate(("veg", "bui", "wat", "bare", "oth"))
                }
                cv2.imwrite(str(b5_dir / f"{src}_b5.png"), color_seg(seg, img))
                if not args.no_masks:
                    # El terreno B5 es la máscara de mejor calidad: reemplaza
                    # la del pase rápido si existe.
                    terr = seg.astype(np.uint8).copy()
                    terr[~valid_nat] = MK.NODATA
                    MK.save_mask(masks_dir / f"{src}_terreno.png", terr)
            print(f"      {len(b5_pct)} frames → {b5_dir}")
    else:
        print("[5/5] Segunda pasada B5 salteada (usá --b5 para activarla).")

    if not args.no_masks:
        print(f"      máscaras de clase: {len(list(masks_dir.glob('*.png')))} "
              f"→ {masks_dir}")

    # ── summary.json (schema canónico) ──────────────────────────────────
    rutas.update({
        "telemetry": args.telemetry,
        "enhanced": enhanced[0] if enhanced else None,
        "evidencias": "outputs/mission/vis",
        "ens_seg": b5_dir_rel,
        "masks": "entrega/masks" if not args.no_masks else None,
    })
    conteos = {
        "vis": _count("outputs/mission/vis", "*_evid.jpg"),
        "high_res": _count("outputs/mission/high_res", "*"),
        "full_res": _count("outputs/mission/full_res", "*"),
        "thumb": _count("outputs/mission/thumb", "*"),
        "enhanced": len(enhanced),
        "ens_seg": len(b5_pct),
        "masks": len(list(masks_dir.glob("*.png"))) if not args.no_masks else 0,
    }

    # Enriquecemos las filas con el diagnóstico de consenso recién calculado,
    # para que "alertas" del summary traiga motivo/diag coherentes con post-vuelo.
    rows_enr = []
    for r in rows:
        rr = dict(r)
        src = str(rr.get("src"))
        if src in diag_por_frame:
            rr["diag"] = diag_por_frame[src]
            rr["alert"] = 1 if IDX.is_alert(diag_por_frame[src]) else 0
        if src in dan:
            rr["danado_pct"] = dan[src]
        rows_enr.append(rr)

    summary = SUM.build_summary(
        rows_enr,
        danado_por_frame=dan,
        terrain_por_frame=b5_pct or None,
        rutas=rutas,
        conteos=conteos,
        nota=("Post-vuelo generado por post_flight.py"
              + (" · ensemble B5" if b5_pct else "")),
        supuestos_perdidas={
            "pop_density": args.pop_density,
            "occupancy": args.occupancy,
            "collapse_frac": args.collapse_frac,
            "fatality": args.fatality_ratio,
        },
    )
    p = summary.get("perdidas") or {}
    if p.get("n_frames_con_dano"):
        print(f"  Estimación de pérdidas (exposición): "
              f"afectados≈{p['personas_afectadas']:.1f} · "
              f"pérdidas≈{p['perdidas_estimadas']:.2f} "
              f"[{p['perdidas_min']:.2f}-{p['perdidas_max']:.2f}] "
              f"sobre {p['n_frames_con_dano']} frames con daño")
    else:
        print("  Estimación de pérdidas: sin frames con daño o sin area_m2 en "
              "la telemetría (CSV viejo)")
    avisos = SUM.validate(summary, n_frames_csv=len(rows))
    if avisos:
        print("  [WARN] avisos del contrato:")
        for a in avisos:
            print(f"         · {a}")

    SUM.write_summary(entrega / "summary.json", summary)

    print(f"\n[OK] Paquete listo en {entrega.resolve()}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def PP_softmax(x: np.ndarray) -> np.ndarray:
    """Softmax sobre el eje de clases de un tensor CHW."""
    e = np.exp(x - x.max(axis=0, keepdims=True))
    return e / np.maximum(e.sum(axis=0, keepdims=True), 1e-12)


def _count(folder: str, pattern: str) -> int:
    p = Path(folder)
    return len(list(p.glob(pattern))) if p.is_dir() else 0


if __name__ == "__main__":
    raise SystemExit(main())
