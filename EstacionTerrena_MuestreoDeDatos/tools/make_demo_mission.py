#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/make_demo_mission.py — Generador de MISIÓN DE DEMOSTRACIÓN (LB135)
========================================================================

⚠ HERRAMIENTA DE DESARROLLO. No forma parte de la estación terrena: existe para
  poder probar la web sin el vuelo real. Con datos reales del pipeline, este
  script no hace falta (y puede borrarse sin afectar nada).

Genera un conjunto de artefactos *sintéticos pero coherentes* que respetan el
contrato de datos del proyecto:

    outputs/mission/telemetry.csv          1 fila por frame (28 columnas: 23 + 5 DPD)
    outputs/mission/vis/<src>_evid.jpg     frame anotado (overlay + cajas + HUD)
    outputs/mission/high_res/<src>.jpg
    outputs/mission/full_res/<src>.jpg
    outputs/mission/thumb/<src>.jpg
    outputs/corridor_map.jpg               corredor apilado (tira horizontal)
    entrega/summary.json                   contrato de post-vuelo
    entrega/ens_seg/<src>_b5.png           solo para un subconjunto (degradación)
    entrega/enhanced/<src>_edsr.jpg        solo para un subconjunto (degradación)

Cómo se construye la coherencia (importante para que la demo "tenga sentido"):
  1. Se dibuja un MUNDO aéreo de 2000×1500 px con: zona rural sana, río, lago,
     trama urbana con calles, un sector inundado, un sector con escombros
     (sismo/viento) y un parche quemado (incendio/erosión).
  2. Junto al RGB se mantienen dos máscaras: ``cls`` (índice de clase 0..4) y
     ``dmg`` (0 = nada, 1 = incendio, 2 = inundación, 3 = sismo/viento).
  3. Cada frame es un recorte de ese mundo a lo largo de la trayectoria de caída
     (deriva hacia el este + descenso de 250 m), así que los porcentajes de
     terreno, el diagnóstico y el % de daño se CALCULAN de las máscaras: no están
     inventados fila por fila.

Uso:
    python tools/make_demo_mission.py                 # 12 frames, todo incluido
    python tools/make_demo_mission.py --frames 20     # más frames
    python tools/make_demo_mission.py --sin-postvuelo # para probar degradación
    python tools/make_demo_mission.py --sin-corredor  # idem, sin corridor_map
    python tools/make_demo_mission.py --clean         # borra lo generado

Requisitos: Pillow (solo este script; la estación terrena no necesita nada).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

# La consola de Windows (cp1252) no puede imprimir ✔/🧍/🚗 cuando la salida se
# redirige o se pipea desde PowerShell: UnicodeEncodeError y el script muere a
# mitad de camino. Reconfigurar stdout a UTF-8 con reemplazo lo hace robusto.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover
    pass

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageStat
except ImportError:  # pragma: no cover
    sys.exit("[!] Este script necesita Pillow:  pip install pillow")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
ENTREGA = ROOT / "entrega"

# --------------------------------------------------------------------------- #
# Vocabulario del pipeline (idéntico al contrato de la web)
# --------------------------------------------------------------------------- #

# Índice de clase -> (nombre, color RGB del frame, color del overlay anotado)
CLASSES = [
    ("vegetation",  (58, 116, 54),  "#35d07f"),   # 0
    ("building",    (168, 162, 158), "#ff9f43"),  # 1
    ("water",       (34, 86, 148),  "#3aa0ff"),   # 2
    ("bare_ground", (166, 133, 92), "#c08552"),   # 3
    ("other",       (96, 99, 104),  "#7f8c9b"),   # 4  (asfalto / misc)
]
VEG, BLD, WAT, BARE, OTH = 0, 1, 2, 3, 4

# Tipos de daño en la máscara `dmg`
DMG_NONE, DMG_FIRE, DMG_FLOOD, DMG_QUAKE = 0, 1, 2, 3

SIN_DESASTRE = "SIN DESASTRE"
AGUA_EXTENSA = "AGUA EXTENSA (lago/rio)"
INCENDIO = "POSIBLE INCENDIO/EROSION"
SISMO = "POSIBLE SISMO/VIENTO"
INUND_URB = "INUNDACION URBANA"
INUND_SEV = "INUNDACION SEVERA"

# Daño base (%) por diagnóstico, para el consenso de los 3 modelos.
DANO_BASE = {
    SIN_DESASTRE: 0.0, AGUA_EXTENSA: 2.0, INCENDIO: 14.0,
    SISMO: 31.0, INUND_URB: 46.0, INUND_SEV: 69.0,
}

# Mundo y tamaños de salida
WORLD_W, WORLD_H = 2000, 1500
FULL = (960, 720)

# ── Parámetros de la trayectoria GPS simulada (extensiones DPD) ────────────
# Coherentes con el perfil de descenso de `build_plan` (≈250 m → ~1.5 m) y con
# el simulador de UART del proyecto de vuelo (deriva hacia el E).
ALT_INI, ALT_FIN = 249.0, 1.5
LAT0, LON0 = -34.6075, -58.6126        # El Palomar (mismo origen que sim_uart.py)
DRIFT_DEG = 0.002                      # ~220 m de deriva total
HIGH = (640, 480)
THUMB = (200, 150)
CORR_TILE = (240, 180)

# GSD supuesto del IMX500 con la óptica del CanSat: ~0,4 mm por píxel y metro de
# altura  ->  a 250 m son ~0,10 m/px. Solo se usa para convertir % a m².
GSD_PER_M = 0.0004


def ascii_txt(v) -> str:
    """Quita acentos: la fuente embebida de PIL no cubre todos los glifos."""
    return unicodedata.normalize("NFD", str(v)).encode("ascii", "ignore").decode("ascii")


def font(size: int):
    """Fuente TrueType con fallback (entornos sin fuentes del sistema)."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# 1. Construcción del mundo (RGB + máscara de clases + máscara de daño)
# --------------------------------------------------------------------------- #

def build_world(rng: random.Random):
    """Dibuja el escenario completo y devuelve (rgb, cls, dmg)."""
    rgb = Image.new("RGB", (WORLD_W, WORLD_H), CLASSES[VEG][1])
    cls = Image.new("L", (WORLD_W, WORLD_H), VEG)     # todo vegetation por defecto
    dmg = Image.new("L", (WORLD_W, WORLD_H), DMG_NONE)
    d_rgb, d_cls, d_dmg = ImageDraw.Draw(rgb), ImageDraw.Draw(cls), ImageDraw.Draw(dmg)

    # ── Textura de vegetación (manchas claras/oscuras + campos de cultivo) ──
    for _ in range(2600):
        x, y = rng.randrange(WORLD_W), rng.randrange(WORLD_H)
        r = rng.randint(6, 34)
        tone = rng.choice([(70, 138, 62), (48, 104, 46), (86, 152, 70), (40, 92, 44)])
        d_rgb.ellipse([x - r, y - r, x + r, y + r], fill=tone)
    # Campos cultivados al oeste (vegetación ordenada en franjas)
    for k in range(14):
        x0 = rng.randint(20, 620); y0 = rng.randint(20, WORLD_H - 200)
        w = rng.randint(90, 220); h = rng.randint(70, 180)
        tone = rng.choice([(96, 148, 62), (74, 128, 58), (120, 158, 74)])
        d_rgb.rectangle([x0, y0, x0 + w, y0 + h], fill=tone)
        d_cls.rectangle([x0, y0, x0 + w, y0 + h], fill=VEG)
        for yy in range(y0, y0 + h, 9):          # surcos
            d_rgb.line([x0, yy, x0 + w, yy], fill=(58, 104, 48), width=2)

    # ── Río serpenteante + lago (agua natural: dmg = 0) ──
    river = [(150, -30), (300, 240), (250, 470), (400, 700), (470, 950),
             (640, 1180), (800, 1380), (900, 1540)]
    smooth = [p for i, p in enumerate(river) for p in (river[i:i + 1] * 2)]
    d_rgb.line(smooth, fill=CLASSES[WAT][1], width=58, joint="curve")
    d_cls.line(smooth, fill=WAT, width=58, joint="curve")
    d_rgb.line(smooth, fill=(52, 112, 176), width=26, joint="curve")   # brillo central
    lake_c = (395, 1130)
    d_rgb.ellipse([lake_c[0] - 205, lake_c[1] - 140, lake_c[0] + 205, lake_c[1] + 140],
                  fill=(30, 80, 140))
    d_cls.ellipse([lake_c[0] - 205, lake_c[1] - 140, lake_c[0] + 205, lake_c[1] + 140], fill=WAT)
    d_rgb.ellipse([lake_c[0] - 150, lake_c[1] - 95, lake_c[0] + 150, lake_c[1] + 95],
                  fill=(44, 104, 168))

    # ── Suelo expuesto natural: caminos de tierra y claros ──
    for _ in range(9):
        x0 = rng.randint(60, 800); y0 = rng.randint(60, WORLD_H - 260)
        pts = [(x0, y0)]
        for _ in range(4):
            pts.append((pts[-1][0] + rng.randint(60, 180), pts[-1][1] + rng.randint(-90, 90)))
        d_rgb.line(pts, fill=(150, 118, 80), width=rng.randint(10, 20), joint="curve")
        d_cls.line(pts, fill=BARE, width=rng.randint(10, 20), joint="curve")
    for _ in range(7):
        x, y = rng.randint(60, 820), rng.randint(60, WORLD_H - 160)
        r = rng.randint(35, 95)
        d_rgb.ellipse([x - r, y - r * 0.6, x + r, y + r * 0.6], fill=(158, 126, 88))
        d_cls.ellipse([x - r, y - r * 0.6, x + r, y + r * 0.6], fill=BARE)

    # ── Trama urbana: manzanas, calles y edificios ──
    TX0, TY0, TX1, TY1 = 900, 220, 1690, 1230
    d_rgb.rectangle([TX0 - 20, TY0 - 20, TX1 + 20, TY1 + 20], fill=(120, 122, 120))
    d_cls.rectangle([TX0 - 20, TY0 - 20, TX1 + 20, TY1 + 20], fill=OTH)   # veredas/lotes
    for y in range(TY0, TY1 + 1, 126):                       # calles horizontales
        d_rgb.line([TX0 - 20, y, TX1 + 20, y], fill=CLASSES[OTH][1], width=17)
        d_cls.line([TX0 - 20, y, TX1 + 20, y], fill=OTH, width=17)
        d_rgb.line([TX0 - 20, y, TX1 + 20, y], fill=(196, 190, 96), width=1)  # línea central
    for x in range(TX0, TX1 + 1, 150):                       # calles verticales
        d_rgb.line([x, TY0 - 20, x, TY1 + 20], fill=CLASSES[OTH][1], width=17)
        d_cls.line([x, TY0 - 20, x, TY1 + 20], fill=OTH, width=17)
    # Edificios dentro de cada manzana
    for by in range(TY0, TY1 - 60, 126):
        for bx in range(TX0, TX1 - 70, 150):
            for _ in range(rng.randint(2, 4)):
                w = rng.randint(28, 62); h = rng.randint(24, 54)
                x0 = bx + rng.randint(14, 150 - w - 14); y0 = by + rng.randint(14, 126 - h - 20)
                tone = rng.choice([(176, 168, 162), (150, 142, 138), (192, 182, 172),
                                   (140, 110, 96), (168, 158, 150)])
                d_rgb.rectangle([x0, y0, x0 + w, y0 + h], fill=tone, outline=(92, 88, 86))
                d_cls.rectangle([x0, y0, x0 + w, y0 + h], fill=BLD)
                d_rgb.line([x0, y0 + h // 2, x0 + w, y0 + h // 2], fill=(120, 114, 110), width=1)

    # ── Inundación urbana: calles y manzanas bajas anegadas (dmg = FLOOD) ──
    FX0, FY0, FX1, FY1 = 900, 640, 1300, 1230
    # Capa de daño de toda la zona baja: los edificios no cambian de clase, pero
    # sí quedan marcados como dañados (es lo que ve el consenso de 3 modelos).
    d_dmg.rectangle([FX0, FY0, FX1, FY1], fill=DMG_FLOOD)
    for _ in range(34):          # borde irregular del anegamiento
        x = rng.randint(FX0 - 40, FX1 + 40); y = rng.randint(FY0 - 30, FY1)
        rx, ry = rng.randint(40, 130), rng.randint(24, 80)
        d_dmg.ellipse([x - rx, y - ry, x + rx, y + ry], fill=DMG_FLOOD)
    for y in range(FY0, FY1, 126):
        d_rgb.line([FX0, y, FX1, y], fill=(46, 96, 150), width=19)
        d_cls.line([FX0, y, FX1, y], fill=WAT, width=19)
        d_dmg.line([FX0, y, FX1, y], fill=DMG_FLOOD, width=19)
    for _ in range(46):                       # manchas de agua entre manzanas
        x = rng.randint(FX0, FX1); y = rng.randint(FY0, FY1)
        rx, ry = rng.randint(26, 90), rng.randint(18, 60)
        d_rgb.ellipse([x - rx, y - ry, x + rx, y + ry], fill=(40, 92, 148))
        d_cls.ellipse([x - rx, y - ry, x + rx, y + ry], fill=WAT)
        d_dmg.ellipse([x - rx, y - ry, x + rx, y + ry], fill=DMG_FLOOD)
    # Edificios parcialmente sumergidos (halo de agua alrededor)
    for _ in range(30):
        x = rng.randint(FX0, FX1); y = rng.randint(FY0, FY1)
        r = rng.randint(18, 40)
        d_rgb.ellipse([x - r, y - r, x + r, y + r], fill=(58, 104, 152))
        d_cls.ellipse([x - r, y - r, x + r, y + r], fill=WAT)
        d_dmg.ellipse([x - r, y - r, x + r, y + r], fill=DMG_FLOOD)

    # ── Sismo/viento: edificios colapsados -> escombros (dmg = QUAKE) ──
    QX0, QY0, QX1, QY1 = 1400, 250, 1790, 620
    for _ in range(58):
        x = rng.randint(QX0, QX1); y = rng.randint(QY0, QY1)
        w = rng.randint(24, 66); h = rng.randint(20, 52)
        tone = rng.choice([(128, 116, 104), (110, 100, 92), (146, 134, 120)])
        d_rgb.rectangle([x, y, x + w, y + h], fill=tone)
        d_cls.rectangle([x, y, x + w, y + h], fill=BARE)      # escombro = suelo expuesto
        d_dmg.rectangle([x, y, x + w, y + h], fill=DMG_QUAKE)
        for _ in range(6):                                    # textura de escombros
            jx = x + rng.randint(-6, w + 6); jy = y + rng.randint(-6, h + 6)
            d_rgb.ellipse([jx - 4, jy - 3, jx + 4, jy + 3], fill=(92, 84, 78))

    # ── Incendio/erosión: parche quemado al este de la ciudad (dmg = FIRE) ──
    BX0, BY0, BX1, BY1 = 1710, 560, 2000, 980
    for _ in range(70):
        x = rng.randint(BX0, BX1); y = rng.randint(BY0, BY1)
        rx, ry = rng.randint(20, 80), rng.randint(14, 55)
        tone = rng.choice([(58, 44, 36), (78, 58, 44), (40, 32, 28), (96, 74, 54)])
        d_rgb.ellipse([x - rx, y - ry, x + rx, y + ry], fill=tone)
        d_cls.ellipse([x - rx, y - ry, x + rx, y + ry], fill=BARE)
        d_dmg.ellipse([x - rx, y - ry, x + rx, y + ry], fill=DMG_FIRE)

    # ── Claro de aterrizaje: suelo expuesto al sureste (frame final) ──
    for cx0, cy0, r in ((1830, 1150, 210), (1690, 1330, 150), (1930, 900, 120)):
        d_rgb.ellipse([cx0 - r, cy0 - r * 0.7, cx0 + r, cy0 + r * 0.7], fill=(154, 122, 86))
        d_cls.ellipse([cx0 - r, cy0 - r * 0.7, cx0 + r, cy0 + r * 0.7], fill=BARE)
        for _ in range(26):                       # surcos de erosión
            x = cx0 + rng.randint(-r, r); y = cy0 + rng.randint(-r // 2, r // 2)
            d_rgb.line([x, y, x + rng.randint(-40, 40), y + rng.randint(-14, 14)],
                       fill=(116, 90, 62), width=rng.randint(2, 5))

    return rgb, cls, dmg


# --------------------------------------------------------------------------- #
# 2. Trayectoria de caída (paracaídas desde ~250 m)
# --------------------------------------------------------------------------- #

# Waypoints de la deriva: (fracción del vuelo, x, y) sobre el mundo.
# Recorren a propósito zonas distintas para que la misión de demo tenga variedad:
# rural -> lago/río -> ciudad inundada -> escombros (sismo) -> incendio -> claro.
WAYPOINTS = [
    (0.00,  330,  300),   # apogeo sobre el sector rural
    (0.10,  470,  620),
    (0.20,  400, 1000),
    (0.30,  560, 1180),   # lago
    (0.40,  820, 1120),   # río hacia el este
    (0.50, 1000,  980),   # entra a la ciudad inundada
    (0.60, 1150,  860),
    (0.70, 1330,  680),
    (0.80, 1520,  450),   # escombros (sismo/viento)
    (0.90, 1760,  640),   # parche de incendio/erosión
    (1.00, 1840, 1120),   # claro de aterrizaje
]


def _interp(p: float) -> tuple[float, float]:
    """Interpola linealmente la posición del waypoint en la fracción de vuelo p."""
    for (p0, x0, y0), (p1, x1, y1) in zip(WAYPOINTS, WAYPOINTS[1:]):
        if p <= p1:
            k = (p - p0) / max(1e-9, (p1 - p0))
            return x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
    return WAYPOINTS[-1][1], WAYPOINTS[-1][2]


def flight_plan(n: int, rng: random.Random) -> list[dict]:
    """
    Devuelve la trayectoria: tiempo, altitud y ventana de recorte en el mundo.
    La caída dura ~85 s (paracaídas desde ~250 m) con deriva hacia el este.
    """
    plan = []
    for i in range(n):
        p = i / max(1, n - 1)
        # Descenso: frenado brusco al abrir el paracaídas y luego casi lineal.
        alt = 249.0 * (1.0 - p) ** 1.08 + rng.uniform(-1.2, 1.2) + 2.4
        t = 3.0 + p * 82.0 + rng.uniform(-0.35, 0.35)
        cx, cy = _interp(p)
        cx += math.sin(p * 5.2) * 26          # oscilación por el viento
        cy += math.cos(p * 3.7) * 22
        # Ventana de recorte: a mayor altura se ve más terreno (GSD mayor).
        side = int(190 + 520 * (alt / 250.0))
        plan.append({"i": i, "t_s": round(max(0.4, t), 2), "alt_m": round(max(1.5, alt), 2),
                     "cx": cx, "cy": cy, "side": side})
    return plan


def crop_world(world: tuple, cx: float, cy: float, side: int):
    """Recorta el mundo (RGB + máscaras) centrado en (cx, cy) con relación 4:3."""
    rgb, cls, dmg = world
    w = side; h = int(side * 0.75)
    x0 = int(min(max(cx - w / 2, 0), max(0, WORLD_W - w)))
    y0 = int(min(max(cy - h / 2, 0), max(0, WORLD_H - h)))
    box = (x0, y0, x0 + w, y0 + h)
    return (rgb.crop(box), cls.crop(box), dmg.crop(box), box)


# --------------------------------------------------------------------------- #
# 3. Métricas por frame (todo se calcula de las máscaras, nada a mano)
# --------------------------------------------------------------------------- #

def pct_from_mask(cls_img: Image.Image) -> dict[str, float]:
    """Porcentaje de cada clase a partir del histograma de la máscara."""
    hist = cls_img.histogram()
    total = sum(hist) or 1
    keys = ["veg", "bui", "wat", "bare", "oth"]
    out = {k: round(100.0 * hist[idx] / total, 2) for k, idx in zip(keys, range(5))}
    # Ajuste fino para que sumen exactamente 100.00 (como haría el pipeline).
    top = max(out, key=out.get)
    out[top] = round(out[top] + (100.0 - sum(out.values())), 2)
    return out


def damage_fraction(cls_img: Image.Image, dmg_img: Image.Image) -> tuple[float, dict[str, float]]:
    """
    % de tejido construido/expuesto con daño + fracción de cada tipo de daño.

    danado_pct = píxeles dañados sobre píxeles de building+bare_ground, que es
    la base sobre la que el consenso de 3 modelos estima daño estructural.
    """
    total = cls_img.size[0] * cls_img.size[1] or 1
    # Base sobre la que se estima daño estructural: edificios, escombros/suelo
    # intervenido y asfalto. El agua libre queda fuera del denominador.
    built = cls_img.point(lambda v: 255 if v in (BLD, BARE, OTH) else 0)
    n_built = built.histogram()[255]
    # multiply sobre máscaras 0/255 equivale a un AND lógico (y acepta modo "L").
    damaged = ImageChops.multiply(built, dmg_img.point(lambda v: 255 if v else 0))
    danado = 100.0 * damaged.histogram()[255] / n_built if n_built else 0.0

    fracs = {}
    for name, code in (("fire", DMG_FIRE), ("flood", DMG_FLOOD), ("quake", DMG_QUAKE)):
        m = dmg_img.point(lambda v, c=code: 255 if v == c else 0)
        fracs[name] = 100.0 * m.histogram()[255] / total
    return round(min(100.0, danado), 2), fracs


def classify(pct: dict, fracs: dict) -> tuple[str, str, float, float]:
    """Diagnóstico de desastre + veredicto ambiental, con las reglas del pipeline."""
    wat, bui, veg, bare, oth = pct["wat"], pct["bui"], pct["veg"], pct["bare"], pct["oth"]

    # El agua de inundación solo existe sobre tejido urbano, así que la fracción
    # de la máscara de daño alcanza para separar "río/lago" de "ciudad inundada".
    # Cuando conviven dos tipos de daño manda el de mayor fracción.
    if fracs["flood"] >= 15.0:
        diag = INUND_SEV
    elif fracs["flood"] >= 5.0:
        diag = INUND_URB
    elif fracs["fire"] >= 6.0 and fracs["fire"] >= fracs["quake"]:
        diag = INCENDIO
    elif fracs["quake"] >= 5.0:
        diag = SISMO
    elif fracs["fire"] >= 6.0:
        diag = INCENDIO
    elif wat >= 28.0:
        diag = AGUA_EXTENSA
    else:
        diag = SIN_DESASTRE

    # USI: carga antrópica ponderada de la superficie (0..1). El agua de
    # inundación computa como estrés porque implica tejido urbano afectado.
    usi = (bui * 1.00 + oth * 0.75 + bare * 0.22 + fracs["flood"] * 0.55) / 100.0
    usi = round(max(0.0, min(1.0, usi)), 3)
    # NDVI proxy a partir de la mezcla de clases (vegetación vs suelo expuesto).
    ndvi = round(max(-1.0, min(1.0, (veg - bare) / (veg + bare + 1e-6))), 3)

    if usi >= 0.42 or bui >= 34:
        verdict = "ALTO ESTRÉS URBANO"
    elif bare >= 40 and veg < 25:
        verdict = "SUELO EXPUESTO"
    elif usi >= 0.18 or (veg + wat) < 60:
        verdict = "ESTRÉS MODERADO"
    else:
        verdict = "ZONA SALUDABLE"
    return diag, verdict, usi, ndvi


def class_mix(pct: dict) -> float:
    """Entropía de Shannon normalizada de las 5 clases (0 = homogéneo, 1 = mezclado)."""
    total = sum(pct.values()) or 1.0
    h = 0.0
    for v in pct.values():
        p = v / total
        if p > 0:
            h -= p * math.log(p)
    return h / math.log(len(CLASSES))


def sample_priority(danado: float, diag: str, alert: int, sharp: float,
                    people: int, usi: float, mix: float,
                    rng: random.Random) -> tuple[str, float]:
    """
    Sampler adaptativo a bordo: puntúa cada frame (0..1) y decide si guardarlo en
    HIGH/MEDIUM/LOW. Pondera daño, tipo de desastre, presencia de personas,
    estrés urbano, nitidez y diversidad de terreno.
    """
    diag_term = 1.0 if diag in (INUND_SEV, INUND_URB, SISMO) else (
        0.55 if diag in (INCENDIO, AGUA_EXTENSA) else 0.0)
    score = (
        0.32 * min(1.0, danado / 60.0) +
        0.15 * diag_term +
        0.13 * min(1.0, people / 6.0) +
        0.10 * usi +
        0.06 * min(1.0, sharp / 300.0) +
        0.04 * alert +
        0.20 * mix
    )
    score = round(max(0.0, min(1.0, score + rng.uniform(-0.025, 0.025))), 3)
    pri = "HIGH" if score >= 0.64 else ("MEDIUM" if score >= 0.26 else "LOW")
    if alert and pri == "LOW":
        pri = "MEDIUM"          # ninguna alerta viaja en prioridad baja
    return pri, score


# --------------------------------------------------------------------------- #
# 4. Render de los artefactos visuales
# --------------------------------------------------------------------------- #

def overlay_rgb(cls_img: Image.Image) -> Image.Image:
    """Convierte la máscara de clases en una capa de color (para el overlay)."""
    pal = []
    for _name, _rgb, hexa in CLASSES:
        h = hexa.lstrip("#")
        pal += [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]
    p = cls_img.copy()
    p.putpalette(pal + [0] * (768 - len(pal)))   # índice de clase -> color del contrato
    return p.convert("RGB")


def detections(rng: random.Random, pct: dict, size: tuple) -> tuple[list, list]:
    """
    Detecciones YOLO simuladas. La cantidad depende de cuánto tejido urbano haya
    en el frame (más ciudad -> más gente y vehículos a la vista).
    """
    w, h = size
    urban = pct["bui"] + pct["oth"] * 0.6
    n_people = 0 if urban < 8 else min(9, int(rng.gauss(urban / 12.0, 1.4)))
    n_veh = 0 if urban < 10 else min(6, int(rng.gauss(urban / 22.0, 1.1)))
    n_people, n_veh = max(0, n_people), max(0, n_veh)

    def box(bw, bh):
        x = rng.randint(int(w * 0.08), int(w * 0.92) - bw)
        y = rng.randint(int(h * 0.14), int(h * 0.90) - bh)
        return [x, y, x + bw, y + bh]

    scale = w / 640.0
    people = [box(max(4, int(7 * scale)), max(8, int(17 * scale))) for _ in range(n_people)]
    vehs = [box(max(8, int(20 * scale)), max(6, int(12 * scale))) for _ in range(n_veh)]
    return people, vehs


def draw_hud(img: Image.Image, row: dict, people: list, vehs: list) -> Image.Image:
    """Frame anotado: overlay de segmentación + cajas + HUD (como a bordo)."""
    rgb = img.convert("RGB")
    ov = overlay_rgb(row["_cls_res"])
    blended = Image.blend(rgb, ov, 0.42)
    d = ImageDraw.Draw(blended)
    w, h = blended.size
    sz_s = max(11, int(w * 0.021))
    sz_l = max(13, int(w * 0.026))
    f_s, f_l = font(sz_s), font(sz_l)

    def textlen(txt, fnt):
        """Ancho de texto con fallback si la fuente no lo soporta."""
        try:
            return d.textlength(txt, font=fnt)
        except (AttributeError, TypeError):
            return len(txt) * sz_l * 0.6

    # Cajas de detección (personas en verde, vehículos en celeste)
    for b in people:
        d.rectangle(b, outline=(61, 220, 132), width=2)
        d.text((b[0], max(0, b[1] - sz_s - 3)), "person", font=f_s, fill=(61, 220, 132))
    for b in vehs:
        d.rectangle(b, outline=(93, 214, 255), width=2)
        d.text((b[0], max(0, b[1] - sz_s - 3)), "vehicle", font=f_s, fill=(93, 214, 255))

    # Banda superior: identificación de la toma + badge de prioridad
    top_h = int(h * 0.085)
    d.rectangle([0, 0, w, top_h], fill=(8, 11, 17))
    d.text((8, 4), ascii_txt(f"LB135 · {row['src']} · t={row['t_s']:.1f}s · alt={row['alt_m']:.1f}m"),
           font=f_l, fill=(216, 227, 240))
    pri_color = {"HIGH": (255, 77, 94), "MEDIUM": (255, 176, 32),
                 "LOW": (150, 165, 180)}[row["sample_pri"]]
    label = f"{row['sample_pri']} {row['sample_score']:.2f}"
    tw = textlen(label, f_l)
    d.rectangle([w - tw - 18, 4, w - 6, top_h - 4], outline=pri_color, width=2)
    d.text((w - tw - 12, 5), label, font=f_l, fill=pri_color)

    # Banda inferior: diagnóstico, daño, mezcla de terreno y métricas ambientales
    bh = int(h * 0.145)
    y0 = h - bh
    d.rectangle([0, y0, w, h], fill=(8, 11, 17))
    l1 = (f"diag: {row['diag']}   alert: {row['alert']}   "
          f"dano: {row['danado_pct']:.1f}%   aff: {row['aff_m2']:.0f} m2")
    l2 = (f"veg {row['veg']:.0f} · bui {row['bui']:.0f} · wat {row['wat']:.0f} · "
          f"bare {row['bare']:.0f} · oth {row['oth']:.0f}")
    l3 = (f"USI {row['usi']:.2f} · NDVI {row['ndvi']:.2f} · {row['verdict']}   "
          f"PERS {row['people']} · VEH {row['vehicles']} · sharp {row['sharp']:.0f}")
    d.text((8, y0 + 4), ascii_txt(l1), font=f_s, fill=(255, 120, 132) if row["alert"] else (216, 227, 240))
    d.text((8, y0 + 6 + sz_s), ascii_txt(l2), font=f_s, fill=(147, 163, 184))
    d.text((8, y0 + 8 + sz_s * 2), ascii_txt(l3), font=f_s, fill=(61, 220, 132))

    # Retícula central + barra de escala (GSD dependiente de la altitud)
    cx, cy = w // 2, h // 2
    d.line([cx - 14, cy, cx + 14, cy], fill=(230, 240, 250), width=1)
    d.line([cx, cy - 14, cx, cy + 14], fill=(230, 240, 250), width=1)
    gsd = row["alt_m"] * GSD_PER_M * (FULL[0] / max(1, w))     # m/px aprox. en esta vista
    bar_m = 10 if gsd * 10 < w * 0.3 else 5
    bar_px = int(bar_m / max(1e-6, gsd))
    if 8 < bar_px < w * 0.6:
        y = y0 - 12
        d.rectangle([10, y, 10 + bar_px, y + 5], outline=(230, 240, 250), width=1)
        d.text((14 + bar_px, y - 3), f"{bar_m} m", font=f_s, fill=(230, 240, 250))
    return blended


def render_ens_seg(cls_img: Image.Image, row: dict) -> Image.Image:
    """Overlay ensemble post-vuelo: máscara dura + leyenda de 5 clases."""
    ov = overlay_rgb(cls_img)
    d = ImageDraw.Draw(ov)
    w, h = ov.size
    f_s = font(max(10, int(w * 0.020)))
    d.rectangle([0, 0, w, 18], fill=(8, 11, 17))
    d.text((6, 3), ascii_txt(f"ens_seg b5 · {row['src']} · dom={row['dom']}"), font=f_s, fill=(216, 227, 240))
    for i, (name, _rgb, hexa) in enumerate(CLASSES):
        y = 24 + i * 16
        d.rectangle([6, y, 16, y + 10], fill=hexa, outline=(0, 0, 0))
        d.text((21, y - 2), f"{name} {row[['veg','bui','wat','bare','oth'][i]]:.1f}%",
               font=f_s, fill=(255, 255, 255))
    return ov


def build_corridor(rows: list[dict], thumbs: list[Image.Image]) -> Image.Image:
    """Corredor de vuelo apilado: tira horizontal en el orden del CSV."""
    tw, th = CORR_TILE
    label_h = 36          # banda para dos líneas de rótulo sin recortes
    gap = 3
    n = len(thumbs)
    W = n * (tw + gap) + gap
    H = th + label_h + gap
    canvas = Image.new("RGB", (W, H), (10, 13, 19))
    d = ImageDraw.Draw(canvas)
    f_s = font(13)
    for i, (row, im) in enumerate(zip(rows, thumbs)):
        x = gap + i * (tw + gap)
        tile = im.resize((tw, th))
        canvas.paste(tile, (x, gap))
        pri = row["sample_pri"]
        color = {"HIGH": (255, 77, 94), "MEDIUM": (255, 176, 32), "LOW": (120, 134, 150)}[pri]
        d.rectangle([x, gap, x + tw - 1, gap + th - 1], outline=color, width=2)
        d.rectangle([x, gap + th, x + tw - 1, H - gap], fill=(14, 19, 27))
        d.text((x + 5, gap + th + 4), f"#{i + 1} {row['src']}", font=f_s, fill=(216, 227, 240))
        d.text((x + 5, gap + th + 20), f"{row['t_s']:.0f}s · {row['alt_m']:.0f}m · {row['sample_pri']}",
               font=font(11), fill=color)
        if row["alert"]:
            d.ellipse([x + tw - 16, gap + 4, x + tw - 6, gap + 14], outline=(255, 77, 94), width=2)
    return canvas


# --------------------------------------------------------------------------- #
# 5. Escritura de artefactos + summary.json
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Genera una misión de demostración para la web")
    ap.add_argument("--frames", type=int, default=12, help="cantidad de frames (default: 12)")
    ap.add_argument("--seed", type=int, default=135, help="semilla aleatoria (default: 135)")
    ap.add_argument("--sin-postvuelo", action="store_true",
                    help="no genera entrega/ (para probar la degradación de la UI)")
    ap.add_argument("--sin-corredor", action="store_true",
                    help="no genera outputs/corridor_map.jpg (idem)")
    ap.add_argument("--clean", action="store_true", help="borra los artefactos generados y sale")
    args = ap.parse_args(argv)

    if args.clean:
        for p in (OUT / "mission", ENTREGA, OUT / "corridor_map.jpg"):
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                p.unlink()
        print("[✔] Artefactos de demostración eliminados.")
        return 0

    rng = random.Random(args.seed)
    n = max(1, args.frames)

    dirs = {
        "vis": OUT / "mission" / "vis",
        "high_res": OUT / "mission" / "high_res",
        "full_res": OUT / "mission" / "full_res",
        "thumb": OUT / "mission" / "thumb",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    if not args.sin_postvuelo:
        (ENTREGA / "ens_seg").mkdir(parents=True, exist_ok=True)
        (ENTREGA / "enhanced").mkdir(parents=True, exist_ok=True)

    print(f"[·] Construyendo mundo sintético ({WORLD_W}×{WORLD_H})…")
    world = build_world(rng)
    plan = flight_plan(n, rng)

    rows: list[dict] = []
    thumbs: list[Image.Image] = []

    for step in plan:
        i = step["i"]
        src = f"cap_{i:04d}"
        rgb_c, cls_c, dmg_c, box = crop_world(world, step["cx"], step["cy"], step["side"])

        # Remuestreo a los tamaños de salida (máscaras en NEAREST para no mezclar clases)
        full = rgb_c.resize(FULL, Image.BILINEAR)
        cls_full = cls_c.resize(FULL, Image.NEAREST)
        dmg_full = dmg_c.resize(FULL, Image.NEAREST)
        high = full.resize(HIGH, Image.BILINEAR)
        cls_high = cls_full.resize(HIGH, Image.NEAREST)
        thumb = high.resize(THUMB, Image.BILINEAR)

        # ── Métricas calculadas sobre las máscaras ──
        pct = pct_from_mask(cls_full)
        danado_mask, fracs = damage_fraction(cls_full, dmg_full)
        diag, verdict, usi, ndvi = classify(pct, fracs)

        # Consenso de 3 modelos: combina la medición sobre la máscara de daño con
        # el prior del diagnóstico, ponderado por el tejido urbano presente.
        urban = min(1.0, (pct["bui"] + pct["bare"] + pct["oth"]) / 38.0)
        danado = round(min(100.0, 0.75 * danado_mask +
                           0.25 * DANO_BASE[diag] * (0.35 + 0.65 * urban)), 2)
        # Nitidez: respuesta de bordes sobre el recorte nativo del sensor.
        edges = ImageStat.Stat(rgb_c.filter(ImageFilter.FIND_EDGES)).mean
        sharp = round(sum(edges) / len(edges) * 15.0, 1)
        alt = step["alt_m"]
        gsd = alt * GSD_PER_M
        area_m2 = (FULL[0] * gsd) * (FULL[1] * gsd)
        aff_m2 = round(area_m2 * danado / 100.0, 1)
        alert = 1 if (diag in (INUND_SEV, SISMO) or danado >= 50) else 0

        people, vehs = detections(rng, pct, HIGH)
        mix = class_mix(pct)
        pri, score = sample_priority(danado, diag, alert, sharp, len(people), usi, mix, rng)

        dom_idx = max(range(5), key=lambda k: [pct["veg"], pct["bui"], pct["wat"],
                                               pct["bare"], pct["oth"]][k])
        p_hpa = round(1013.25 * (1 - 2.25577e-5 * alt) ** 5.25588 + rng.uniform(-0.4, 0.4), 2)
        temp_c = round(23.5 + (249 - alt) * 0.012 + rng.uniform(-0.5, 0.5), 2)

        # ── Extensiones DPD: GPS + estimación de pérdidas humanas ──────────
        # Trayectoria simulada: deriva hacia el E desde el punto de eyección
        # (mismo modelo que sim_uart.py). La estimación replica la fórmula de
        # cansat/casualties.py (proyecto de vuelo); el demo no puede importarla
        # porque son proyectos separados, así que se mantiene sincronizada acá.
        frac_desc = 1.0 - min(1.0, max(0.0, (alt - ALT_FIN) / max(1.0, ALT_INI - ALT_FIN)))
        lat = round(LAT0 + frac_desc * DRIFT_DEG, 5)
        lon = round(LON0 + frac_desc * DRIFT_DEG * 1.3, 5)
        sup = {"pop_density": 1500.0, "occupancy": 0.6,
               "collapse_frac": 0.3, "fatality": 0.1}
        expuestas = sup["pop_density"] * (area_m2 / 1e6) * sup["occupancy"]
        afectadas = expuestas * (danado / 100.0)
        perdidas = afectadas * sup["collapse_frac"] * sup["fatality"]

        row = {
            "t_s": step["t_s"], "alt_m": alt, "p_hpa": p_hpa, "temp_c": temp_c,
            "veg": pct["veg"], "bui": pct["bui"], "wat": pct["wat"],
            "bare": pct["bare"], "oth": pct["oth"], "dom": dom_idx,
            "usi": usi, "ndvi": ndvi, "verdict": verdict,
            "people": len(people), "vehicles": len(vehs),
            "danado_pct": danado, "aff_m2": aff_m2, "diag": diag, "alert": alert,
            "sharp": round(sharp, 1), "src": src,
            "sample_pri": pri, "sample_score": score,
            "lat": lat, "lon": lon,
            "hum_pct": round(min(98.0, 48.0 + 16.0 * frac_desc
                                 + (10.0 if diag == INUND_SEV else 0.0)
                                 + rng.uniform(-2.0, 2.0)), 1),
            "area_m2": int(area_m2),
            "personas_afectadas": round(afectadas, 1),
            "perdidas_est": round(perdidas, 2),
            # internos (no van al CSV)
            "_expuestas": round(expuestas, 1),
            "_cls_res": cls_high,
        }

        # ── Archivos del frame ──
        full.save(dirs["full_res"] / f"{src}.jpg", quality=92)
        high.save(dirs["high_res"] / f"{src}.jpg", quality=88)
        thumb.save(dirs["thumb"] / f"{src}.jpg", quality=80)
        draw_hud(high, row, people, vehs).save(dirs["vis"] / f"{src}_evid.jpg", quality=90)

        # Post-vuelo: solo un subconjunto tiene ens_seg / enhanced (a propósito,
        # para ejercitar la degradación elegante de la UI).
        if not args.sin_postvuelo:
            if i % 2 == 0:
                render_ens_seg(cls_high, row).save(ENTREGA / "ens_seg" / f"{src}_b5.png")
            if pri == "HIGH" or i in (0, n - 1):
                enh = high.resize((high.width * 2, high.height * 2), Image.LANCZOS)
                enh = enh.filter(ImageFilter.UnsharpMask(radius=1.6, percent=90, threshold=2))
                enh.save(ENTREGA / "enhanced" / f"{src}_edsr.jpg", quality=92)

        rows.append(row)
        thumbs.append(thumb)
        print(f"    {src}  t={row['t_s']:6.1f}s alt={row['alt_m']:6.1f}m  "
              f"{diag:<24} {pri:<6} score={score:.2f} daño={danado:5.1f}%  "
              f"🧍{row['people']} 🚗{row['vehicles']}  "
              f"[veg{pct['veg']:5.1f} bui{pct['bui']:5.1f} wat{pct['wat']:5.1f} "
              f"bar{pct['bare']:5.1f} oth{pct['oth']:5.1f} | "
              f"fl{fracs['flood']:4.1f} qk{fracs['quake']:4.1f} fr{fracs['fire']:4.1f} "
              f"USI{usi:.2f} mix{mix:.2f}] {verdict}")

    # ── telemetry.csv ──
    csv_path = OUT / "mission" / "telemetry.csv"
    cols = ["t_s", "alt_m", "p_hpa", "temp_c", "veg", "bui", "wat", "bare", "oth", "dom",
            "usi", "ndvi", "verdict", "people", "vehicles", "danado_pct", "aff_m2",
            "diag", "alert", "sharp", "src", "sample_pri", "sample_score",
            # Extensiones DPD (mismo orden que CSV_COLUMNS del pipeline)
            "lat", "lon", "hum_pct", "area_m2", "personas_afectadas", "perdidas_est"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ── telemetry.jsonl ─────────────────────────────────────────────────────
    # El pipeline de vuelo deja también este archivo con los campos que no van
    # al CSV (incertidumbre, tiempos por etapa, % sin datos). Sin él, el panel
    # de Muestreo de la estación queda vacío en las demos.
    jsonl_path = OUT / "mission" / "telemetry.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            unc = round(min(0.95, 0.12 + r["danado_pct"] / 220.0
                            + rng.uniform(-0.03, 0.06)), 3)
            fh.write(json.dumps({
                "src": r["src"], "uncert": unc,
                "ms_seg": int(160 + rng.uniform(-20, 60)),
                "ms_dmg": int(40 + rng.uniform(-10, 30)),
                "ms_total": int(230 + rng.uniform(-30, 90)),
                "nodata_pct": round(rng.uniform(0.0, 3.5), 1),
                "danado2_edif_pct": round(min(100.0, r["danado_pct"] * rng.uniform(1.0, 2.4)), 1),
                "sharp_ok": True,
                "area_m2": r["area_m2"],
                "sample_pri": r["sample_pri"],
                "sample_score": r["sample_score"],
                "supuestos": sup,
            }, ensure_ascii=False) + "\n")

    # ── corridor_map.jpg ──
    if not args.sin_corredor:
        build_corridor(rows, thumbs).save(OUT / "corridor_map.jpg", quality=88)

    # ── entrega/summary.json (contrato v2) ──
    # Mismo schema que post_flight.py (fuente: cansat/summary.py, SCHEMA_VERSION
    # 2): `alertas` es list[dict] y `archivos` va separado en {rutas, conteos}.
    # Antes la demo escribía el formato legacy plano y el frontend tipaba
    # `alertas: unknown[]` y solo usaba `.length`: la estación se probaba contra
    # un contrato que no era el que produce el vuelo real.
    if not args.sin_postvuelo:
        veredictos: dict[str, int] = {}
        for r in rows:
            veredictos[r["verdict"]] = veredictos.get(r["verdict"], 0) + 1
        perd_est = sum(r["perdidas_est"] for r in rows)
        frames_con_dano = sum(1 for r in rows if r["danado_pct"] > 0)
        summary = {
            "schema_version": 2,
            "mision": "LB135",
            "n_frames": len(rows),
            "alt_max_m": round(max(r["alt_m"] for r in rows), 2),
            "alt_min_m": round(min(r["alt_m"] for r in rows), 2),
            "veredictos": veredictos,
            "personas_total": sum(r["people"] for r in rows),
            "vehiculos_total": sum(r["vehicles"] for r in rows),
            # Estimación de pérdidas humanas (DPD). Misma estructura que
            # cansat/casualties.py::estimar_mision del proyecto de vuelo.
            "perdidas": {
                "area_relevada_m2": round(sum(r["area_m2"] for r in rows), 1),
                "area_danada_m2": round(sum(r["area_m2"] * r["danado_pct"] / 100.0
                                           for r in rows), 1),
                "personas_expuestas": round(sum(r["_expuestas"] for r in rows), 1),
                "personas_afectadas": round(sum(r["personas_afectadas"] for r in rows), 1),
                "perdidas_estimadas": round(perd_est, 2),
                "perdidas_min": round(perd_est * 0.5, 2),
                "perdidas_max": round(perd_est * 2.0, 2),
                "supuestos": sup,
                "n_frames_con_dano": frames_con_dano,
                "nota": ("Estimación de EXPOSICIÓN con supuestos declarados, no una "
                         "predicción de víctimas."),
            },
            "alertas": [
                {"src": r["src"], "t_s": r["t_s"], "alt_m": r["alt_m"],
                 "diag": r["diag"], "danado_pct": r["danado_pct"],
                 "sample_pri": r["sample_pri"],
                 "motivo": f"{r['diag']} · daño {r['danado_pct']:.1f}%"}
                for r in rows if r["alert"] == 1
            ],
            "danado_pct_por_frame": {r["src"]: r["danado_pct"] for r in rows},
            "danado_pct_prom": round(sum(r["danado_pct"] for r in rows) / len(rows), 2),
            "terrain_b5_por_frame": {
                r["src"]: {k: r[k] for k in ("veg", "bui", "wat", "bare", "oth")} for r in rows
            },
            "archivos": {
                "rutas": {
                    "telemetry": "outputs/mission/telemetry.csv",
                    "corridor_map": ("outputs/corridor_map.jpg"
                                     if not args.sin_corredor else None),
                    "vis": "outputs/mission/vis",
                    "ens_seg": "entrega/ens_seg",
                    "enhanced": "entrega/enhanced",
                },
                "conteos": {
                    "vis": len(rows),
                    "high_res": len(rows),
                    "full_res": len(rows),
                    "thumb": len(rows),
                    "ens_seg": len(list((ENTREGA / "ens_seg").glob("*_b5.png"))),
                    "enhanced": len(list((ENTREGA / "enhanced").glob("*_edsr.jpg"))),
                },
            },
            "generado": datetime.now(timezone.utc).isoformat(),
            "nota": "Misión de demostración generada por tools/make_demo_mission.py",
        }
        (ENTREGA / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[✔] Misión demo lista: {len(rows)} frames")
    print(f"    {csv_path.relative_to(ROOT)}")
    if not args.sin_corredor:
        print(f"    {(OUT / 'corridor_map.jpg').relative_to(ROOT)}")
    if not args.sin_postvuelo:
        print(f"    {(ENTREGA / 'summary.json').relative_to(ROOT)}")
    print("\n    Ahora:  python web_server.py   ->   http://localhost:8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
