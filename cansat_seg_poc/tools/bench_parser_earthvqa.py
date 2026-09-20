"""
Benchmark del parser de la Consulta Terrestre contra los QA de EarthVQA.

Mide lo que SÍ se puede medir honestamente sin responder VQA:

  · **Cobertura**: fracción de preguntas que el parser mapea a una plantilla
    soportada (global y por ``Type`` de EarthVQA).
  · **Precisión de mapeo**: sobre una muestra estratificada por tipo que se
    etiqueta a mano (SI/NO) en el CSV que emite el propio benchmark.

⚠ NO mide exactitud de respuesta: eso exigiría ejecutar el motor sobre las
máscaras refinadas de EarthVQA y comparar respuestas libres (números, Yes/No,
categorías). Se declara en la ``nota`` del JSON.

⚠ Licencia: EarthVQA (RSIDEA, Wuhan University) es académico, NO comercial.
El QA no se versiona (``dataset/`` está en .gitignore). Está en Hugging Face
gated: https://huggingface.co/datasets/Kingdrone-Junjue/EarthVLSet
(sin aceptar las condiciones de la cuenta, la descarga devuelve 403).

Uso:
    python tools/bench_parser_earthvqa.py --qa dataset/earthvqa/Val_QA.json
    python tools/bench_parser_earthvqa.py --qa ... --muestra 100 --seed 0
    # etiquetar la columna veredicto del CSV (SI/NO) y volver a correr.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cansat.consultas import parsear                        # noqa: E402

#: Capacidad del parser asumiendo que la misión tiene todas las máscaras.
#: La métrica mide LENGUAJE (mapeo pregunta → plantilla), no disponibilidad.
DISPONIBLES = {"terreno", "dano2", "flood", "vias"}

NOTA = (
    "Mide cobertura del parser y precisión de sus DECISIONES sobre una muestra "
    "etiquetada a mano (mapeo correcto en filas soportadas; rechazo correcto en "
    "no soportadas). NO mide exactitud de respuesta VQA (respuestas libres + "
    "máscaras refinadas de EarthVQA fuera de este benchmark). EarthVQA: uso "
    "académico, no comercial."
)


def cargar_qa(path: Path) -> list[dict]:
    """Aplana el JSON de EarthVQA ``{imagen: [{Type, Question, Answer}, …]}``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    filas: list[dict] = []
    for imagen, qas in data.items():
        for qa in qas or []:
            preg = str(qa.get("Question") or "").strip()
            if preg:
                filas.append({
                    "imagen": imagen,
                    "tipo": str(qa.get("Type") or "sin_tipo"),
                    "pregunta": preg,
                    "respuesta": str(qa.get("Answer") or ""),
                })
    return filas


def clasificar(pregunta: str) -> dict:
    """Spec del parser para una pregunta (soportada o no), sin ejecutar."""
    return parsear(pregunta, DISPONIBLES)


def cobertura(filas: list[dict]) -> dict:
    """Cobertura global y por tipo + plantillas más disparadas."""
    por_tipo: dict[str, Counter] = {}
    plantillas: Counter = Counter()
    for f in filas:
        spec = clasificar(f["pregunta"])
        c = por_tipo.setdefault(f["tipo"], Counter())
        c["n"] += 1
        if spec.get("soportada"):
            c["ok"] += 1
            plantillas[spec["plantilla"]] += 1
    return {
        "global": {
            "n": len(filas),
            "soportadas": sum(c["ok"] for c in por_tipo.values()),
            "cobertura": round(sum(c["ok"] for c in por_tipo.values())
                               / max(len(filas), 1), 4),
        },
        "por_tipo": {
            t: {"n": c["n"], "soportadas": c["ok"],
                "cobertura": round(c["ok"] / max(c["n"], 1), 4)}
            for t, c in sorted(por_tipo.items())
        },
        "plantillas": dict(plantillas.most_common()),
    }


def muestra_estratificada(filas: list[dict], n: int, seed: int) -> list[dict]:
    """
    Muestra proporcional por ``Type`` de preguntas ÚNICAS, determinista.

    EarthVQA repite la misma pregunta en cientos de imágenes; muestrear filas
    daría una precisión artificialmente confiada con pocas familias. Acá se
    deduplica por texto: con ``--muestra 100`` entran TODAS las preguntas
    distintas del QA (a la fecha, 51). El peso por tipo sigue el conteo real.
    """
    rng = random.Random(seed)
    por_tipo: dict[str, list[dict]] = {}
    vistas: set[tuple[str, str]] = set()
    for f in filas:
        key = (f["tipo"], f["pregunta"])
        if key in vistas:
            continue
        vistas.add(key)
        por_tipo.setdefault(f["tipo"], []).append(f)
    for v in por_tipo.values():
        rng.shuffle(v)
    out: list[dict] = []
    i = 0
    tocando = True
    while tocando and len(out) < min(n, len(vistas)):
        tocando = False
        for tipo in sorted(por_tipo):
            if i < len(por_tipo[tipo]) and len(out) < n:
                out.append(por_tipo[tipo][i])
                tocando = True
        i += 1
    out.sort(key=lambda f: (f["tipo"], f["pregunta"]))
    return out


def evaluar_muestra(ruta_csv: Path, muestra: list[dict]) -> dict:
    """
    Métricas de la muestra usando la columna ``veredicto`` (SI/NO) del CSV.

    El veredicto etiqueta a mano si la DECISIÓN del parser es correcta:
      · fila soportada → ¿el mapeo propuesto es el correcto?
      · fila no soportada → ¿era correcto rechazarla?
    Se reportan precisiones separadas para no mezclar cobertura con acierto.

    Si el CSV no existe se escribe con ``veredicto`` vacío y se devuelve
    ``etiquetados=0`` (la cobertura ya es válida; la precisión queda pendiente).
    """
    existentes: dict[tuple, str] = {}
    if ruta_csv.is_file():
        with ruta_csv.open("r", encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                existentes[(r.get("imagen", ""), r.get("pregunta", ""))] = \
                    (r.get("veredicto") or "").strip().upper()
    else:
        ruta_csv.parent.mkdir(parents=True, exist_ok=True)
        with ruta_csv.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["imagen", "tipo", "pregunta", "soportada",
                         "plantilla", "sujetos", "veredicto"])
            for f in muestra:
                spec = clasificar(f["pregunta"])
                wr.writerow([f["imagen"], f["tipo"], f["pregunta"],
                             "SI" if spec.get("soportada") else "NO",
                             spec.get("plantilla") or "",
                             json.dumps({k: spec.get(k)
                                         for k in ("a", "b", "c")},
                                        ensure_ascii=False),
                             ""])

    n_sop = n_no = si_sop = si_no = 0
    for f in muestra:
        spec = clasificar(f["pregunta"])
        v = existentes.get((f["imagen"], f["pregunta"]), "")
        if spec.get("soportada"):
            n_sop += 1
            si_sop += int(v.startswith("S"))
        else:
            n_no += 1
            si_no += int(v.startswith("S"))
    etiquetados = si_sop + si_no
    return {
        "n": len(muestra),
        "n_soportadas": n_sop,
        "n_no_soportadas": n_no,
        "etiquetados": etiquetados,
        "precision_mapeo": round(si_sop / n_sop, 4) if n_sop else None,
        "precision_rechazo": round(si_no / n_no, 4) if n_no else None,
        "exactitud": round((si_sop + si_no) / len(muestra), 4) if muestra else None,
        "archivo": str(ruta_csv),
        "pendiente": ("etiquetar la columna veredicto (SI/NO) del CSV y "
                      "re-correr para obtener la precisión"
                      if not etiquetados else None),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark del parser vs EarthVQA QA")
    ap.add_argument("--qa", default="dataset/earthvqa/Val_QA.json",
                    help="Val_QA.json (o Test_QA.json) de EarthVQA")
    ap.add_argument("--muestra", type=int, default=100,
                    help="tamaño de la muestra estratificada para precisión")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/earthvqa_parser.json")
    args = ap.parse_args(argv)

    p = Path(args.qa)
    if not p.is_file():
        print(f"[ERROR] No existe {p}.\n"
              f"        EarthVQA está gated en Hugging Face:\n"
              f"        https://huggingface.co/datasets/Kingdrone-Junjue/EarthVLSet\n"
              f"        Aceptá las condiciones con tu cuenta y bajá Val_QA.json "
              f"(≈7.5 MB).")
        return 1

    filas = cargar_qa(p)
    if not filas:
        print("[ERROR] QA vacío o con formato inesperado.")
        return 1
    cob = cobertura(filas)
    muestra = muestra_estratificada(filas, args.muestra, args.seed)
    ruta_csv = ROOT / "outputs" / f"earthvqa_parser_muestra_{args.muestra}_{args.seed}.csv"
    prec = evaluar_muestra(ruta_csv, muestra)

    print(f"QA: {p} · {cob['global']['n']} preguntas")
    print(f"  Cobertura global: {cob['global']['cobertura'] * 100:.1f}% "
          f"({cob['global']['soportadas']}/{cob['global']['n']})")
    for tipo, c in cob["por_tipo"].items():
        print(f"    {tipo:<32} {c['cobertura'] * 100:5.1f}%  ({c['soportadas']}/{c['n']})")
    print(f"  Plantillas: {cob['plantillas']}")
    if prec["etiquetados"]:
        print(f"  Muestra n={prec['n']}: {prec['n_soportadas']} soportadas / "
              f"{prec['n_no_soportadas']} rechazadas")
        print(f"    precisión de mapeo   : {prec['precision_mapeo'] * 100:.1f}%")
        print(f"    precisión de rechazo : {prec['precision_rechazo'] * 100:.1f}%")
        print(f"    exactitud de decisión: {prec['exactitud'] * 100:.1f}%")
    else:
        print(f"  Precisión pendiente: etiquetá {prec['archivo']} y re-corré.")

    payload = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "script": "tools/bench_parser_earthvqa.py",
        "qa": str(p),
        "preguntas_unicas_totales": len({f["pregunta"] for f in filas}),
        "cobertura": cob,
        "muestra": prec,
        "nota": NOTA,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
