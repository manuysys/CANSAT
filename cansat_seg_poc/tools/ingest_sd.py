"""
Ingesta de la microSD del CanSat a la PC — verificación de integridad.

Paso que el DPD pide explícitamente en la operación: *"Post-recuperación:
descarga de datos almacenados (microSD), verificación de integridad de imágenes
y telemetría, respaldo de la información"*. Hasta ahora `post_flight.py` asumía
que la carpeta ya estaba copiada a la PC, sin ninguna verificación.

Qué hace
--------
1. Copia los vuelos (`vuelos/<TS>/`) de la SD a la PC **sin pisar** lo que ya
   exista (a menos que se pase ``--force``).
2. Calcula **SHA-256** de cada archivo copiado y escribe
   ``ingesta_<TS>/manifest.sha256`` + ``ingesta_<TS>/resumen.json``.
3. **Verifica la pareja imagen ↔ telemetría**: cada fila del CSV tiene que
   tener su frame (o su evidencia) y viceversa. Reporta faltantes/sobrantes
   sin abortar.
4. Sugiere el comando de post-vuelo con las rutas ya resueltas.

Uso (desde la PC, con la SD montada):
    python tools/ingest_sd.py --src E:/vuelos                 # Windows
    python tools/ingest_sd.py --src /media/pi/vuelos          # Linux
    python tools/ingest_sd.py --src E:/vuelos --dst vuelos --force
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# La consola de Windows (cp1252) no imprime ▸/✓ si la salida se redirige.
try:
    import sys as _sys

    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover
    pass

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def sha256(path: Path, bloque: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(bloque), b""):
            h.update(chunk)
    return h.hexdigest()


def copiar_vuelo(origen: Path, destino: Path, force: bool) -> list[Path]:
    """Copia recursiva archivo por archivo; no pisa salvo --force."""
    copiados: list[Path] = []
    for src in sorted(origen.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(origen)
        dst = destino / rel
        if dst.exists() and not force:
            print(f"    [=] ya existe, se saltea: {rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copiados.append(dst)
    return copiados


def verificar_telemetria(vuelo: Path) -> dict:
    """Verifica pares imagen ↔ telemetría del vuelo copiado."""
    info: dict = {"csv": None, "jsonl": None, "n_filas": 0,
                  "srcs_sin_frame": [], "frames_sin_src": [],
                  "imagenes": 0, "avisos": []}
    csvs = sorted(vuelo.glob("telemetry*.csv"))
    if not csvs:
        info["avisos"].append("no hay telemetry*.csv en el vuelo")
        return info
    # Preferir el versionado (telemetry_<ts>.csv) sobre la copia canónica.
    csv_p = next((c for c in csvs if c.name != "telemetry.csv"), csvs[-1])
    info["csv"] = str(csv_p)
    jsonl_p = csv_p.with_suffix(".jsonl")
    info["jsonl"] = str(jsonl_p) if jsonl_p.is_file() else None

    with csv_p.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    info["n_filas"] = len(rows)
    srcs = {str(r.get("src") or "").strip() for r in rows if r.get("src")}

    imagenes = [p for p in vuelo.rglob("*") if p.suffix.lower() in IMG_EXTS]
    info["imagenes"] = len(imagenes)
    stems = {p.stem.split("_evid")[0].split("_edsr")[0].split("_b5")[0]
             for p in imagenes}

    info["srcs_sin_frame"] = sorted(s for s in srcs if s and s not in stems)[:20]
    info["frames_sin_src"] = sorted(s for s in stems if s and s not in srcs)[:20]
    if not info["jsonl"]:
        info["avisos"].append(
            "sin telemetry.jsonl: la estación no tendrá el panel de muestreo "
            "(incertidumbre/tiempos); ¿se copió todo?")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ingesta y verificación de la SD del CanSat")
    ap.add_argument("--src", required=True,
                    help="carpeta de vuelos en la SD (p. ej. E:/vuelos)")
    ap.add_argument("--dst", default="vuelos", help="destino en la PC")
    ap.add_argument("--force", action="store_true", help="pisar archivos existentes")
    ap.add_argument("--solo-verificar", action="store_true",
                    help="no copiar: sólo verificar un vuelo ya en la PC")
    args = ap.parse_args(argv)

    origen = Path(args.src)
    if not origen.is_dir():
        print(f"[ERROR] No existe {origen}. ¿Montaste la SD?")
        return 1

    # Cada subcarpeta de la SD es un vuelo (vuelos/<TS>/).
    vuelos = [d for d in sorted(origen.iterdir()) if d.is_dir()]
    if not vuelos and (origen / "telemetry.csv").is_file():
        vuelos = [origen]          # pasaron la carpeta del vuelo directamente
    if not vuelos:
        print(f"[ERROR] {origen} no tiene subcarpetas de vuelo.")
        return 1

    sello = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = Path(args.dst)
    reporte_dir = destino / f"ingesta_{sello}"
    print("=" * 74)
    print(f"  Ingesta de SD — {len(vuelos)} vuelo(s)")
    print(f"  Origen : {origen}")
    print(f"  Destino: {destino}")
    print("=" * 74)

    manifiesto: list[str] = []
    resumen: dict = {"generado": datetime.now(timezone.utc).isoformat(),
                     "origen": str(origen), "vuelos": {}}

    for v in vuelos:
        print(f"\n  ▸ Vuelo {v.name}")
        if args.solo_verificar:
            destino_vuelo = v
            copiados: list[Path] = []
        else:
            destino_vuelo = destino / v.name
            copiados = copiar_vuelo(v, destino_vuelo, args.force)
            print(f"    copiados: {len(copiados)} archivo(s)")

        print("    verificando integridad (SHA-256)...")
        for p in sorted(destino_vuelo.rglob("*")):
            if p.is_file():
                manifiesto.append(f"{sha256(p)}  {p.relative_to(destino).as_posix()}")

        info = verificar_telemetria(destino_vuelo)
        resumen["vuelos"][v.name] = info
        print(f"    telemetría: {info['n_filas']} filas · "
              f"{info['imagenes']} imágenes · jsonl={'sí' if info['jsonl'] else 'NO'}")
        if info["srcs_sin_frame"]:
            print(f"    [WARN] {len(info['srcs_sin_frame'])} src del CSV sin imagen "
                  f"(ej: {', '.join(info['srcs_sin_frame'][:3])})")
        if info["frames_sin_src"]:
            print(f"    [WARN] {len(info['frames_sin_src'])} imágenes sin fila en el CSV "
                  f"(ej: {', '.join(info['frames_sin_src'][:3])})")
        for a in info["avisos"]:
            print(f"    [WARN] {a}")

    reporte_dir.mkdir(parents=True, exist_ok=True)
    (reporte_dir / "manifest.sha256").write_text("\n".join(manifiesto) + "\n",
                                                 encoding="utf-8")
    (reporte_dir / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 74)
    print(f"  ✓ {len(manifiesto)} archivo(s) hasheados")
    print(f"  ✓ manifiesto: {reporte_dir / 'manifest.sha256'}")
    print(f"  ✓ resumen   : {reporte_dir / 'resumen.json'}")
    print("\n  Para verificar de nuevo más tarde:")
    print(f"    python -c \"import hashlib,pathlib; ...\"   # o: sha256sum -c "
          f"{reporte_dir.as_posix()}/manifest.sha256")
    if not args.solo_verificar:
        primero = next(iter(resumen['vuelos']))
        print("\n  Post-vuelo sugerido:")
        print(f"    python post_flight.py --frames <carpeta-de-frames> "
              f"--telemetry {destino.as_posix()}/{primero}/telemetry.csv "
              f"--no-pipeline --b5")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
