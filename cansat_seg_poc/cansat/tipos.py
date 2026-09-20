"""
tipos.py - Política de decisión del clasificador de TIPO de desastre.

Contexto honesto (2026-09-20): el clasificador de 7 clases NO generaliza a
eventos nuevos — leave-one-event-out (LOEO) dio 0.248 ponderado; solo los
incendios generalizan (0.755 / 0.712 en eventos held-out). El 0.978 que se
midió con split aleatorio era fuga de evento (tiles del mismo evento en train
y val). Ver ``outputs/disaster_type_loeo.json`` y MODELS.yaml.

Política ``fire_only_v1``: el sistema SOLO afirma ``incendio`` cuando el top-1
es ``incendio`` y su confianza supera el umbral calibrado; cualquier otra
clase (o incendio bajo umbral) se ABSTIENE. La salida queda auditada en el
JSONL con estado explícito (``tipo_estado``), y la UI solo muestra badge
cuando el estado es ``confirmado_por_modelo``.

Función pura: sin torch, sin cv2, sin ONNX — testeable en cualquier lado.
"""
from __future__ import annotations

DISASTER_CLASSES = ["huracan", "inundacion", "sismo", "incendio", "volcan",
                    "tornado", "otro"]

# Clases habilitadas por política: solo incendio puede confirmarse.
HABILITADAS: dict[str, set[str]] = {"fire_only_v1": {"incendio"}}

# Umbral provisional: lo reemplaza el valor recomendado por el sweep de
# validación por evento (tools/evaluar_umbral_incendio.py). Si el sweep no
# alcanza el criterio de aceptación, queda este y se documenta.
UMBRAL_INCENDIO_DEFAULT = 0.70


def decidir_tipo(probs, umbral: float = UMBRAL_INCENDIO_DEFAULT,
                 politica: str = "fire_only_v1",
                 habilitadas: set[str] | None = None) -> dict:
    """Decide el tipo de desastre a partir de las probabilidades del modelo.

    Devuelve un dict con ``clase`` (top-1 crudo), ``conf``, ``estado``,
    ``confiable`` y ``nota``. Estados posibles:
      · confirmado_por_modelo    — clase habilitada y conf >= umbral
      · abstencion_por_confianza — clase habilitada pero conf < umbral
      · clase_no_habilitada      — el top-1 no puede afirmarse en esta política
      · modelo_deshabilitado     — no corrió el modelo (lo setea el pipeline)

    Fail-safe: una política desconocida abstiene todo (no lanza excepción).
    """
    probs = [float(p) for p in probs]
    if not probs:
        return {"clase": None, "conf": 0.0, "estado": "modelo_deshabilitado",
                "confiable": False, "nota": "sin probabilidades"}
    top1 = max(range(len(probs)), key=lambda i: probs[i])
    conf = probs[top1]
    clase = (DISASTER_CLASSES[top1] if top1 < len(DISASTER_CLASSES)
             else str(top1))
    hab = habilitadas if habilitadas is not None else HABILITADAS.get(politica, set())
    if clase not in hab:
        estado = "clase_no_habilitada"
        nota = f"{clase} no habilitada en {politica}"
    elif conf < umbral:
        estado = "abstencion_por_confianza"
        nota = f"incendio {conf:.2f} < umbral {umbral:.2f}"
    else:
        estado = "confirmado_por_modelo"
        nota = ""
    return {"clase": clase, "conf": conf, "estado": estado,
            "confiable": estado == "confirmado_por_modelo", "nota": nota}
