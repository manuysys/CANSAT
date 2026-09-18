"""
cansat — biblioteca compartida del proyecto CanSat LB135.

Nace para eliminar la duplicación que había llevado a que el mismo concepto
estuviera implementado de 2 a 62 veces, con variantes sutiles y a veces
contradictorias, repartidas en los 64 scripts de la raíz.

Módulos
-------
``indices``      USI / GVI / veredicto / diagnóstico de desastre. **FUENTE ÚNICA.**
``preprocess``   Normalización y preprocesado de imagen.  (antes: 62 variantes)
``metrics``      Matriz de confusión, IoU, F1, acuerdo.   (antes: 3 variantes)
``nodata``       Máscara de píxeles sin datos (borde vs umbral global).
``onnxio``       Nombres de nodo leídos del modelo, no hardcodeados.
``checkpoints``  Formato único de checkpoint + ``weights_only=True``.
``protocol``     Paquete de radio/UART LB135.             (antes: 3 formatos)
``summary``      Schema canónico de ``entrega/summary.json`` (antes: 2 schemas).
``paths``        Rutas resueltas desde el archivo, no desde el CWD.
``seed``         Reproducibilidad y device con fallback.

Convención: los scripts de la raíz siguen corriendo con CWD = raíz del
proyecto, así que ``from cansat.indices import usi`` funciona sin instalar.
Con ``pip install -e .`` funciona desde cualquier directorio.
"""

__version__ = "1.0.0"

import contextlib as _contextlib
import sys as _sys


def enable_utf8_stdout() -> None:
    """
    Fuerza stdout/stderr a UTF-8 en Windows.

    La consola cp1252 no puede codificar ``→``, ``≈``, ``✔`` ni emojis: cuando
    la salida se redirige o se pipea desde PowerShell, ``print`` lanza
    UnicodeEncodeError y el script muere A MITAD DE CAMINO. En el pipeline de
    vuelo eso pasa DESPUÉS de escribir la telemetría, así que se pierde la
    copia final a ``telemetry.csv`` (el contrato que lee la estación terrena).

    Se aplica al importar el paquete porque todos los scripts de vuelo y
    post-vuelo importan ``cansat.*``. Es idempotente.
    """
    for stream in (_sys.stdout, _sys.stderr):
        with _contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")
    # En Linux/Pi el stream ya está en UTF-8: reconfigurar es un no-op barato.


enable_utf8_stdout()

# E402/F401: el import va DESPUÉS de enable_utf8_stdout() a propósito; el
# reexport `paths` es parte de la API del paquete.
from . import paths  # noqa: E402,F401
