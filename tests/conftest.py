"""Configuración compartida de pytest."""
import sys
from pathlib import Path

# Los scripts del proyecto asumen CWD = raíz, así que la agregamos al path para
# poder importar `cansat` y los módulos de la raíz desde tests/.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
