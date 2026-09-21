"""
Acceso a modelos ONNX sin nombres de nodo hardcodeados.

Antes, los literales ``"logits"`` y ``"input"`` (y ``"pre"``/``"post"`` en el
siamés) estaban **quemados en ~10 archivos**. Cualquier re-export con otro
nombre rompía todo con un mensaje críptico de ONNX Runtime.

Acá los nombres se leen del propio modelo.
"""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Sequence

import numpy as np


def _name_score(logical: str, real: str) -> int:
    """Cuánto se parece una clave lógica a un nombre real de entrada."""
    ln, rn = logical.lower(), real.lower()
    if ln == rn:
        return 3
    if rn.startswith(ln) or rn.endswith(ln):
        return 2
    return 1 if ln in rn else 0


def _aviso_cv2_dnn(cv2_mod) -> str | None:
    """
    Aviso si el OpenCV instalado es 5.x (medido: cv2.dnn divergente).

    Devuelve el texto del aviso o ``None``. Separado de ``_init_cv2`` para
    poder testearlo sin un modelo real.
    """
    try:
        major = int(cv2_mod.__version__.split(".")[0])
    except (AttributeError, ValueError):
        return None
    if major >= 5:
        return (f"cv2 {cv2_mod.__version__} con backend DNN. Se midió que 5.0.0 "
                f"difiere del ONNX de vuelo (argmax 0.01 vs ORT); con 4.x el "
                f"acuerdo es >0.99. Validar en la placa o fijar opencv<5.")
    return None


def map_feed(
    input_names: Sequence[str], feed: dict[str, np.ndarray], label: str = "model"
) -> tuple[dict[str, str], bool]:
    """
    Resuelve claves lógicas (``"pre"``, ``"post"``, ``"input"``) a los nombres
    reales de las entradas del ONNX. Devuelve ``(mapping, positional)``.

    ⚠ Antes el caso multi-entrada era **posicional a ciegas**:
      ``dict(zip(self.input_names, feed.values()))``. Si un re-export cambiaba
      el orden de ``pre``/``post`` del siamés, los tensores se intercambiaban en
      silencio y el modelo predecía cualquier cosa sin un solo warning. Ahora se
      matchea por nombre (exacto o por prefijo/sufijo) y, si no hay match, el
      fallback posicional se marca para que el llamador avise.

    Algoritmo: para cada clave lógica gana el nombre real no usado con mejor
    score. Si todas las claves matchean y no sobran entradas, ``positional`` es
    ``False``.
    """
    logicals = list(feed.keys())
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for logical in logicals:
        best, best_score = None, 0
        for name in input_names:
            if name in used:
                continue
            s = _name_score(str(logical), name)
            if s > best_score:
                best, best_score = name, s
        if best is not None:
            mapping[logical] = best
            used.add(best)

    if len(mapping) == len(logicals) == len(input_names):
        return mapping, False
    if len(input_names) == len(logicals):
        return dict(zip(logicals, input_names, strict=True)), True
    raise KeyError(
        f"[{label}] el modelo espera {list(input_names)} y se le pasó {sorted(logicals)}"
    )


class OnnxModel:
    """Wrapper mínimo sobre ``onnxruntime.InferenceSession``."""

    def __init__(
        self,
        path: str | Path,
        providers: Sequence[str] | None = None,
        required: bool = True,
        label: str = "",
        hint: str = "",
        backend: str | None = None,
    ):
        self.path = Path(path)
        self.label = label or self.path.stem
        self.required = required
        self.hint = hint
        self.sess = None
        self._net = None
        self.backend = ""
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.input_shape: tuple | None = None
        self._feed_map: dict[str, str] | None = None
        self._positional_feed = False

        if not self.path.is_file():
            msg = f"[{self.label}] no existe el modelo ONNX: {self.path}"
            if required:
                raise FileNotFoundError(
                    msg
                    + (
                        f"\n  → {hint}"
                        if hint
                        else "\n  → Regeneralo con el script de export correspondiente "
                        "(ver MODELS.yaml) o pasá la ruta correcta por CLI."
                    )
                )
            print(f"  [WARN] {msg}" + (f" — {hint}" if hint else "") + " Modelo desactivado.")
            return

        if backend in (None, "auto", "onnxruntime"):
            try:
                self._init_onnxruntime(providers)
                return
            except ImportError:
                if backend == "onnxruntime":
                    raise
                # Sin onnxruntime (caso Raspberry Pi Zero v1 / ARMv6): seguir
                # con OpenCV DNN, que ya es dependencia del pipeline.
        self._init_cv2()

    def _init_onnxruntime(self, providers) -> None:
        import onnxruntime as ort

        self.sess = ort.InferenceSession(
            str(self.path), providers=list(providers) if providers else None
        )
        self.backend = "onnxruntime"
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]
        shp = self.sess.get_inputs()[0].shape
        # [1, 3, 320, 320] → 320 ; ['batch', 3, 'h', 'w'] → None (dinámico)
        self.input_shape = tuple(d if isinstance(d, int) else None for d in shp)

    def _init_cv2(self) -> None:
        """
        Backend de respaldo: ``cv2.dnn``. Necesario en placas sin wheels de
        onnxruntime (Pi Zero v1, ARMv6). Limitaciones frente a ORT:

          · no expone el shape de entrada ni la cantidad de clases → ``size_px``
            y ``n_classes`` devuelven ``None`` (el pipeline usa ``--img-size``);
          · **no lee pesos externos** (``.onnx.data``): el modelo tiene que ser
            autocontenido. Usar ``python tools/onnx_inline.py`` para empotrar.
        """
        try:
            import cv2
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                f"[{self.label}] no hay onnxruntime ni OpenCV para inferir."
            ) from e
        try:
            self._net = cv2.dnn.readNetFromONNX(str(self.path))
        except cv2.error as e:
            data = self.path.with_suffix(self.path.suffix + ".data")
            pista = (
                "\n  → OpenCV DNN no lee pesos externos. Si al lado hay un "
                f"{self.path.name}.data, empotralo en el .onnx con:\n"
                f"      python tools/onnx_inline.py \"{self.path}\""
            )
            if data.is_file():
                pista = f"\n  → El .data está presente: {data.name}." + pista
            detalle = " ".join(str(e).split())[:300]
            raise RuntimeError(
                f"[{self.label}] OpenCV DNN no pudo cargar {self.path.name}: {detalle}{pista}"
            ) from e
        self.backend = "opencv-dnn"
        # ⚠ Medición 2026-09-21: OpenCV 5.0.0 da logits absurdos en cv2.dnn con
        # el ONNX de vuelo. requirements-flight.txt fija <5; esto avisa igual.
        if aviso := _aviso_cv2_dnn(cv2):
            print(f"  [WARN] {self.label}: {aviso}")
        try:
            self.output_names = list(self._net.getUnconnectedOutLayersNames())
        except cv2.error:  # pragma: no cover
            self.output_names = []

    # ── estado ────────────────────────────────────────────────────────── #
    def __bool__(self) -> bool:
        return self.sess is not None or self._net is not None

    @property
    def available(self) -> bool:
        return bool(self)

    @property
    def n_classes(self) -> int | None:
        """Cantidad de clases de la salida (dim 1 de un NCHW), o ``None``."""
        if not self.sess:
            return None
        shp = self.sess.get_outputs()[0].shape
        if len(shp) == 4 and isinstance(shp[1], int):
            return shp[1]
        return None

    @property
    def size_px(self) -> int | None:
        """Lado de la entrada cuadrada fija, o ``None`` si no se puede saber."""
        if not self.input_shape or len(self.input_shape) != 4:
            return None
        h, w = self.input_shape[2], self.input_shape[3]
        return h if (isinstance(h, int) and h == w) else None

    def describe(self) -> str:
        if not self:
            return f"{self.label}: NO DISPONIBLE ({self.path})"
        mb = os.path.getsize(self.path) / 1e6
        shp = list(self.input_shape or []) if self.sess else ["shape via --img-size"]
        return (
            f"{self.label}: {self.path.name} {mb:.1f} MB · "
            f"[{self.backend}] in={self.input_names or '(cv2.dnn)'}{shp} · "
            f"out={self.output_names}"
        )

    # ── inferencia ────────────────────────────────────────────────────── #
    def run(self, feed: dict[str, np.ndarray], output: str | None = None) -> np.ndarray:
        """
        Corre el modelo resolviendo los nombres de nodo desde el archivo.

        ``feed`` puede usar claves lógicas: si el modelo tiene una sola entrada
        y se pasa ``{"input": x}``, se remapea al nombre real.
        """
        if not self:
            raise RuntimeError(f"[{self.label}] el modelo no está cargado.")
        if self.sess is not None:
            real = self._resolve_feed(feed)
            out_name = output or self.output_names[0]
            return self.sess.run([out_name], real)[0]
        return self._run_cv2(feed, output)

    def _run_cv2(self, feed: dict[str, np.ndarray], output: str | None = None) -> np.ndarray:
        import cv2

        for name, arr in feed.items():
            blob = np.ascontiguousarray(arr, dtype=np.float32)
            try:
                self._net.setInput(blob, str(name))
            except cv2.error:
                # Nombre no reconocido (modelos de una sola entrada): sin nombre.
                self._net.setInput(blob)
        out_name = output or (self.output_names[0] if self.output_names else "")
        return self._net.forward(out_name) if out_name else self._net.forward()

    def _resolve_feed(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if set(feed) == set(self.input_names):
            return dict(feed)
        # Un solo nodo de entrada: aceptar cualquier clave que le manden.
        if len(self.input_names) == 1 and len(feed) == 1:
            return {self.input_names[0]: next(iter(feed.values()))}
        # Multi-entrada: resolver por nombre y cachear el mapeo. Si cae a
        # posicional se avisa UNA vez (no por frame).
        if self._feed_map is None or set(self._feed_map) != set(feed):
            self._feed_map, positional = map_feed(self.input_names, feed, self.label)
            self._positional_feed = positional
            if positional:
                print(f"  [WARN] {self.label}: no pude mapear {sorted(feed)} por nombre "
                      f"a {self.input_names}; uso ORDEN POSICIONAL. Verificá que el "
                      f"export declare las entradas en el orden esperado.")
        return {real: feed[logical] for logical, real in self._feed_map.items()}

    def argmax(self, feed: dict[str, np.ndarray], axis: int = 1) -> np.ndarray:
        """Salida → mapa de clases (sin el eje de clases)."""
        out = self.run(feed)
        return np.argmax(out[0], axis=axis - 1) if out.ndim == 4 else np.argmax(out, axis=0)


def load_optional(
    path: str | Path,
    label: str = "",
    providers: Sequence[str] | None = None,
    hint: str = "",
    backend: str | None = None,
) -> OnnxModel:
    """Carga un modelo opcional: si falta, devuelve un objeto falsy en vez de romper."""
    return OnnxModel(path, providers=providers, required=False, label=label,
                     hint=hint, backend=backend)


def load_required(
    path: str | Path,
    label: str = "",
    providers: Sequence[str] | None = None,
    hint: str = "",
    backend: str | None = None,
) -> OnnxModel:
    """
    Carga un modelo obligatorio: si falta, ``FileNotFoundError`` con mensaje útil.

    ``backend``: ``"onnxruntime"`` | ``"cv2"`` | ``None`` (auto: ORT si está
    instalado, si no ``cv2.dnn`` — el camino de la Pi Zero v1 sin wheels de ORT).
    """
    return OnnxModel(path, providers=providers, required=True, label=label,
                     hint=hint, backend=backend)
