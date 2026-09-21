"""
Tests de ``cansat.onnxio`` — mapeo de entradas por nombre.

Cubren el hallazgo: el caso multi-entrada (siamés) mapeaba por POSICIÓN
(``zip(input_names, feed.values())``). Un re-export con otro orden intercambia
``pre``/``post`` en silencio.
"""
import numpy as np
import pytest

from cansat.onnxio import map_feed


def _t(v=0.0):
    return np.full((1, 3, 8, 8), v, dtype=np.float32)


def test_nombres_exactos():
    feed = {"pre": _t(1), "post": _t(2)}
    m, positional = map_feed(["pre", "post"], feed, "siam")
    assert not positional
    assert m == {"pre": "pre", "post": "post"}


def test_orden_del_dict_no_importa():
    """El bug: el mapeo posicional dependía del orden de inserción."""
    feed = {"post": _t(2), "pre": _t(1)}          # orden invertido a propósito
    m, positional = map_feed(["pre", "post"], feed, "siam")
    assert not positional
    assert m["pre"] == "pre" and m["post"] == "post"


def test_prefijos_pre_post():
    feed = {"pre": _t(1), "post": _t(2)}
    m, positional = map_feed(["pre_input", "post_input"], feed, "siam")
    assert not positional
    assert m == {"pre": "pre_input", "post": "post_input"}


def test_single_input_acepta_clave_logica():
    m, positional = map_feed(["input"], {"input": _t()}, "seg")
    assert not positional and m == {"input": "input"}


def test_fallback_posicional_se_marca():
    """Entradas genéricas (input_1/input_2): cae a posicional y hay que avisar."""
    feed = {"pre": _t(1), "post": _t(2)}
    m, positional = map_feed(["input_1", "input_2"], feed, "siam")
    assert positional is True
    assert m == {"pre": "input_1", "post": "input_2"}


def test_cantidad_incompatible_lanza():
    with pytest.raises(KeyError):
        map_feed(["solo_uno"], {"pre": _t(1), "post": _t(2)}, "siam")

class _Cv2Falso:
    def __init__(self, version):
        self.__version__ = version


def test_aviso_cv2_5_dnn():
    from cansat.onnxio import _aviso_cv2_dnn
    assert _aviso_cv2_dnn(_Cv2Falso("5.0.0")) is not None
    assert "opencv<5" in _aviso_cv2_dnn(_Cv2Falso("5.0.0"))
    assert _aviso_cv2_dnn(_Cv2Falso("4.13.0")) is None


def test_aviso_cv2_sin_version_no_revienta():
    from cansat.onnxio import _aviso_cv2_dnn
    assert _aviso_cv2_dnn(object()) is None
    assert _aviso_cv2_dnn(_Cv2Falso("no-numerica")) is None
