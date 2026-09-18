"""
DEPRECADO — usar ``cansat.crf.dense_crf``.

Este módulo tenía un "Dense CRF" que no usaba la imagen de guía (la variable
``I`` se calculaba y nunca se leía), aplicaba ``sigmaColor=0.051`` sobre valores
en [0,1] —escala equivocada, así que no suavizaba nada— y reiniciaba el
mean-field desde el unary en cada iteración.

Se mantiene como shim para no romper los ``from crf_refine import dense_crf``
existentes. La implementación correcta vive en ``cansat/crf.py`` y los motivos
están documentados ahí.
"""
from __future__ import annotations

from cansat.crf import dense_crf as _dense_crf

__all__ = ["dense_crf"]


def dense_crf(probs, img, iters: int = 5, sxy: float = 3.0,
              srgb: float = 13.0, compat: float = 5.0):
    """
    Firma vieja (``sxy``/``srgb``) adaptada a la implementación correcta.

    ⚠ ``srgb`` ya NO se divide por 255. El bug histórico era justamente ese:
    ``srgb=13`` en escala 0-255 dividido por 255 daba 0.051 sobre
    probabilidades en [0,1], y con ese valor el filtro no promediaba nada.
    ``cansat.crf`` espera ``sigma_color`` en escala 0-255 (lo traduce a ``eps``
    del filtro guiado), así que se pasa tal cual.
    """
    return _dense_crf(probs, img, iters=iters,
                      sigma_color=max(1.0, float(srgb)),
                      sigma_space=sxy, compat=compat)
