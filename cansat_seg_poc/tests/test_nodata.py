"""
Tests de ``cansat.nodata`` — la máscara de píxeles sin datos.

Cubren el hallazgo 1.7: el umbral global de luminancia descartaba sombras y
asfalto oscuro de los frames de cámara real, y sobre ese remanente se calculaban
porcentajes, USI, veredicto, sampler y alerta.
"""
import numpy as np
import pytest

from cansat import nodata as ND


@pytest.fixture
def tile_con_borde():
    """Tile 64x64 con contenido válido y un borde negro de 8 px (estilo LoveDA)."""
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    img[:8, :] = 0
    img[-8:, :] = 0
    img[:, :8] = 0
    img[:, -8:] = 0
    return img


@pytest.fixture
def frame_con_sombra():
    """Frame de cámara: contenido normal con una mancha OSCURA en el medio (sombra)."""
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    img[24:40, 24:40] = 5        # sombra profunda, pero NO toca el borde
    return img


# ── threshold_mask (comportamiento viejo) ───────────────────────────────── #
def test_threshold_desactivado_con_cero():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    assert ND.threshold_mask(img, 0).all()


def test_threshold_marca_el_borde_negro(tile_con_borde):
    m = ND.threshold_mask(tile_con_borde, 15)
    assert not m[:8, :].any()          # borde superior = sin datos
    assert m[32, 32]                   # centro = con datos


def test_threshold_se_come_la_sombra_interna(frame_con_sombra):
    """
    EL BUG. Con umbral global, una sombra en el medio del frame se marca como
    "sin datos" aunque sea terreno perfectamente válido.
    """
    m = ND.threshold_mask(frame_con_sombra, 15)
    assert not m[32, 32], "el umbral global descarta la sombra (comportamiento viejo)"


# ── border_mask (comportamiento nuevo) ──────────────────────────────────── #
def test_border_mask_marca_el_borde_negro(tile_con_borde):
    m = ND.border_mask(tile_con_borde, 15)
    assert not m[:8, :].any()
    assert not m[-8:, :].any()
    assert m[32, 32]


def test_border_mask_conserva_la_sombra_interna(frame_con_sombra):
    """La corrección: la sombra NO toca el borde, así que es terreno válido."""
    m = ND.border_mask(frame_con_sombra, 15)
    assert m[32, 32], "la sombra interna debe conservarse como dato válido"


def test_border_mask_con_imagen_toda_oscura(tile=None):
    """
    Un frame entero oscuro (vuelo nocturno, tapa de la lente, subexposición)
    sí es todo sin-datos: la mancha toca el borde por todos lados.
    """
    img = np.full((32, 32, 3), 3, dtype=np.uint8)
    m = ND.border_mask(img, 15)
    assert not m.any()


def test_border_mask_con_imagen_toda_clara():
    img = np.full((32, 32, 3), 200, dtype=np.uint8)
    assert ND.border_mask(img, 15).all()


def test_border_mask_desactivado_con_umbral_cero():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    assert ND.border_mask(img, 0).all()


def test_border_mask_sin_negro_no_descarta_nada():
    img = np.full((32, 32, 3), 60, dtype=np.uint8)
    assert ND.border_mask(img, 15).all()


# ── valid_at_size ───────────────────────────────────────────────────────── #
def test_valid_at_size_reescala_con_vecino_mas_cercano():
    m = np.zeros((8, 8), dtype=bool)
    m[:, 4:] = True
    out = ND.valid_at_size(m, 4)
    assert out.shape == (4, 4)
    assert out.dtype == bool
    assert not out[:, :2].any()
    assert out[:, 2:].all()


def test_valid_at_size_no_inventa_valores_intermedios():
    """INTER_NEAREST: una máscara booleana no puede quedar en 0.5."""
    m = np.zeros((16, 16), dtype=bool)
    m[8:, :] = True
    out = ND.valid_at_size(m, 8)
    assert set(np.unique(out).tolist()) <= {False, True}


# ── terrain_percentages ─────────────────────────────────────────────────── #
def test_percentages_suman_100_con_todo_valido():
    seg = np.array([[0, 0, 1, 1], [2, 2, 3, 4]], dtype=np.int64)
    valid = np.ones((2, 4), dtype=bool)
    pcts, dom, n = ND.terrain_percentages(seg, valid, 5)
    assert sum(pcts) == pytest.approx(100.0)
    assert n == 8
    assert dom in (0, 1)                    # empate entre veg y building


def test_percentages_excluyen_los_pixeles_invalidos():
    seg = np.zeros((4, 4), dtype=np.int64)  # todo vegetación
    seg[0, :] = 1                           # una fila de edificios
    valid = np.ones((4, 4), dtype=bool)
    valid[0, :] = False                     # pero esa fila no cuenta
    pcts, _dom, n = ND.terrain_percentages(seg, valid, 5)
    assert pcts[0] == pytest.approx(100.0)  # 100 % vegetación
    assert pcts[1] == 0.0
    assert n == 12


def test_percentages_sin_pixeles_validos_no_inventa_una_clase():
    """
    BUG PREEXISTENTE: con 0 píxeles válidos, argmax sobre [0,0,0,0,0] devolvía 0,
    o sea "vegetation" como clase dominante, y el veredicto salía
    "ZONA SALUDABLE". Ahora devuelve dominante = -1.
    """
    seg = np.zeros((4, 4), dtype=np.int64)
    valid = np.zeros((4, 4), dtype=bool)
    pcts, dom, n = ND.terrain_percentages(seg, valid, 5)
    assert pcts == [0.0] * 5
    assert dom == -1
    assert n == 0
