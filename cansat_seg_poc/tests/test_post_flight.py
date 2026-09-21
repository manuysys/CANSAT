"""Tests de post_flight.py — helpers del post-vuelo (sin modelos ONNX).

Cubre la resolución del SegFormer B5 (el bug original salteaba toda la segunda
pasada en silencio), el buscador de frames, el orden estable y `sliding_logits`
con frame más chico que la ventana (el índice negativo de la auditoría).
"""

import numpy as np

import post_flight as PF


# ── resolve_b5 ──────────────────────────────────────────────────────────── #
def test_resolve_b5_explicito_inexistente(capsys):
    assert PF.resolve_b5("no/existe.onnx") is None
    assert "No encontré ningún SegFormer B5" in capsys.readouterr().out


def test_resolve_b5_elige_el_primer_candidato_existente(tmp_path, monkeypatch):
    a = tmp_path / "b5_512.onnx"       # NO existe
    b = tmp_path / "b5_320.onnx"
    b.write_bytes(b"modelo")
    monkeypatch.setattr(PF, "B5_CANDIDATES", (str(a), str(b)))
    assert PF.resolve_b5(None) == b


def test_resolve_b5_explicito_existente(tmp_path):
    p = tmp_path / "elegido.onnx"
    p.write_bytes(b"modelo")
    assert PF.resolve_b5(str(p)) == p


# ── frames ──────────────────────────────────────────────────────────────── #
def test_frame_path_extensiones_y_glob(tmp_path):
    (tmp_path / "cam_001.jpg").write_bytes(b"x")
    (tmp_path / "cam_002.bmp").write_bytes(b"x")
    assert PF.frame_path(tmp_path, "cam_001").suffix == ".jpg"
    assert PF.frame_path(tmp_path, "cam_002").suffix == ".bmp"
    assert PF.frame_path(tmp_path, "no_existe") is None


def test_frame_sort_key_usa_t_s_y_no_revienta_con_src_textual():
    early = {"t_s": "10.0", "src": "cam_010"}
    late = {"t_s": "20.5", "src": "cam_002"}
    sin_ts = {"t_s": "", "src": "cam_001"}
    assert PF.frame_sort_key(early, 0) < PF.frame_sort_key(late, 0)
    # Antes era int(src) y explotaba con cam_001: ahora t_s inválida va al final.
    assert PF.frame_sort_key(late, 0) < PF.frame_sort_key(sin_ts, 0)
    # Empate de t_s: desempata el src como texto y el índice.
    assert PF.frame_sort_key(early, 1) > PF.frame_sort_key(early, 0)


# ── visualización ───────────────────────────────────────────────────────── #
def test_color_seg_devuelve_overlay_del_tamano_de_la_imagen():
    seg = np.array([[0, 1], [2, 3]], np.uint8)
    img = np.zeros((8, 16, 3), np.uint8)
    out = PF.color_seg(seg, img)
    assert out.shape == img.shape and out.dtype == np.uint8


# ── sliding_logits (frame más chico que la ventana) ─────────────────────── #
class _SessStub:
    """Sesión ONNX falsa: devuelve logits 0 del tamaño de la ventana."""

    n_classes = 5
    size_px = None

    def run(self, _feed):
        return np.zeros((1, self.n_classes, 32, 32), np.float32)


def test_sliding_logits_frame_menor_que_ventana_no_usa_indices_negativos():
    img = np.full((20, 24, 3), 120, np.uint8)
    out = PF.sliding_logits(_SessStub(), img, win=32, stride=16)
    assert out.shape == (5, 20, 24)
    assert np.isfinite(out).all()


def test_sliding_logits_frame_grande_suma_ventanas():
    img = np.full((64, 64, 3), 120, np.uint8)
    out = PF.sliding_logits(_SessStub(), img, win=32, stride=16)
    assert out.shape == (5, 64, 64)
    assert np.isfinite(out).all()
