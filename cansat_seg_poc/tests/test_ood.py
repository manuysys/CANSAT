"""Tests de cansat/ood.py — proxy de fuera-de-distribución (drift)."""

import numpy as np
import pytest

from cansat import ood as OOD


def _ref(**over):
    base = {
        "fuente": "test",
        "clases_prom": [0.4, 0.1, 0.1, 0.1, 0.3],
        "umbral_js": 0.1,
        "veg_exg_pct_media": 40.0,
        "veg_exg_pct_std": 10.0,
        "shadow_pct_media": 12.0,
        "shadow_pct_std": 4.0,
    }
    base.update(over)
    return base


# ── JS ──────────────────────────────────────────────────────────────────── #
def test_js_identicas_es_cero():
    p = [0.2, 0.3, 0.5]
    assert OOD.js_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_js_soporte_disjunto_es_uno():
    assert OOD.js_divergence([1, 0], [0, 1]) == pytest.approx(1.0, abs=1e-9)


def test_js_simetrica_y_normaliza():
    a, b = [2.0, 1.0, 1.0], [1.0, 1.0, 2.0]   # sin normalizar
    assert OOD.js_divergence(a, b) == pytest.approx(OOD.js_divergence(b, a), abs=1e-9)
    assert OOD.js_divergence(a, b) == pytest.approx(OOD.js_divergence([0.5, 0.25, 0.25],
                                                                     [0.25, 0.25, 0.5]),
                                                    abs=1e-9)


def test_js_formas_distintas_falla():
    with pytest.raises(ValueError):
        OOD.js_divergence([1, 2], [1, 2, 3])


# ── score ───────────────────────────────────────────────────────────────── #
def test_score_sin_referencia_no_inventa():
    res = OOD.score([0.4, 0.1, 0.1, 0.1, 0.3], referencia=None)
    assert res["ood_score"] is None and res["ood_flag"] is None


def test_score_frame_tipico_no_marca():
    res = OOD.score([0.4, 0.1, 0.1, 0.1, 0.3], veg_exg_pct=40.0, shadow_pct=12.0,
                    referencia=_ref())
    assert res["ood_flag"] is False
    assert res["ood_score"] == pytest.approx(0.0, abs=1e-6)


def test_score_clases_muy_distintas_marca():
    # Casi todo edificio: lejísimos de la referencia de Val.
    res = OOD.score([0.02, 0.9, 0.03, 0.02, 0.03], veg_exg_pct=40.0,
                    shadow_pct=12.0, referencia=_ref())
    assert res["ood_flag"] is True
    assert res["componentes"]["js"] > res["componentes"]["umbral_js"]


def test_score_z_de_exg_marca_aunque_las_clases_coincidan():
    res = OOD.score([0.4, 0.1, 0.1, 0.1, 0.3], veg_exg_pct=90.0,  # +5σ
                    shadow_pct=12.0, referencia=_ref())
    assert res["ood_flag"] is True
    assert res["componentes"]["z_veg_exg_pct"] == pytest.approx(5.0, abs=0.01)


# ── referencia ──────────────────────────────────────────────────────────── #
def test_cargar_referencia_tolerante(tmp_path):
    assert OOD.cargar_referencia(tmp_path / "no.json") is None
    roto = tmp_path / "roto.json"
    roto.write_text("{no es json", encoding="utf-8")
    assert OOD.cargar_referencia(roto) is None


def test_referencia_real_del_repo_si_existe():
    from cansat import paths as PROJ
    ruta = PROJ.ROOT / OOD.REFERENCIA_DEFECTO
    if not ruta.is_file():
        pytest.skip("referencia OOD no construida todavía")
    ref = OOD.cargar_referencia(ruta)
    assert ref is not None
    assert len(ref["clases_prom"]) == 5
    assert 0.0 < ref["umbral_js"] <= 1.0
    assert np.isclose(sum(ref["clases_prom"]), 1.0, atol=0.02)
