"""Split conformal para el umbral de incendio (cansat/conformal.py)."""
from __future__ import annotations

from cansat import conformal


def test_cuantil_conformal_conocido():
    s = [0.1, 0.2, 0.3, 0.4, 0.5]
    # n=5, alpha=0.2 -> k=ceil(6*0.8)=5 -> s[4]
    assert conformal.cuantil_conformal(s, 0.2) == 0.5


def test_alpha_muy_chico_sin_datos_abstiene():
    # k=ceil(6*0.99)=6 > n=5 -> sin datos para ese alpha: fail-safe
    assert conformal.cuantil_conformal([0.1, 0.2, 0.3, 0.4, 0.5], 0.01) == 1.0


def test_sin_calibracion_abstiene_todo():
    assert conformal.cuantil_conformal([], 0.05) == 1.0


def test_recall_y_fpr_medidos():
    rec, ok, tot = conformal.recall([0.9, 0.4], 0.5)
    assert (rec, ok, tot) == (0.5, 1, 2)
    fp, nfp, n = conformal.fpr([0.2, 0.6], 0.5)
    assert (fp, nfp, n) == (0.5, 1, 2)


def test_alpha_mayor_umbral_menor_o_igual():
    s = [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
    t1 = conformal.cuantil_conformal(s, 0.05)
    t2 = conformal.cuantil_conformal(s, 0.20)
    assert t2 <= t1


def test_evaluar_csv(tmp_path):
    p = tmp_path / "probs.csv"
    p.write_text(
        "evento,es_incendio,p_incendio\n"
        "xbd:socal-fire,1,0.95\n"
        "xbd:socal-fire,1,0.85\n"
        "xbd:hurricane-harvey,0,0.10\n"
        "xbd:hurricane-harvey,0,0.20\n"
        "xbd:hurricane-harvey,0,0.30\n"
        "xbd:hurricane-harvey,0,0.40\n",
        encoding="utf-8")
    res = conformal.evaluar(p, alphas=(0.25,))
    assert res["n_incendio"] == 2 and res["n_no_incendio"] == 4
    f = res["filas"][0]
    # n=4, alpha=0.25 -> k=ceil(5*0.75)=4 -> tau = 0.40 -> recall 1.0
    assert f["umbral_tau"] == 0.4
    assert f["recall_incendio"] == 1.0
    assert "no garantizado" in res["nota"]
