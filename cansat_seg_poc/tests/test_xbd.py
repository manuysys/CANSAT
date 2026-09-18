"""
Tests de ``cansat.xbd`` — split por desastre y métrica de daño.

Lo que fijan: que ningún desastre aparezca en train y val a la vez (la fuga
geográfica que inflaba el IoU), y que la métrica de selección sea la de la
misión (daño sobre edificios).
"""
import numpy as np
import pytest

from cansat.xbd import grupo, iou_dano_edificios, split_por_desastre


def _fila(nombre):
    return {"name": nombre, "image": f"{nombre}.png", "mask": f"{nombre}.png"}


def test_grupo_extrae_el_desastre():
    assert grupo("hurricane-harvey_00000426") == "hurricane-harvey"
    assert grupo("socal-fire_00001001_post_disaster") == "socal-fire"
    assert grupo("palu-tsunami_00000003") == "palu-tsunami"


def test_split_no_mezcla_desastres():
    rows = [_fila(f"{d}_{i:04d}") for d in
            ("hurricane-harvey", "socal-fire", "palu-tsunami", "midwest-flooding")
            for i in range(10)]
    train, val, val_g = split_por_desastre(rows, holdout_frac=0.25, seed=42)
    assert val_g, "el holdout no puede quedar vacío"
    g_train = {grupo(r["name"]) for r in train}
    g_val = {grupo(r["name"]) for r in val}
    assert not (g_train & g_val), "hay desastres en train Y val (fuga)"
    assert g_val == set(val_g)
    assert len(train) + len(val) == len(rows)


def test_split_es_reproducible():
    rows = [_fila(f"{d}_{i:04d}") for d in "abcdef" for i in range(5)]
    a = split_por_desastre(rows, 0.33, seed=7)
    b = split_por_desastre(rows, 0.33, seed=7)
    assert [r["name"] for r in a[1]] == [r["name"] for r in b[1]]


def test_split_con_un_solo_desastre_reserva_ese_desastre():
    """Caso borde: si hay un único grupo, val = ese grupo y train queda vacío."""
    rows = [_fila(f"hurricane-x_{i}") for i in range(4)]
    train, _val, val_g = split_por_desastre(rows, 0.2)
    assert val_g == ["hurricane-x"] and not train


# ── Métrica de daño sobre edificios ───────────────────────────────────────
def test_iou_dano_ignora_el_fondo():
    pred = np.array([[2, 0], [0, 0]], np.uint8)
    gt = np.array([[2, 0], [0, 0]], np.uint8)
    assert iou_dano_edificios(pred, gt) == pytest.approx(1.0)
    # Un falso positivo SOBRE FONDO no entra en la unión: en la misión el daño
    # se enmascara con los edificios predichos, así que ahí no cuenta.
    pred_fp = np.array([[2, 2], [0, 0]], np.uint8)
    assert iou_dano_edificios(pred_fp, gt) == pytest.approx(1.0)
    # En cambio, predecir daño donde el GT dice edificio INTACTO sí penaliza.
    pred_fp2 = np.array([[2, 0], [2, 0]], np.uint8)
    gt2 = np.array([[2, 0], [1, 0]], np.uint8)
    assert iou_dano_edificios(pred_fp2, gt2) == pytest.approx(0.5)


def test_iou_dano_sin_edificios_ni_predicciones_es_cero():
    pred = np.zeros((4, 4), np.uint8)
    gt = np.zeros((4, 4), np.uint8)
    assert iou_dano_edificios(pred, gt) == 0.0


def test_iou_dano_con_gt_intacto_no_infla():
    """Un edificio intacto mal predicho como dañado baja el IoU."""
    pred = np.array([[2]], np.uint8)
    gt = np.array([[1]], np.uint8)          # intacto
    assert iou_dano_edificios(pred, gt) == 0.0
