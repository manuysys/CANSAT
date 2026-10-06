"""Tests de cansat.metrics (confusión, IoU por clase, mIoU, acuerdo)."""
from __future__ import annotations

import numpy as np

from cansat import metrics as M


def test_confusion_basica():
    gt = np.array([0, 1, 2, 2])
    pred = np.array([0, 1, 1, 2])
    conf = M.confusion(gt, pred, 3)
    assert conf.shape == (3, 3)
    assert conf[0, 0] == 1 and conf[1, 1] == 1 and conf[2, 2] == 1
    assert conf[2, 1] == 1          # GT 2 predicho como 1


def test_confusion_ignora_255():
    gt = np.array([0, 1, 255])
    pred = np.array([0, 1, 1])
    conf = M.confusion(gt, pred, 2, ignore_index=M.IGNORE_INDEX)
    assert conf.sum() == 2


def test_from_confusion_perfecto():
    m = M.from_confusion(np.diag([10, 20, 30]).astype(np.int64))
    assert m.miou == 1.0
    assert m.pixel_acc == 1.0
    assert np.allclose(m.iou, 1.0)
    d = m.as_dict(["a", "b", "c"])
    assert d["miou"] == 1.0 and d["por_clase"]["b"]["iou"] == 1.0


def test_accumulate_suma_confusiones():
    total = np.zeros((2, 2), np.int64)
    M.accumulate(total, np.array([0, 1]), np.array([0, 1]), 2)
    M.accumulate(total, np.array([1, 1]), np.array([1, 0]), 2)
    assert total[0, 0] == 1 and total[1, 1] == 2 and total[1, 0] == 1


def test_agreement():
    a = np.array([[0, 1], [2, 2]])
    assert M.agreement(a, a.copy()) == 1.0
    b = np.array([[0, 0], [2, 2]])
    assert M.agreement(a, b) == 0.75
