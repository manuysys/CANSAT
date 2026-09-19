"""Tests de cansat/population.py — densidad real por GPS (grilla WorldPop)."""
import csv

from cansat import population as POP


def _grid(tmp_path, filas):
    p = tmp_path / "grid.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "dens_km2"])
        w.writerows(filas)
    return p


def test_lookup_celda_exacta(tmp_path):
    p = _grid(tmp_path, [[-34.6, -58.6, 6500.0]])
    d, f = POP.densidad(-34.6, -58.6, path=p)
    assert d == 6500.0 and f == "worldpop"


def test_celda_redondeada_al_centro_mas_cercano(tmp_path):
    p = _grid(tmp_path, [[-34.6, -58.6, 6500.0]])
    d, f = POP.densidad(-34.62, -58.63, path=p)
    assert d == 6500.0 and f == "worldpop"


def test_sin_fix_usa_default(tmp_path):
    p = _grid(tmp_path, [[-34.6, -58.6, 6500.0]])
    d, f = POP.densidad(0.0, 0.0, default=1500.0, path=p)
    assert d == 1500.0 and f == "supuesto"


def test_celda_inexistente_usa_default(tmp_path):
    p = _grid(tmp_path, [[-34.6, -58.6, 6500.0]])
    d, f = POP.densidad(-20.0, -60.0, default=1500.0, path=p)
    assert d == 1500.0 and f == "supuesto"


def test_sin_grilla_no_explota(tmp_path):
    d, f = POP.densidad(-34.6, -58.6, default=1200.0,
                        path=tmp_path / "no-existe.csv")
    assert f == "supuesto" and d == 1200.0
