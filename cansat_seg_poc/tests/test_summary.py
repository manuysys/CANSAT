"""
Tests de ``cansat.summary`` — el schema de ``entrega/summary.json``.

Cubren el hallazgo 1.3: los dos productores del archivo escribían schemas
incompatibles y el frontend tuvo que tipar ``alertas`` como ``unknown[]``.
"""
import pytest

from cansat import summary as S


def fila(src="cap_0000", t=0.0, alt=250.0, verdict="ZONA SALUDABLE",
         alert=0, danado=0.0, diag="SIN DESASTRE", **kw):
    base = {"src": src, "t_s": t, "alt_m": alt, "verdict": verdict,
            "people": 2, "vehicles": 3, "alert": alert, "danado_pct": danado,
            "diag": diag, "sample_pri": "HIGH",
            "veg": 50.0, "bui": 10.0, "wat": 10.0, "bare": 20.0, "oth": 10.0}
    base.update(kw)
    return base


# ── build_summary ───────────────────────────────────────────────────────── #
def test_build_summary_schema_version():
    s = S.build_summary([fila()])
    assert s["schema_version"] == S.SCHEMA_VERSION


def test_build_summary_rechaza_cero_filas():
    with pytest.raises(ValueError):
        S.build_summary([])


def test_alertas_siempre_lista_de_dict_con_src():
    """El contrato: ``alertas`` es SIEMPRE list[dict] con 'src'."""
    rows = [fila("a", alert=0), fila("b", alert=1, danado=45.0,
                                     diag="INUNDACION SEVERA"), fila("c", alert=1)]
    s = S.build_summary(rows)
    assert isinstance(s["alertas"], list)
    assert len(s["alertas"]) == 2
    for a in s["alertas"]:
        assert isinstance(a, dict)
        assert a.get("src")
        assert a["src"] in ("b", "c")


def test_motivo_se_genera_si_no_viene():
    s = S.build_summary([fila("b", alert=1, danado=45.98, diag="INUNDACION SEVERA")])
    assert "INUNDACION SEVERA" in s["alertas"][0]["motivo"]
    assert "46.0" in s["alertas"][0]["motivo"]


def test_confianza_limitada_vacia_sin_estres():
    s = S.build_summary([fila("a"), fila("b", haze_pct=20.0, humidex=30.0)])
    conf = s["confianza_limitada"]
    assert conf["n_frames"] == 0
    assert conf["frames"] == []
    from cansat import stress as ST
    assert conf["umbrales"] == {"haze_pct": ST.CONTAM_DENSA,
                                "humidex": ST.HEAT_PELIGRO}


def test_confianza_limitada_marca_bruma_y_calor():
    rows = [fila("a", haze_pct=60.0),
            fila("b", humidex=50.0),
            fila("c", haze_pct=10.0, humidex=25.0)]
    s = S.build_summary(rows)
    conf = s["confianza_limitada"]
    assert conf["n_frames"] == 2
    por_src = {f["src"]: f["motivos"] for f in conf["frames"]}
    assert any("bruma densa" in m for m in por_src["a"])
    assert any("calor peligroso" in m for m in por_src["b"])
    assert "c" not in por_src


def test_alt_max_min():
    s = S.build_summary([fila("a", alt=250.0), fila("b", alt=100.0),
                         fila("c", alt=1.5)])
    assert s["alt_max_m"] == 250.0
    assert s["alt_min_m"] == 1.5


def test_conteos_de_personas_y_vehiculos():
    s = S.build_summary([fila("a", people=2, vehicles=3),
                         fila("b", people=5, vehicles=1)])
    assert s["personas_total"] == 7
    assert s["vehiculos_total"] == 4


def test_veredictos_es_un_conteo():
    s = S.build_summary([fila("a", verdict="ZONA SALUDABLE"),
                         fila("b", verdict="ZONA SALUDABLE"),
                         fila("c", verdict="ALTO ESTRÉS URBANO")])
    assert s["veredictos"] == {"ZONA SALUDABLE": 2, "ALTO ESTRÉS URBANO": 1}


def test_danado_promedio():
    # Ojo: `fila()` defaultea src="cap_0000", así que sin src distintos las dos
    # filas colisionan en danado_pct_por_frame y el promedio sale del último.
    s = S.build_summary([fila("a", danado=10.0), fila("b", danado=30.0)])
    assert s["danado_pct_prom"] == 20.0
    assert s["danado_pct_por_frame"] == {"a": 10.0, "b": 30.0}


def test_src_duplicados_se_avisan(capsys):
    """Hallazgo del propio test: dos filas con el mismo src se pisan en silencio."""
    S.build_summary([fila("x", danado=10.0), fila("x", danado=90.0)])
    assert "duplicado" in capsys.readouterr().out


def test_archivos_separa_rutas_de_conteos():
    """
    ANTES: un productor ponía rutas y el otro conteos ENTEROS en el mismo campo.
    Ahora van en dos claves separadas.
    """
    s = S.build_summary([fila()],
                        rutas={"corridor_map": "entrega/corridor_map.jpg"},
                        conteos={"vis": 12, "enhanced": 3})
    assert s["archivos"]["rutas"]["corridor_map"] == "entrega/corridor_map.jpg"
    assert s["archivos"]["conteos"]["vis"] == 12
    assert isinstance(s["archivos"]["rutas"]["corridor_map"], str)
    assert isinstance(s["archivos"]["conteos"]["vis"], int)


def test_terrain_por_frame_se_deriva_si_no_se_pasa():
    s = S.build_summary([fila("a", veg=10.0, bui=20.0, wat=30.0, bare=25.0, oth=15.0)])
    assert s["terrain_b5_por_frame"]["a"]["bui"] == 20.0


# ── normalize_summary: los dos formatos legacy ──────────────────────────── #
LEGACY_POST_FLIGHT = {
    "mision": "LB135", "n_frames": 2,
    "alertas": ["cap_0004", "cap_0005"],                    # ← list[str]
    "archivos": {                                            # ← rutas planas
        "corredor": "entrega/corridor_map.jpg",
        "enhanced": ["a.jpg", "b.jpg"],
        "evidencias": "outputs/mission/vis",
        "telemetria_csv": "outputs/mission/telemetry.csv",
        "seg_b5": "entrega/b5_seg",
    },
    "veredictos": {"ZONA SALUDABLE": 2},
}

LEGACY_DEMO = {
    "mision": "LB135", "n_frames": 2,
    "alertas": [{"src": "cap_0004", "diag": "INUNDACION SEVERA",   # ← list[dict]
                 "danado_pct": 45.98, "motivo": "x"}],
    "archivos": {"telemetry": "outputs/mission/telemetry.csv",      # ← conteos
                 "corridor_map": "outputs/corridor_map.jpg",
                 "vis": 12, "high_res": 12, "ens_seg": 6},
    "veredictos": {"ZONA SALUDABLE": 2},
}


def test_normaliza_alertas_legacy_de_strings():
    n = S.normalize_summary(LEGACY_POST_FLIGHT)
    assert all(isinstance(a, dict) for a in n["alertas"])
    assert [a["src"] for a in n["alertas"]] == ["cap_0004", "cap_0005"]


def test_normaliza_alertas_legacy_de_dicts():
    n = S.normalize_summary(LEGACY_DEMO)
    assert n["alertas"][0]["diag"] == "INUNDACION SEVERA"
    assert n["alertas"][0]["danado_pct"] == 45.98


def test_los_dos_legacy_convergen_al_mismo_schema():
    """EL OBJETIVO: dos productores distintos → un solo schema de salida."""
    a = S.normalize_summary(LEGACY_POST_FLIGHT)
    b = S.normalize_summary(LEGACY_DEMO)
    assert set(a["archivos"]) == set(b["archivos"]) == {"rutas", "conteos"}
    assert all(isinstance(x, dict) for x in a["alertas"])
    assert all(isinstance(x, dict) for x in b["alertas"])


def test_renombra_las_claves_legacy():
    n = S.normalize_summary(LEGACY_POST_FLIGHT)
    rutas = n["archivos"]["rutas"]
    assert "corridor_map" in rutas and "corredor" not in rutas
    assert "vis" in rutas and "evidencias" not in rutas
    assert "ens_seg" in rutas and "seg_b5" not in rutas
    assert "telemetry" in rutas and "telemetria_csv" not in rutas


def test_lista_de_rutas_se_vuelve_conteo_y_primera_ruta():
    n = S.normalize_summary(LEGACY_POST_FLIGHT)
    assert n["archivos"]["conteos"]["enhanced"] == 2
    assert n["archivos"]["rutas"]["enhanced"] == "a.jpg"


def test_normalize_none_y_no_dict():
    assert S.normalize_summary(None) is None
    assert S.normalize_summary([1, 2])["raw"] == [1, 2]


def test_normalize_es_idempotente():
    """Normalizar dos veces no debe cambiar nada."""
    n1 = S.normalize_summary(LEGACY_DEMO)
    n2 = S.normalize_summary(n1)
    assert n1["archivos"] == n2["archivos"]
    assert n1["alertas"] == n2["alertas"]


# ── validate ────────────────────────────────────────────────────────────── #
def test_validate_sin_avisos():
    s = S.build_summary([fila("a"), fila("b")])
    assert S.validate(s, n_frames_csv=2) == []


def test_validate_detecta_n_frames_inconsistente():
    s = S.build_summary([fila("a"), fila("b")])
    avisos = S.validate(s, n_frames_csv=5)
    assert any("n_frames" in a for a in avisos)


def test_validate_detecta_veredictos_que_no_suman():
    s = S.build_summary([fila("a"), fila("b")])
    s["veredictos"] = {"ZONA SALUDABLE": 99}
    assert any("veredictos suman" in a for a in S.validate(s, 2))


def test_validate_detecta_danio_fuera_de_rango():
    s = S.build_summary([fila()])
    s["danado_pct_prom"] = 150.0
    assert any("danado_pct_prom" in a for a in S.validate(s, 1))


def test_validate_detecta_alerta_sin_src():
    s = S.build_summary([fila()])
    s["alertas"] = [{"diag": "x"}]
    assert any("sin 'src'" in a for a in S.validate(s, 1))


# ── round-trip a disco ──────────────────────────────────────────────────── #
def test_write_read_roundtrip(tmp_path):
    s = S.build_summary([fila("a"), fila("b", alert=1, danado=30.0,
                                      diag="POSIBLE SISMO/VIENTO")])
    p = S.write_summary(tmp_path / "sub" / "summary.json", s)
    assert p.is_file()
    back = S.read_summary(p)
    assert back["alertas"][0]["diag"] == "POSIBLE SISMO/VIENTO"
    assert back["n_frames"] == 2


def test_read_summary_de_archivo_inexistente(tmp_path):
    assert S.read_summary(tmp_path / "nope.json") is None


def test_read_summary_de_archivo_corrupto(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{esto no es json", encoding="utf-8")
    assert S.read_summary(p) is None


def test_acentos_se_preservan(tmp_path):
    """ensure_ascii=False: el archivo debe ser legible a simple vista."""
    s = S.build_summary([fila(verdict="ALTO ESTRÉS URBANO")])
    p = S.write_summary(tmp_path / "s.json", s)
    assert "ALTO ESTRÉS URBANO" in p.read_text(encoding="utf-8")
