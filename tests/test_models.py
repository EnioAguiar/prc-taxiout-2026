import numpy as np
import pandas as pd

from models import (
    TwoStageNM,
    apply_calibration,
    apply_lines,
    calib_groups,
    combine,
    copied_from_sched,
    fit_calibration,
    fit_lines,
    isotonic_fit,
    line_rows,
    nm_groups,
)


def test_combina_pela_esperanca_da_mistura():
    p = np.array([0.0, 1.0, 0.5, 0.5])
    ms = np.array([5000.0, 5000.0, 5000.0, np.nan])  # sem SCHED: só o regressor
    reg = np.array([900.0, 900.0, 900.0, 900.0])
    assert combine(p, ms, reg).tolist() == [900.0, 5000.0, 2950.0, 900.0]


def test_previsao_nunca_negativa():
    assert combine(np.array([0.0]), np.array([0.0]), np.array([-50.0])).tolist() == [0.0]


def test_rotulo_de_copia_usa_60_s_e_trata_nulos_como_normais():
    t = pd.Timestamp("2025-07-01 10:00", tz="UTC")
    s = pd.Timedelta(seconds=1)
    df = pd.DataFrame({
        "SCHED_TIME_UTC_mvt": [t, t, t, t, pd.NaT],
        "BLOCK_TIME_UTC_mvt": [t + 60 * s, t - 60 * s, t + 61 * s, pd.NaT, t],
    })
    assert copied_from_sched(df).tolist() == [True, True, False, False, False]


def test_retas_por_grupo_recuperam_a_relacao_e_usam_fallback():
    ms = np.array([0.0, 1000.0, 2000.0, 3000.0] * 20 + [0.0, 1000.0])
    keys = np.array(["LIRF"] * 80 + ["EDDF"] * 2)
    y = np.where(keys == "LIRF", -2818 + 1.062 * ms, 900.0)
    lines, fallback = fit_lines(ms, y, keys, min_rows=50)
    assert set(lines) == {"LIRF"}  # EDDF tem 2 linhas: usa a reta global
    a, b = lines["LIRF"]
    assert abs(a + 2818) < 1e-4 and abs(b - 1.062) < 1e-7
    out = apply_lines(np.array([3000.0, 1000.0, np.nan]), np.array(["LIRF", "EDDF", "LIRF"]),
                      lines, fallback)
    assert abs(out[0] - (-2818 + 1.062 * 3000)) < 1e-3
    assert out[1] == max(0.0, fallback[0] + fallback[1] * 1000.0)
    assert np.isnan(out[2])


def test_retas_nunca_preveem_negativo():
    lines = {"LIRF": (-2818.0, 1.062)}
    assert apply_lines(np.array([0.0]), np.array(["LIRF"]), lines, (0.0, 0.0)).tolist() == [0.0]


def test_reta_so_troca_voos_sem_nm_acima_do_limiar():
    nm = np.array([1, 1, 1, 0, 1], bool)
    ms = np.array([30_000.0, 3_000.0, np.nan, 30_000.0, 21_600.0])
    line = np.array([29_000.0, 2_000.0, np.nan, 29_000.0, 20_000.0])
    assert line_rows(nm, ms, line, 21_600).tolist() == [True, False, False, False, False]
    assert line_rows(nm, ms, line, 0).tolist() == [True, True, False, False, True]


def test_grupos_com_corte_de_2h():
    df = pd.DataFrame({
        "AIRPORT": pd.Categorical(["LIRF", "LIRF", "EDDF"]),
        "to_takeoff_from_SCHED_TIME_UTC_mvt": [9000.0, 600.0, np.nan],
    })
    assert nm_groups(df, split_ms=True).tolist() == ["LIRF|>2h", "LIRF|<=2h", "EDDF|<=2h"]
    assert nm_groups(df, split_ms=False).tolist() == ["LIRF", "LIRF", "EDDF"]


def test_calibracao_corrige_p_e_fica_monotona_em_0_1():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20_000)
    label = rng.binomial(1, p**2)  # a taxa real de cópia é p², o classificador diz p
    grade = np.linspace(0, 1, 21)
    out = apply_calibration(grade, np.array(["g"] * grade.size), {"g": isotonic_fit(p, label)})
    assert np.all(np.diff(out) >= 0)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert abs(out[10] - 0.25) < 0.05  # p bruto 0,5 → taxa real 0,25


def test_celula_da_calibracao_usa_lirf_nm_e_faixa_de_ms():
    df = pd.DataFrame({
        "AIRPORT": pd.Categorical(["LIRF", "LIRF", "EDDF", "EDDF"]),
        "nm_missing": [1, 0, 1, 0],
        "to_takeoff_from_SCHED_TIME_UTC_mvt": [50_000.0, 9_000.0, 600.0, np.nan],
    })
    assert calib_groups(df).tolist() == [
        "LIRF|semNM|>12h", "LIRF|comNM|2-6h", "outros|semNM|<=2h", "outros|comNM|sem_ms",
    ]


def test_celula_sem_as_duas_classes_fica_com_p_bruto():
    p = np.linspace(0.0, 1.0, 1000)
    keys = np.array(["so_normais"] * 500 + ["misto"] * 500)
    label = np.concatenate([np.zeros(500), (p[500:] > 0.75).astype(float)])
    cal = fit_calibration(p, label, keys)
    assert set(cal) == {"misto"}
    assert apply_calibration(p, keys, cal)[:500].tolist() == p[:500].tolist()


class _Fixo:
    """Modelo de mentira: devolve sempre o mesmo vetor."""

    def __init__(self, valores) -> None:
        self.valores = np.asarray(valores, float)

    def predict(self, x) -> np.ndarray:
        return self.valores


def _modelo_nm(hibrido: bool, p, reg) -> TwoStageNM:
    m = TwoStageNM({"nm_hibrido": hibrido, "nm_min_ms": 0})
    m.cols = ["nm_missing"]
    m.cls, m.reg = _Fixo(p), _Fixo(reg)
    m.lines, m.fallback = {"LIRF": (0.0, 1.0)}, (0.0, 1.0)  # reta = ms
    return m


def test_hibrido_mistura_a_reta_em_vez_de_substituir():
    df = pd.DataFrame({
        "AIRPORT": pd.Categorical(["LIRF", "LIRF"]),
        "nm_missing": [1, 1],
        "to_takeoff_from_SCHED_TIME_UTC_mvt": [10_000.0, 10_000.0],
    })
    p, reg = np.array([0.25, 1.0]), np.array([900.0, 900.0])
    assert _modelo_nm(False, p, reg).predict(df).tolist() == [10_000.0, 10_000.0]
    assert _modelo_nm(True, p, reg).predict(df).tolist() == [0.25 * 10_000 + 0.75 * 900, 10_000.0]
