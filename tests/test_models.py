import numpy as np
import pandas as pd

from models import (
    TwoStageNM,
    apply_lines,
    combine,
    copied_from_sched,
    fit_lines,
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
