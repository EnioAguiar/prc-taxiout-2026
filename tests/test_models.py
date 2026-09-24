import numpy as np
import pandas as pd

from models import apply_lines, combine, copied_from_sched, fit_lines


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
