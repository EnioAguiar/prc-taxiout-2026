import numpy as np
import pandas as pd
import pytest

import models
from models import (
    apply_lines, build_model, combine, copied_from_sched, fit_lines, line_rows, nm_groups,
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


class _ModeloDaSeed:
    """Modelo falso: prevê a própria seed em todas as linhas."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.best_iter = None

    def fit(self, train, cols, run=None, valid=None) -> "_ModeloDaSeed":
        self.treinou = (len(train), tuple(cols), valid)
        self.best_iter = 100 + int(self.cfg["seed"])
        return self

    def predict(self, df) -> np.ndarray:
        return np.full(len(df), float(self.cfg["seed"]))


class _Run:
    """Run falso: só guarda as mensagens de progresso."""

    def __init__(self) -> None:
        self.linhas = []

    def log(self, msg: str) -> None:
        self.linhas.append(msg)


@pytest.fixture
def modelo_da_seed(monkeypatch):
    monkeypatch.setitem(models.MODELS, "falso", _ModeloDaSeed)
    return "falso"


def test_media_de_seeds_usa_seeds_consecutivas_e_loga_o_progresso(modelo_da_seed):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    run = _Run()

    m = build_model({"model": modelo_da_seed, "seed": 3, "seeds": 4})
    m.fit(df, ["x"], run=run, valid=None)

    assert [c["seed"] for c in m.cfgs] == [3, 4, 5, 6]
    assert m.predict(df).tolist() == [4.5, 4.5, 4.5]  # média de 3, 4, 5 e 6
    assert m.best_iter == 103  # o da primeira cópia
    assert run.linhas == [f"seed {k}/4 (seed={k + 2})" for k in range(1, 5)]


def test_cada_copia_treina_com_seeds_1_para_nao_recursar(modelo_da_seed):
    m = build_model({"model": modelo_da_seed, "seed": 0, "seeds": 2})

    assert [c["seeds"] for c in m.cfgs] == [1, 1]


def test_todas_as_copias_veem_o_mesmo_treino_e_holdout(modelo_da_seed):
    df = pd.DataFrame({"x": [1.0, 2.0]})
    hold = pd.DataFrame({"x": [9.0]})

    m = build_model({"model": modelo_da_seed, "seed": 0, "seeds": 3}).fit(df, ["x"], valid=hold)

    assert [c.treinou for c in m.models] == [(2, ("x",), hold)] * 3


def test_sem_seeds_ou_com_uma_seed_o_modelo_e_o_de_hoje(modelo_da_seed):
    assert isinstance(build_model({"model": modelo_da_seed, "seed": 0}), _ModeloDaSeed)
    assert isinstance(build_model({"model": modelo_da_seed, "seed": 0, "seeds": 1}), _ModeloDaSeed)
