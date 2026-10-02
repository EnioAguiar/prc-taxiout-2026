import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import roma  # noqa: E402

N = 4 * roma.PARAMS["min_data_in_leaf"]


def _frame(apt, ms, stand=None, n=None) -> pd.DataFrame:
    """Quadro mínimo com as colunas que `roma` lê (as numéricas que faltam entram nulas)."""
    n = n or len(ms)
    df = pd.DataFrame({
        "AIRPORT": apt if isinstance(apt, list) else [apt] * n,
        roma.MS: np.asarray(ms, float),
        "hour": np.linspace(0, 23, n), "dow": np.arange(n) % 7,
        "nm_missing": np.zeros(n),
        "to_takeoff_from_AOBT_3_flt": np.full(n, 900.0),
        "apt_dep_prev_30m": np.arange(n) % 11,
    })
    for c in roma.CAT:
        df[c] = "x"
    df["STAND_mvt"] = stand if stand is not None else "A1"
    return df


def test_o_alvo_do_modelo_e_g_igual_a_d_menos_o_taxi():
    """Com G constante, Ĝ reproduz esse G e T̂ = D − Ĝ volta ao táxi verdadeiro."""
    rng = np.random.default_rng(0)
    ms = rng.uniform(2000, 9000, N)
    y = ms - 1200.0  # G = 1200 s em todas as linhas
    df = _frame("LIRF", ms)
    out = roma.aplicar(df, roma.ajustar(df, y))

    assert np.allclose(out["roma_g_hat"], 1200.0, atol=60)
    assert np.allclose(out["roma_t_hat"], y, atol=60)


def test_fora_do_lirf_as_colunas_saem_nulas():
    """A decomposição só vale em Roma: outro aeroporto não recebe nada."""
    ms = np.linspace(600, 9000, N)
    treino = _frame("LIRF", ms)
    modelo = roma.ajustar(treino, ms - 1200.0)
    out = roma.aplicar(_frame(["LIRF", "LFPG", "EHAM"], [3000.0, 3000.0, 3000.0]), modelo)

    assert out["roma_g_hat"].notna().tolist() == [True, False, False]
    assert out[roma.COLS].iloc[1:].isna().all(axis=None)


def test_o_taxi_reconstruido_nunca_e_negativo():
    """Ĝ maior que D viraria táxi negativo; o piso é 0."""
    ms = np.linspace(4000, 9000, N)
    df = _frame("LIRF", ms)
    modelo = roma.ajustar(df, ms - 3000.0)  # aprende G ≈ 3000
    out = roma.aplicar(_frame("LIRF", [60.0]), modelo)  # D = 60 s, bem abaixo de Ĝ

    assert out["roma_t_hat"].iloc[0] == 0.0


def test_treino_sem_linhas_do_lirf_nao_gera_modelo():
    """Sem partidas de Roma não há o que ajustar, e aplicar devolve tudo nulo."""
    ms = np.linspace(600, 9000, N)
    assert roma.ajustar(_frame("LFPG", ms), ms - 1200.0) is None
    assert roma.aplicar(_frame("LIRF", [3000.0]), None)[roma.COLS].isna().all(axis=None)


def test_linha_sem_d_fica_nula():
    """Sem MVT − SCHED não dá para reconstruir o táxi: a linha não recebe coluna."""
    ms = np.linspace(600, 9000, N)
    treino = _frame("LIRF", ms)
    out = roma.aplicar(_frame("LIRF", [np.nan, 3000.0]), roma.ajustar(treino, ms - 1200.0))

    assert out[roma.COLS].iloc[0].isna().all()
    assert out[roma.COLS].iloc[1].notna().all()


def test_por_bloco_usa_o_modelo_do_proprio_bloco():
    """Cada linha lê o modelo do seu bloco de meses, nunca o do vizinho."""
    ms = np.linspace(600, 9000, N)
    df = _frame("LIRF", ms)
    modelos = {0: roma.ajustar(df, ms - 300.0), 1: roma.ajustar(df, ms - 3000.0)}
    alvo = _frame("LIRF", [5000.0] * 4)
    out = roma.aplicar_por_bloco(alvo, np.array([0, 1, 1, 0]), modelos)

    g = out["roma_g_hat"].to_numpy()
    assert np.allclose(g[[0, 3]], 300.0, atol=60)
    assert np.allclose(g[[1, 2]], 3000.0, atol=60)


def test_as_colunas_saem_na_ordem_e_no_indice_das_linhas():
    ms = np.linspace(600, 9000, N)
    treino = _frame("LIRF", ms)
    alvo = _frame("LIRF", [1000.0, 2000.0, 3000.0]).set_index(pd.Index([7, 8, 9]))
    out = roma.aplicar(alvo, roma.ajustar(treino, ms - 1200.0))

    assert out.index.tolist() == [7, 8, 9]
    assert list(out.columns) == roma.COLS


def test_categoria_nova_nao_quebra_a_previsao():
    """Stand que não estava no treino vira categoria nula, não erro."""
    ms = np.linspace(600, 9000, N)
    treino = _frame("LIRF", ms, stand="A1")
    out = roma.aplicar(_frame("LIRF", [3000.0], stand="ZZ9"), roma.ajustar(treino, ms - 1200.0))

    assert out["roma_g_hat"].notna().all()
