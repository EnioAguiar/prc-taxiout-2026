import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import refcel  # noqa: E402


def _frame(apt, stand, pista) -> pd.DataFrame:
    return pd.DataFrame({"AIRPORT": apt, "STAND_mvt": stand, "RUNWAY_mvt": pista})


def test_celula_pequena_cai_para_a_chave_mais_geral():
    """Stand com menos de MIN_VOOS voos não vira célula: a linha usa aeroporto × pista."""
    n = refcel.MIN_VOOS
    df = _frame(["A"] * (3 * n), ["cheio"] * (2 * n) + ["raro"] * 2 + ["outro"] * (n - 2),
                ["09"] * (3 * n))
    y = np.r_[np.full(2 * n, 600.0), np.full(2, 5000.0), np.full(n - 2, 900.0)]
    out = refcel.aplicar(df, refcel.ajustar(df, y))

    raras = out[np.array(df["STAND_mvt"]) == "raro"]
    assert (raras["cel_nivel"] == 2).all()  # aeroporto × pista
    assert (out[np.array(df["STAND_mvt"]) == "cheio"]["cel_nivel"] == 0).all()
    assert raras["cel_p50"].iloc[0] == 600.0  # a mediana do aeroporto, não os 5000 dela


def test_estatisticas_sao_do_grupo_e_cortam_o_alvo():
    """P50/P90/desvio/tamanho saem da célula, com o alvo cortado em CORTE_S."""
    n = refcel.MIN_VOOS
    df = _frame(["A"] * n, ["1"] * n, ["09"] * n)
    y = np.r_[np.full(n - 1, 500.0), 86400.0]  # um voo de um dia
    t = refcel.ajustar(df, y)
    out = refcel.aplicar(df, t)
    assert out["cel_n"].iloc[0] == n
    assert out["cel_p50"].iloc[0] == 500.0
    assert out["cel_p90"].iloc[0] <= refcel.CORTE_S  # o voo de 24 h vira 7200, não 86400


def test_linha_sem_celula_conhecida_fica_nan():
    """Aeroporto que não estava no treino não inventa estatística."""
    n = refcel.MIN_VOOS
    treino = _frame(["A"] * n, ["1"] * n, ["09"] * n)
    t = refcel.ajustar(treino, np.full(n, 700.0))
    out = refcel.aplicar(_frame(["Z"], ["9"], ["27"]), t)
    assert out[refcel.COLS].isna().all().all()


def test_por_bloco_usa_a_tabela_do_proprio_bloco():
    """Cada linha lê as estatísticas do seu bloco de meses, nunca as do vizinho."""
    n = refcel.MIN_VOOS
    df = _frame(["A"] * n, ["1"] * n, ["09"] * n)
    tabelas = {0: refcel.ajustar(df, np.full(n, 100.0)),
               1: refcel.ajustar(df, np.full(n, 900.0))}
    alvo = _frame(["A"] * 4, ["1"] * 4, ["09"] * 4)
    out = refcel.aplicar_por_bloco(alvo, np.array([0, 1, 1, 0]), tabelas)
    assert list(out["cel_p50"]) == [100.0, 900.0, 900.0, 100.0]
