"""Decomposição `T = D − G` nas partidas de Roma (`--roma-tdg`).

Identidade exata nos dados: o táxi é `T = D − G`, com

- `D = MVT − SCHED` (`to_takeoff_from_SCHED_TIME_UTC_mvt`), **observado** também no ranking;
- `G = BLOCK − SCHED`, o atraso de portão, que é justamente o que falta nas partidas.

Como `G = D − T`, o alvo do modelo sai das próprias linhas de treino sem tocar no `BLOCK`
das linhas previstas (que o ranking esconde). Prever `G` e reconstruir `T̂ = max(D − Ĝ, 0)`
embute a subtração que uma árvore não faz sozinha: com as mesmas colunas e os mesmos dados,
o holdout jan+jul de 2025 dá RMSE 643,5 s por `D − Ĝ` contra 1.220,3 s prevendo `T` direto
(no LIRF sem registro NM, 4.049,5 contra 8.803,5).

Só vale no LIRF: `corr(T, D)` é 0,90 lá e ≤ 0,44 nos outros aeroportos (medido em
`docs/research/2026-10-02-roma-t-d-g.md`). Fora do LIRF as colunas saem nulas.

`Ĝ` sozinho **não** bate a campeã (643,5 contra 542,6 no LIRF): a base de dois estágios já
ancora em `D` e o `AOBT_3` cobre `G` em 98,5 % das linhas. O ganho medido vem de entregar
`roma_g_hat`/`roma_t_hat` ao **corretor**, que decide onde usá-las
(`docs/research/2026-10-02-roma-t-d-g.md`).

Nada aqui lê o alvo das linhas previstas: o modelo sai só das linhas passadas a `ajustar`,
e quem chama (`stack.py`, `train.py`) usa sempre meses de fora do bloco.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

import features as F

AEROPORTO = "LIRF"
MS = "to_takeoff_from_SCHED_TIME_UTC_mvt"
COLS = ["roma_g_hat", "roma_t_hat"]

# Categóricas do voo (o aeroporto é constante aqui) e numéricas que o ranking também tem.
CAT = [c for c in F.CATEGORICAL if c != F.AIRPORT]
NUM_PREFIXOS = ("to_takeoff_from_", "gap_", "round_", "apt_dep_", "apt_arr_", "rwy_dep_")
NUM_FIXAS = ["hour", "dow", "nm_missing"]

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
              verbose=-1, num_threads=12, seed=0, deterministic=True, force_row_wise=True)
RODADAS = 400


def linhas(df: pd.DataFrame) -> np.ndarray:
    """As linhas de Roma: só nelas o modelo treina e só nelas as colunas saem preenchidas."""
    return (df[F.AIRPORT].astype("string") == AEROPORTO).fillna(False).to_numpy()


def colunas_numericas(df: pd.DataFrame) -> list[str]:
    return [*NUM_FIXAS, *sorted(c for c in df.columns if c.startswith(NUM_PREFIXOS))]


def _entradas(df: pd.DataFrame, nums: list[str],
              cats: dict[str, pd.CategoricalDtype]) -> pd.DataFrame:
    """Entradas na ordem fixada no treino: coluna que falta no quadro entra nula."""
    X = pd.DataFrame(index=df.index)
    for c in nums:
        X[c] = df[c].to_numpy(float) if c in df.columns else np.nan
    for c, dtype in cats.items():
        texto = df[c].astype("string") if c in df else pd.Series(pd.NA, index=df.index, dtype="string")
        X[c] = texto.where(texto.isin(dtype.categories)).astype(dtype)  # categoria nova vira nula
    return X


def _vocabulario(df: pd.DataFrame) -> dict[str, pd.CategoricalDtype]:
    return {c: pd.CategoricalDtype(sorted({v for v in df[c].astype("string").dropna()}))
            for c in CAT if c in df.columns}


def ajustar(df: pd.DataFrame, y) -> tuple | None:
    """Modelo de `G = D − T` nas partidas do LIRF de `df`; `None` quando não há linhas."""
    sel = linhas(df)
    ms = df[MS].to_numpy(float)
    alvo = ms - np.asarray(y, float)
    sel &= np.isfinite(ms) & np.isfinite(alvo)
    if sel.sum() < 2 * PARAMS["min_data_in_leaf"]:
        return None
    treino = df.loc[sel]
    cats, nums = _vocabulario(treino), colunas_numericas(treino)
    ds = lgb.Dataset(_entradas(treino, nums, cats), alvo[sel], free_raw_data=False)
    return lgb.train(PARAMS, ds, RODADAS), cats, nums


def aplicar(df: pd.DataFrame, modelo: tuple | None) -> pd.DataFrame:
    """`roma_g_hat` e `roma_t_hat = max(D − Ĝ, 0)` das linhas de `df`; nulo fora do LIRF."""
    out = pd.DataFrame({c: np.full(len(df), np.nan) for c in COLS}, index=df.index)
    if modelo is None:
        return out
    booster, cats, nums = modelo
    sel = linhas(df) & np.isfinite(df[MS].to_numpy(float))
    onde = np.flatnonzero(sel)
    if not onde.size:
        return out
    g = booster.predict(_entradas(df.iloc[onde], nums, cats))
    out.iloc[onde, out.columns.get_loc("roma_g_hat")] = g
    out.iloc[onde, out.columns.get_loc("roma_t_hat")] = np.clip(
        df[MS].to_numpy(float)[onde] - g, 0.0, None)
    return out


def aplicar_por_bloco(df: pd.DataFrame, bloco: np.ndarray,
                      modelos: dict[int, tuple | None]) -> pd.DataFrame:
    """Como `aplicar`, mas cada linha usa o modelo do seu bloco (meses de fora dele)."""
    out = pd.DataFrame({c: np.full(len(df), np.nan) for c in COLS}, index=df.index)
    bloco = np.asarray(bloco)
    for k, modelo in modelos.items():
        sel = np.flatnonzero(bloco == k)
        if sel.size:
            out.iloc[sel] = aplicar(df.iloc[sel], modelo).to_numpy()
    return out
