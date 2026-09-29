"""Estatísticas do taxi-out da célula (aeroporto, stand, pista) para o corretor.

A base só recebe o P10 da célula (`features.ref_p10`, "taxi sem impedimento"). Nos
aeroportos sem ADS-B (LTFM, LFPG, LEMD, EGLL) a célula é quase toda a informação que
existe: o diagnóstico de 29/09 (`docs/research/2026-09-29-base-sem-adsb.md`) mostra que
nenhum agrupamento novo — companhia, tipo, destino, fila de pushback, configuração de
pista — explica o resíduo da v28, enquanto o desvio do taxi-out *dentro* da célula vai
de 211 s (LEMD) a 351 s (EGLL). O corretor recebe hoje a previsão e o pátio do stand
(`stand_p`), nunca o nível nem a dispersão da célula.

Este bloco dá quatro números por voo, com queda para chaves mais gerais quando a célula
tem menos de `MIN_VOOS` voos:

- `cel_p50`, `cel_p90`: mediana e P90 do taxi-out da célula (o P10 já existe na base);
- `cel_dp`: desvio-padrão da célula, a incerteza que a previsão deveria respeitar;
- `cel_n`: quantos voos sustentam a célula;
- `cel_nivel`: qual chave casou (0 = aeroporto+stand+pista, 3 = só aeroporto).

O alvo é cortado em `CORTE_S` antes de agregar (mesmo corte do regressor da base): sem
isso um único voo de horas move a mediana e destrói o P90 da célula.

Nada aqui lê o alvo das linhas previstas: as tabelas saem só das linhas passadas a
`ajustar`, e quem chama (`stack.py`, `train.py`) usa sempre meses de fora do bloco.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import features as F

# Da chave mais específica à mais geral; o stand sozinho vale mais que a pista sozinha.
CHAVES = [
    [F.AIRPORT, "STAND_mvt", "RUNWAY_mvt"],
    [F.AIRPORT, "STAND_mvt"],
    [F.AIRPORT, "RUNWAY_mvt"],
    [F.AIRPORT],
]
MIN_VOOS = 10
CORTE_S = 7200.0  # mesmo corte do regressor da base (--reg-corte)

ESTATISTICAS = ["cel_p50", "cel_p90", "cel_dp", "cel_n"]
COLS = [*ESTATISTICAS, "cel_nivel"]


def _chave(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Chave de texto do grupo; coluna ausente ou nula vira `?` (como em features._key)."""
    k = df[cols[0]].astype("string").fillna("?")
    for col in cols[1:]:
        k = k + "|" + df[col].astype("string").fillna("?")
    return k.to_numpy()


def ajustar(df: pd.DataFrame, y) -> list[pd.DataFrame]:
    """Uma tabela por chave: mediana, P90, desvio e tamanho das células com voos bastantes."""
    alvo = np.clip(np.asarray(y, float), 0.0, CORTE_S)
    ok = np.isfinite(alvo)
    tabelas = []
    for cols in CHAVES:
        g = pd.DataFrame({"k": _chave(df, cols)[ok], "y": alvo[ok]}).groupby("k")["y"]
        t = pd.DataFrame({"cel_p50": g.median(), "cel_p90": g.quantile(0.9),
                          "cel_dp": g.std(), "cel_n": g.size().astype(float)})
        tabelas.append(t[t["cel_n"] >= MIN_VOOS])
    return tabelas


def aplicar(df: pd.DataFrame, tabelas: list[pd.DataFrame]) -> pd.DataFrame:
    """Colunas `cel_*` das linhas de `df`, da chave mais específica que casar."""
    out = pd.DataFrame({c: np.full(len(df), np.nan) for c in COLS}, index=df.index)
    falta = np.ones(len(df), bool)
    for nivel, (cols, t) in enumerate(zip(CHAVES, tabelas)):
        if t.empty or not falta.any():
            continue
        k = pd.Series(_chave(df, cols))
        achou = falta & k.isin(t.index).to_numpy()
        if not achou.any():
            continue
        sub = t.reindex(k[achou].to_numpy())
        for c in ESTATISTICAS:
            out.iloc[np.flatnonzero(achou), out.columns.get_loc(c)] = sub[c].to_numpy()
        out.iloc[np.flatnonzero(achou), out.columns.get_loc("cel_nivel")] = float(nivel)
        falta &= ~achou
    return out


def aplicar_por_bloco(df: pd.DataFrame, bloco: np.ndarray,
                      tabelas: dict[int, list[pd.DataFrame]]) -> pd.DataFrame:
    """Como `aplicar`, mas cada linha usa as tabelas do seu bloco (meses de fora dele)."""
    out = pd.DataFrame({c: np.full(len(df), np.nan) for c in COLS}, index=df.index)
    bloco = np.asarray(bloco)
    for k, tabs in tabelas.items():
        sel = np.flatnonzero(bloco == k)
        if sel.size:
            out.iloc[sel] = aplicar(df.iloc[sel], tabs).to_numpy()
    return out
