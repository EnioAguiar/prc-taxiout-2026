"""Aviões no solo (`--superficie`): quantos aviões dividem o pátio com cada decolagem.

Ideia do preditor de taxi-out do ATD2 da NASA: o que estica o táxi não é só o voo, é
quanta gente está se movendo no solo quando ele sai do stand e quanta gente entra no
pátio enquanto ele roda. O bloco reconstrói esse cenário **só com o que existe na hora
de prever o ranking de 2026**:

- de cada decolagem `j` sabemos a hora de decolar `MVT_j` (está no ranking) e a previsão
  da base `pred_j`, então o push estimado é `o_j = MVT_j − pred_j`. O off-block de
  verdade das decolagens (`BLOCK_TIME_UTC_mvt` das linhas DEP) é o alvo da competição e
  **nunca é lido aqui** — nem o da própria linha, nem o dos vizinhos;
- de cada pouso `k` sabemos o pouso `L_k` (`MVT_TIME_UTC_mvt`) e o in-block `B_k`
  (`BLOCK_TIME_UTC_mvt`) de verdade, lidos dos parquets brutos (`PHASE_mvt == "ARR"`,
  como em `plano13._pousos`): o in-block dos pousos existe também no ranking de 2026.

As contagens saem por aeroporto e dentro do mesmo quadro (as decolagens `j` são as linhas
do próprio quadro — cegas, holdout ou ranking):

- `sup_dep_taxiando_push`: decolagens `j ≠ i` com `o_j ≤ o_i < MVT_j`, isto é, que
  estimamos estar taxiando no push estimado de `i`;
- `sup_dep_decolam_durante`: decolagens `j ≠ i` com `o_i < MVT_j < MVT_i`, que decolam
  durante o táxi estimado de `i` (variante `_rwy`: mesma pista);
- `sup_arr_taxiando_push`: pousos com `L_k ≤ o_i < B_k` (taxiando para o pátio no push);
- `sup_arr_pousam_durante`: pousos com `o_i < L_k < MVT_i`;
- `sup_dep_decolam_durante_min` e `sup_arr_pousam_durante_min`: as duas contagens
  "durante" por minuto de táxi previsto (`pred_i / 60`), para separar fila de táxi longo.

Tudo é vetorizado: dentro de cada aeroporto os horários viram vetores ordenados e cada
contagem é uma diferença de `searchsorted` (O(n log n), segundos para 2 M linhas).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import features as F
from cache import DATA
from externos import BLOCK, TIME
from plano13 import _brutos

# Pouso anterior à janela do quadro ainda pode estar taxiando dentro dela.
MARGEM = pd.Timedelta("6h")
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
ARR_COLS = ["PHASE_mvt", "ADES_mvt", TIME, BLOCK]

COLS = ["sup_dep_taxiando_push", "sup_dep_decolam_durante", "sup_dep_decolam_durante_rwy",
        "sup_arr_taxiando_push", "sup_arr_pousam_durante",
        "sup_dep_decolam_durante_min", "sup_arr_pousam_durante_min"]


def _segundos(s: pd.Series) -> np.ndarray:
    """Horário em segundos desde 1970 (NaT vira NaN)."""
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return (t - EPOCH).dt.total_seconds().to_numpy(float)


def _grupos(chave: np.ndarray) -> dict:
    """Posições de cada chave (aeroporto, ou aeroporto e pista)."""
    return pd.Series(np.arange(chave.size)).groupby(chave, sort=False).indices


def _chegadas(apts: set[str], t0: pd.Timestamp, t1: pd.Timestamp, dados: Path) -> pd.DataFrame:
    """Pouso `L` e in-block `B` (segundos) de cada chegada dos parquets que cobrem a janela.

    Só linhas `ARR`: o BLOCK lido é o in-block de quem pousou, nunca o off-block de uma
    decolagem. Chegada com in-block antes do pouso (0,02 % do dado) é descartada.
    """
    partes = []
    for ini, fim, p in _brutos(dados):
        if fim <= t0 - MARGEM or ini > t1:
            continue
        raw = pd.read_parquet(p, columns=ARR_COLS)
        arr = raw[(raw["PHASE_mvt"] == "ARR") & raw["ADES_mvt"].isin(apts)]
        del raw
        partes.append(pd.DataFrame({"apt": arr["ADES_mvt"].astype(str).to_numpy(),
                                    "L": _segundos(arr[TIME]), "B": _segundos(arr[BLOCK])}))
    if not partes:
        return pd.DataFrame({"apt": pd.Series(dtype="object"), "L": pd.Series(dtype=float),
                             "B": pd.Series(dtype=float)})
    ch = pd.concat(partes, ignore_index=True).dropna()
    return ch[ch["B"] >= ch["L"]]


def _conta_decolagens(out: dict, apt: np.ndarray, rwy: np.ndarray, empurra: np.ndarray,
                      mvt: np.ndarray, taxi: np.ndarray) -> None:
    """Vizinhas do mesmo quadro: taxiando no push de `i` e decolando durante o táxi de `i`.

    `o_j ≤ MVT_j` sempre (o táxi previsto nunca é negativo), então
    `#{o_j ≤ o_i < MVT_j} = #{o_j ≤ o_i} − #{MVT_j ≤ o_i}`: duas buscas binárias em vez de
    varrer intervalos. A própria linha entra na primeira contagem e é descontada.

    A diferença das buscas só vale quando o intervalo `(o_i, MVT_i)` existe; com táxi
    previsto zero ele é vazio e o piso 0 é o que responde (empates em `MVT_i`).
    """
    for pos in _grupos(apt).values():
        pos = pos[np.isfinite(mvt[pos])]
        if not pos.size:
            continue
        oi, mi = empurra[pos], mvt[pos]
        ordem_o, ordem_m = np.sort(oi), np.sort(mi)
        out["sup_dep_taxiando_push"][pos] = (np.searchsorted(ordem_o, oi, "right")
                                             - np.searchsorted(ordem_m, oi, "right")
                                             - (taxi[pos] > 0))
        out["sup_dep_decolam_durante"][pos] = np.maximum(
            np.searchsorted(ordem_m, mi, "left") - np.searchsorted(ordem_m, oi, "right"), 0)
    for pos in _grupos(rwy).values():
        pos = pos[np.isfinite(mvt[pos])]
        if not pos.size:
            continue
        ordem_m = np.sort(mvt[pos])
        out["sup_dep_decolam_durante_rwy"][pos] = np.maximum(
            np.searchsorted(ordem_m, mvt[pos], "left")
            - np.searchsorted(ordem_m, empurra[pos], "right"), 0)


def _conta_chegadas(out: dict, apt: np.ndarray, empurra: np.ndarray, mvt: np.ndarray,
                    ch: pd.DataFrame) -> None:
    """Pousos taxiando no push de `i` e pousos que acontecem durante o táxi de `i`."""
    vazio = np.empty(0)
    porto = {nome: (np.sort(g["L"].to_numpy()), np.sort(g["B"].to_numpy()))
             for nome, g in ch.groupby("apt", sort=False)}
    for nome, pos in _grupos(apt).items():
        pos = pos[np.isfinite(mvt[pos])]
        if not pos.size:
            continue
        pousos, blocos = porto.get(nome, (vazio, vazio))
        oi, mi = empurra[pos], mvt[pos]
        out["sup_arr_taxiando_push"][pos] = (np.searchsorted(pousos, oi, "right")
                                             - np.searchsorted(blocos, oi, "right"))
        out["sup_arr_pousam_durante"][pos] = np.maximum(
            np.searchsorted(pousos, mi, "left") - np.searchsorted(pousos, oi, "right"), 0)


def contagens(df: pd.DataFrame, pred: np.ndarray, dados: Path = DATA) -> pd.DataFrame:
    """Colunas `sup_*` das linhas de `df`, na ordem delas (`corrector_frame`).

    `pred` é o táxi previsto de cada linha pela base — é dele que sai o push estimado
    `o_i = MVT_i − pred_i`. Previsão negativa (rara) é tratada como zero, para que o push
    nunca caia depois da decolagem.
    """
    n = len(df)
    out = {c: np.full(n, np.nan) for c in COLS}
    if n == 0:
        return pd.DataFrame(out, index=df.index)
    mvt = _segundos(df[TIME])
    taxi = np.clip(np.asarray(pred, float), 0.0, None)
    empurra = mvt - taxi
    apt = df[F.AIRPORT].astype(str).to_numpy()
    rwy = (df[F.AIRPORT].astype(str) + "|" + df["RUNWAY_mvt"].astype(str)).to_numpy()
    _conta_decolagens(out, apt, rwy, empurra, mvt, taxi)

    conhecido = np.isfinite(mvt)
    if conhecido.any():
        t0 = EPOCH + pd.to_timedelta(float(np.nanmin(empurra[conhecido])), unit="s")
        t1 = EPOCH + pd.to_timedelta(float(mvt[conhecido].max()), unit="s")
        ch = _chegadas(set(apt[conhecido]), t0, t1, dados)
        if not ch.empty:
            _conta_chegadas(out, apt, empurra, mvt, ch)

    por_min = np.where(taxi > 0, taxi / 60.0, np.nan)
    out["sup_dep_decolam_durante_min"] = out["sup_dep_decolam_durante"] / por_min
    out["sup_arr_pousam_durante_min"] = out["sup_arr_pousam_durante"] / por_min
    return pd.DataFrame(out, index=df.index)
