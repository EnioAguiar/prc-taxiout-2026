"""Contexto do aeroporto na hora da decolagem: taxi-in das chegadas e vizinhos de MVT−AOBT_3.

As colunas `ctx_*` ficam fora de `features.feature_columns` (a base não muda) e entram só no
corretor (`stack.corrector_frame`). Nada aqui lê o alvo nem o in-block das decolagens: só as
chegadas (`TAXITIME_SEC_mvt` das ARR existe também no ranking) e horários planejados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import features as F

WINDOWS_MIN = (15, 60)
MAX_IDADE_STAND = 86400  # chegada mais velha que um dia não diz nada sobre o stand
TAXI_IN_MAX = 3600
PROXY_MAX = 7200

COLS = [
    *[f"ctx_arr_{q}_{k}_{w}" for k in ("apt", "rwy") for q in ("tin", "n") for w in WINDOWS_MIN],
    "ctx_stand_ult_idade",
    "ctx_stand_ult_tin",
    *[f"ctx_viz_{lado}_{k}_{w}" for k in ("apt", "rwy") for lado in ("pas", "fut")
      for w in WINDOWS_MIN],
    "ctx_viz_proprio_menos_pas",
    "ctx_viz_proprio_menos_fut",
]

_NAO_VISTO = np.iinfo(np.int64).max  # código de grupo que não casa com nenhum outro


def _segundos(s: pd.Series) -> np.ndarray:
    """Epoch em segundos; NaT vira NaN."""
    v = s.to_numpy("datetime64[s]").astype("float64")
    v[s.isna().to_numpy()] = np.nan
    return v


def _codigos(dep: pd.DataFrame, arr: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Mesmo código inteiro para o mesmo grupo nas duas pontas; NaN vira grupo próprio."""
    def chave(df: pd.DataFrame) -> pd.Series:
        k = df[cols[0]].astype("string")
        for col in cols[1:]:
            k = k + "|" + df[col].astype("string")
        return k

    codes, _ = pd.factorize(pd.concat([chave(dep), chave(arr)], ignore_index=True))
    return codes[: len(dep)].astype("int64"), codes[len(dep):].astype("int64")


def _chaves(codes: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Grupo e instante em um inteiro só: ordenar/buscar por grupo sem fatiar por grupo."""
    k = (codes << 32) + t.astype("int64")
    return np.where(codes < 0, _NAO_VISTO, k)


def _acumulado(codes: np.ndarray, t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eventos ordenados por (grupo, instante) e soma acumulada dos valores."""
    ok = np.isfinite(t) & (codes >= 0)
    codes, t, v = codes[ok], t[ok], v[ok]
    ordem = np.lexsort((t, codes))
    chaves = _chaves(codes[ordem], t[ordem])
    return chaves, np.concatenate([[0.0], np.cumsum(v[ordem], dtype="float64")])


def _media(chaves: np.ndarray, cs: np.ndarray, lo: np.ndarray, hi: np.ndarray,
           lado_lo: str, lado_hi: str) -> tuple[np.ndarray, np.ndarray]:
    n = np.searchsorted(chaves, hi, lado_hi) - (i := np.searchsorted(chaves, lo, lado_lo))
    soma = cs[i + n] - cs[i]
    return np.where(n > 0, soma / np.maximum(n, 1), np.nan), n


def _chave_consulta(codes: np.ndarray, t: np.ndarray, desloc: int) -> np.ndarray:
    """Chave da consulta; instante inválido vai para um grupo que não existe."""
    bruto = np.where(np.isfinite(t), t + desloc, 0).astype("int64")
    return np.where(np.isfinite(t) & (codes >= 0), (codes << 32) + bruto, _NAO_VISTO)


def contexto(raw: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por decolagem de `raw`: `MVT_ID_mvt` e as 20 colunas `ctx_*`."""
    fase = raw["PHASE_mvt"].to_numpy()
    cols = [F.ID, F.AIRPORT, "RUNWAY_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt"]
    dep = raw.loc[fase == "DEP", [*cols, "AOBT_3_flt"]]
    arr = raw.loc[fase == "ARR", [*cols, "BLOCK_TIME_UTC_mvt", F.TARGET]]

    d_mvt = _segundos(dep["MVT_TIME_UTC_mvt"])
    a_mvt = _segundos(arr["MVT_TIME_UTC_mvt"])
    a_block = _segundos(arr["BLOCK_TIME_UTC_mvt"])
    a_tin = arr[F.TARGET].to_numpy("float64").clip(0, TAXI_IN_MAX)
    proxy = d_mvt - _segundos(dep["AOBT_3_flt"])
    proxy[~((proxy >= 0) & (proxy <= PROXY_MAX))] = np.nan

    grupos = {"apt": [F.AIRPORT], "rwy": [F.AIRPORT, "RUNWAY_mvt"]}
    feats: dict[str, np.ndarray] = {}
    for nome, chave in grupos.items():
        d_cod, a_cod = _codigos(dep, arr, chave)
        chaves, cs = _acumulado(a_cod, a_mvt, a_tin)
        for w in WINDOWS_MIN:  # chegadas em (MVT − w, MVT]
            lo = _chave_consulta(d_cod, d_mvt, -w * 60)
            hi = _chave_consulta(d_cod, d_mvt, 0)
            media, n = _media(chaves, cs, lo, hi, "right", "right")
            feats[f"ctx_arr_tin_{nome}_{w}"] = media
            feats[f"ctx_arr_n_{nome}_{w}"] = n.astype("float64")

        chaves, cs = _acumulado(d_cod, d_mvt, proxy)
        for w in WINDOWS_MIN:
            passado = _media(chaves, cs, _chave_consulta(d_cod, d_mvt, -w * 60),
                             _chave_consulta(d_cod, d_mvt, 0), "left", "left")[0]
            futuro = _media(chaves, cs, _chave_consulta(d_cod, d_mvt, 0),
                            _chave_consulta(d_cod, d_mvt, w * 60), "right", "right")[0]
            feats[f"ctx_viz_pas_{nome}_{w}"] = passado
            feats[f"ctx_viz_fut_{nome}_{w}"] = futuro

    feats["ctx_viz_proprio_menos_pas"] = proxy - feats["ctx_viz_pas_apt_60"]
    feats["ctx_viz_proprio_menos_fut"] = proxy - feats["ctx_viz_fut_apt_60"]
    feats.update(_stand(dep, arr, d_mvt, a_block, a_tin))

    out = pd.DataFrame({F.ID: dep[F.ID].to_numpy()})
    for col in COLS:
        out[col] = feats[col]
    return out


def _stand(dep: pd.DataFrame, arr: pd.DataFrame, d_mvt: np.ndarray, a_block: np.ndarray,
           a_tin: np.ndarray) -> dict[str, np.ndarray]:
    """Última chegada no mesmo aeroporto × stand com in-block ≤ MVT: idade e taxi-in."""
    d_cod, a_cod = _codigos(dep, arr, [F.AIRPORT, "STAND_mvt"])
    sem_stand = dep["STAND_mvt"].isna().to_numpy()
    idade = np.full(len(dep), np.nan)
    tin = np.full(len(dep), np.nan)

    ok = np.isfinite(a_block) & ~arr["STAND_mvt"].isna().to_numpy()
    if ok.any():
        ordem = np.lexsort((a_block[ok], a_cod[ok]))
        chaves = _chaves(a_cod[ok][ordem], a_block[ok][ordem])
        blocos = a_block[ok][ordem]
        tins = a_tin[ok][ordem]
        pos = np.searchsorted(chaves, _chave_consulta(d_cod, d_mvt, 0), "right") - 1
        achou = (pos >= 0) & ~sem_stand & np.isfinite(d_mvt)
        achou[achou] = (chaves[pos[achou]] >> 32) == d_cod[achou]
        alvo = pos[achou]
        idade[achou] = d_mvt[achou] - blocos[alvo]
        tin[achou] = tins[alvo]

    velha = idade > MAX_IDADE_STAND
    idade[velha] = np.nan
    tin[velha] = np.nan
    return {"ctx_stand_ult_idade": idade, "ctx_stand_ult_tin": tin}
