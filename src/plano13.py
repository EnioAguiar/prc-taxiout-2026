"""Sinais do plano 13 no corretor: METAR, rotação no stand, consistência NM e companhia.

Quatro blocos de colunas, sempre na ordem do quadro que recebem:

- `metar`: a observação mais recente do aeroporto até 2 h antes do movimento (IEM ASOS) —
  temperatura, spread, vento, rajada, visibilidade, teto, fenômenos e `met_degelo`;
- `rotacao`: o último pouso no mesmo aeroporto e stand nas 24 h anteriores ao movimento —
  se é da mesma companhia, do mesmo tipo e há quantos segundos (`rot_*`). Os pousos vêm
  dos parquets brutos que cobrem o período do quadro (os meses de 2025 e o ranking de
  2026), então o valor de um voo é o mesmo em qualquer recorte; o BLOCK da própria
  decolagem nunca entra;
- `consistencia`: duração NM planejada × realizada e se ADEP/ADES/tipo do movimento batem
  com os do plano de voo (`nm_*`);
- `coluna_cia`: a companhia como categoria, com vocabulário fixo (as `CIAS_TOP` mais
  frequentes do treino, o resto vira `outro`) — o mesmo em treino, holdout e ranking.

    bin/run src/plano13.py baixar   # data/externo/metar/ (fora do git), pula o que já existe

Fonte do METAR: IEM ASOS (Iowa State University, <https://mesonet.agron.iastate.edu/>,
dados públicos).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
from cache import DATA
from externos import BLOCK, RAIZ, TIME, _baixar, _cia

APTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")

# METAR
TOL_METAR = pd.Timedelta("2h")   # observação mais velha que isso não descreve o voo
NA_METAR = ["null", "M"]
MET_COLS = ["met_temp", "met_spread", "met_vento", "met_rajada", "met_vis", "met_teto",
            "met_neve", "met_congel", "met_trovoada", "met_chuva", "met_nevoa"]
DEGELO_C = 3.0       # temperatura (°C) abaixo da qual o degelo fica provável
DEGELO_SPREAD = 3.0  # spread (°C) que indica umidade suficiente para gelo

# Rotação no stand
IDADE_MAX = pd.Timedelta(seconds=86400)  # pouso mais velho que isso não é a rotação
ARR_COLS = ["PHASE_mvt", "ADES_mvt", "STAND_mvt", BLOCK, "FLIGHT_mvt", "AIRCRAFT_TYPE_mvt"]

# Companhia
CIAS_TOP = 150
OUTRA = "outro"

# Download: uma requisição por aeroporto, todo o período de uma vez.
URL_METAR = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={apt}"
             "&data=tmpf&data=dwpf&data=sknt&data=gust&data=vsby&data=wxcodes&data=skyc1"
             "&data=skyl1&year1=2025&month1=1&day1=1&year2=2026&month2=8&day2=1"
             "&tz=Etc/UTC&format=onlycomma&latlon=no&missing=null&trace=T&direct=no")


def baixar(raiz: Path = RAIZ) -> None:
    """METAR de 2025 e 2026 de cada aeroporto em `raiz`/metar/."""
    for apt in APTS:
        print(_baixar(URL_METAR.format(apt=apt), raiz / "metar" / f"{apt}.csv"), flush=True)


def _hora(df: pd.DataFrame) -> pd.Series:
    """Hora do movimento em UTC."""
    return pd.to_datetime(df[TIME], utc=True)


def _observacoes(raiz: Path) -> pd.DataFrame:
    """Todos os METAR dos nossos aeroportos, ordenados no tempo."""
    partes = []
    for apt in APTS:
        p = raiz / "metar" / f"{apt}.csv"
        if not p.exists():
            raise SystemExit(f"falta {p.name} em {p.parent}; rode: bin/run src/plano13.py baixar")
        m = pd.read_csv(p, na_values=NA_METAR, low_memory=False)
        for col in ("tmpf", "dwpf", "sknt", "gust", "vsby", "skyl1"):
            m[col] = pd.to_numeric(m[col], errors="coerce")
        wx = m["wxcodes"].astype("string").fillna("")
        partes.append(pd.DataFrame({
            "t": pd.to_datetime(m["valid"], utc=True, errors="coerce"),
            "apt": apt,
            "met_temp": (m["tmpf"] - 32) / 1.8,
            "met_spread": (m["tmpf"] - m["dwpf"]) / 1.8,
            "met_vento": m["sknt"],
            "met_rajada": m["gust"],
            "met_vis": m["vsby"],
            "met_teto": m["skyl1"],
            "met_neve": wx.str.contains("SN").astype(float),
            "met_congel": wx.str.contains("FZ").astype(float),
            "met_trovoada": wx.str.contains("TS").astype(float),
            "met_chuva": wx.str.contains("RA").astype(float),
            "met_nevoa": wx.str.contains("FG|BR").astype(float),
        }))
    obs = pd.concat(partes, ignore_index=True)
    return obs.dropna(subset=["t"]).sort_values("t", ignore_index=True)


def metar(df: pd.DataFrame, raiz: Path = RAIZ) -> pd.DataFrame:
    """Colunas `met_*` do METAR do aeroporto até `TOL_METAR` antes de cada movimento."""
    consulta = pd.DataFrame({
        "i": np.arange(len(df)),
        "t": _hora(df).reset_index(drop=True),
        "apt": df[F.AIRPORT].astype(str).reset_index(drop=True),
    }).dropna(subset=["t"]).sort_values("t", ignore_index=True)
    casado = pd.merge_asof(consulta, _observacoes(raiz), on="t", by="apt",
                           direction="backward", tolerance=TOL_METAR)
    out = casado.set_index("i")[MET_COLS].reindex(range(len(df)))
    # degelo provável: frio e (ar úmido ou precipitação); sem observação, sem valor.
    out["met_degelo"] = ((out["met_temp"] <= DEGELO_C)
                         & ((out["met_spread"] <= DEGELO_SPREAD)
                            | (out[["met_neve", "met_congel", "met_chuva"]].sum(axis=1) > 0))
                         ).astype(float).where(out["met_temp"].notna())
    out.index = df.index
    return out


def _brutos(dados: Path) -> list[tuple[pd.Timestamp, pd.Timestamp, Path]]:
    """(início, fim, arquivo) de cada parquet bruto: os meses de 2025 e o ranking de 2026."""
    janelas = []
    for p in sorted(dados.glob("training_2025-*.parquet")):
        ini, fim = p.stem.split("_")[1:3]
        janelas.append((pd.Timestamp(ini, tz="UTC"), pd.Timestamp(fim, tz="UTC"), p))
    ranking = dados / "ranking.parquet"
    if ranking.exists():
        janelas.append((pd.Timestamp("2026-01-01", tz="UTC"),
                        pd.Timestamp("2027-01-01", tz="UTC"), ranking))
    return janelas


def _pousos(t0: pd.Timestamp, t1: pd.Timestamp, dados: Path) -> pd.DataFrame:
    """Pousos com stand e in-block dos parquets que cobrem [t0 − `IDADE_MAX`, t1]."""
    partes = []
    for ini, fim, p in _brutos(dados):
        if fim <= t0 - IDADE_MAX or ini > t1:
            continue
        raw = pd.read_parquet(p, columns=ARR_COLS)
        arr = raw[raw["PHASE_mvt"] == "ARR"].dropna(subset=[BLOCK, "STAND_mvt"])
        del raw
        partes.append(pd.DataFrame({
            "t_a": pd.to_datetime(arr[BLOCK], utc=True),
            "apt": arr["ADES_mvt"].astype(str),
            "st": arr["STAND_mvt"].astype(str),
            "cia_a": _cia(arr["FLIGHT_mvt"]).astype(str),
            "tipo_a": arr["AIRCRAFT_TYPE_mvt"].astype(str),
        }))
    colunas = ["t_a", "apt", "st", "cia_a", "tipo_a"]
    if not partes:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in colunas}
                            ).astype({"t_a": "datetime64[ns, UTC]"})
    pousos = pd.concat(partes, ignore_index=True)
    return pousos.dropna(subset=["t_a"]).sort_values("t_a", ignore_index=True)[colunas]


def rotacao(df: pd.DataFrame, dados: Path = DATA) -> pd.DataFrame:
    """Colunas `rot_*`: o último pouso no mesmo aeroporto e stand nas 24 h anteriores.

    Só pousos com in-block anterior à hora do movimento entram, então nada do futuro nem
    o BLOCK da própria decolagem chega ao corretor. Os parquets lidos saem do período de
    `df`, e não do recorte: um voo tem o mesmo valor no treino, no holdout e no ranking.
    """
    consulta = pd.DataFrame({
        "i": np.arange(len(df)),
        "t": _hora(df).reset_index(drop=True),
        "apt": df[F.AIRPORT].astype(str).reset_index(drop=True),
        "st": df["STAND_mvt"].astype(str).reset_index(drop=True),
        "cia": _cia(df["FLIGHT_mvt"]).astype(str).reset_index(drop=True),
        "tipo": df["AIRCRAFT_TYPE_mvt"].astype(str).reset_index(drop=True),
    }).dropna(subset=["t"]).sort_values("t", ignore_index=True)
    vazio = pd.DataFrame(np.nan, index=df.index,
                         columns=["rot_mesma_cia", "rot_mesmo_tipo", "rot_idade"])
    if consulta.empty:
        return vazio
    pousos = _pousos(consulta["t"].iloc[0], consulta["t"].iloc[-1], dados)
    if pousos.empty:
        return vazio
    casado = pd.merge_asof(consulta, pousos, left_on="t", right_on="t_a", by=["apt", "st"],
                           direction="backward", tolerance=IDADE_MAX).set_index("i")
    achou = casado["t_a"].notna()
    out = pd.DataFrame({
        "rot_mesma_cia": (casado["cia"] == casado["cia_a"]).astype(float).where(achou),
        "rot_mesmo_tipo": (casado["tipo"] == casado["tipo_a"]).astype(float).where(achou),
        "rot_idade": (casado["t"] - casado["t_a"]).dt.total_seconds(),
    }).reindex(range(len(df)))
    out.index = df.index
    return out


def consistencia(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas `nm_*`: duração NM planejada × realizada e batidas com o plano de voo."""
    def duracao(fim: str, inicio: str) -> pd.Series:
        return (pd.to_datetime(df[fim], utc=True, errors="coerce")
                - pd.to_datetime(df[inicio], utc=True, errors="coerce")).dt.total_seconds()

    def igual(a: str, b: str) -> pd.Series:
        return (df[a].astype(str) == df[b].astype(str)).astype(float)

    plan, real = duracao("ARVT_1_flt", "EOBT_1_flt"), duracao("ARVT_3_flt", "AOBT_3_flt")
    return pd.DataFrame({
        "nm_dur_plan": plan,
        "nm_dur_real": real,
        "nm_dur_dif": real - plan,
        "nm_dur_razao": real / plan.where(plan > 0),  # plano de duração nula não divide
        "nm_ades_igual": igual("ADES_mvt", "ADES_flt"),
        "nm_ades_filed_igual": igual("ADES_flt", "ADES_FILED_flt"),
        "nm_tipo_igual": igual("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"),
        "nm_adep_igual": igual("ADEP_mvt", "ADEP_flt"),
    }, index=df.index)


def vocabulario(df: pd.DataFrame, n: int = CIAS_TOP) -> list[str]:
    """As `n` companhias mais frequentes de `df`: o vocabulário fixo do treino."""
    return sorted(_cia(df["FLIGHT_mvt"]).value_counts().index[:n])


def coluna_cia(df: pd.DataFrame, cias: list[str]) -> pd.Categorical:
    """Companhia de cada voo como categoria; fora do vocabulário vira `outro`."""
    cia = _cia(df["FLIGHT_mvt"]).astype(str)
    return pd.Categorical(cia.where(cia.isin(set(cias)), OUTRA), categories=[*cias, OUTRA])


def colunas_p13(df: pd.DataFrame, cias: list[str], raiz: Path = RAIZ,
                dados: Path = DATA) -> pd.DataFrame:
    """Todas as colunas do plano 13 das linhas de `df`, para o `corrector_frame`.

    Números como float e `cia` categórica, sempre na ordem das linhas de `df`.
    """
    partes = [metar(df, raiz), rotacao(df, dados), consistencia(df)]
    out = pd.concat([p.reset_index(drop=True).astype(float) for p in partes], axis=1)
    out["cia"] = coluna_cia(df, cias)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("acao", choices=["baixar"])
    ap.add_argument("--raiz", type=Path, default=RAIZ)
    baixar(ap.parse_args().raiz)
