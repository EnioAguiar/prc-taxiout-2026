"""Dados externos abertos no corretor: companhia, séries diárias e OPDI (plano 12).

Três blocos de colunas `ext_*`, sempre na ordem do quadro que recebem:

- `CopiaCia`: taxa de cópia do SCHED (|BLOCK − SCHED| ≤ 60 s) entre os voos com
  MVT − SCHED > 1 h, por (aeroporto, companhia, com/sem NM), suavizada para a taxa do
  aeroporto. `fit` guarda contagens por mês; `transform` soma só os meses de treino e
  nunca o mês da própria linha, então nenhuma linha vê o próprio BLOCK.
- `diarias`: séries diárias da EUROCONTROL (aderência a slot e atraso pré-partida total
  e de ATC) por (dia UTC, aeroporto).
- `opdi`: tempo em solo da aeronave desde o pouso anterior, do flight list do OPDI.

    bin/run src/externos.py baixar   # data/externo/ (fora do git), pula o que já existe

Fontes: séries diárias da EUROCONTROL (<https://ansperformance.eu/csv/>, dados públicos)
e OPDI v0.0.2 (<https://www.opdi.aero/>, open data com atribuição).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
from cache import DATA, split_paths

RAIZ = DATA / "externo"
TIME = "MVT_TIME_UTC_mvt"
SCHED = "SCHED_TIME_UTC_mvt"
BLOCK = "BLOCK_TIME_UTC_mvt"

# CopiaCia
K_SUAVIZA = 20      # peso da taxa do aeroporto no grupo (aeroporto, companhia, NM)
COPIA_S = 60        # |BLOCK − SCHED| que conta como cópia do horário programado
ATRASO_MIN_S = 3600  # só voos com MVT − SCHED acima disso entram na taxa
MESES_2025 = tuple(range(1, 13))

# OPDI
TOL_CALLSIGN_S = 600  # decolagem do OPDI com o mesmo callsign no mesmo aeroporto
TOL_APT_S = 90        # sem callsign, só o aeroporto: tolerância curta
OPDI_COLS = ["icao24", "flt_id", "adep", "ades", "first_seen", "last_seen"]

# Séries diárias: colunas usadas de cada CSV da EUROCONTROL.
SERIES = {
    "atfm_slot_adherence": ["FLT_DEP_1", "FLT_DEP_REG_1", "FLT_DEP_OUT_LATE_1"],
    "all_pre_departure_delays": ["FLT_DEP_IFR_2", "DLY_ALL_PRE_2"],
    "atc_pre_departure_delays": ["FLT_DEP_3", "DLY_ATC_PRE_3"],
}

# Download
ANOS = (2025, 2026)
URL_CSV = "https://www.eurocontrol.int/performance/data/download/csv/{nome}.csv"
URL_OPDI = ("https://www.eurocontrol.int/performance/data/download/OPDI/v002/"
            "flight_list/flight_list_{aaaamm}.parquet")
MESES_OPDI = (*(f"2025{m:02d}" for m in MESES_2025), "202601", "202607")


def _baixar(url: str, destino: Path) -> str:
    """Baixa `url` em `destino`; arquivo que já existe é pulado."""
    if destino.exists():
        return f"{destino.name}: já existe"
    destino.parent.mkdir(parents=True, exist_ok=True)
    parte = destino.with_suffix(destino.suffix + ".part")
    subprocess.run(["curl", "-sfL", "--retry", "5", "-o", str(parte), url], check=True)
    parte.rename(destino)  # arquivo pela metade nunca vira arquivo bom
    return f"{destino.name}: {destino.stat().st_size / 1e6:.1f} MB"


def baixar(raiz: Path = RAIZ) -> None:
    """Séries diárias da EUROCONTROL e flight lists do OPDI em `raiz`."""
    for nome in SERIES:
        for ano in ANOS:
            print(_baixar(URL_CSV.format(nome=f"{nome}_{ano}"), raiz / f"{nome}_{ano}.csv"),
                  flush=True)
    for aaaamm in MESES_OPDI:
        print(_baixar(URL_OPDI.format(aaaamm=aaaamm),
                      raiz / "opdi" / f"flight_list_{aaaamm}.parquet"), flush=True)


def _cia(flight: pd.Series) -> pd.Series:
    """Companhia: as 2–3 primeiras letras de `FLIGHT_mvt` (LH982 → LH); '?' sem prefixo."""
    return flight.astype("string").str.extract(r"^([A-Z]{2,3})", expand=False).fillna("?")


def _razao(a: pd.Series, b: pd.Series) -> pd.Series:
    """a/b, sem valor onde não há denominador."""
    return a / b.where(b > 0)


class CopiaCia:
    """Taxa de cópia do SCHED por (aeroporto, companhia, com/sem NM), aprendida por mês.

    Nos voos com MVT − SCHED > 1 h, parte das companhias tem o off-block oficial copiado
    do horário programado (o táxi vira o atraso inteiro) e parte não; a taxa do grupo,
    suavizada para a do aeroporto com k = `K_SUAVIZA`, diz quanto disso esperar.
    """

    def __init__(self, k: float = K_SUAVIZA) -> None:
        self.k = k
        self.contagens: pd.DataFrame | None = None  # (aeroporto, cia, nm, mês) → n, cópias

    def fit(self, raw_dep: pd.DataFrame) -> "CopiaCia":
        """Soma as contagens das DEP brutas de `raw_dep`; pode ser chamado mês a mês."""
        t = pd.to_datetime(raw_dep[TIME], utc=True)
        sched = pd.to_datetime(raw_dep[SCHED], utc=True)
        block = pd.to_datetime(raw_dep[BLOCK], utc=True)
        tab = pd.DataFrame({
            "apt": raw_dep[F.AIRPORT].astype("string"),
            "cia": _cia(raw_dep["FLIGHT_mvt"]),
            "nm": raw_dep["FLIGHT_ID_mvt"].isna().astype("int8"),
            "mes": t.dt.month,
            "n": 1.0,
            "copias": ((block - sched).dt.total_seconds().abs() <= COPIA_S).astype(float),
        })[(t - sched).dt.total_seconds() > ATRASO_MIN_S]
        novo = tab.groupby(["apt", "cia", "nm", "mes"])[["n", "copias"]].sum()
        self.contagens = novo if self.contagens is None else self.contagens.add(novo, fill_value=0)
        return self

    def transform(self, df: pd.DataFrame, meses_treino) -> pd.DataFrame:
        """`ext_taxa_cia` e `ext_taxa_cia_ms` das linhas de `df`, na ordem delas.

        Só os meses de `meses_treino` contam, e nunca o mês da própria linha: a taxa de
        um voo de março nunca foi aprendida com o BLOCK de março.
        """
        tab = self.contagens.reset_index()
        tab = tab[tab["mes"].isin({int(m) for m in meses_treino})]
        tab["chave"] = tab["apt"] + "|" + tab["cia"] + "|" + tab["nm"].astype(str)
        mes_txt = "|" + tab["mes"].astype(str)
        por_chave = tab.groupby("chave")[["n", "copias"]].sum()
        por_chave_mes = tab.groupby(tab["chave"] + mes_txt)[["n", "copias"]].sum()
        por_apt = tab.groupby("apt")[["n", "copias"]].sum()
        por_apt_mes = tab.groupby(tab["apt"] + mes_txt)[["n", "copias"]].sum()

        apt = df[F.AIRPORT].astype("string")
        chave = apt + "|" + _cia(df["FLIGHT_mvt"]) + "|" + df["nm_missing"].astype(int).astype(str)
        t = pd.to_datetime(df[TIME], utc=True)
        mes = "|" + t.dt.month.astype("Int64").astype("string")

        def fora_do_mes(k: pd.Series, k_mes: pd.Series, total, por_mes, col) -> pd.Series:
            """Soma dos meses de treino menos a do mês da própria linha."""
            return k.map(total[col]).fillna(0.0) - k_mes.map(por_mes[col]).fillna(0.0)

        n_apt = fora_do_mes(apt, apt + mes, por_apt, por_apt_mes, "n")
        c_apt = fora_do_mes(apt, apt + mes, por_apt, por_apt_mes, "copias")
        n = fora_do_mes(chave, chave + mes, por_chave, por_chave_mes, "n")
        c = fora_do_mes(chave, chave + mes, por_chave, por_chave_mes, "copias")

        taxa_apt = c_apt / n_apt.where(n_apt > 0)  # aeroporto sem histórico fica sem taxa
        taxa = (c + self.k * taxa_apt) / (n + self.k)
        ms = (t - pd.to_datetime(df[SCHED], utc=True)).dt.total_seconds()
        return pd.DataFrame({"ext_taxa_cia": taxa, "ext_taxa_cia_ms": taxa * ms}, index=df.index)


def _serie(raiz: Path, nome: str) -> pd.DataFrame:
    """Uma série diária da EUROCONTROL somada por (dia, aeroporto), de todos os anos."""
    arquivos = sorted(raiz.glob(f"{nome}_*.csv"))
    if not arquivos:
        raise SystemExit(f"falta {nome}_*.csv em {raiz}; rode: bin/run src/externos.py baixar")
    cols = SERIES[nome]
    df = pd.concat((pd.read_csv(p, usecols=["FLT_DATE", "APT_ICAO", *cols]) for p in arquivos),
                   ignore_index=True)
    chave = (pd.to_datetime(df["FLT_DATE"]).dt.strftime("%Y-%m-%d").astype("string")
             + "|" + df["APT_ICAO"].astype("string"))
    return df.groupby(chave)[cols].sum(min_count=1)  # dia sem medida continua sem medida


def diarias(df: pd.DataFrame, raiz: Path = RAIZ) -> pd.DataFrame:
    """Séries diárias do dia UTC e do aeroporto de cada linha; sem dado, sem valor."""
    t = pd.to_datetime(df[TIME], utc=True)
    chave = t.dt.strftime("%Y-%m-%d").astype("string") + "|" + df[F.AIRPORT].astype("string")
    slot, pre, atc = (_serie(raiz, nome) for nome in SERIES)
    reg = chave.map(slot["FLT_DEP_REG_1"])
    return pd.DataFrame({
        "ext_reg_frac": _razao(reg, chave.map(slot["FLT_DEP_1"])),
        "ext_late_frac": _razao(chave.map(slot["FLT_DEP_OUT_LATE_1"]), reg),
        "ext_reg_n": reg.astype(float),
        "ext_pre_min": _razao(chave.map(pre["DLY_ALL_PRE_2"]), chave.map(pre["FLT_DEP_IFR_2"])),
        "ext_atc_min": _razao(chave.map(atc["DLY_ATC_PRE_3"]), chave.map(atc["FLT_DEP_3"])),
    }, index=df.index)


def _flight_list(raiz: Path, meses: list[str]) -> pd.DataFrame:
    """Voos do OPDI dos meses pedidos, com o solo desde o voo anterior da mesma aeronave."""
    partes = []
    for aaaamm in meses:
        p = raiz / "opdi" / f"flight_list_{aaaamm}.parquet"
        if not p.exists():
            raise SystemExit(f"falta {p.name} em {p.parent}; rode: bin/run src/externos.py baixar")
        partes.append(pd.read_parquet(p, columns=OPDI_COLS))
    voos = pd.concat(partes, ignore_index=True)
    for col in ("first_seen", "last_seen"):
        voos[col] = pd.to_datetime(voos[col], utc=True)
    voos = voos.sort_values(["icao24", "first_seen"], ignore_index=True)
    mesma = voos["icao24"].eq(voos["icao24"].shift())
    voos["ext_solo_s"] = (
        (voos["first_seen"] - voos["last_seen"].shift()).dt.total_seconds().where(mesma)
    )
    anterior = voos["ades"].shift().where(mesma)
    igual = (anterior == voos["adep"]).fillna(False).astype(float)
    voos["ext_mesmo_apt"] = igual.where(anterior.notna() & voos["adep"].notna())
    return voos


def _asof(esq: pd.DataFrame, dir_: pd.DataFrame, por: list[str], tol_s: int) -> pd.DataFrame:
    """Linhas de `esq` com a decolagem do OPDI mais próxima em até `tol_s`, casando por `por`."""
    saida = ["i", "ext_solo_s", "ext_mesmo_apt"]
    if esq.empty or dir_.empty:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in saida})
    casado = pd.merge_asof(esq, dir_, left_on="t", right_on="first_seen", by=por,
                           tolerance=pd.Timedelta(seconds=tol_s), direction="nearest")
    return casado[casado["first_seen"].notna()][saida]


def opdi(df: pd.DataFrame, raiz: Path = RAIZ) -> pd.DataFrame:
    """`ext_solo_s` e `ext_mesmo_apt` de cada linha; só os meses de `df` são lidos."""
    t = pd.to_datetime(df[TIME], utc=True).reset_index(drop=True)
    apt = df[F.AIRPORT].astype("string").reset_index(drop=True)
    voos = _flight_list(raiz, sorted(t.dropna().dt.strftime("%Y%m").unique()))
    voos = voos[voos["adep"].isin(set(apt.dropna()))]  # só decolagens dos nossos aeroportos
    voos = pd.DataFrame({
        "apt": voos["adep"].astype("string"),
        "cs": voos["flt_id"].astype("string").str.strip(),
        "first_seen": voos["first_seen"],
        "ext_solo_s": voos["ext_solo_s"],
        "ext_mesmo_apt": voos["ext_mesmo_apt"],
    }).sort_values("first_seen", ignore_index=True)

    esq = pd.DataFrame({
        "i": np.arange(len(df)),
        "t": t,
        "apt": apt,
        "cs": df["CALLSIGN_flt"].astype("string").str.strip().reset_index(drop=True),
    }).dropna(subset=["t"]).sort_values("t", ignore_index=True)

    tem_cs = esq["cs"].notna() & (esq["cs"] != "")
    casado = _asof(esq[tem_cs], voos[voos["cs"].notna() & (voos["cs"] != "")],
                   ["apt", "cs"], TOL_CALLSIGN_S)
    resto = esq[~esq["i"].isin(casado["i"])]  # sem callsign no OPDI: só aeroporto e hora
    casado = pd.concat([casado, _asof(resto, voos, ["apt"], TOL_APT_S)], ignore_index=True)

    solo, mesmo = np.full(len(df), np.nan), np.full(len(df), np.nan)
    onde = casado["i"].to_numpy(int)
    solo[onde] = casado["ext_solo_s"].to_numpy(float)
    mesmo[onde] = casado["ext_mesmo_apt"].to_numpy(float)
    return pd.DataFrame({"ext_solo_s": solo, "ext_mesmo_apt": mesmo}, index=df.index)


def colunas_ext(df: pd.DataFrame, copia: CopiaCia, meses_treino,
                raiz: Path = RAIZ) -> dict[str, np.ndarray]:
    """Todas as colunas `ext_*` das linhas de `df`, prontas para o `corrector_frame`."""
    partes = [copia.transform(df, meses_treino), diarias(df, raiz), opdi(df, raiz)]
    return {c: p[c].to_numpy(float) for p in partes for c in p.columns}


def copia_cia_2025(run=None) -> CopiaCia:
    """`CopiaCia` ajustada nas DEP brutas de 2025, um mês por vez (um parquet na RAM)."""
    copia = CopiaCia()
    for p in split_paths("full2025"):
        raw = F.load([p])
        copia.fit(raw[raw["PHASE_mvt"] == "DEP"])
        del raw
    if run:
        run.log(f"taxa de cópia: {len(copia.contagens):,} grupos (aeroporto, cia, NM, mês)")
    return copia


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("acao", choices=["baixar"])
    ap.add_argument("--raiz", type=Path, default=RAIZ)
    baixar(ap.parse_args().raiz)
