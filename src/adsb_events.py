"""Eventos por voo a partir dos recortes do adsb.lol e casamento com as decolagens.

Decolagem = ponto no chão seguido, no mesmo avião, de ponto no ar com Δt < TAKEOFF_GAP_S.
Off-block observado = primeiro ponto do segmento contínuo no chão (lacunas < SEG_GAP_S)
que termina nessa decolagem. Teste de 1 dia em EDDM (README, item A): |erro| mediano 82 s.

Casamento com o movimento DEP: mesma callsign e decolagem a ±CS_TOL_S; senão a decolagem
mais próxima a ±TIME_TOL_S; um evento serve a um voo só.

Uso:
    bin/run src/adsb_events.py [--raiz DIR]    # grava <raiz>/events.parquet
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adsb import RAIZ  # noqa: E402

TAKEOFF_GAP_S = 120
SEG_GAP_S = 600
CS_TOL_S = 300
TIME_TOL_S = 90
DATA = Path(__file__).resolve().parent.parent / "data"

EVENT_COLS = ["apt", "takeoff", "first_ground", "first_move", "gs0", "n_chao", "gap_max", "frac_mlat",
              "lat0", "lon0", "callsign", "reg", "tipo", "icao"]
FEATURES = ["adsb_taxi", "adsb_taxi_move", "adsb_gs0", "adsb_takeoff_err", "adsb_n_chao", "adsb_gap_max",
            "adsb_frac_mlat", "adsb_lat0", "adsb_lon0"]
MOVE_KT = 1.0  # acima disso o avião está andando (pushback/táxi)


def add_features(df: pd.DataFrame, raiz: Path = RAIZ) -> pd.DataFrame:
    """Junta as colunas FEATURES por MVT_ID_mvt (NaN sem evento ou sem events.parquet).

    Escreve as nove colunas no próprio `df`. O `drop` + `merge` de antes copiava o quadro
    inteiro duas vezes (2 × 1,7 GB em full2025) só para acrescentá-las; como `MVT_ID_mvt` é
    único nos eventos, o left join é uma busca posicional e o resultado é o mesmo.
    """
    sobrando = [c for c in FEATURES if c in df.columns]
    if sobrando:
        df.drop(columns=sobrando, inplace=True)
    path = raiz / "events.parquet"
    ev = pd.read_parquet(path, columns=["MVT_ID_mvt", *FEATURES]) if path.exists() else None
    if ev is None or ev.empty:
        for c in FEATURES:
            df[c] = np.nan
        return df
    chaves = pd.Index(ev["MVT_ID_mvt"])
    if chaves.has_duplicates:
        raise ValueError("events.parquet com MVT_ID_mvt repetido: o join duplicaria linhas")
    onde = chaves.get_indexer(pd.Index(df["MVT_ID_mvt"]))
    achou = onde >= 0
    for c in FEATURES:
        v = ev[c].to_numpy(float)
        df[c] = np.where(achou, v[onde], np.nan)  # onde = −1 pega o último; a máscara descarta
    return df


def decolagens(cut: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por decolagem vista no recorte (colunas EVENT_COLS)."""
    d = cut.sort_values(["icao", "t"], kind="stable").reset_index(drop=True)
    icao = d["icao"].to_numpy()
    t = d["t"].to_numpy(float)
    chao = d["chao"].to_numpy(bool)
    same_prev = np.r_[False, icao[1:] == icao[:-1]]
    gap_prev = np.r_[np.inf, np.diff(t)]
    cont = same_prev & np.r_[False, chao[:-1]] & chao & (gap_prev < SEG_GAP_S)
    run = np.cumsum(chao & ~cont)  # id do segmento no chão (vale só onde chao)
    nxt_same = np.r_[same_prev[1:], False]
    nxt_air = np.r_[~chao[1:], False]
    nxt_gap = np.r_[gap_prev[1:], np.inf]
    last = np.flatnonzero(chao & nxt_same & nxt_air & (nxt_gap < TAKEOFF_GAP_S))
    if last.size == 0:
        return pd.DataFrame(columns=EVENT_COLS)

    g = d.loc[chao].assign(run=run[chao], gap=np.where(cont, gap_prev, 0.0)[chao],
                           mlat=(d["fonte"].astype(str) == "mlat").to_numpy()[chao])
    g["t_move"] = g["t"].where(g["gs"] > MOVE_KT)
    agg = g.groupby("run").agg(first_ground=("t", "min"), first_move=("t_move", "min"),
                               gs0=("gs", "first"), n_chao=("t", "size"),
                               gap_max=("gap", "max"), frac_mlat=("mlat", "mean"),
                               lat0=("lat", "first"), lon0=("lon", "first"))
    ev = pd.DataFrame({
        "run": run[last],
        "apt": d["apt"].to_numpy()[last + 1],
        "takeoff": t[last + 1],
        # callsign do ponto no ar (a do chão costuma faltar antes do transponder completo)
        "callsign": d["callsign"].astype(str).to_numpy()[last + 1],
        "reg": d["reg"].astype(str).to_numpy()[last],
        "tipo": d["tipo"].astype(str).to_numpy()[last],
        "icao": icao[last],
    }).join(agg, on="run")
    ev["callsign"] = ev["callsign"].replace({"nan": None, "None": None})
    return ev[EVENT_COLS].reset_index(drop=True)


def casar(dep: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """dep: MVT_ID_mvt, apt, mvt (epoch s), callsign. Devolve MVT_ID_mvt, mvt + colunas do evento."""
    ev = ev.reset_index(drop=True).assign(ev_id=lambda x: x.index)
    ev["apt"] = ev["apt"].astype(str)
    dep = dep.assign(apt=dep["apt"].astype(str))
    cs = dep.dropna(subset=["callsign"]).merge(
        ev.dropna(subset=["callsign"])[["ev_id", "apt", "callsign", "takeoff"]], on=["apt", "callsign"])
    pares = [cs[(cs["takeoff"] - cs["mvt"]).abs() < CS_TOL_S].assign(pri=0)]
    # proximidade no tempo, por aeroporto: vizinho anterior e posterior (o mais perto pode já
    # ter dono; aí o outro lado serve)
    for apt, dd in dep.groupby("apt"):
        ee = ev[ev["apt"] == apt].sort_values("takeoff")
        if ee.empty:
            continue
        dd = dd.sort_values("mvt")
        for direcao in ("backward", "forward"):
            m = pd.merge_asof(dd, ee[["ev_id", "takeoff"]], left_on="mvt", right_on="takeoff",
                              direction=direcao, tolerance=TIME_TOL_S).dropna(subset=["ev_id"])
            pares.append(m.assign(pri=1))
    p = pd.concat(pares, ignore_index=True)
    p["dist"] = (p["takeoff"] - p["mvt"]).abs()
    p = p.sort_values(["pri", "dist"], kind="stable")
    # guloso: par melhor primeiro, cada voo e cada evento uma vez só
    usados_v, usados_e, keep = set(), set(), []
    for i, v, e in zip(p.index, p["MVT_ID_mvt"].to_numpy(), p["ev_id"].to_numpy()):
        if v not in usados_v and e not in usados_e:
            usados_v.add(v)
            usados_e.add(e)
            keep.append(i)
    p = p.loc[keep]
    out = p[["MVT_ID_mvt", "mvt", "ev_id"]].merge(ev.drop(columns=["apt", "callsign"]), on="ev_id")
    return out.drop(columns="ev_id")


def features(m: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "MVT_ID_mvt": m["MVT_ID_mvt"].to_numpy(),
        "adsb_taxi": (m["mvt"] - m["first_ground"]).to_numpy(float),
        "adsb_taxi_move": (m["mvt"] - m["first_move"]).to_numpy(float),
        "adsb_gs0": m["gs0"].to_numpy(float),
        "adsb_takeoff_err": (m["takeoff"] - m["mvt"]).to_numpy(float),
        "adsb_n_chao": m["n_chao"].to_numpy(float),
        "adsb_gap_max": m["gap_max"].to_numpy(float),
        "adsb_frac_mlat": m["frac_mlat"].to_numpy(float),
        "adsb_lat0": m["lat0"].to_numpy(float),
        "adsb_lon0": m["lon0"].to_numpy(float),
    })


def carregar_deps() -> pd.DataFrame:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]
    paths = sorted(DATA.glob("training_2025-*.parquet")) + [DATA / "ranking.parquet"]
    d = pd.concat((pd.read_parquet(p, columns=cols) for p in paths), ignore_index=True)
    d = d[d["PHASE_mvt"] == "DEP"]
    mvt = pd.to_datetime(d["MVT_TIME_UTC_mvt"], utc=True)
    return pd.DataFrame({
        "MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy(),
        "apt": d["ADEP_mvt"].astype(str).to_numpy(),
        "mvt": (mvt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(),
        "dia": mvt.dt.strftime("%Y-%m-%d").to_numpy(),
        "callsign": d["CALLSIGN_flt"].fillna(d["FLIGHT_mvt"]).to_numpy(),
    })


def eventos_do_dia(cut_dir: Path, dia: str) -> pd.DataFrame:
    """Decolagens do dia; inclui as 2 h finais do dia anterior para o táxi que cruza a meia-noite."""
    ini = pd.Timestamp(dia, tz="UTC").timestamp()
    partes = []
    ontem = cut_dir / f"{(date.fromisoformat(dia) - timedelta(days=1)).isoformat()}.parquet"
    if ontem.exists():
        partes.append(pd.read_parquet(ontem, filters=[("t", ">=", ini - 7200)]))
    partes.append(pd.read_parquet(cut_dir / f"{dia}.parquet"))
    ev = decolagens(pd.concat(partes, ignore_index=True))
    return ev[(ev["takeoff"] >= ini) & (ev["takeoff"] < ini + 86400)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=RAIZ)
    a = ap.parse_args()
    cut_dir = a.raiz / "cut"
    deps = carregar_deps()
    dias = sorted(p.stem for p in cut_dir.glob("*.parquet"))
    out = []
    for i, dia in enumerate(dias, 1):
        dd = deps[deps["dia"] == dia]
        if dd.empty:
            continue
        m = casar(dd, eventos_do_dia(cut_dir, dia))
        out.append(features(m))
        if i % 20 == 0 or i == len(dias):
            print(f"[{i}/{len(dias)}] {dia}: {len(m)}/{len(dd)} casados", flush=True)
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(a.raiz / "events.parquet", compression="zstd", index=False)
    print(f"{len(res):,} decolagens com evento ADS-B → {a.raiz / 'events.parquet'}")


if __name__ == "__main__":
    main()
