"""Caçador de regras 4: deslocamentos exatos BLOCK = X + c por grupo (picos), medidos na v30.

Nos 10 meses de treino, para cada horário X e grupo, acha o deslocamento c (em passos de
60 s) com mais massa em |BLOCK − X − c| ≤ 30 s. Um pico forte e estreito é um horário
derivado de outro (ex.: BLOCK = AOBT_3 − 5 min). No holdout, mede nos voos do grupo: a
fração que cai no pico e quanto do erro² da v30 está nesses voos (o teto de acertá-los).
Lê só as colunas necessárias (cabe junto com um envio na RAM).
"""
import sys
sys.path.insert(0, "src")
import glob
import numpy as np
import pandas as pd
from cache import TRUTH, load_split
from experiment import lottery_mask
from features import ID, AIRPORT

HORARIOS = ["SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt",
            "ARVT_1_flt", "ARVT_3_flt"]
COLS = ["PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", *HORARIOS]
MIN_N = 500

partes = []
for f in sorted(glob.glob("data/training_2025-*.parquet")):
    if "2025-01-01" in f or "2025-07-01" in f:
        continue
    r = pd.read_parquet(f, columns=COLS)
    partes.append(r[r.PHASE_mvt == "DEP"])
tr = pd.concat(partes, ignore_index=True)
tr[AIRPORT] = tr["ADEP_mvt"]

hcols = [ID, AIRPORT, "FLIGHT_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", TRUTH, *HORARIOS,
         "to_takeoff_from_SCHED_TIME_UTC_mvt", "to_takeoff_from_LOBT_flt", "to_takeoff_from_IOBT_flt",
         "to_takeoff_from_EOBT_1_flt", "to_takeoff_from_AOBT_3_flt"]
h = load_split("holdout2025")[hcols]
h = h.merge(pd.read_parquet("runs/20260929-154118-v29_mapa_cf.parquet")[[ID, "pred"]], on=ID)
y, pr = h[TRUTH].to_numpy(float), h["pred"].to_numpy(float)
lot = lottery_mask(h, y)
e2 = (y - pr) ** 2
Ts = e2[~lot].sum()


def chaves(df):
    cia = df["FLIGHT_mvt"].astype(str).str.extract(r"^([A-Z]{2,3})")[0].fillna("?")
    apt = df[AIRPORT].astype(str)
    nm = np.where(df["FLIGHT_ID_mvt"].isna(), "semNM", "NM")
    return {"apt": apt, "apt|nm": apt + "|" + nm, "apt|cia": apt + "|" + cia}


ktr, kho = chaves(tr), chaves(h)
blk = pd.to_datetime(tr["BLOCK_TIME_UTC_mvt"], utc=True)
mvt_h = pd.to_datetime(h["MVT_TIME_UTC_mvt"], utc=True)
linhas = []
for X in HORARIOS:
    d = (blk - pd.to_datetime(tr[X], utc=True)).dt.total_seconds()
    gap_h = (mvt_h - pd.to_datetime(h[X], utc=True)).dt.total_seconds().to_numpy(float)
    for gnome, g in ktr.items():
        ok = d.notna()
        passo = (d[ok] / 60).round().astype(int)
        cont = pd.DataFrame({"g": g[ok].to_numpy(), "c": passo.to_numpy()}).value_counts()
        tot = g[ok].value_counts()
        topo = cont.groupby(level=0).head(1).reset_index(name="k")
        topo["n"] = topo["g"].map(tot)
        topo = topo[topo.n >= MIN_N]
        topo["frac"] = topo.k / topo.n
        # pico de verdade: massa no minuto do pico bem acima da vizinhança (±2..5 min)
        viz = cont.reset_index(name="kk")
        for _, t in topo[topo.frac >= 0.05].iterrows():
            v = viz[(viz.g == t.g) & (viz.c.sub(t.c).abs().between(2, 5))].kk.mean()
            pico = t.k / max(v if v == v else 0, 1)
            sel = (kho[gnome] == t.g).to_numpy()
            alvo = gap_h - 60 * t.c  # y se BLOCK = X + c
            no_pico = sel & (np.abs(y - alvo) <= 30)
            linhas.append({"X": X.replace("_TIME_UTC_mvt", "").replace("_flt", ""), "grupo": f"{gnome}={t.g}",
                           "c_min": int(t.c), "frac_treino": round(t.frac, 3), "pico/vizinhos": round(pico, 1),
                           "n_holdout": int(sel.sum()), "no_pico_holdout": int(no_pico.sum()),
                           "erro2_sem_lot_no_pico_%": round(100 * e2[no_pico & ~lot].sum() / Ts, 3)})
    print(f"{X}: feito", flush=True)

res = pd.DataFrame(linhas)
res = res[(res["pico/vizinhos"] >= 3)].sort_values("erro2_sem_lot_no_pico_%", ascending=False)
res.to_csv(".superpowers/noite2/cacador4.csv", index=False)
print(res.head(30).to_string(index=False))
