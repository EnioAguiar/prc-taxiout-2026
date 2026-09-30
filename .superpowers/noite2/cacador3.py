"""Caçador de regras 3: janelas BLOCK − X por grupo, aprendidas no treino e medidas na v30.

Para cada horário X e cada grupo (aeroporto, NM, companhia, pista, tipo de voo e cruzamentos),
a janela [lo, hi] de BLOCK − X nos 10 meses de treino vira o intervalo de y
[MVT − X − hi, MVT − X − lo]; a previsão da v30 no holdout é projetada nele. Guarda o ganho
de erro² (completo e sem loteria) e a taxa de violação no holdout.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from cache import load_split, TRUTH
from experiment import lottery_mask
from features import ID, AIRPORT

HORARIOS = ["SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt",
            "AOBT_3_flt", "ARVT_3_flt", "MVT_TIME_UTC_mvt"]
QS = (0.0, 1e-4, 1e-3)
MIN_N = 300

tr = load_split("full2025")
tr = tr[~pd.to_datetime(tr["MVT_TIME_UTC_mvt"]).dt.month.isin([1, 7])].reset_index(drop=True)
h = load_split("holdout2025")
v30 = pd.read_parquet("runs/20260929-154118-v29_mapa_cf.parquet")[[ID, "pred"]]
h = h.merge(v30, on=ID, validate="one_to_one").reset_index(drop=True)
y = h[TRUTH].to_numpy(float)
pr = h["pred"].to_numpy(float)
lot = lottery_mask(h, y)
e0 = (y - pr) ** 2
T, Ts = e0.sum(), e0[~lot].sum()


def grupos(df: pd.DataFrame) -> dict[str, pd.Series]:
    cia = df["FLIGHT_mvt"].astype(str).str.extract(r"^([A-Z]{2,3})")[0].fillna("?")
    apt = df[AIRPORT].astype(str)
    nm = np.where(df["FLIGHT_ID_mvt"].isna(), "semNM", "NM")
    return {
        "global": pd.Series("*", index=df.index),
        "apt": apt,
        "apt|nm": apt + "|" + nm,
        "apt|cia": apt + "|" + cia,
        "apt|nm|cia": apt + "|" + nm + "|" + cia,
        "apt|pista": apt + "|" + df["RUNWAY_mvt"].astype(str),
        "apt|tipo_voo": apt + "|" + df["FLIGHT_TYPE_flt"].astype(str),
        "cia": cia,
    }


gtr, gho = grupos(tr), grupos(h)
blk_tr = pd.to_datetime(tr["BLOCK_TIME_UTC_mvt"], utc=True)
mvt_ho = pd.to_datetime(h["MVT_TIME_UTC_mvt"], utc=True)
linhas = []
for X in HORARIOS:
    d_tr = (blk_tr - pd.to_datetime(tr[X], utc=True)).dt.total_seconds()
    gap_ho = (mvt_ho - pd.to_datetime(h[X], utc=True)).dt.total_seconds().to_numpy(float)  # MVT − X
    for gnome in gtr:
        chave_tr, chave_ho = gtr[gnome], gho[gnome]
        ok = d_tr.notna()
        stats = d_tr[ok].groupby(chave_tr[ok])
        n = stats.size()
        for q in QS:
            lo = stats.quantile(q) if q else stats.min()
            hi = stats.quantile(1 - q) if q else stats.max()
            tab = pd.DataFrame({"n": n, "lo": lo, "hi": hi})
            tab = tab[tab.n >= MIN_N]
            lo_ho = chave_ho.map(tab["lo"]).to_numpy(float)
            hi_ho = chave_ho.map(tab["hi"]).to_numpy(float)
            ymin, ymax = gap_ho - hi_ho, gap_ho - lo_ho  # y = MVT − BLOCK = gap − (BLOCK − X)
            tem = ~np.isnan(ymin)
            novo = np.where(tem, np.clip(pr, np.where(tem, ymin, -np.inf), np.where(tem, ymax, np.inf)), pr)
            novo = np.clip(novo, 0, None)
            mexe = np.abs(novo - pr) > 1
            if not mexe.any():
                continue
            e1 = (y - novo) ** 2
            viol = tem & ((y < ymin - 1) | (y > ymax + 1))
            linhas.append({
                "X": X.replace("_TIME_UTC_mvt", "").replace("_flt", ""), "grupo": gnome, "q": q,
                "grupos_uteis": len(tab), "mexe": int(mexe.sum()),
                "viola_%": round(100 * viol[tem].mean(), 3),
                "ganho_completo_%": round(100 * (e0.sum() - e1.sum()) / T, 3),
                "ganho_sem_lot_%": round(100 * (e0[~lot].sum() - e1[~lot].sum()) / Ts, 3),
            })
    print(f"{X}: feito", flush=True)

res = pd.DataFrame(linhas).sort_values("ganho_sem_lot_%", ascending=False)
res.to_csv(".superpowers/noite2/cacador3.csv", index=False)
print(res.head(25).to_string(index=False))
