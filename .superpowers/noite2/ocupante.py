"""Próximo ocupante do stand identificado pelo icao24 do OPDI (holdout jan+jul 2025, v30)."""
import sys
sys.path.insert(0, "src")
import glob
import numpy as np
import pandas as pd
import externos as E
from cache import load_split, TRUTH
from experiment import lottery_mask
from features import ID, AIRPORT

V30 = "runs/20260929-154118-v29_mapa_cf.parquet"
p = pd.read_parquet(V30)
h = load_split("holdout2025")
d = p[[ID, "pred"]].merge(h, on=ID, validate="one_to_one").reset_index(drop=True)
y = d[TRUTH].to_numpy(float)
pr = d["pred"].to_numpy(float)
lot = lottery_mask(d, y)
mvt = pd.to_datetime(d["MVT_TIME_UTC_mvt"], utc=True)

# pousos dos meses do holdout (e o dia anterior a cada mês não importa: janela de 24 h)
cols = ["PHASE_mvt", "ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "CALLSIGN_flt", "FLIGHT_mvt"]
arr = []
for f in sorted(glob.glob("data/training_2025-*.parquet")):
    if not any(m in f for m in ("2025-01-01", "2025-07-01", "2024-12", "2025-06-01")):
        continue
    r = pd.read_parquet(f, columns=cols)
    arr.append(r[r.PHASE_mvt == "ARR"])
arr = pd.concat(arr, ignore_index=True).dropna(subset=["BLOCK_TIME_UTC_mvt", "STAND_mvt"])
arr["t_land"] = pd.to_datetime(arr["MVT_TIME_UTC_mvt"], utc=True)
arr["t_a"] = pd.to_datetime(arr["BLOCK_TIME_UTC_mvt"], utc=True)

voos = E._flight_list(E.RAIZ, ["202412", "202501", "202506", "202507", "202508"]
                      if (E.RAIZ / "opdi" / "flight_list_202412.parquet").exists()
                      else ["202501", "202506", "202507", "202508"])
voos["cs"] = voos["flt_id"].astype("string").str.strip()


def casa(esq: pd.DataFrame, apt_col: str, t_col: str, lado: str) -> np.ndarray:
    """icao24 de cada linha de `esq` (callsign ±600 s, senão só aeroporto ±90 s)."""
    v = voos[voos[lado].isin(set(esq[apt_col]))]
    v = pd.DataFrame({"apt": v[lado].astype(str), "cs": v["cs"], "t_v": v[t_col], "icao24": v["icao24"]})
    q = pd.DataFrame({"i": np.arange(len(esq)), "t": esq["_t"].to_numpy(), "apt": esq[apt_col].astype(str).to_numpy(),
                      "cs": esq["CALLSIGN_flt"].astype("string").str.strip().to_numpy()})
    q["cs"] = q["cs"].astype("string")
    v["cs"] = v["cs"].astype("string")
    v = v.dropna(subset=["t_v"]).sort_values("t_v")
    q = q.dropna(subset=["t"]).sort_values("t")
    ok = q.cs.notna() & (q.cs != "")
    a = pd.merge_asof(q[ok], v[v.cs.notna()], left_on="t", right_on="t_v", by=["apt", "cs"],
                      tolerance=pd.Timedelta(seconds=600), direction="nearest")
    a = a[a.icao24.notna()]
    resto = q[~q.i.isin(a.i)]
    b = pd.merge_asof(resto, v.drop(columns="cs"), left_on="t", right_on="t_v", by="apt",
                      tolerance=pd.Timedelta(seconds=90), direction="nearest")
    b = b[b.icao24.notna()]
    out = np.full(len(esq), None, dtype=object)
    out[a.i.to_numpy()] = a.icao24.to_numpy()
    out[b.i.to_numpy()] = b.icao24.to_numpy()
    return out, set(a.i)


d["_t"] = mvt
dep_icao, dep_cs = casa(d, AIRPORT, "first_seen", "adep")
arr["_t"] = arr["t_land"]
arr_icao, arr_cs = casa(arr, "ADES_mvt", "last_seen", "ades")
arr["icao24"] = arr_icao

# último pouso no mesmo aeroporto e stand antes do MVT (24 h)
q = pd.DataFrame({"i": np.arange(len(d)), "t": mvt, "apt": d[AIRPORT].astype(str),
                  "st": d["STAND_mvt"].astype(str)}).dropna(subset=["t"]).sort_values("t")
a = pd.DataFrame({"t_a": arr["t_a"], "apt": arr["ADES_mvt"].astype(str), "st": arr["STAND_mvt"].astype(str),
                  "icao_a": arr["icao24"]}).sort_values("t_a")
m = pd.merge_asof(q, a, left_on="t", right_on="t_a", by=["apt", "st"], direction="backward",
                  tolerance=pd.Timedelta(hours=24)).set_index("i").reindex(range(len(d)))
gap = (mvt - m["t_a"]).dt.total_seconds().to_numpy(float)  # MVT − in-block do último pouso
icao_a = m["icao_a"].to_numpy(object)
tem = ~np.isnan(gap) & pd.notna(icao_a) & pd.notna(dep_icao)
outro = tem & (icao_a != dep_icao)   # próximo ocupante: y ≥ gap
proprio = tem & (icao_a == dep_icao)  # o próprio avião: y ≤ gap

viol_outro = outro & (y < gap - 60)
viol_proprio = proprio & (y > gap + 60)
print(f"DEP com icao24 {pd.notna(dep_icao).mean():.2f} · pousos com icao24 {pd.notna(arr_icao).mean():.2f}")
print(f"último pouso = outro avião: {outro.sum()} (viola y ≥ gap: {viol_outro.sum()})")
print(f"último pouso = o próprio:   {proprio.sum()} (viola y ≤ gap: {viol_proprio.sum()})")

r = lambda q_: np.sqrt(np.mean((y - q_) ** 2))
rs = lambda q_: np.sqrt(np.mean((y[~lot] - q_[~lot]) ** 2))
base = pr.copy()
piso = np.where(outro, np.maximum(base, gap), base)
teto = np.where(proprio, np.minimum(base, gap), base)
ambos = np.where(proprio, np.minimum(piso, gap), piso)
for nome, q_ in (("v30", base), ("piso (próximo ocupante)", piso), ("teto (próprio avião)", teto), ("ambos", ambos)):
    print(f"{nome:26s} completo {r(q_):7.2f} · sem loteria {rs(q_):7.2f} · mexe {(np.abs(q_ - base) > 1).sum()}")
mexe = np.abs(ambos - base) > 1
print(pd.DataFrame({"apt": d[AIRPORT].astype(str)[mexe], "nm": d.FLIGHT_ID_mvt.isna()[mexe],
                    "ganho": ((y - base) ** 2 - (y - ambos) ** 2)[mexe]}).groupby(["apt", "nm"])["ganho"]
      .agg(["size", "sum"]).sort_values("sum").to_string())

# diagnóstico dos "outro avião"
arr_cs_mask = np.zeros(len(arr), bool); arr_cs_mask[list(arr_cs)] = True
dep_cs_mask = np.zeros(len(d), bool); dep_cs_mask[list(dep_cs)] = True
o = pd.DataFrame({"gap_min": gap / 60, "viola": viol_outro, "dep_cs": dep_cs_mask,
                  "tipo_igual": False})[outro]
print(o.groupby(pd.cut(o.gap_min, [0, 5, 15, 30, 60, 120, 300, 1440]), observed=True)["viola"].agg(["size", "mean"]).round(2).to_string())
print("DEP casado por callsign:", o.groupby("dep_cs")["viola"].agg(["size", "mean"]).round(2).to_string())

print("--- piso só com pouso recente de outro avião")
for lim in (10, 15, 20, 30):
    sel = outro & (gap <= lim * 60)
    q_ = np.where(sel, np.maximum(base, gap), base)
    muda = np.abs(q_ - base) > 1
    print(f"gap ≤ {lim:2d} min: aplica {sel.sum():6d}, mexe {muda.sum():5d}, viola {(sel & (y < gap - 60)).sum():4d} · "
          f"completo {r(q_):7.2f} · sem loteria {rs(q_):7.2f} · normais "
          f"{np.sqrt(np.mean((y[(y<3600)&~lot]-q_[(y<3600)&~lot])**2)):.2f} vs {np.sqrt(np.mean((y[(y<3600)&~lot]-base[(y<3600)&~lot])**2)):.2f}")
