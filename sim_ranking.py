"""Throwaway: reproduz o caminho do submit em jan+jul/2025 (montados à parte, alvo apagado)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import features as F
import train as T

paths = sorted(Path("data").glob("training_*.parquet"))
hold = [p for p in paths if "-01-01_" in p.name or "-07-01_" in p.name]
fit = [p for p in paths if p not in hold]

tr = F.build(F.load(fit))
tr = tr[tr[F.TARGET].notna()]

raw = F.load(hold)
truth = raw.loc[raw.PHASE_mvt == "DEP", [F.ID, F.TARGET]].set_index(F.ID)[F.TARGET]
dep = raw.PHASE_mvt == "DEP"
raw.loc[dep, ["BLOCK_TIME_UTC_mvt", F.TARGET]] = pd.NaT, np.nan
rk = F.build(raw)
for c in ["apt_dep_prev_10m", "rwy_dep_prev_10m", "apt_arr_prev_10m"]:
    print(c, "p50", rk[c].median(), "p95", rk[c].quantile(0.95))

(pred,) = T.fit_predict(tr, [rk], [], rounds=int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
p = pd.Series(pred, index=rk[F.ID].to_numpy()).reindex(truth.index)
p = p.fillna(p.median())
ok = (truth > 0) & (truth < 3 * 3600)
print("RMSE completo:", T.rmse(truth.to_numpy(), p.to_numpy()))
print("RMSE sem outliers:", T.rmse(truth[ok].to_numpy(), p[ok].to_numpy()))
