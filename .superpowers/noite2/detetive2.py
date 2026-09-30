"""Detetive do resíduo 2: quanto das colunas brutas ainda explica o erro da v30 (teto otimista).

LightGBM no resíduo y − pred da v30, 5 dobras por dia dentro do holdout jan+jul 2025, com
todas as colunas do holdout (numéricas, datas como hora/minuto, categóricas brutas) e a
própria previsão. Ganho = RMSE da v30 menos RMSE da v30 + resíduo previsto.
"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import pandas as pd
import stack
from cache import TRUTH, load_split
from experiment import lottery_mask
from features import ID

h = load_split("holdout2025")
h = h.merge(pd.read_parquet("runs/20260929-154118-v29_mapa_cf.parquet")[[ID, "pred", "dia"]], on=ID)
y, pr = h[TRUTH].to_numpy(float), h["pred"].to_numpy(float)
lot = lottery_mask(h, y)
fora = {TRUTH, "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", ID, "FLIGHT_ID_mvt", "dia", "FLIGHT_mvt", "CALLSIGN_flt"}
X = pd.DataFrame(index=h.index)
for c in h.columns:
    if c in fora or c.startswith("gap_BLOCK") or "BLOCK" in c:
        continue
    s = h[c]
    if str(s.dtype).startswith("datetime"):
        t = pd.to_datetime(s, utc=True)
        X[c + "_h"] = t.dt.hour + t.dt.minute / 60
    elif s.dtype == object or str(s.dtype) in ("category", "string", "str"):
        X[c] = s.astype(str).astype("category")
    else:
        X[c] = pd.to_numeric(s, errors="coerce")
alvo = y - pr
dobra = stack.day_folds(h["dia"].to_numpy())
params = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              max_cat_to_onehot=8, cat_smooth=20, num_threads=12, verbose=-1, seed=0)
corr = np.zeros(len(y))
imp = pd.Series(0.0, index=X.columns)
for k in np.unique(dobra):
    tr, te = dobra != k, dobra == k
    # os voos-loteria saem do treino: só ensinariam ruído
    m = lgb.train(params, lgb.Dataset(X[tr & ~lot], alvo[tr & ~lot]), 300)
    corr[te] = m.predict(X[te])
    imp += pd.Series(m.feature_importance("gain"), index=X.columns)
novo = np.clip(pr + corr, 0, None)
r = lambda q, s=slice(None): np.sqrt(np.mean((y[s] - q[s]) ** 2))
nrm = (y < 3600) & h["FLIGHT_ID_mvt"].notna().to_numpy()
print(f"completo    {r(pr):8.2f} → {r(novo):8.2f}")
print(f"sem loteria {r(pr, ~lot):8.2f} → {r(novo, ~lot):8.2f}")
print(f"normais NM  {r(pr, nrm):8.2f} → {r(novo, nrm):8.2f}")
print((imp / imp.sum()).sort_values(ascending=False).head(20).round(3).to_string())
