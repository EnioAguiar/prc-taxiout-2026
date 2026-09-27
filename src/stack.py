"""Empilhamento fora do fold sobre uma corrida base (plano 4, tarefa 4b).

Um LightGBM aprende a correção `y − pred_base` no holdout jan+jul/2025 com 5 folds
por dia (cada voo é previsto por um modelo que não viu o dia dele). Entradas: `pred`,
aeroporto, `nm_missing`, hora, `to_takeoff_from_*` e, sem `--sem-adsb`, as colunas
`adsb_*` de `events.parquet` (NaN sem cobertura). A corrida é gravada em `runs/` e no
`experiments.jsonl`, então passa pelo `compare.py` como qualquer outra.

Uso:
    bin/run src/stack.py <nome> [--base <id>] [--sem-adsb]
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import features as F
from adsb import RAIZ
from adsb_events import FEATURES as ADSB
from cache import TRUTH, load_split
from experiment import RUNS, metrics
from runlog import ROOT, Run

FOLDS = 5
ROUNDS = 300
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              verbose=-1, num_threads=12, seed=0, deterministic=True, force_row_wise=True)


def day_folds(days: np.ndarray, k: int = FOLDS) -> np.ndarray:
    """Fold de cada linha: dias ordenados distribuídos em rodízio (um dia inteiro por fold)."""
    uniq = np.array(sorted(set(days)))
    return pd.Series(np.arange(uniq.size) % k, index=uniq)[days].to_numpy()


def oof_correction(X: pd.DataFrame, y: np.ndarray, base: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.empty(len(y))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr] - base[tr]), ROUNDS)
        out[te] = base[te] + m.predict(X[te])
    return np.clip(out, 0, None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("nome")
    ap.add_argument("--base", help="id da corrida base (padrão: campeã)")
    ap.add_argument("--sem-adsb", action="store_true", help="controle: mesmo empilhamento sem adsb_*")
    ap.add_argument("--nota", default="")
    a = ap.parse_args()
    base_id = a.base or json.loads((ROOT / "champion.json").read_text())["id"]
    cfg = {"model": "stack", "base": base_id, "adsb": not a.sem_adsb, "folds": FOLDS,
           "rounds": ROUNDS, "seed": PARAMS["seed"]}

    with Run(a.nome, cfg) as run:
        run.set(nota=a.nota)
        with run.phase("dados", 0.3):
            base = pd.read_parquet(RUNS / f"{base_id}.parquet")
            hold = load_split("holdout2025")
            hold = hold.set_index(F.ID).loc[base[F.ID]].reset_index()  # mesma ordem da base
            cols = [F.AIRPORT, "nm_missing", "hour", *[c for c in hold if c.startswith("to_takeoff_from_")]]
            X = hold[cols].copy()
            X[F.AIRPORT] = X[F.AIRPORT].astype("category")
            X["pred"] = base["pred"].to_numpy()
            if not a.sem_adsb:
                ev = pd.read_parquet(RAIZ / "events.parquet", columns=[F.ID, *ADSB])
                X = X.join(hold[[F.ID]].merge(ev, on=F.ID, how="left").drop(columns=F.ID))
                X["adsb_menos_pred"] = X["adsb_taxi_move"] - X["pred"]
                run.log(f"adsb: {X['adsb_taxi'].notna().mean():.1%} dos voos com evento")
        with run.phase("treino", 0.6):
            y = hold[TRUTH].to_numpy(float)
            days = base["dia"].to_numpy()
            pred = oof_correction(X, y, base["pred"].to_numpy(float), day_folds(days))
        with run.phase("métricas", 0.1):
            m = metrics(hold, pred)
            path = RUNS / f"{run.id}.parquet"
            pd.DataFrame({F.ID: base[F.ID], "dia": days, TRUTH: y, "pred": pred}).to_parquet(path, index=False)
        run.metric(**m)
        run.set(previsoes=str(path.relative_to(ROOT)))
        run.log(" · ".join(f"{k} {v}" for k, v in m.items() if not isinstance(v, dict)))


if __name__ == "__main__":
    main()
