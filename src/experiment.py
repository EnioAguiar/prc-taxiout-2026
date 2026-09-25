"""Experimento na simulação calibrada do ranking.

    bin/run src/experiment.py <nome> --model single [--rounds 400] [--nota "texto"]
    bin/run src/experiment.py <nome> --model two_stage [--cls-rounds 400] [--reg-rounds 400]
    bin/run src/experiment.py <nome> --model two_stage_nm [--nm-split-ms]

Treina em train2025 (10 meses), prevê holdout2025 (jan+jul/2025 montados como o
ranking) e registra RMSE completo, sem outliers, por grupo e por aeroporto.
Previsões em runs/<id>.parquet; resumo em experiments.jsonl.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import features as F
from cache import TRUTH, load_split
from models import MODELS, leaky_columns, prepare
from runlog import ROOT, Run

RUNS = ROOT / "runs"
MAX_NORMAL_S = 3 * 3600


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def metrics(df: pd.DataFrame, pred: np.ndarray) -> dict:
    y = df[TRUTH].to_numpy(float)
    pred = np.asarray(pred, float)
    nm = df["FLIGHT_ID_mvt"].isna().to_numpy()
    normal = (y > 0) & (y < MAX_NORMAL_S)
    tail = y > 3600
    m = {
        "completo": rmse(y, pred),
        "sem_outliers": rmse(y[normal], pred[normal]),
        "nm_presente": rmse(y[~nm], pred[~nm]),
        "nm_ausente": rmse(y[nm], pred[nm]),
        "y_gt_1h": rmse(y[tail], pred[tail]),
    }
    m = {k: round(v, 2) for k, v in m.items()}
    groups = df.groupby(F.AIRPORT, observed=True).indices
    m["por_aeroporto"] = {str(a): round(rmse(y[i], pred[i]), 1) for a, i in groups.items()}
    return m


def config(a: argparse.Namespace) -> dict:
    if a.model == "single":
        return {"model": a.model, "rounds": a.rounds, "seed": a.seed}
    cfg = {
        "model": a.model,
        "cls_rounds": a.cls_rounds,
        "reg_rounds": a.reg_rounds,
        "seed": a.seed,
    }
    if a.model == "two_stage_nm":
        cfg["nm_split_ms"] = a.nm_split_ms
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("nome")
    ap.add_argument("--model", choices=sorted(MODELS), default="single")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--cls-rounds", type=int, default=400)
    ap.add_argument("--reg-rounds", type=int, default=400)
    ap.add_argument("--nm-split-ms", action="store_true",
                    help="two_stage_nm: retas separadas para atraso > 2 h (célula de Roma)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nota", default="")
    a = ap.parse_args()
    cfg = config(a)

    with Run(a.nome, cfg) as run:
        run.set(nota=a.nota)
        with run.phase("dados", 0.15):
            train, hold = load_split("train2025"), load_split("holdout2025")
            rk = load_split("ranking2026")
            cols = prepare(train, [hold])
            drop = leaky_columns(train, rk, cols)
            cols = [c for c in cols if c not in drop]
            del rk
            run.log(
                f"treino {len(train):,} · holdout {len(hold):,} · {len(cols)} features"
                f" · ignoradas: {drop or 'nenhuma'}"
            )
        with run.phase("treino", 0.75):
            model = MODELS[a.model](cfg).fit(train, cols, run=run, valid=hold)
        with run.phase("métricas", 0.10):
            pred = model.predict(hold)
            m = metrics(hold, pred)
            RUNS.mkdir(exist_ok=True)
            path = RUNS / f"{run.id}.parquet"
            pd.DataFrame({
                F.ID: hold[F.ID].to_numpy(),
                "dia": hold["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").to_numpy(),
                TRUTH: hold[TRUTH].to_numpy(),
                "pred": pred,
            }).to_parquet(path, index=False)
        run.metric(**m)
        run.set(best_iter=model.best_iter, previsoes=str(path.relative_to(ROOT)))
        run.log(" · ".join(f"{k} {v}" for k, v in m.items() if k != "por_aeroporto"))
        worst = sorted(m["por_aeroporto"].items(), key=lambda kv: -kv[1])
        run.log("por aeroporto: " + ", ".join(f"{k} {v}" for k, v in worst))


if __name__ == "__main__":
    main()
