"""Empilhamento sobre uma corrida base: um LightGBM aprende a correção `y − pred_base`.

Duas simulações do mesmo corretor:

- padrão (plano 4): 5 folds por dia no holdout jan+jul/2025 — cada voo é corrigido por um
  modelo que não viu o dia dele;
- `--crossfit` (plano 5): o corretor treina no ano inteiro, sobre as previsões da base
  fora do bloco (`crossfit.oof_base` nas linhas cegas de `blind2025`), e é medido no
  holdout com as previsões da corrida base.

Entradas do corretor: `pred`, aeroporto, `nm_missing`, hora, `to_takeoff_from_*` e, sem
`--sem-adsb`, as colunas `adsb_*` de `events.parquet` (NaN sem cobertura); a saída é
`pred + correção` com piso 0. A corrida é gravada em `runs/` e no `experiments.jsonl`,
então passa pelo `compare.py` como qualquer outra.

Uso:
    bin/run src/stack.py <nome> [--base <id>] [--crossfit [--seeds N]] [--sem-adsb]

Com `--crossfit`, `--seeds N` (N > 1) manda a base de cada bloco ser a média de N seeds:
sobrescreve `seeds` na config da corrida base. A base do holdout vem pronta de `--base`,
que já deve ser a corrida de N seeds.
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import features as F
from adsb_events import FEATURES as ADSB
from cache import TRUTH, load_split
from crossfit import oof_base
from experiment import RUNS, metrics, rmse
from runlog import REGISTRY, ROOT, Run

FOLDS = 5
ROUNDS = 300
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              verbose=-1, num_threads=12, seed=0, deterministic=True, force_row_wise=True)


def base_config(base_id: str) -> dict:
    """Config com que a corrida base foi medida (a última linha dela no registro)."""
    for line in reversed(REGISTRY.read_text().splitlines()):
        rec = json.loads(line) if line.strip() else {}
        if rec.get("id") == base_id:
            return rec["config"]
    raise SystemExit(f"corrida base {base_id} não está em {REGISTRY.name}")


def config_da_base(base_id: str, seeds: int) -> dict:
    """Config dos blocos: a da corrida base, com média de N seeds quando N > 1."""
    cfg = base_config(base_id)
    return {**cfg, "seeds": seeds} if seeds > 1 else cfg


def corrector_frame(df: pd.DataFrame, pred: np.ndarray, adsb: bool = True) -> pd.DataFrame:
    """Entradas do corretor: a previsão da base, o contexto do voo e o rastro ADS-B."""
    cols = [F.AIRPORT, "nm_missing", "hour", *[c for c in df if c.startswith("to_takeoff_from_")]]
    X = df[cols].copy()
    X[F.AIRPORT] = X[F.AIRPORT].astype("category")
    X["pred"] = np.asarray(pred, float)
    if adsb:
        X = X.join(df[ADSB])
        X["adsb_menos_pred"] = X["adsb_taxi_move"] - X["pred"]
    return X


def fit_corrector(X: pd.DataFrame, y: np.ndarray, base: np.ndarray) -> lgb.Booster:
    """Corretor L2 na diferença entre o alvo e a previsão da base."""
    return lgb.train(PARAMS, lgb.Dataset(X, y - base), ROUNDS)


def apply_corrector(model: lgb.Booster, X: pd.DataFrame, base: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(base, float) + model.predict(X), 0, None)


def day_folds(days: np.ndarray, k: int = FOLDS) -> np.ndarray:
    """Fold de cada linha: dias ordenados distribuídos em rodízio (um dia inteiro por fold)."""
    uniq = np.array(sorted(set(days)))
    return pd.Series(np.arange(uniq.size) % k, index=uniq)[days].to_numpy()


def oof_correction(X: pd.DataFrame, y: np.ndarray, base: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.empty(len(y))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        model = fit_corrector(X[tr], y[tr], base[tr])
        out[te] = apply_corrector(model, X[te], base[te])
    return out


def holdout_da_base(base: pd.DataFrame) -> pd.DataFrame:
    """Holdout na mesma ordem das previsões da corrida base."""
    return load_split("holdout2025").set_index(F.ID).loc[base[F.ID]].reset_index()


def simulacao_folds(
    run: Run, base_id: str, adsb: bool
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Corretor em 5 folds por dia dentro do holdout; devolve (holdout, base, previsão)."""
    with run.phase("dados", 0.3):
        base = pd.read_parquet(RUNS / f"{base_id}.parquet")
        hold = holdout_da_base(base)
        X = corrector_frame(hold, base["pred"].to_numpy(float), adsb)
        if adsb:
            run.log(f"adsb: {X['adsb_taxi'].notna().mean():.1%} dos voos com evento")
    with run.phase("treino", 0.6):
        pred = oof_correction(X, hold[TRUTH].to_numpy(float), base["pred"].to_numpy(float),
                              day_folds(base["dia"].to_numpy()))
    return hold, base, pred


def simulacao_crossfit(
    run: Run, base_id: str, cfg_base: dict, adsb: bool
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Corretor treinado no ano fora do bloco; devolve (holdout, base, previsão)."""
    with run.phase("dados", 0.1):
        base = pd.read_parquet(RUNS / f"{base_id}.parquet")
        hold = holdout_da_base(base)
        train, blind = load_split("train2025"), load_split("blind2025")
        rk = load_split("ranking2026")
        run.log(f"treino {len(train):,} · cegas {len(blind):,} · holdout {len(hold):,}")
    with run.phase("base fora do bloco", 0.7):
        oof = oof_base(cfg_base, train, blind, rk, run)
        del train, rk
        caminho_oof = RUNS / f"{run.id}_oof.parquet"
        oof.to_parquet(caminho_oof, index=False)
        run.set(oof=str(caminho_oof.relative_to(ROOT)))
        run.log(f"fora do bloco: {len(oof):,} previsões · rmse {rmse(oof[TRUTH], oof['pred']):.2f}")
    with run.phase("corretor", 0.1):
        pred_oof = oof["pred"].to_numpy(float)
        cegas = blind.set_index(F.ID).loc[oof[F.ID]].reset_index()  # mesma ordem do oof
        del blind
        X_oof = corrector_frame(cegas, pred_oof, adsb)
        del cegas
        if adsb:
            run.log(f"adsb no treino do corretor: {X_oof['adsb_taxi'].notna().mean():.1%}")
        model = fit_corrector(X_oof, oof[TRUTH].to_numpy(float), pred_oof)
        del X_oof
        pred_base = base["pred"].to_numpy(float)
        pred = apply_corrector(model, corrector_frame(hold, pred_base, adsb), pred_base)
    return hold, base, pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("nome")
    ap.add_argument("--base", help="id da corrida base (padrão: campeã)")
    ap.add_argument("--crossfit", action="store_true",
                    help="corretor treinado no ano, nas previsões da base fora do bloco")
    ap.add_argument("--sem-adsb", action="store_true", help="controle: mesmo empilhamento sem adsb_*")
    ap.add_argument("--seeds", type=int, default=1,
                    help="--crossfit: base de cada bloco é a média de N seeds")
    ap.add_argument("--nota", default="")
    a = ap.parse_args()
    if a.seeds > 1 and not a.crossfit:
        ap.error("--seeds só vale com --crossfit (a base do holdout vem pronta em --base)")
    base_id = a.base or json.loads((ROOT / "champion.json").read_text())["id"]
    adsb = not a.sem_adsb
    if a.crossfit:
        cfg = {"model": "stack_cf", "base": base_id,
               "base_config": config_da_base(base_id, a.seeds),
               "adsb": adsb, "rounds": ROUNDS, "seed": PARAMS["seed"]}
    else:
        cfg = {"model": "stack", "base": base_id, "adsb": adsb, "folds": FOLDS,
               "rounds": ROUNDS, "seed": PARAMS["seed"]}

    with Run(a.nome, cfg) as run:
        run.set(nota=a.nota)
        if a.crossfit:
            hold, base, pred = simulacao_crossfit(run, base_id, cfg["base_config"], adsb)
        else:
            hold, base, pred = simulacao_folds(run, base_id, adsb)
        with run.phase("métricas", 0.1):
            m = metrics(hold, pred)
            path = RUNS / f"{run.id}.parquet"
            pd.DataFrame({
                F.ID: base[F.ID].to_numpy(),
                "dia": base["dia"].to_numpy(),
                TRUTH: hold[TRUTH].to_numpy(float),
                "pred": pred,
            }).to_parquet(path, index=False)
        run.metric(**m)
        run.set(previsoes=str(path.relative_to(ROOT)))
        run.log(" · ".join(f"{k} {v}" for k, v in m.items() if not isinstance(v, dict)))


if __name__ == "__main__":
    main()
