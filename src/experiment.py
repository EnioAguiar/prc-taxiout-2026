"""Experimento na simulação calibrada do ranking.

    bin/run src/experiment.py <nome> --model single [--rounds 400] [--nota "texto"]
    bin/run src/experiment.py <nome> --model two_stage [--cls-rounds 400] [--reg-rounds 400]
    bin/run src/experiment.py <nome> --model two_stage_nm [--nm-split-ms] [--nm-min-ms S]

Qualquer modelo aceita `--seeds N`: a previsão passa a ser a média de N cópias treinadas
com as seeds `--seed`, `--seed`+1, …; N = 1 (padrão) é o comportamento de sempre.

Qualquer modelo aceita `--sem-feature COLUNA` (pode repetir): tira a coluna da base e
grava `sem_features` na config; sem a flag, nada muda.

Treina em train2025 (10 meses), prevê holdout2025 (jan+jul/2025 montados como o
ranking) e registra RMSE completo, sem outliers, por grupo, por fatia de erro
(voos normais com NM, alarmes falsos, cauda que é cópia, sem loteria) e por
aeroporto. Previsões em runs/<id>.parquet; resumo em experiments.jsonl.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import features as F
import memoria
from cache import TRUTH, load_split
from models import MODELS, build_model, leaky_columns, prepare
from runlog import ROOT, Run

RUNS = ROOT / "runs"
MAX_NORMAL_S = 3 * 3600
TAIL_S = 3600  # acima de 1 h o erro deixa de ser "voo normal"
NEAR_PLAN_S = 300  # ±5 min de um horário planejado = o alvo é cópia dele
PLAN_GAPS = [f"to_takeoff_from_{c}" for c in F.PLAN_REFS]


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2))) if y.size else float("nan")


def near_plan(df: pd.DataFrame, y) -> np.ndarray:
    """y a ≤ 5 min de algum MVT − horário planejado (horário nulo nunca está perto)."""
    gaps = df.reindex(columns=PLAN_GAPS).to_numpy(float)
    return (np.abs(np.asarray(y, float)[:, None] - gaps) <= NEAR_PLAN_S).any(1)


def lottery_mask(df: pd.DataFrame, y) -> np.ndarray:
    """Voos "loteria": y > 3 h que nenhum horário planejado explica.

    São 11 no holdout da campeã, 32 % do erro²: imprevisíveis e donos do ruído
    entre seeds. As métricas "sem loteria" medem o resto, que dá para melhorar.
    """
    return (np.asarray(y, float) > MAX_NORMAL_S) & ~near_plan(df, y)


def metrics(df: pd.DataFrame, pred: np.ndarray) -> dict:
    y = df[TRUTH].to_numpy(float)
    pred = np.asarray(pred, float)
    nm = df["FLIGHT_ID_mvt"].isna().to_numpy()
    normal = (y > 0) & (y < MAX_NORMAL_S)
    tail = y > TAIL_S
    near = near_plan(df, y)
    normais_nm = ~tail & ~nm
    falso = ~tail & (pred > TAIL_S)
    keep = ~lottery_mask(df, y)
    m = {
        "completo": rmse(y, pred),
        "sem_outliers": rmse(y[normal], pred[normal]),
        "nm_presente": rmse(y[~nm], pred[~nm]),
        "nm_ausente": rmse(y[nm], pred[nm]),
        "y_gt_1h": rmse(y[tail], pred[tail]),
        "normais_nm": rmse(y[normais_nm], pred[normais_nm]),
        "cauda_copia": rmse(y[tail & near], pred[tail & near]),
        "sem_loteria": rmse(y[keep], pred[keep]),
    }
    m = {k: round(v, 2) for k, v in m.items()}
    err2 = (y - pred) ** 2
    m["alarmes_falsos"] = {
        "n": int(falso.sum()),
        "parte_erro2": round(float(err2[falso].sum() / err2.sum()), 4),
    }
    groups = df.groupby(F.AIRPORT, observed=True).indices
    m["por_aeroporto"] = {str(a): round(rmse(y[i], pred[i]), 1) for a, i in groups.items()}
    return m


def config(a: argparse.Namespace) -> dict:
    if a.model == "single":
        cfg = {"model": a.model, "rounds": a.rounds, "seed": a.seed}
    else:
        cfg = {
            "model": a.model,
            "cls_rounds": a.cls_rounds,
            "reg_rounds": a.reg_rounds,
            "seed": a.seed,
        }
        if a.model == "two_stage_nm":
            cfg["nm_split_ms"] = a.nm_split_ms
            cfg["nm_min_ms"] = a.nm_min_ms
    if a.seeds > 1:  # config de uma seed continua idêntica às antigas
        cfg["seeds"] = a.seeds
    if a.janela_lobt:  # config sem a janela continua idêntica às antigas
        cfg["janela_lobt"] = True
    if a.reg_corte:  # config sem o corte continua idêntica às antigas
        cfg["reg_corte"] = a.reg_corte
    if a.reg_sem_lirf_nm:  # idem
        cfg["reg_sem_lirf_nm"] = True
    if a.sem_feature:  # config sem a flag continua idêntica às antigas
        cfg["sem_features"] = list(a.sem_feature)
    if a.base_ctx:  # idem
        cfg["base_ctx"] = True
    if a.base_p13:  # idem
        cfg["base_p13"] = True
    if a.base_mapa:  # idem
        cfg["base_mapa"] = True
    if a.base_por_apt:  # idem
        cfg["base_por_apt"] = True
    if a.cls_peso:  # idem
        cfg["cls_peso"] = a.cls_peso
    if a.cat_max:  # idem
        cfg["cat_max"] = a.cat_max
    if a.motor != "lgb":  # idem
        cfg["motor"] = a.motor
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
    ap.add_argument("--nm-min-ms", type=float, default=0.0,
                    help="two_stage_nm: só usa a reta com atraso acima de S segundos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1,
                    help="média das previsões de N cópias, com seeds seed..seed+N−1")
    ap.add_argument("--janela-lobt", action="store_true",
                    help="prende a previsão em MVT − LOBT ± 3606 s e zera p fora da janela")
    ap.add_argument("--sem-feature", action="append", default=[], metavar="COLUNA",
                    help="tira a coluna da base (pode repetir); nome inexistente é erro")
    ap.add_argument("--reg-corte", type=float, default=0.0, metavar="S",
                    help="corta o alvo do regressor em S segundos (só no treino)")
    ap.add_argument("--reg-sem-lirf-nm", action="store_true",
                    help="tira do treino do regressor as linhas de LIRF sem NM")
    ap.add_argument("--base-ctx", action="store_true",
                    help="a base também usa as colunas ctx_* (contexto de chegadas, stand, vizinhos)")
    ap.add_argument("--base-p13", action="store_true",
                    help="a base também usa as colunas do plano 13 (METAR, rotação, consistência NM, cia)")
    ap.add_argument("--base-mapa", action="store_true",
                    help="a base também usa as colunas map_* (distância stand → cabeceira, src/mapa.py)")
    ap.add_argument("--base-por-apt", action="store_true",
                    help="two_stage*: soma ao regressor global um regressor por aeroporto (média)")
    ap.add_argument("--cls-peso", choices=["abs", "quad"],
                    help="two_stage*: pesa o classificador pelo custo de errar p (|MVT−SCHED−900| ou ²)")
    ap.add_argument("--cat-max", type=int, default=0, metavar="N",
                    help="limita cada categórica às N−1 mais frequentes (GPU exige N ≤ 256)")
    ap.add_argument("--motor", choices=["lgb", "xgb"], default="lgb",
                    help="two_stage*: LightGBM (padrão) ou XGBoost na GPU")
    ap.add_argument("--nota", default="")
    a = ap.parse_args()
    if a.model != "two_stage_nm" and (a.nm_split_ms or a.nm_min_ms):
        ap.error("--nm-split-ms e --nm-min-ms só valem com --model two_stage_nm")
    if a.model == "single" and (a.reg_corte or a.reg_sem_lirf_nm):
        ap.error("--reg-corte e --reg-sem-lirf-nm não valem com --model single")
    cfg = config(a)

    with Run(a.nome, cfg) as run:
        run.set(nota=a.nota)
        with run.phase("dados", 0.15):
            train, hold = load_split("train2025"), load_split("holdout2025")
            rk = load_split("ranking2026")
            cols = prepare(train, [hold], cfg.get("sem_features", ()),
                   cfg.get("base_ctx", False), cfg.get("base_p13", False),
                   int(cfg.get("cat_max", 0)), cfg.get("base_mapa", False))
            drop = leaky_columns(train, rk, cols)
            cols = [c for c in cols if c not in drop]
            del rk
            memoria.soltar()  # o ranking sai do heap antes do treino, não só da variável
            run.log(
                f"treino {len(train):,} · holdout {len(hold):,} · {len(cols)} features"
                f" · ignoradas: {drop or 'nenhuma'}"
            )
        with run.phase("treino", 0.75):
            model = build_model(cfg).fit(train, cols, run=run, valid=hold)
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
        af = m["alarmes_falsos"]
        run.log(
            " · ".join(f"{k} {v}" for k, v in m.items() if not isinstance(v, dict))
            + f" · alarmes falsos {af['n']} voos = {100 * af['parte_erro2']:.1f}% do erro²"
        )
        worst = sorted(m["por_aeroporto"].items(), key=lambda kv: -kv[1])
        run.log("por aeroporto: " + ", ".join(f"{k} {v}" for k, v in worst))


if __name__ == "__main__":
    main()
