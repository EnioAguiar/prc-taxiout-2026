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

Quando a corrida base usa `janela_lobt`, o corretor de `--crossfit` ganha `dist_lo` e
`dist_hi` (folga até as bordas da janela do LOBT, NaN sem LOBT) e sua saída é projetada
na janela antes do piso 0; sem `janela_lobt` na base, tudo segue como antes.

Uso:
    bin/run src/stack.py <nome> [--base <id>] [--crossfit [--seeds N] [--conjunto]] [--sem-adsb]
                                [--sem-feature COLUNA] [--externos] [--plano13]

`--sem-feature COLUNA` (pode repetir, só com `--crossfit`) tira a coluna da base e do
corretor e grava `sem_features` na config da corrida e na `base_config`; sem a flag, nada
muda. Nos folds a base vem pronta e ninguém confere o nome, então a flag é recusada.

`--conjunto` (só com `--crossfit`) troca o corretor único pela média de três treinados
nas mesmas entradas: LightGBM global, um LightGBM por aeroporto (aeroporto sem modelo
próprio usa o global) e CatBoost; grava `corretor: "conjunto"` na config, que o
`train.py` lê no envio.

Com `--crossfit`, `--seeds N` (N > 1) manda a base de cada bloco ser a média de N seeds:
sobrescreve `seeds` na config da corrida base. A base do holdout vem pronta de `--base`,
que já deve ser a corrida de N seeds.

`--externos` (só com `--crossfit`) soma ao corretor as colunas `ext_*` de
`src/externos.py` (taxa de cópia do SCHED por companhia, séries diárias da EUROCONTROL e
tempo em solo do OPDI) e grava `externos: true` na config. A taxa de cópia é aprendida só
nos meses do oof (os 10 do treino) e nunca no mês da própria linha.

`--plano13` (só com `--crossfit`) soma ao corretor os quatro sinais de `src/plano13.py`
(METAR do aeroporto, rotação no stand, consistência do plano NM e a companhia como
categoria) e grava `plano13: true` na config. O vocabulário de companhias é fixado nas
linhas de treino e vale igual para as cegas, o holdout e o ranking.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

import contexto
import features as F
from adsb_events import FEATURES as ADSB
from cache import TRUTH, load_split
from crossfit import oof_base
from externos import colunas_ext, copia_cia_2025
from experiment import RUNS, metrics, rmse
from models import janela_lobt, limitar_janela
from plano13 import colunas_p13, vocabulario
from runlog import REGISTRY, ROOT, Run

FOLDS = 5
ROUNDS = 300
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              verbose=-1, num_threads=12, seed=0, deterministic=True, force_row_wise=True)
PARAMS_AEROPORTO = {**PARAMS, "min_data_in_leaf": 100}  # menos dados por modelo
CATBOOST = dict(iterations=800, depth=8, learning_rate=0.08, loss_function="RMSE", random_seed=0)


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


def corrector_frame(df: pd.DataFrame, pred: np.ndarray, adsb: bool = True,
                    janela: bool = False, sem: Iterable[str] = (),
                    externos: dict | None = None,
                    plano13: pd.DataFrame | None = None) -> pd.DataFrame:
    """Entradas do corretor: a previsão da base, o contexto do voo e o rastro ADS-B.

    `sem` tira as colunas com esses nomes (nome que não está no quadro é ignorado: a
    lista é a mesma da base, que tem colunas que o corretor não usa).

    `externos` (`src/externos.py`, `--externos`) são colunas `ext_*` prontas, uma por
    linha e na mesma ordem de `df`; sem elas o quadro é o de sempre.

    `plano13` (`src/plano13.py`, `--plano13`) é o quadro com as colunas `met_*`, `rot_*`,
    `nm_*` e `cia`, na mesma ordem de `df`; `cia` entra categórica.
    """
    cols = [F.AIRPORT, "nm_missing", "hour", *[c for c in df if c.startswith("to_takeoff_from_")],
            *[c for c in contexto.COLS if c in df]]
    X = df[cols].copy()
    X[F.AIRPORT] = X[F.AIRPORT].astype("category")
    X["pred"] = np.asarray(pred, float)
    if janela:
        lo, hi = janela_lobt(df)
        X["dist_lo"] = X["pred"].to_numpy(float) - lo  # folga até o fundo da janela do LOBT
        X["dist_hi"] = hi - X["pred"].to_numpy(float)
    if adsb:
        X = X.join(df[ADSB])
        X["adsb_menos_pred"] = X["adsb_taxi_move"] - X["pred"]
    for nome, valores in (externos or {}).items():
        X[nome] = np.asarray(valores, float)
    if plano13 is not None:
        for nome in plano13.columns:
            X[nome] = plano13[nome].array  # posicional; `cia` continua categórica
    return X.drop(columns=[c for c in sem if c in X.columns])


class Conjunto:
    """Média simples de três corretores sobre as mesmas entradas.

    (1) LightGBM global; (2) um LightGBM por aeroporto (aeroporto sem modelo próprio cai
    no global); (3) CatBoost com as categóricas em texto.
    """

    def __init__(self, global_, aeroportos: dict, catboost) -> None:
        self.global_, self.aeroportos, self.catboost = global_, aeroportos, catboost

    def _por_aeroporto(self, X: pd.DataFrame) -> np.ndarray:
        out = np.asarray(self.global_.predict(X), float)  # aeroporto novo usa o global
        aero = X[F.AIRPORT].astype(str).to_numpy()
        for nome, modelo in self.aeroportos.items():
            sel = aero == nome
            if sel.any():
                out[sel] = modelo.predict(X[sel])
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        glob = np.asarray(self.global_.predict(X), float)
        cat = np.asarray(self.catboost.predict(catboost_frame(X)), float)
        return (glob + self._por_aeroporto(X) + cat) / 3


def colunas_cat(X: pd.DataFrame) -> list[str]:
    """Colunas categóricas do quadro: o aeroporto e, com `--plano13`, a companhia."""
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def catboost_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Entradas do CatBoost: as mesmas do LightGBM, com as categóricas em texto."""
    Xc = X.copy()
    for col in colunas_cat(X):
        Xc[col] = Xc[col].astype(str)
    return Xc


def fit_catboost(X: pd.DataFrame, alvo: np.ndarray):
    """CatBoost determinístico na correção; GPU quando há placa, senão 12 threads."""
    from catboost import CatBoostRegressor, utils

    params = dict(CATBOOST, verbose=0)
    if utils.get_gpu_device_count() > 0:
        params["task_type"] = "GPU"
    else:
        params["thread_count"] = 12
    modelo = CatBoostRegressor(**params)
    modelo.fit(catboost_frame(X), alvo, cat_features=colunas_cat(X))
    return modelo


def fit_corrector(X: pd.DataFrame, y: np.ndarray, base: np.ndarray,
                  conjunto: bool = False) -> lgb.Booster | Conjunto:
    """Corretor L2 na diferença entre o alvo e a previsão da base.

    Com `conjunto`, devolve a média de três corretores sobre as mesmas entradas.
    """
    alvo = np.asarray(y, float) - np.asarray(base, float)
    global_ = lgb.train(PARAMS, lgb.Dataset(X, alvo), ROUNDS)
    if not conjunto:
        return global_
    aeroportos = {}
    aero = X[F.AIRPORT].astype(str).to_numpy()
    for nome in np.unique(aero):
        sel = aero == nome
        aeroportos[nome] = lgb.train(PARAMS_AEROPORTO, lgb.Dataset(X[sel], alvo[sel]), ROUNDS)
    return Conjunto(global_, aeroportos, fit_catboost(X, alvo))


def apply_corrector(model: lgb.Booster | Conjunto, X: pd.DataFrame, base: np.ndarray,
                    df: pd.DataFrame | None = None) -> np.ndarray:
    """Base mais a correção, com piso 0; com `df`, projetada antes na janela do LOBT."""
    pred = np.asarray(base, float) + model.predict(X)
    return limitar_janela(pred, df) if df is not None else np.clip(pred, 0, None)


def previsao_corrigida(model: lgb.Booster | Conjunto, df: pd.DataFrame, base: np.ndarray,
                       adsb: bool, janela: bool, sem: Iterable[str] = (),
                       externos: dict | None = None,
                       plano13: pd.DataFrame | None = None) -> np.ndarray:
    """Previsão dos voos de `df` corrigida: com `janela`, dentro da janela do LOBT."""
    X = corrector_frame(df, base, adsb, janela, sem, externos, plano13)
    return apply_corrector(model, X, base, df if janela else None)


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
    run: Run, base_id: str, adsb: bool, sem: Iterable[str] = ()
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Corretor em 5 folds por dia dentro do holdout; devolve (holdout, base, previsão)."""
    with run.phase("dados", 0.3):
        base = pd.read_parquet(RUNS / f"{base_id}.parquet")
        hold = holdout_da_base(base)
        X = corrector_frame(hold, base["pred"].to_numpy(float), adsb, sem=sem)
        if adsb:
            run.log(f"adsb: {X['adsb_taxi'].notna().mean():.1%} dos voos com evento")
    with run.phase("treino", 0.6):
        pred = oof_correction(X, hold[TRUTH].to_numpy(float), base["pred"].to_numpy(float),
                              day_folds(base["dia"].to_numpy()))
    return hold, base, pred


def simulacao_crossfit(
    run: Run, base_id: str, cfg_base: dict, adsb: bool, conjunto: bool = False,
    sem: Iterable[str] = (), externos: bool = False, plano13: bool = False,
    fila: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Corretor treinado no ano fora do bloco; devolve (holdout, base, previsão)."""
    janela = bool(cfg_base.get("janela_lobt"))
    with run.phase("dados", 0.1):
        base = pd.read_parquet(RUNS / f"{base_id}.parquet")
        hold = holdout_da_base(base)
        train, blind = load_split("train2025"), load_split("blind2025")
        rk = load_split("ranking2026")
        cias = vocabulario(train) if plano13 else None  # vocabulário fixo do treino
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
        ext_cegas = ext_hold = None
        if externos:
            # meses do oof: os 10 do treino, nunca jan/jul — nem as cegas nem o holdout
            # veem a taxa de cópia do próprio mês.
            meses = sorted({int(m) for m in oof["mes"]})
            copia = copia_cia_2025(run)
            ext_cegas, ext_hold = (colunas_ext(d, copia, meses) for d in (cegas, hold))
            run.log(f"externos: meses de treino da taxa de cópia {meses}")
        p13_cegas = p13_hold = None
        if plano13:
            p13_cegas, p13_hold = (colunas_p13(d, cias, com_fila=fila) for d in (cegas, hold))
            run.log(f"plano 13: {len(cias)} companhias no vocabulário · rotação em "
                    f"{p13_cegas['rot_idade'].notna().mean():.1%} das cegas")
        X_oof = corrector_frame(cegas, pred_oof, adsb, janela, sem, ext_cegas, p13_cegas)
        del cegas
        if adsb:
            run.log(f"adsb no treino do corretor: {X_oof['adsb_taxi'].notna().mean():.1%}")
        model = fit_corrector(X_oof, oof[TRUTH].to_numpy(float), pred_oof, conjunto)
        del X_oof
        pred_base = base["pred"].to_numpy(float)
        pred = previsao_corrigida(model, hold, pred_base, adsb, janela, sem, ext_hold, p13_hold)
    return hold, base, pred


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("nome")
    ap.add_argument("--base", help="id da corrida base (padrão: campeã)")
    ap.add_argument("--crossfit", action="store_true",
                    help="corretor treinado no ano, nas previsões da base fora do bloco")
    ap.add_argument("--sem-adsb", action="store_true", help="controle: mesmo empilhamento sem adsb_*")
    ap.add_argument("--seeds", type=int, default=1,
                    help="--crossfit: base de cada bloco é a média de N seeds")
    ap.add_argument("--conjunto", action="store_true",
                    help="--crossfit: corretor = média de global, por aeroporto e CatBoost")
    ap.add_argument("--sem-feature", action="append", default=[], metavar="COLUNA",
                    help="tira a coluna da base e do corretor (pode repetir)")
    ap.add_argument("--externos", action="store_true",
                    help="--crossfit: soma as colunas ext_* (companhia, séries diárias, OPDI)")
    ap.add_argument("--plano13", action="store_true",
                    help="--crossfit: soma METAR, rotação no stand, consistência NM e a companhia")
    ap.add_argument("--fila", action="store_true",
                    help="--plano13: soma contagens de fila, pista e pátio do stand (diagnóstico de Roma)")
    ap.add_argument("--nota", default="")
    return ap


def config_da_corrida(a: argparse.Namespace, base_id: str) -> dict:
    """Config gravada no registro: com `--conjunto`, a chave `corretor`.

    Com `--sem-feature`, `sem_features` no topo (corretor) e na `base_config` (blocos);
    com `--externos`, `externos: true`; com `--plano13`, `plano13: true`.
    """
    adsb = not a.sem_adsb
    sem = list(a.sem_feature)
    if not a.crossfit:
        cfg = {"model": "stack", "base": base_id, "adsb": adsb, "folds": FOLDS,
               "rounds": ROUNDS, "seed": PARAMS["seed"]}
        if sem:
            cfg["sem_features"] = sem
        return cfg
    cfg_base = config_da_base(base_id, a.seeds)
    if sem:
        cfg_base = {**cfg_base, "sem_features": sem}
    cfg = {"model": "stack_cf", "base": base_id,
           "base_config": cfg_base,
           "adsb": adsb, "rounds": ROUNDS, "seed": PARAMS["seed"]}
    if a.conjunto:
        cfg["corretor"] = "conjunto"
    if a.externos:
        cfg["externos"] = True
    if a.plano13:
        cfg["plano13"] = True
    if a.fila:
        cfg["fila"] = True
    if sem:
        cfg["sem_features"] = sem
    return cfg


def main() -> None:
    ap = parser()
    a = ap.parse_args()
    if a.seeds > 1 and not a.crossfit:
        ap.error("--seeds só vale com --crossfit (a base do holdout vem pronta em --base)")
    if a.conjunto and not a.crossfit:
        ap.error("--conjunto só vale com --crossfit")
    if a.sem_feature and not a.crossfit:
        ap.error("--sem-feature só vale com --crossfit (nos folds nada confere o nome)")
    if a.externos and not a.crossfit:
        ap.error("--externos só vale com --crossfit (os folds não têm meses de treino separados)")
    if a.fila and not a.plano13:
        ap.error("--fila só vale com --plano13")
    if a.plano13 and not a.crossfit:
        ap.error("--plano13 só vale com --crossfit (os folds não têm vocabulário de treino)")
    base_id = a.base or json.loads((ROOT / "champion.json").read_text())["id"]
    adsb = not a.sem_adsb
    cfg = config_da_corrida(a, base_id)

    with Run(a.nome, cfg) as run:
        run.set(nota=a.nota)
        sem = cfg.get("sem_features", ())
        if a.crossfit:
            hold, base, pred = simulacao_crossfit(run, base_id, cfg["base_config"], adsb,
                                                  a.conjunto, sem, a.externos, a.plano13,
                                                  a.fila)
        else:
            hold, base, pred = simulacao_folds(run, base_id, adsb, sem)
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
