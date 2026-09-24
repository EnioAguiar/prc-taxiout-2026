"""Modelos com interface única.

    cols = prepare(train, [outros...])
    modelo = MODELS[nome](cfg).fit(train, cols, run=run, valid=holdout)
    pred = modelo.predict(df)
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

import features as F
from cache import TRUTH

PARAMS = dict(
    objective="regression",  # L2 no alvo bruto, alinhado ao RMSE
    metric="rmse",
    learning_rate=0.05,
    num_leaves=255,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    cat_smooth=20,
    max_cat_to_onehot=8,
    num_threads=12,  # 6 núcleos físicos × 2; 24 threads é 2–4× mais lento (benchmark 24/09)
    verbose=-1,
)


def prepare(train: pd.DataFrame, others: list[pd.DataFrame]) -> list[str]:
    """Referência P10 (só do treino) e o mesmo vocabulário de categorias em todos."""
    ref = F.fit_reference(train)
    for df in (train, *others):
        df["ref_p10"] = F.apply_reference(df, ref)
    F.as_categories([train, *others])
    return F.feature_columns(train)


def leaky_columns(train: pd.DataFrame, ranking: pd.DataFrame, cols: list[str]) -> list[str]:
    """Colunas preenchidas no treino mas apagadas no ranking: o modelo não pode usá-las."""
    return [
        c for c in cols
        if c in train and c in ranking
        and train[c].isna().mean() < 0.5 and ranking[c].isna().mean() > 0.95
    ]


class SingleLGBM:
    """Um LightGBM L2 no alvo bruto (modelo da v2)."""

    def __init__(self, cfg: dict) -> None:
        self.rounds = int(cfg.get("rounds", 400))
        self.best_iter: int | None = None

    def fit(self, train, cols, run=None, valid=None) -> "SingleLGBM":
        self.cols = cols
        data = lgb.Dataset(train[cols], train[F.TARGET])
        callbacks = [run.lgb_callback(self.rounds)] if run else []
        valid_sets, curve = None, {}
        if valid is not None:
            valid_sets = [lgb.Dataset(valid[cols], valid[TRUTH], reference=data)]
            callbacks.append(lgb.record_evaluation(curve))
        self.model = lgb.train(
            PARAMS, data, self.rounds, valid_sets=valid_sets,
            valid_names=["holdout"] if valid_sets else None,
            callbacks=callbacks,
        )
        if curve:
            self.best_iter = int(np.argmin(curve["holdout"]["rmse"])) + 1
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.cols])


SCHED_GAP = "to_takeoff_from_SCHED_TIME_UTC_mvt"  # MVT − SCHED, em segundos
COPY_TOL_S = 60


def copied_from_sched(df: pd.DataFrame) -> pd.Series:
    """Rótulo do estágio 1: BLOCK oficial a ≤ 60 s do horário programado."""
    gap = (df["BLOCK_TIME_UTC_mvt"] - df["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    return gap.abs() <= COPY_TOL_S


def combine(p, ms, reg) -> np.ndarray:
    """Esperança da mistura p·(MVT−SCHED) + (1−p)·regressor, com piso 0.

    Sem SCHED (ms nulo) usa só o regressor. Nunca argmax: errar a classe custa horas².
    """
    p, ms, reg = (np.asarray(v, float) for v in (p, ms, reg))
    mix = np.where(np.isnan(ms), reg, p * ms + (1 - p) * reg)
    return np.clip(mix, 0, None)


class TwoStage:
    """Classificador 'BLOCK copiado do SCHED' + regressor L2 nos voos normais."""

    def __init__(self, cfg: dict) -> None:
        self.cls_rounds = int(cfg.get("cls_rounds", 400))
        self.reg_rounds = int(cfg.get("reg_rounds", 400))
        self.best_iter: int | None = None

    def fit(self, train, cols, run=None, valid=None) -> "TwoStage":
        self.cols = cols
        copied = copied_from_sched(train)
        cls_params = {**PARAMS, "objective": "binary", "metric": "binary_logloss"}

        def cb(rounds: int, label: str, start: float) -> list:
            return [run.lgb_callback(rounds, label, start=start, span=0.5)] if run else []

        self.cls = lgb.train(
            cls_params, lgb.Dataset(train[cols], copied.astype("int8")), self.cls_rounds,
            callbacks=cb(self.cls_rounds, "classificador", 0.0),
        )
        normal = train[~copied]
        self.reg = lgb.train(
            PARAMS, lgb.Dataset(normal[cols], normal[F.TARGET]), self.reg_rounds,
            callbacks=cb(self.reg_rounds, "regressor", 0.5),
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.cols]
        return combine(self.cls.predict(x), df[SCHED_GAP], self.reg.predict(x))


MODELS = {"single": SingleLGBM, "two_stage": TwoStage}
