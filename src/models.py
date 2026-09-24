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


MODELS = {"single": SingleLGBM}
