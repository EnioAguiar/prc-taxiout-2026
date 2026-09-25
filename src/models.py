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
    deterministic=True,  # mesma seed → mesmo modelo, mesmo com 12 threads
    force_row_wise=True,
    verbose=-1,
)


def params_for(cfg: dict) -> dict:
    return {**PARAMS, "seed": int(cfg.get("seed", 0))}


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
        self.params = params_for(cfg)

    def fit(self, train, cols, run=None, valid=None) -> "SingleLGBM":
        self.cols = cols
        data = lgb.Dataset(train[cols], train[F.TARGET])
        callbacks = [run.lgb_callback(self.rounds)] if run else []
        valid_sets, curve = None, {}
        if valid is not None:
            valid_sets = [lgb.Dataset(valid[cols], valid[TRUTH], reference=data)]
            callbacks.append(lgb.record_evaluation(curve))
        self.model = lgb.train(
            self.params, data, self.rounds, valid_sets=valid_sets,
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
        self.params = params_for(cfg)

    def fit(self, train, cols, run=None, valid=None) -> "TwoStage":
        self.cols = cols
        copied = copied_from_sched(train)
        cls_params = {**self.params, "objective": "binary", "metric": "binary_logloss"}

        def cb(rounds: int, label: str, start: float) -> list:
            return [run.lgb_callback(rounds, label, start=start, span=0.5)] if run else []

        self.cls = lgb.train(
            cls_params, lgb.Dataset(train[cols], copied.astype("int8")), self.cls_rounds,
            callbacks=cb(self.cls_rounds, "classificador", 0.0),
        )
        normal = train[~copied]
        self.reg = lgb.train(
            self.params, lgb.Dataset(normal[cols], normal[F.TARGET]), self.reg_rounds,
            callbacks=cb(self.reg_rounds, "regressor", 0.5),
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.cols]
        return combine(self.cls.predict(x), df[SCHED_GAP], self.reg.predict(x))


NM_MIN_ROWS = 50  # grupos menores usam a reta global
NM_SPLIT_S = 7200  # célula de Roma: atraso > 2 h


def nm_groups(df: pd.DataFrame, split_ms: bool) -> np.ndarray:
    """Chave da reta: aeroporto, ou aeroporto × (atraso > 2 h)."""
    keys = df[F.AIRPORT].astype(str).to_numpy()
    if not split_ms:
        return keys
    late = np.where(df[SCHED_GAP].to_numpy(float) > NM_SPLIT_S, ">2h", "<=2h")
    return np.char.add(np.char.add(keys.astype(str), "|"), late)


def _line(ms: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    b, a = np.polyfit(ms, y, 1)
    return float(a), float(b)


def fit_lines(ms, y, keys, min_rows: int = NM_MIN_ROWS):
    """Mínimos quadrados y = a + b·ms por grupo (L2, alinhado ao RMSE) e reta global."""
    ms, y, keys = np.asarray(ms, float), np.asarray(y, float), np.asarray(keys)
    ok = ~np.isnan(ms) & ~np.isnan(y)
    ms, y, keys = ms[ok], y[ok], keys[ok]
    lines = {
        str(k): _line(ms[keys == k], y[keys == k])
        for k in np.unique(keys)
        if (keys == k).sum() >= min_rows
    }
    return lines, _line(ms, y)


def apply_lines(ms, keys, lines: dict, fallback: tuple[float, float]) -> np.ndarray:
    ms, keys = np.asarray(ms, float), np.asarray(keys)
    a = np.array([lines.get(str(k), fallback)[0] for k in keys])
    b = np.array([lines.get(str(k), fallback)[1] for k in keys])
    return np.clip(a + b * ms, 0, None)  # NaN em ms continua NaN


def line_rows(nm, ms, line_pred, min_ms: float) -> np.ndarray:
    """Voos em que a reta substitui o dois estágios: sem NM, com reta e atraso > limiar."""
    nm, ms, line_pred = np.asarray(nm, bool), np.asarray(ms, float), np.asarray(line_pred, float)
    over = np.nan_to_num(ms, nan=-np.inf) > min_ms if min_ms > 0 else np.ones(ms.size, bool)
    return nm & ~np.isnan(line_pred) & over


class TwoStageNM(TwoStage):
    """Dois estágios para voos com NM; reta por aeroporto no atraso para voos sem NM."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.split_ms = bool(cfg.get("nm_split_ms", False))
        self.min_ms = float(cfg.get("nm_min_ms", 0))

    def fit(self, train, cols, run=None, valid=None) -> "TwoStageNM":
        super().fit(train, cols, run=run, valid=valid)
        nm = train[train["nm_missing"] == 1]
        self.lines, self.fallback = fit_lines(
            nm[SCHED_GAP], nm[F.TARGET], nm_groups(nm, self.split_ms)
        )
        if run:
            run.log(
                f"retas NM ausente: {len(self.lines)} grupos"
                f" · global a={self.fallback[0]:.0f} b={self.fallback[1]:.3f}"
            )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = super().predict(df)
        nm = (df["nm_missing"] == 1).to_numpy()
        lines = apply_lines(df[SCHED_GAP], nm_groups(df, self.split_ms), self.lines, self.fallback)
        use = line_rows(nm, df[SCHED_GAP], lines, self.min_ms)
        pred[use] = lines[use]
        return pred


MODELS = {"single": SingleLGBM, "two_stage": TwoStage, "two_stage_nm": TwoStageNM}
