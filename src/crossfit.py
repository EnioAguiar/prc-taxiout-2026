"""Previsões da base fora do bloco, por meses (plano 5, tarefa 2).

Os meses presentes no treino viram blocos de dois meses consecutivos. Para cada
bloco, um modelo da base treina nas linhas normais dos *outros* meses e prevê as
linhas cegas (`blind2025`, DEP com BLOCK e alvo apagados) dos meses do bloco —
P10 e vocabulário de categorias refeitos só com o treino daquele bloco.

    oof = oof_base(cfg_da_base, train2025, blind2025, ranking2026, run)
"""

from __future__ import annotations

import pandas as pd

import features as F
from cache import TRUTH
from models import MODELS, leaky_columns, prepare

TIME = "MVT_TIME_UTC_mvt"


def month_blocks(months) -> dict[int, int]:
    """Mês → índice do bloco: meses únicos ordenados, agrupados 2 a 2."""
    uniq = sorted({int(m) for m in months})
    return {m: i // 2 for i, m in enumerate(uniq)}


def oof_base(
    cfg: dict, train: pd.DataFrame, blind: pd.DataFrame, ranking_cols_ref: pd.DataFrame,
    run=None,
) -> pd.DataFrame:
    """Uma previsão por DEP cega, sempre de um modelo que não viu o mês dela."""
    train_mes = train[TIME].dt.month
    blind_mes = blind[TIME].dt.month
    blocos: dict[int, list[int]] = {}
    for mes, k in month_blocks(train_mes).items():
        blocos.setdefault(k, []).append(mes)

    partes = []
    for k, meses in sorted(blocos.items()):
        if run:
            run.log(f"bloco {k + 1}/{len(blocos)} meses {meses}")
        tr = train[~train_mes.isin(meses)].copy()  # prepare muta os frames: cópias por bloco
        te = blind[blind_mes.isin(meses) & blind[TRUTH].notna()].copy()
        cols = prepare(tr, [te])
        drop = leaky_columns(tr, ranking_cols_ref, cols)
        cols = [c for c in cols if c not in drop]
        model = MODELS[cfg["model"]](cfg).fit(tr, cols, run=run)
        partes.append(pd.DataFrame({
            F.ID: te[F.ID].to_numpy(),
            "dia": te[TIME].dt.strftime("%Y-%m-%d").to_numpy(),
            "mes": te[TIME].dt.month.to_numpy(),
            TRUTH: te[TRUTH].to_numpy(),
            "pred": model.predict(te),
        }))
        del tr, te, model  # RSS: um bloco de cada vez

    out = pd.concat(partes, ignore_index=True)
    return out.sort_values(F.ID, ignore_index=True)
