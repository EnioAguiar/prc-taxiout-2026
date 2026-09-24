"""Treino e submissão.

    python src/train.py validate        # treina sem jan/jul de 2025, valida em jan+jul 2025
    python src/train.py submit N        # treina em 2025 inteiro, gera submissions/<TEAM>_vN.parquet
"""

import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import features as F

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "submissions"

VALID_MONTHS = {1, 7}  # imita o ranking: um mês de inverno e um de pico de verão

PARAMS = dict(
    objective="regression",  # L2, alinhado ao RMSE
    learning_rate=0.05,
    num_leaves=255,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    cat_smooth=20,
    max_cat_to_onehot=8,
    verbose=-1,
)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def training_deps() -> pd.DataFrame:
    paths = sorted(DATA.glob("training_*.parquet"))
    if not paths:
        sys.exit(f"Nenhum training_*.parquet em {DATA}. Rode: python src/s3.py download")
    dep = F.build(F.load(paths))
    # Sem cortar outliers: a verdade oficial inclui taxi-outs de horas (em geral
    # BLOCK_TIME = horário programado), e eles pesam ~40% do erro quadrático.
    return dep[dep[F.TARGET].notna()]


def leaky_columns(train: pd.DataFrame, ranking: pd.DataFrame, cols: list[str]) -> list[str]:
    """Colunas presentes no treino mas apagadas no ranking: o modelo não pode depender delas."""
    return [
        c
        for c in cols
        if c in train and c in ranking
        and train[c].isna().mean() < 0.5 and ranking[c].isna().mean() > 0.95
    ]


def fit_predict(
    train: pd.DataFrame, others: list[pd.DataFrame], drop: list[str], rounds: int
) -> list[np.ndarray]:
    ref = F.fit_reference(train)
    for df in (train, *others):
        df["ref_p10"] = F.apply_reference(df, ref)
    F.as_categories([train, *others])
    cols = [c for c in F.feature_columns(train) if c not in drop]

    def progress(env: lgb.callback.CallbackEnv) -> None:
        done = env.iteration + 1
        if done % 100 == 0 or done == rounds:
            print(f"rodada {done}/{rounds} ({100 * done // rounds}%)", flush=True)

    model = lgb.train(
        PARAMS,
        lgb.Dataset(train[cols], train[F.TARGET]),
        num_boost_round=rounds,
        callbacks=[progress],
    )
    imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("top features:", ", ".join(imp.index[:10]))
    return [model.predict(df[cols]) for df in others]


def validate() -> None:
    dep = training_deps()
    month = dep["MVT_TIME_UTC_mvt"].dt.month
    train, valid = dep[~month.isin(VALID_MONTHS)].copy(), dep[month.isin(VALID_MONTHS)].copy()
    y = valid[F.TARGET].to_numpy()

    median = train.groupby([F.AIRPORT, "STAND_mvt", "RUNWAY_mvt"])[F.TARGET].median()
    base = valid.join(median.rename("m"), on=[F.AIRPORT, "STAND_mvt", "RUNWAY_mvt"])["m"]
    base = base.fillna(valid[F.AIRPORT].map(train.groupby(F.AIRPORT)[F.TARGET].median()))
    print(f"treino {len(train):,}  validação {len(valid):,}")
    print(f"RMSE baseline (mediana aeroporto/stand/pista): {rmse(y, base.to_numpy()):.2f} s")

    ranking = DATA / "ranking.parquet"
    drop = []
    if ranking.exists():
        rk = F.build(F.load([ranking]))
        drop = leaky_columns(dep, rk, F.feature_columns(dep))
        print("ignoradas (vazias no ranking):", drop or "nenhuma")

    (pred,) = fit_predict(train, [valid], drop, rounds=1500)
    print(f"RMSE LightGBM: {rmse(y, pred):.2f} s")
    for apt, g in valid.assign(p=pred).groupby(F.AIRPORT):
        print(f"  {apt}: {rmse(g[F.TARGET].to_numpy(), g['p'].to_numpy()):.1f} s  ({len(g):,})")


def submit(version: int) -> None:
    load_dotenv(ROOT / ".env")
    team = os.environ.get("TEAM_NAME")
    if not team:
        sys.exit("Falta TEAM_NAME no .env")

    dep = training_deps()
    rk = F.build(F.load([DATA / "ranking.parquet"]))
    template = pd.read_parquet(DATA / "submitting.parquet")
    drop = leaky_columns(dep, rk, F.feature_columns(dep))
    print("ignoradas (vazias no ranking):", drop or "nenhuma")

    (pred,) = fit_predict(dep, [rk], drop, rounds=1500)
    by_id = pd.Series(pred, index=rk[F.ID].to_numpy())
    out = template[[F.ID]].copy()
    out[F.TARGET] = out[F.ID].map(by_id)
    missing = out[F.TARGET].isna()
    if missing.any():
        # Decolagem sem MVT_TIME no ranking não gera features: usa a mediana das previsões.
        print(f"aviso: {missing.sum()} linhas sem previsão, preenchidas com a mediana")
        out.loc[missing, F.TARGET] = float(np.median(pred))

    OUT.mkdir(exist_ok=True)
    path = OUT / f"{team}_v{version}.parquet"
    out.to_parquet(path, index=False)
    print(f"gerado {path} ({len(out):,} linhas). Envie com: python src/s3.py submit {path}")


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["validate"]:
            validate()
        case ["submit", n] if n.isdigit():
            submit(int(n))
        case _:
            sys.exit(__doc__)
