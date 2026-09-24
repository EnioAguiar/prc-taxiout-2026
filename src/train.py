"""Versão final para envio, a partir do campeão (champion.json).

    bin/run src/train.py submit N    # gera submissions/<TEAM>_vN.parquet (NÃO envia)

Envio separado, só depois de aprovado: .venv/bin/python src/s3.py submit <arquivo>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import features as F
from cache import DATA, load_split
from compare import CHAMPION
from models import MODELS, prepare
from runlog import ROOT, Run

OUT = ROOT / "submissions"
ROUNDS_SCALE = 1.2  # full2025 tem 2,085 M linhas contra 1,741 M do train2025


def leaky_columns(train: pd.DataFrame, ranking: pd.DataFrame, cols: list[str]) -> list[str]:
    """Colunas preenchidas no treino mas apagadas no ranking: o modelo não pode usá-las."""
    return [
        c for c in cols
        if c in train and c in ranking
        and train[c].isna().mean() < 0.5 and ranking[c].isna().mean() > 0.95
    ]


def final_config(champ: dict) -> dict:
    cfg = dict(champ["config"])
    if champ.get("best_iter"):
        cfg["rounds"] = champ["best_iter"]
    for key in ("rounds", "cls_rounds", "reg_rounds"):
        if key in cfg:
            cfg[key] = round(cfg[key] * ROUNDS_SCALE)
    return cfg


def build_submission(template: pd.DataFrame, ids, pred) -> pd.DataFrame:
    by_id = pd.Series(np.asarray(pred, float), index=np.asarray(ids))
    out = template[[F.ID]].copy()
    out[F.TARGET] = out[F.ID].map(by_id)
    missing = out[F.TARGET].isna()
    if missing.any():
        # Decolagem sem MVT_TIME no ranking não gera features: usa a mediana das previsões.
        out.loc[missing, F.TARGET] = float(np.median(pred))
    return out


def submit(version: int) -> None:
    load_dotenv(ROOT / ".env")
    team = os.environ.get("TEAM_NAME") or sys.exit("Falta TEAM_NAME no .env")
    champ = json.loads(CHAMPION.read_text())
    cfg = final_config(champ)

    with Run(f"submit_v{version}", {**cfg, "campeao": champ["id"]}) as run:
        with run.phase("dados", 0.15):
            full, rk = load_split("full2025"), load_split("ranking2026")
            template = pd.read_parquet(DATA / "submitting.parquet")
            cols = prepare(full, [rk])
            drop = leaky_columns(full, rk, cols)
            cols = [c for c in cols if c not in drop]
            run.log(f"treino {len(full):,} · ranking {len(rk):,} · ignoradas: {drop or 'nenhuma'}")
        with run.phase("treino", 0.75):
            model = MODELS[cfg["model"]](cfg).fit(full, cols, run=run)
        with run.phase("arquivo", 0.10):
            out = build_submission(template, rk[F.ID], model.predict(rk))
            OUT.mkdir(exist_ok=True)
            path = OUT / f"{team}_v{version}.parquet"
            out.to_parquet(path, index=False)
            run.set(arquivo=str(path.relative_to(ROOT)))
            run.log(
                f"gerado {path.name}: {len(out):,} linhas, mediana {out[F.TARGET].median():.0f} s. "
                "NÃO enviado."
            )


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["submit", n] if n.isdigit():
            submit(int(n))
        case _:
            sys.exit(__doc__)
