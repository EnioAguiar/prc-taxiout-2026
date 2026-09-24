"""Versão final para envio, a partir do campeão (champion.json).

    bin/run src/train.py submit N [--forcar]   # gera submissions/<TEAM>_vN.parquet (NÃO envia)

Aborta se o código mudou desde a promoção do campeão (src_hash); --forcar ignora.
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
import runlog
from cache import DATA, load_split
from compare import CHAMPION
from models import MODELS, leaky_columns, prepare
from runlog import ROOT, Run

OUT = ROOT / "submissions"
ROUNDS_SCALE = 1.2  # full2025 tem 2,085 M linhas contra 1,741 M do train2025


def check_code(champ: dict, forcar: bool) -> str:
    """O campeão vale para o código que o mediu; mudou, refaça o experimento."""
    atual = runlog.src_hash()
    esperado = champ.get("src_hash")
    if esperado and esperado != atual and not forcar:
        raise SystemExit(
            f"o código mudou desde o campeão {champ['id']} (src_hash {esperado} → {atual}): "
            "refaça o experimento dele e promova de novo, ou repita com --forcar"
        )
    return atual


def final_config(champ: dict) -> dict:
    cfg = dict(champ["config"])
    if champ.get("best_iter"):
        cfg["rounds"] = champ["best_iter"]
    for key in ("rounds", "cls_rounds", "reg_rounds"):
        if key in cfg:
            cfg[key] = round(cfg[key] * ROUNDS_SCALE)
    return cfg


def build_submission(
    template: pd.DataFrame, ids, pred, max_fill_frac: float = 0.001
) -> tuple[pd.DataFrame, int]:
    """Previsões na ordem do template; poucos IDs sem previsão viram a mediana."""
    by_id = pd.Series(np.asarray(pred, float), index=np.asarray(ids))
    out = template[[F.ID]].copy()
    out[F.TARGET] = out[F.ID].map(by_id)
    missing = out[F.TARGET].isna()
    n = int(missing.sum())
    if n > max_fill_frac * len(out):
        raise SystemExit(
            f"{n:,} de {len(out):,} IDs do template ficaram sem previsão "
            f"(limite {max_fill_frac:.1%}): ranking e template não batem, confira os IDs"
        )
    if n:
        # Decolagem sem MVT_TIME no ranking não gera features: usa a mediana das previsões.
        out.loc[missing, F.TARGET] = float(np.median(pred))
    return out, n


def submit(version: int, forcar: bool = False) -> None:
    load_dotenv(ROOT / ".env")
    team = os.environ.get("TEAM_NAME") or sys.exit("Falta TEAM_NAME no .env")
    champ = json.loads(CHAMPION.read_text())
    check_code(champ, forcar)
    cfg = final_config(champ)

    with Run(f"submit_v{version}", {**cfg, "campeao": champ["id"], "forcar": forcar}) as run:
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
            out, preenchidas = build_submission(template, rk[F.ID], model.predict(rk))
            OUT.mkdir(exist_ok=True)
            path = OUT / f"{team}_v{version}.parquet"
            out.to_parquet(path, index=False)
            run.set(arquivo=str(path.relative_to(ROOT)), preenchidas=preenchidas)
            run.log(
                f"gerado {path.name}: {len(out):,} linhas, mediana {out[F.TARGET].median():.0f} s, "
                f"preenchidas com a mediana: {preenchidas}. NÃO enviado."
            )


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["submit", n] if n.isdigit():
            submit(int(n))
        case ["submit", n, "--forcar"] if n.isdigit():
            submit(int(n), forcar=True)
        case _:
            sys.exit(__doc__)
