"""Features prontas em data/cache/, refeitas só quando features.py ou os dados mudam.

Splits:
    train2025    10 meses de 2025 (sem jan e jul), com alvo
    holdout2025  jan + jul/2025 montados como o ranking (BLOCK e alvo apagados nas
                 DEP antes das features); a verdade fica na coluna y_true
    blind2025    12 meses de 2025 montados como o ranking (igual ao holdout2025,
                 sem filtro de mês); usado nas previsões fora do bloco
    full2025     12 meses de 2025, com alvo (treino da versão final)
    ranking2026  ranking.parquet

    bin/run src/cache.py    # monta os 5 em sequência (pico ~5 GB; rodar sozinho)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import contexto
import features as F
from adsb_events import add_features

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
HOLDOUT_MONTHS = {1, 7}
TRUTH = "y_true"
BLOCK = "BLOCK_TIME_UTC_mvt"
SPLITS = ("train2025", "holdout2025", "blind2025", "full2025", "ranking2026")


def _month(p: Path) -> int:
    """training_2025-01-01_2025-02-01.parquet -> 1"""
    return int(p.name.split("_")[1][5:7])


def split_paths(name: str) -> list[Path]:
    training = sorted(DATA.glob("training_2025-*.parquet"))
    match name:
        case "train2025":
            return [p for p in training if _month(p) not in HOLDOUT_MONTHS]
        case "holdout2025":
            return [p for p in training if _month(p) in HOLDOUT_MONTHS]
        case "full2025" | "blind2025":
            return training
        case "ranking2026":
            return [DATA / "ranking.parquet"]
    raise ValueError(f"split desconhecido: {name}")


def cache_key(paths: list[Path]) -> str:
    """Muda quando features.py, contexto.py, cache.py (montagem/filtros) ou os dados mudam."""
    h = hashlib.sha256(Path(F.__file__).read_bytes())
    h.update(Path(contexto.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    for p in paths:
        h.update(f"{p.name}:{p.stat().st_size}".encode())
    return h.hexdigest()[:12]


def build_blind(raw: pd.DataFrame) -> pd.DataFrame:
    """Monta as DEP como no ranking: BLOCK e alvo apagados antes das features."""
    dep = raw["PHASE_mvt"] == "DEP"
    truth = pd.Series(raw.loc[dep, F.TARGET].to_numpy(), index=raw.loc[dep, F.ID].to_numpy())
    blind = raw.copy()
    blind.loc[dep, BLOCK] = pd.NaT
    blind.loc[dep, F.TARGET] = np.nan
    out = F.build(blind)
    out[TRUTH] = out[F.ID].map(truth)
    return out


def load_split(name: str) -> pd.DataFrame:
    paths = split_paths(name)
    if not paths or any(not p.exists() for p in paths):
        raise SystemExit(f"Faltam dados de {name}. Rode: .venv/bin/python src/s3.py download")
    target = CACHE / f"{name}-{cache_key(paths)}.parquet"
    if target.exists():
        return add_features(pd.read_parquet(target))
    raw = F.load(paths)
    df = build_blind(raw) if name in ("holdout2025", "blind2025") else F.build(raw)
    df = df.merge(contexto.contexto(raw), on=F.ID, how="left")
    del raw
    if name in ("train2025", "full2025"):
        df = df[df[F.TARGET].notna()]
    df = df.reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    df.to_parquet(tmp, compression="zstd", index=False)
    tmp.replace(target)  # arquivo pela metade nunca vira cache válido
    for old in CACHE.glob(f"{name}-*.parquet"):
        if old != target:
            old.unlink()
    return add_features(df)


if __name__ == "__main__":
    from runlog import LOGS, Run

    with Run("cache", {"splits": list(SPLITS)}, registry=LOGS / "infra.jsonl") as run:
        for split in SPLITS:
            with run.phase(split, 1 / len(SPLITS)):
                run.log(f"{split}: {len(load_split(split)):,} linhas")
