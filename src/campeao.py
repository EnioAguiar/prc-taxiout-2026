"""Campeã v2: média simples de corretores (membros) sobre a mesma base, mais pós-regras.

    {"base", "oof", "membros": [{"id", "config"}], "pos_regras", "rmse_simulacao",
     "sem_loteria", "enviada": {"versao", "sem_loteria"} | null}

`oof` é a corrida cujo oof fora do bloco os corretores novos reaproveitam (`--reusar-oof`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cache import TRUTH
from features import ID
from runlog import REGISTRY, ROOT

CAMPEA = ROOT / "champion.json"


def registro(run_id: str) -> dict:
    for line in reversed(REGISTRY.read_text().splitlines()):
        rec = json.loads(line) if line.strip() else {}
        if rec.get("id") == run_id:
            return rec
    raise SystemExit(f"{run_id} não está em {REGISTRY.name}")


def previsao(ids: list[str]) -> pd.DataFrame:
    """Previsão média dos membros no holdout, alinhada por voo."""
    partes = [pd.read_parquet(ROOT / registro(i)["previsoes"]) for i in ids]
    base = partes[0][[ID, "dia", TRUTH]].copy()
    preds = [base[[ID]].merge(p[[ID, "pred"]], on=ID, validate="one_to_one")["pred"].to_numpy(float)
             for p in partes]
    if any(len(p) != len(base) for p in preds):
        raise SystemExit("membros com holdouts diferentes")
    base["pred"] = np.mean(preds, axis=0)
    return base


def ultimo_por_nome(nome: str) -> dict:
    """A última corrida gravada com esse `nome` (o id leva data e hora na frente)."""
    for line in reversed(REGISTRY.read_text().splitlines()):
        rec = json.loads(line) if line.strip() else {}
        if rec.get("nome") == nome:
            return rec
    raise SystemExit(f"nenhuma corrida com nome {nome} em {REGISTRY.name}")


def nova(membros: list[dict], pos_regras: list[str] | None = None,
         enviada: dict | None = None, medir: bool = True) -> dict:
    """Campeã v2 a partir de registros do `experiments.jsonl`; `medir` calcula a simulação
    da média (lê o holdout real; os testes passam `medir=False`)."""
    bases = {m["config"]["base"] for m in membros}
    if len(bases) != 1:
        raise ValueError(f"membros precisam da mesma base: {sorted(bases)}")
    ids = [m["id"] for m in membros]
    return {
        "base": bases.pop(),
        "oof": membros[0]["config"].get("reusar_oof") or membros[0]["id"],
        "membros": [{"id": m["id"], "config": m["config"]} for m in membros],
        "pos_regras": list(pos_regras) if pos_regras is not None else ["roma"],
        "rmse_simulacao": None, "sem_loteria": None,
        "enviada": enviada,
    } | (_metricas(ids) if medir else {})


def _metricas(ids: list[str]) -> dict:
    """RMSE completo e sem loteria da média (sem as pós-regras, como a simulação de sempre)."""
    p = previsao(ids)
    from compare import slice_masks  # import tardio: compare importa campeao
    y, pr = p[TRUTH].to_numpy(float), p["pred"].to_numpy(float)
    keep = slice_masks(p)["sem loteria"]
    return {"rmse_simulacao": round(float(np.sqrt(np.mean((y - pr) ** 2))), 2),
            "sem_loteria": round(float(np.sqrt(np.mean((y[keep] - pr[keep]) ** 2))), 2)}


def carregar(caminho: Path = CAMPEA) -> dict:
    return json.loads(Path(caminho).read_text())


def salvar(champ: dict, caminho: Path = CAMPEA) -> None:
    Path(caminho).write_text(json.dumps(champ, indent=2, ensure_ascii=False) + "\n")
