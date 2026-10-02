"""Campeã v3: média simples de corretores (membros), mais pós-regras.

    {"membros": [{"id", "config"}], "pos_regras", "rmse_simulacao",
     "sem_loteria", "enviada": {"versao", "sem_loteria"} | null}

Os membros podem estar sobre **bases diferentes** (foi daí que veio o ganho de 02/10): a
base de cada um vive em `config.base`/`config.base_config` (`base_do_membro()`) e o
conjunto delas sai de `bases()`. Um candidato de corretor roda sobre a base do membro de
quem ele saiu (`esteira.membro_de_origem`); `principal()` só serve de padrão para o
`stack.py --base` rodado à mão.
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


def base_do_membro(m: dict) -> dict:
    """A base de um membro: a corrida da base e a corrida dona do oof fora do bloco dela.

    `oof` é quem os corretores novos daquela base reaproveitam (`--reusar-oof`); quando o
    próprio membro calculou o seu, é ele mesmo."""
    return {"base": m["config"]["base"], "oof": m["config"].get("reusar_oof") or m["id"]}


def bases(champ: dict) -> list[dict]:
    """Uma entrada por base distinta dos membros, na ordem em que aparecem."""
    saida: dict[str, dict] = {}
    for m in champ["membros"]:
        saida.setdefault(m["config"]["base"], base_do_membro(m))
    return list(saida.values())


def principal(champ: dict) -> dict:
    """A base do primeiro membro: o padrão do `stack.py --base` rodado à mão."""
    return bases(champ)[0]


def nova(membros: list[dict], pos_regras: list[str] | None = None,
         enviada: dict | None = None, medir: bool = True) -> dict:
    """Campeã v3 a partir de registros do `experiments.jsonl`; `medir` calcula a simulação
    da média (lê o holdout real; os testes passam `medir=False`)."""
    ids = [m["id"] for m in membros]
    return {
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
