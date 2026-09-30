"""Régua da esteira: seleção na metade A dos dias, confirmação cega na metade B.

A: ganho `sem_loteria` ≥ GANHO_A e IC baixo > 0 (bootstrap pareado por dia do compare.py).
B: ganho `sem_loteria` > 0 e IC baixo > IC_B. Dias todos: ganho `completo` ≥ COMPLETO_MIN.
B nunca escolhe: só confirma a proposta que A escolheu.
"""
from __future__ import annotations

import numpy as np

import campeao
from cache import TRUTH
from compare import paired_bootstrap, slice_masks

GANHO_A, IC_B, COMPLETO_MIN = 0.3, -0.3, -0.5


def metades(dias: np.ndarray, semente: int = 0) -> np.ndarray:
    unicos = np.array(sorted(set(np.asarray(dias).astype(str))))
    if semente:
        unicos = np.random.default_rng(semente).permutation(unicos)
    lado = {d: "A" if i % 2 == 0 else "B" for i, d in enumerate(unicos)}
    return np.array([lado[d] for d in np.asarray(dias).astype(str)])


def propostas(membros: list[str], novo: str, mesma_base: bool) -> dict[str, list[str]]:
    if not mesma_base:
        return {"sozinho": [novo]}
    p = {f"troca:{m}": [novo if x == m else x for x in membros] for m in membros}
    return p | {"soma": [*membros, novo], "sozinho": [novo]}


def decidir(y, base, novo, dias, sem_lot, semente: int = 0) -> dict:
    y, base, novo = (np.asarray(v, float) for v in (y, base, novo))
    dias = np.asarray(dias).astype(str)
    lado = metades(dias, semente)

    def boot(mask):
        return paired_bootstrap(y[mask], base[mask], novo[mask], dias[mask])

    a, b = boot((lado == "A") & sem_lot), boot((lado == "B") & sem_lot)
    completo = paired_bootstrap(y, base, novo, dias)["ganho"]
    if not (a["ganho"] >= GANHO_A and a["ic_baixo"] > 0):
        return {"aprovado": False, "a": a, "b": None, "completo": completo,
                "motivo": "A: não seleciona"}
    if not (b["ganho"] > 0 and b["ic_baixo"] > IC_B):
        return {"aprovado": False, "a": a, "b": b, "completo": completo,
                "motivo": "B: não confirma"}
    if completo < COMPLETO_MIN:
        return {"aprovado": False, "a": a, "b": b, "completo": completo,
                "motivo": "completo piora"}
    return {"aprovado": True, "a": a, "b": b, "completo": completo, "motivo": "aprovado"}


def avaliar(membros: list[str], novo: str, semente: int = 0) -> dict:
    """Melhor proposta pela metade A; só ela é confirmada em B."""
    mesma = campeao.registro(novo)["config"]["base"] == campeao.registro(membros[0])["config"]["base"]
    ref = campeao.previsao(membros)
    y, dias = ref[TRUTH].to_numpy(float), ref["dia"].to_numpy()
    sem_lot = slice_masks(ref)["sem loteria"]
    lado = metades(dias, semente)
    base_pred = ref["pred"].to_numpy(float)
    escolha, melhor = None, -np.inf
    m = (lado == "A") & sem_lot
    todas = propostas(membros, novo, mesma)
    for nome, ids in todas.items():
        p = campeao.previsao(ids)["pred"].to_numpy(float)
        g = paired_bootstrap(y[m], base_pred[m], p[m], dias[m])["ganho"]
        if g > melhor:
            escolha, melhor = (nome, ids, p), g
    nome, ids, p = escolha
    r = decidir(y, base_pred, p, dias, sem_lot, semente)
    return r | {"proposta": nome, "membros": ids, "avaliadas": len(todas)}
