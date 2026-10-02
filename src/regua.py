"""Régua da esteira: seleção na metade A dos dias, confirmação cega na metade B.

A: ganho `sem_loteria` ≥ GANHO_A e IC baixo > 0 (bootstrap pareado por dia do compare.py).
B: ganho `sem_loteria` > 0 (só o sinal; meia amostra deixa o IC largo demais para bases novas).
Dias todos: ganho `sem_loteria` com IC baixo > 0 — o critério que previu o oficial da v29 e da
v32 — e ganho `completo` ≥ COMPLETO_MIN. B nunca escolhe: só confirma a proposta que A escolheu.

Calibração de 30/09: com `IC_B = −0,3` a régua reprovou a base com plano 13 (candidato 76:
A +2,37, B +0,50 com IC −0,93 a +1,95, dias todos sem loteria +1,45 com IC +0,09 a +3,05).
"""
from __future__ import annotations

import numpy as np

import campeao
from cache import TRUTH
from features import ID
from compare import paired_bootstrap, slice_masks

GANHO_A, COMPLETO_MIN = 0.3, -0.5


def metades(dias: np.ndarray, semente: int = 0) -> np.ndarray:
    unicos = np.array(sorted(set(np.asarray(dias).astype(str))))
    if semente:
        unicos = np.random.default_rng(semente).permutation(unicos)
    lado = {d: "A" if i % 2 == 0 else "B" for i, d in enumerate(unicos)}
    return np.array([lado[d] for d in np.asarray(dias).astype(str)])


def propostas(membros: list[str], novo: str) -> dict[str, list[str]]:
    """Trocar o novo por cada membro, somá-lo aos atuais, ou ficar só com ele.

    Vale mesmo quando o novo está sobre outra base: o envio agrupa os membros por
    `base_config` e treina uma base final por grupo (`train.grupos_por_base`)."""
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
    if not b["ganho"] > 0:
        return {"aprovado": False, "a": a, "b": b, "completo": completo,
                "motivo": "B: não confirma"}
    todos = boot(sem_lot)
    if not todos["ic_baixo"] > 0:
        return {"aprovado": False, "a": a, "b": b, "completo": completo,
                "motivo": "sem loteria (dias todos): IC toca 0"}
    if completo < COMPLETO_MIN:
        return {"aprovado": False, "a": a, "b": b, "completo": completo,
                "motivo": "completo piora"}
    return {"aprovado": True, "a": a, "b": b, "completo": completo, "motivo": "aprovado"}


def _melhor(membros: list[str], todas: dict[str, list[str]], semente: int) -> dict:
    """Entre as propostas que passam em A, a de maior ganho; só ela é confirmada em B."""
    ref = campeao.previsao(membros)
    y, dias = ref[TRUTH].to_numpy(float), ref["dia"].to_numpy()
    sem_lot = slice_masks(ref)["sem loteria"]
    lado = metades(dias, semente)
    base_pred = ref["pred"].to_numpy(float)
    escolha, melhor = None, -np.inf
    passou, melhor_passou = None, -np.inf
    m = (lado == "A") & sem_lot
    for nome, ids in todas.items():
        p = _alinhado(ref, ids)
        r_a = paired_bootstrap(y[m], base_pred[m], p[m], dias[m])
        g = r_a["ganho"]
        if g > melhor:
            escolha, melhor = (nome, ids, p), g
        if g >= GANHO_A and r_a["ic_baixo"] > 0 and g > melhor_passou:
            passou, melhor_passou = (nome, ids, p), g
    nome, ids, p = passou or escolha  # sem nenhuma passando, reprova com a de maior ganho
    r = decidir(y, base_pred, p, dias, sem_lot, semente)
    return r | {"proposta": nome, "membros": ids, "avaliadas": len(todas)}


def _alinhado(ref, ids: list[str]) -> np.ndarray:
    """Previsão média dos `ids` na ordem de voos de `ref` (base nova grava na ordem dela)."""
    p = ref[[ID]].merge(campeao.previsao(ids)[[ID, "pred"]], on=ID, how="left",
                        validate="one_to_one")["pred"].to_numpy(float)
    if np.isnan(p).any():
        raise SystemExit(f"{'+'.join(ids)} não cobre o mesmo holdout da campeã")
    return p


def avaliar(membros: list[str], novo: str, semente: int = 0) -> dict:
    """Candidato de corretor: trocar por um membro, somar aos atuais, ou ficar só com ele."""
    return _melhor(membros, propostas(membros, novo), semente)


def avaliar_conjunto(membros: list[str], novos: list[str], semente: int = 0) -> dict:
    """Candidato de base nova: os corretores refeitos sobre ela trocam a campeã inteira
    (`base_nova`) ou **somam** aos membros atuais (`soma`), que é a mistura de duas bases."""
    return _melhor(membros, {"base_nova": list(novos), "soma": [*membros, *novos]}, semente)
