"""Comparação pareada entre dois experimentos (bootstrap por dia).

    bin/run src/compare.py <id_novo> [<id_base>] [--promover] [--aceitar-fragil]

Sem <id_base>, compara com a média dos membros da campeã (champion.json v2). Veredito MELHOR exige ganho ≥ 10 s,
IC 95% > 0, ganho sem os 10 maiores voos > 0 e ≥ 10 % do cheio, e ganho com IC > 0 em cada
mês. Se só os critérios de robustez falham: FRÁGIL (não promove sem --aceitar-fragil).
O ganho nos voos normais com NM e o ganho sem os voos "loteria" saem como informação
para decidir; nenhum dos dois entra no veredito.
--promover grava a nova campeã quando o veredito é MELHOR e a base é a própria campeã
(ou o único membro dela); sem campeã, promove direto.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import campeao
from cache import TRUTH, load_split
from campeao import CAMPEA as CHAMPION
from experiment import TAIL_S, lottery_mask
from features import ID, PLAN_REFS
from runlog import REGISTRY, ROOT

MIN_GAIN_S = 10.0


def paired_bootstrap(y, base, new, groups, n: int = 1000, seed: int = 0) -> dict:
    """Ganho = RMSE(base) − RMSE(novo); reamostra dias inteiros com reposição."""
    y, base, new = (np.asarray(v, float) for v in (y, base, new))
    codes, _ = pd.factorize(np.asarray(groups))
    k = codes.max() + 1
    cnt = np.bincount(codes, minlength=k)
    sse_b = np.bincount(codes, (y - base) ** 2, minlength=k)
    sse_n = np.bincount(codes, (y - new) ** 2, minlength=k)
    gain = np.sqrt(sse_b.sum() / cnt.sum()) - np.sqrt(sse_n.sum() / cnt.sum())
    draws = np.random.default_rng(seed).integers(0, k, size=(n, k))
    c = cnt[draws].sum(1)
    boot = np.sqrt(sse_b[draws].sum(1) / c) - np.sqrt(sse_n[draws].sum(1) / c)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"ganho": float(gain), "ic_baixo": float(lo), "ic_alto": float(hi)}


def is_better(res: dict, min_gain: float = MIN_GAIN_S) -> bool:
    return res["ganho"] >= min_gain and res["ic_baixo"] > 0


TOP_K = 10
MIN_SHARE_WITHOUT_TOP = 0.10  # ganho sem os 10 maiores voos ≥ 10 % do ganho cheio


def gain_without_top(y, base, new, k: int = TOP_K) -> float:
    """Ganho de RMSE depois de tirar os k voos que mais contribuíram para o ganho."""
    y, base, new = (np.asarray(v, float) for v in (y, base, new))
    contrib = (y - base) ** 2 - (y - new) ** 2
    keep = np.ones(y.size, bool)
    keep[np.argsort(contrib)[-k:]] = False
    rmse = lambda p: np.sqrt(np.mean((y[keep] - p[keep]) ** 2))  # noqa: E731
    return float(rmse(base) - rmse(new))


def by_month(y, base, new, days) -> dict:
    """Bootstrap pareado por dia, separado por mês (jan e jul no holdout)."""
    days = np.asarray(days).astype(str)
    months = np.array([d[5:7] for d in days])
    y, base, new = (np.asarray(v, float) for v in (y, base, new))
    return {
        m: paired_bootstrap(y[months == m], base[months == m], new[months == m], days[months == m])
        for m in sorted(set(months))
    }


HOLD_COLS = [ID, "FLIGHT_ID_mvt", *(f"to_takeoff_from_{c}" for c in PLAN_REFS)]


def slice_masks(m: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fatias informativas alinhadas às linhas de `m`, com as features do holdout.

    Voos normais com NM são 43 % do erro² e é neles que dá para melhorar de verdade;
    os 11 voos "loteria" são 32 % do erro² e só adicionam ruído ao ganho.
    """
    hold = load_split("holdout2025")[HOLD_COLS]
    j = m[[ID]].merge(hold, on=ID, how="left", validate="one_to_one")
    y = m[TRUTH].to_numpy(float)
    return {
        "voos normais com NM": (y <= TAIL_S) & j["FLIGHT_ID_mvt"].notna().to_numpy(),
        "sem loteria": ~lottery_mask(j, y),
    }


def verdict(res: dict, sem_top: float, meses: dict) -> str:
    """MELHOR: ganho ≥ 10 s com IC > 0, sobrevive sem os 10 maiores voos e vale em cada mês.

    FRÁGIL: passa o critério antigo mas falha a robustez (caso v3 → v4).
    """
    if not is_better(res):
        return "não comprovado"
    robust = sem_top > 0 and sem_top >= MIN_SHARE_WITHOUT_TOP * res["ganho"]
    each_month = all(r["ganho"] > 0 and r["ic_baixo"] > 0 for r in meses.values())
    return "MELHOR" if robust and each_month else "FRÁGIL"


def registry_entry(run_id: str) -> dict:
    for line in REGISTRY.read_text().splitlines():
        rec = json.loads(line)
        if rec["id"] == run_id:
            return rec
    raise SystemExit(f"{run_id} não está em {REGISTRY.name}")


def may_promote(base_id: str | None, champion: dict | None) -> bool:
    """Só se destrona a campeã comparando com ela (ou com o único membro dela)."""
    if not champion or base_id is None:
        return True
    ids = [m["id"] for m in champion["membros"]]
    return ids == [base_id]


def promote(rec: dict) -> None:
    if (rec.get("config") or {}).get("pseudo_vazado"):
        # alvo destilado de um modelo que viu jan/jul de 2025: este holdout não a mede
        raise SystemExit(f"{rec['id']} tem `pseudo_vazado`: o holdout não julga essa corrida"
                         " (docs/research/2026-10-03-pseudo-rotulo.md); não promove")
    atual = campeao.carregar() if CHAMPION.exists() else None
    campeao.salvar(campeao.nova([rec], enviada=atual.get("enviada") if atual else None))
    print(f"campeã agora: {rec['id']} (1 membro)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("novo")
    ap.add_argument("base", nargs="?")
    ap.add_argument("--promover", action="store_true")
    ap.add_argument("--aceitar-fragil", action="store_true",
                    help="promove veredito FRÁGIL (usar só com teto.py ≥ 3 s e ok do usuário)")
    a = ap.parse_args()

    new = registry_entry(a.novo)
    champ = campeao.carregar() if CHAMPION.exists() else None
    if a.base:
        base = registry_entry(a.base)
        pb = pd.read_parquet(ROOT / base["previsoes"])
        nome_base, rmse_base = base["id"], base["metricas"]["completo"]
    elif champ:
        pb = campeao.previsao([m["id"] for m in champ["membros"]])
        nome_base = "campeã (" + " + ".join(m["id"] for m in champ["membros"]) + ")"
        rmse_base = champ["rmse_simulacao"]
    else:
        print("ainda não há campeã")
        if a.promover:
            promote(new)
        return

    pn = pd.read_parquet(ROOT / new["previsoes"])
    m = pb.merge(pn[[ID, "pred"]], on=ID, suffixes=("_base", "_novo"))
    if not len(m) == len(pb) == len(pn):
        raise SystemExit("as previsões não cobrem o mesmo holdout; refaça o experimento base")
    y, pb_, pn_ = m[TRUTH], m["pred_base"], m["pred_novo"]
    res = paired_bootstrap(y, pb_, pn_, m["dia"])
    sem_top = gain_without_top(y, pb_, pn_)
    meses = by_month(y, pb_, pn_, m["dia"])
    v = verdict(res, sem_top, meses)
    print(f"{nome_base} → {new['id']}")
    print(f"  RMSE simulação {rmse_base} → {new['metricas']['completo']}")
    print(f"  ganho {res['ganho']:.1f} s (IC 95% {res['ic_baixo']:.1f} a {res['ic_alto']:.1f})")
    print(f"  sem os {TOP_K} maiores voos: {sem_top:.1f} s")
    for mes, r in meses.items():
        print(f"  mês {mes}: ganho {r['ganho']:.1f} s (IC 95% {r['ic_baixo']:.1f} a {r['ic_alto']:.1f})")
    print("  informativo (fora do veredito):")
    for nome, mask in slice_masks(m).items():
        r = paired_bootstrap(y[mask], pb_[mask], pn_[mask], m["dia"][mask])
        print(
            f"    {nome} ({int(mask.sum()):,} voos): ganho {r['ganho']:.1f} s"
            f" (IC 95% {r['ic_baixo']:.1f} a {r['ic_alto']:.1f})"
        )
    print(f"  veredito: {v}")
    promote_ok = v == "MELHOR" or (v == "FRÁGIL" and a.aceitar_fragil)
    if a.promover and promote_ok:
        if may_promote(a.base, champ):
            promote(new)
        else:
            print(
                f"não promovido: a base {a.base} não é a campeã; "
                "compare sem <id_base> ou refaça a corrida da campeã"
            )
    elif a.promover and v == "FRÁGIL":
        print("não promovido: ganho frágil. Só com --aceitar-fragil, depois do teto.py e do ok do usuário")


if __name__ == "__main__":
    main()
