"""Comparação pareada entre dois experimentos (bootstrap por dia).

    bin/run src/compare.py <id_novo> [<id_base>] [--promover]

Sem <id_base>, compara com o campeão (champion.json). Veredito "MELHOR" exige
ganho ≥ 10 s na simulação e IC 95% acima de zero. --promover grava o novo
campeão quando o veredito é "MELHOR" e a base é o próprio campeão (ou uma
repetição da configuração dele); sem campeão, promove direto.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from cache import TRUTH
from features import ID
from runlog import REGISTRY, ROOT

CHAMPION = ROOT / "champion.json"
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


def registry_entry(run_id: str) -> dict:
    for line in REGISTRY.read_text().splitlines():
        rec = json.loads(line)
        if rec["id"] == run_id:
            return rec
    raise SystemExit(f"{run_id} não está em {REGISTRY.name}")


def may_promote(base: dict, champion: dict | None) -> bool:
    """Só se pode destronar o campeão comparando com ele (ou com uma repetição dele)."""
    if not champion:
        return True
    return base["id"] == champion["id"] or base["config"] == champion["config"]


def promote(rec: dict) -> None:
    champ = {
        "id": rec["id"], "config": rec["config"], "best_iter": rec.get("best_iter"),
        "rmse_simulacao": rec["metricas"]["completo"],
        "src_hash": rec.get("src_hash"), "git_commit": rec.get("git_commit"),
    }
    CHAMPION.write_text(json.dumps(champ, indent=2, ensure_ascii=False) + "\n")
    print(f"campeão agora: {rec['id']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("novo")
    ap.add_argument("base", nargs="?")
    ap.add_argument("--promover", action="store_true")
    a = ap.parse_args()

    new = registry_entry(a.novo)
    champ = json.loads(CHAMPION.read_text()) if CHAMPION.exists() else None
    if a.base:
        base = registry_entry(a.base)
    elif champ:
        base = registry_entry(champ["id"])
    else:
        print("ainda não há campeão")
        if a.promover:
            promote(new)
        return

    pb = pd.read_parquet(ROOT / base["previsoes"])
    pn = pd.read_parquet(ROOT / new["previsoes"])
    m = pb.merge(pn[[ID, "pred"]], on=ID, suffixes=("_base", "_novo"))
    if not len(m) == len(pb) == len(pn):
        raise SystemExit("as previsões não cobrem o mesmo holdout; refaça o experimento base")
    res = paired_bootstrap(m[TRUTH], m["pred_base"], m["pred_novo"], m["dia"])
    better = is_better(res)
    print(f"{base['id']} → {new['id']}")
    print(f"  RMSE simulação {base['metricas']['completo']} → {new['metricas']['completo']}")
    print(f"  ganho {res['ganho']:.1f} s (IC 95% {res['ic_baixo']:.1f} a {res['ic_alto']:.1f})")
    print(f"  veredito: {'MELHOR' if better else 'não comprovado'} (exige ≥ {MIN_GAIN_S:.0f} s e IC > 0)")
    if a.promover and better:
        if may_promote(base, champ):
            promote(new)
        else:
            print(
                f"não promovido: a base {base['id']} não é o campeão {champ['id']}; "
                "compare sem <id_base> ou refaça a corrida do campeão"
            )


if __name__ == "__main__":
    main()
