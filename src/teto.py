"""Teto do ganho no ranking, antes de enviar.

    bin/run src/teto.py <base.parquet> <novo.parquet> --oficial-base 338.67 \
        [--min-ms 21600] [--salvar submissions/<TEAM>_vN.parquet]

Compara dois arquivos de envio nas linhas onde diferem (> 1 s) e têm atraso
MVT − SCHED acima de --min-ms. Oráculo otimista do mecanismo "BLOCK copiado do
SCHED": y = MVT − SCHED nessas linhas. O teto é o ganho de RMSE oficial se o
oráculo fosse verdade; ganho simulado > 2 × teto = a simulação mede folga que o
modelo final não tem.

--salvar grava um candidato: o arquivo base com as previsões do novo só nessas
linhas (NÃO envia).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cache import load_split
from features import ID, TARGET
from models import SCHED_GAP


def ceiling_gain(base, new, oracle, n_total: int, base_official: float) -> float:
    base, new, oracle = (np.asarray(v, float) for v in (base, new, oracle))
    dsse = np.sum((new - oracle) ** 2 - (base - oracle) ** 2)
    return float(base_official - np.sqrt(max(base_official**2 + dsse / n_total, 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("novo")
    ap.add_argument("--oficial-base", type=float, required=True)
    ap.add_argument("--min-ms", type=float, default=0.0)
    ap.add_argument("--salvar")
    a = ap.parse_args()

    b, n = pd.read_parquet(a.base), pd.read_parquet(a.novo)
    rk = load_split("ranking2026")[[ID, SCHED_GAP]]
    m = b.merge(n, on=ID, suffixes=("_base", "_novo")).merge(rk, on=ID, how="left")
    if not len(m) == len(b) == len(n):
        raise SystemExit("os dois arquivos não têm os mesmos IDs")
    ms = m[SCHED_GAP].to_numpy(float)
    diff = np.abs(m[f"{TARGET}_novo"] - m[f"{TARGET}_base"]).to_numpy() > 1
    rows = diff & (np.nan_to_num(ms, nan=-np.inf) > a.min_ms)
    teto = ceiling_gain(
        m.loc[rows, f"{TARGET}_base"], m.loc[rows, f"{TARGET}_novo"], ms[rows], len(m), a.oficial_base
    )
    print(f"linhas diferentes: {diff.sum():,} · acima de {a.min_ms:.0f} s: {rows.sum():,}")
    print(f"teto do ganho oficial (oráculo y = MVT − SCHED): {teto:.2f} s")
    if a.salvar:
        out = b.copy()
        out.loc[rows, TARGET] = m.loc[rows, f"{TARGET}_novo"].to_numpy()
        out.to_parquet(a.salvar, index=False)
        print(f"candidato gravado em {a.salvar} ({len(out):,} linhas, {rows.sum():,} trocadas). NÃO enviado.")


if __name__ == "__main__":
    main()
