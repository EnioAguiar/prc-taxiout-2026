# Plano 3a — regra de promoção robusta, seeds e variante > 6 h (v5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Impedir que ganhos frágeis (sorte de poucos voos) virem campeão, medir o ruído do próprio treino e testar a variante recomendada pelo diagnóstico (retas só para voos sem NM com atraso > 6 h), gerando a v5 só se o ganho for sólido.

**Architecture:** `compare.py` ganha dois critérios de robustez (ganho sem os 10 maiores voos; ganho em jan e jul separadamente) e o veredito `FRÁGIL`. Os modelos recebem `seed` e rodam determinísticos. `TwoStageNM` ganha `nm_min_ms` (só troca a previsão acima do limiar). Novo `teto.py` calcula, antes de enviar, o ganho máximo possível no ranking e pode montar o arquivo candidato a partir de dois envios já feitos.

**Tech Stack:** Python 3.14, pandas 3, numpy, LightGBM 4.7, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-pipeline-melhoria-continua-design.md`; evidência: `docs/research/2026-09-24-diagnostico-v4.md`.

## Global Constraints

- Processo pesado por `bin/run`, supervisionado (hub `start`), um por vez; `num_threads=12`; RAM ≤ 7 GB.
- L2 no alvo bruto; só piso 0.
- `--promover` só sem base explícita. Veredito `FRÁGIL` nunca é promovido automaticamente.
- Nada é enviado ao placar sem ok explícito do usuário; implementador nunca roda `s3.py submit`.
- `experiments.jsonl` e `champion.json` versionados; textos em português.
- Commit + push na branch de trabalho ao fim de cada tarefa.

## Fatos que o plano usa (diagnóstico da v4)

- v3 → v4: ganho simulado 42,27 s; sem os 10 maiores voos −9,71 s; jan IC baixo −9,34; jul IC baixo −0,78; oficial −1,5 s.
- v2 → v3: sem os 10 maiores +10,18 s (15,2 %); jan IC baixo +2,98; jul +32,80; oficial −46 s.
- Retas só em `ms` > 6 h: holdout 334,51 s; sem top-10 +1,90 (3,5 % do ganho); jan IC baixo +0,19; jul +18,04; dano nos voos normais sem NM zero (1.081,9 vs 1.083,2); teto no ranking −8,5 s.
- Envios: `submissions/outgoing-boat_v3.parquet` (oficial 338,67) e `outgoing-boat_v4.parquet` (oficial 337,18); diferem só nas linhas `nm_missing`.

## Estrutura de arquivos

| Arquivo | Ação | Responsabilidade |
|---|---|---|
| `src/compare.py` | alterar | `gain_without_top`, `by_month`, `verdict`, veredito `FRÁGIL`, `--aceitar-fragil` |
| `src/models.py` | alterar | `seed` e determinismo; `nm_min_ms`; `line_rows` |
| `src/experiment.py` | alterar | `--seed`, `--nm-min-ms`; erro para flags do modelo errado |
| `src/teto.py` | criar | teto no ranking e montagem do candidato |
| `tests/test_compare.py`, `tests/test_models.py`, `tests/test_teto.py` | alterar/criar | contratos |
| `README.md` | alterar | regra nova, ruído, resultado |

---

### Task 1: Regra de promoção robusta em `compare.py`

**Files:**
- Modify: `src/compare.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Produces: `TOP_K = 10`, `MIN_SHARE_WITHOUT_TOP = 0.10`; `gain_without_top(y, base, new, k=TOP_K) -> float`; `by_month(y, base, new, days) -> dict[str, dict]` (chave `"01"`, `"07"`…; valor = saída de `paired_bootstrap` no mês); `verdict(res, sem_top, meses) -> str` em `{"MELHOR", "FRÁGIL", "não comprovado"}`.

- [ ] **Step 1: Testes que falham**

Acrescentar a `tests/test_compare.py` (imports no topo):

```python
import pandas as pd

from compare import by_month, gain_without_top, verdict


def _jan_jul(n=20_000, seed=3):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", periods=31).append(pd.date_range("2025-07-01", periods=31))
    dia = np.array(days.strftime("%Y-%m-%d"))[rng.integers(0, 62, n)]
    y = rng.normal(900, 400, n)
    return rng, y, dia


def test_ganho_espalhado_e_melhor():
    rng, y, dia = _jan_jul()
    base = y + rng.normal(0, 300, y.size)
    new = y + rng.normal(0, 200, y.size)
    res = paired_bootstrap(y, base, new, dia)
    assert verdict(res, gain_without_top(y, base, new), by_month(y, base, new, dia)) == "MELHOR"

    idx = rng.choice(y.size, 10, replace=False)
    base[idx] = y[idx] + 40_000  # 10 voos em que a base erra horas (= TOP_K)
    new[idx] = y[idx]
    base = y + rng.normal(0, 300, y.size)
    new = base + rng.normal(0, 30, y.size)  # um pouco pior em quase tudo
    idx = rng.choice(y.size, 5, replace=False)
    base[idx] = y[idx] + 40_000  # 5 voos em que a base erra horas
    new[idx] = y[idx]
    res = paired_bootstrap(y, base, new, dia)
    sem_top = gain_without_top(y, base, new)
    assert res["ganho"] >= 10 and sem_top <= 0
    assert verdict(res, sem_top, by_month(y, base, new, dia)) == "FRÁGIL"
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/bin/python -m pytest tests/test_compare.py -q`
Expected: FAIL (`cannot import name 'by_month'`).

- [ ] **Step 3: Implementar**

Em `src/compare.py`, depois de `is_better`:

```python
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


def verdict(res: dict, sem_top: float, meses: dict) -> str:
    """MELHOR: ganho ≥ 10 s com IC > 0, sobrevive sem os 10 maiores voos e vale em cada mês.

    FRÁGIL: passa o critério antigo mas falha a robustez (caso v3 → v4).
    """
    if not is_better(res):
        return "não comprovado"
    robust = sem_top > 0 and sem_top >= MIN_SHARE_WITHOUT_TOP * res["ganho"]
    each_month = all(r["ganho"] > 0 and r["ic_baixo"] > 0 for r in meses.values())
    return "MELHOR" if robust and each_month else "FRÁGIL"
```

Em `main()`, trocar o bloco a partir de `res = paired_bootstrap(...)` até o fim da função por:

```python
    y, pb_, pn_ = m[TRUTH], m["pred_base"], m["pred_novo"]
    res = paired_bootstrap(y, pb_, pn_, m["dia"])
    sem_top = gain_without_top(y, pb_, pn_)
    meses = by_month(y, pb_, pn_, m["dia"])
    v = verdict(res, sem_top, meses)
    print(f"{base['id']} → {new['id']}")
    print(f"  RMSE simulação {base['metricas']['completo']} → {new['metricas']['completo']}")
    print(f"  ganho {res['ganho']:.1f} s (IC 95% {res['ic_baixo']:.1f} a {res['ic_alto']:.1f})")
    print(f"  sem os {TOP_K} maiores voos: {sem_top:.1f} s")
    for mes, r in meses.items():
        print(f"  mês {mes}: ganho {r['ganho']:.1f} s (IC 95% {r['ic_baixo']:.1f} a {r['ic_alto']:.1f})")
    print(f"  veredito: {v}")
    promote_ok = v == "MELHOR" or (v == "FRÁGIL" and a.aceitar_fragil)
    if a.promover and promote_ok:
        if may_promote(base, champ):
            promote(new)
        else:
            print(
                f"não promovido: a base {base['id']} não é o campeão {champ['id']}; "
                "compare sem <id_base> ou refaça a corrida do campeão"
            )
    elif a.promover and v == "FRÁGIL":
        print("não promovido: ganho frágil. Só com --aceitar-fragil, depois do teto.py e do ok do usuário")
```

E acrescentar o argumento depois de `--promover`:

```python
    ap.add_argument("--aceitar-fragil", action="store_true",
                    help="promove veredito FRÁGIL (usar só com teto.py ≥ 3 s e ok do usuário)")
```

Atualizar o docstring do módulo: "Veredito MELHOR exige ganho ≥ 10 s, IC 95% > 0, ganho sem os 10 maiores voos > 0 e ≥ 10 % do cheio, e ganho com IC > 0 em cada mês. Se só os critérios de robustez falham: FRÁGIL (não promove sem --aceitar-fragil)."

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/bin/python -m pytest -q` → Expected: 15 passed.

- [ ] **Step 5: Conferir contra o diagnóstico (leve, sem treino)**

Run: `bin/run src/compare.py 20260924-194654-nm_retas 20260924-194237-dois_estagios_r`
Expected: ganho 42,3; sem os 10 maiores ≈ −9,7; mês 01 IC baixo < 0; veredito `FRÁGIL`.

- [ ] **Step 6: Commit**

```bash
git add src/compare.py tests/test_compare.py
git commit -m "compare: robustez (sem os 10 maiores voos, por mês) e veredito FRÁGIL"
git push
```

---

### Task 2: Seeds e treino determinístico

**Files:**
- Modify: `src/models.py` (`PARAMS`, `SingleLGBM`, `TwoStage`)
- Modify: `src/experiment.py` (`config`, argumento `--seed`)

**Interfaces:**
- Produces: cfg `seed: int` (padrão 0) em todos os modelos; `models.params_for(cfg) -> dict`; CLI `--seed N`.

- [ ] **Step 1: Determinismo e seed**

Em `PARAMS`, acrescentar:

```python
    deterministic=True,  # mesma seed → mesmo modelo, mesmo com 12 threads
    force_row_wise=True,
```

Depois de `PARAMS`, acrescentar:

```python
def params_for(cfg: dict) -> dict:
    return {**PARAMS, "seed": int(cfg.get("seed", 0))}
```

Em `SingleLGBM.__init__` e `TwoStage.__init__`, acrescentar `self.params = params_for(cfg)`. Em `SingleLGBM.fit`, trocar `PARAMS` por `self.params`. Em `TwoStage.fit`, trocar `cls_params = {**PARAMS, ...}` por `cls_params = {**self.params, "objective": "binary", "metric": "binary_logloss"}` e o `PARAMS` do regressor por `self.params`.

- [ ] **Step 2: CLI**

Em `experiment.py`, acrescentar `ap.add_argument("--seed", type=int, default=0)` e, em `config()`, incluir `"seed": a.seed` em todos os retornos (no `single` e no dict dos dois estágios).

- [ ] **Step 3: Verificar determinismo (rápido, sem holdout inteiro)**

Run:
```bash
bin/run -c "
import numpy as np, pandas as pd, lightgbm as lgb, sys; sys.path.insert(0,'src')
from models import params_for
rng=np.random.default_rng(0); X=pd.DataFrame(rng.normal(size=(20000,8))); y=X[0]*3+rng.normal(size=20000)
p=lambda s: lgb.train(params_for({'seed':s}), lgb.Dataset(X,y), 50).predict(X)
print(np.abs(p(0)-p(0)).max(), np.abs(p(0)-p(1)).max())"
```
Expected: primeiro número `0.0`; segundo > 0.

Run: `.venv/bin/python -m pytest -q` → Expected: 15 passed.

- [ ] **Step 4: Commit**

```bash
git add src/models.py src/experiment.py
git commit -m "Seeds e treino determinístico (deterministic, force_row_wise)"
git push
```

---

### Task 3: Retas só acima do limiar e reparos da revisão

**Files:**
- Modify: `src/models.py` (`line_rows`, `TwoStageNM`)
- Modify: `src/experiment.py` (`--nm-min-ms`, validação de flags)
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `line_rows(nm, ms, line_pred, min_ms) -> np.ndarray[bool]`; cfg `nm_min_ms: float` (s, padrão 0); CLI `--nm-min-ms S`.

- [ ] **Step 1: Testes que falham**

Acrescentar a `tests/test_models.py`:

```python
from models import line_rows, nm_groups


def test_reta_so_troca_voos_sem_nm_acima_do_limiar():
    nm = np.array([1, 1, 1, 0, 1], bool)
    ms = np.array([30_000.0, 3_000.0, np.nan, 30_000.0, 21_600.0])
    line = np.array([29_000.0, 2_000.0, np.nan, 29_000.0, 20_000.0])
    assert line_rows(nm, ms, line, 21_600).tolist() == [True, False, False, False, False]
    assert line_rows(nm, ms, line, 0).tolist() == [True, True, False, False, True]


def test_grupos_com_corte_de_2h():
    df = pd.DataFrame({
        "AIRPORT": pd.Categorical(["LIRF", "LIRF", "EDDF"]),
        "to_takeoff_from_SCHED_TIME_UTC_mvt": [9000.0, 600.0, np.nan],
    })
    assert nm_groups(df, split_ms=True).tolist() == ["LIRF|>2h", "LIRF|<=2h", "EDDF|<=2h"]
    assert nm_groups(df, split_ms=False).tolist() == ["LIRF", "LIRF", "EDDF"]
```

(`F.AIRPORT == "AIRPORT"`.)

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/bin/python -m pytest tests/test_models.py -q` → Expected: FAIL (`cannot import name 'line_rows'`).

- [ ] **Step 3: Implementar**

Em `models.py`, antes de `class TwoStageNM`:

```python
def line_rows(nm, ms, line_pred, min_ms: float) -> np.ndarray:
    """Voos em que a reta substitui o dois estágios: sem NM, com reta e atraso > limiar."""
    nm, ms, line_pred = np.asarray(nm, bool), np.asarray(ms, float), np.asarray(line_pred, float)
    over = np.nan_to_num(ms, nan=-np.inf) > min_ms if min_ms > 0 else np.ones(ms.size, bool)
    return nm & ~np.isnan(line_pred) & over
```

Em `TwoStageNM.__init__`: `self.min_ms = float(cfg.get("nm_min_ms", 0))`. Em `TwoStageNM.predict`, trocar `use = nm & ~np.isnan(lines)` por:

```python
        use = line_rows(nm, df[SCHED_GAP], lines, self.min_ms)
```

Em `experiment.py`: acrescentar `ap.add_argument("--nm-min-ms", type=float, default=0.0, help="two_stage_nm: só usa a reta com atraso acima de S segundos")`; em `config()`, no ramo `two_stage_nm`, acrescentar `cfg["nm_min_ms"] = a.nm_min_ms`; logo depois de `a = ap.parse_args()`:

```python
    if a.model != "two_stage_nm" and (a.nm_split_ms or a.nm_min_ms):
        ap.error("--nm-split-ms e --nm-min-ms só valem com --model two_stage_nm")
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/bin/python -m pytest -q` → Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models.py src/experiment.py tests/test_models.py
git commit -m "two_stage_nm: reta só acima do limiar (--nm-min-ms); flags validadas"
git push
```

---

### Task 4: `teto.py` — teto no ranking e candidato

**Files:**
- Create: `src/teto.py`
- Test: `tests/test_teto.py`

**Interfaces:**
- Consumes: `cache.load_split("ranking2026")`, `models.SCHED_GAP`, `features.ID`, `features.TARGET`.
- Produces: `ceiling_gain(base, new, oracle, n_total, base_official) -> float` (s de RMSE, positivo = melhora); CLI `bin/run src/teto.py <base.parquet> <novo.parquet> --oficial-base S [--min-ms S] [--salvar arquivo]`.

- [ ] **Step 1: Teste que falha**

`tests/test_teto.py`:

```python
import numpy as np

from teto import ceiling_gain


def test_teto_so_ganha_se_o_novo_chega_mais_perto_do_oraculo():
    oracle = np.array([10_000.0, 20_000.0])
    base = np.array([5_000.0, 5_000.0])
    assert ceiling_gain(base, oracle, oracle, 1_000, 300.0) > 0
    assert ceiling_gain(base, base, oracle, 1_000, 300.0) == 0
    assert ceiling_gain(oracle, base, oracle, 1_000, 300.0) < 0
    # conta exata: ΔSSE = −(5000² + 15000²) em 1000 voos, a partir de 300 s
    expected = 300 - np.sqrt(max(300**2 - (5000**2 + 15000**2) / 1000, 0))
    assert abs(ceiling_gain(base, oracle, oracle, 1_000, 300.0) - expected) < 1e-9
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/bin/python -m pytest tests/test_teto.py -q` → Expected: FAIL (`No module named 'teto'`).

- [ ] **Step 3: Implementar `src/teto.py`**

```python
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
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/bin/python -m pytest -q` → Expected: 18 passed.

- [ ] **Step 5: Conferir contra o diagnóstico**

Run: `bin/run src/teto.py submissions/outgoing-boat_v3.parquet submissions/outgoing-boat_v4.parquet --oficial-base 338.6742 --min-ms 21600`
Expected: 96 linhas acima de 21600 s; teto ≈ 8,5 s (diagnóstico: −8,5 s).

- [ ] **Step 6: Commit**

```bash
git add src/teto.py tests/test_teto.py
git commit -m "teto.py: teto do ganho no ranking e montagem de candidato"
git push
```

---

### Task 5: Novo campeão de referência, ruído e variante > 6 h

**Files:**
- Modify (gerado): `experiments.jsonl`, `champion.json`

- [ ] **Step 1: Reiniciar o campeão com o código novo**

A v4 (campeã atual) é `FRÁGIL` pela regra nova, e o código mudou (seed/determinismo). A referência passa a ser o dois estágios medido no código novo:

```bash
git rm -q champion.json
```

Run (supervisionado, `prc-exp`): `bin/run src/experiment.py ref_dois_estagios_s0 --model two_stage --seed 0 --nota "referência do plano 3a (código determinístico)"`
Run: `bin/run src/compare.py <id_ref_s0> --promover` → Expected: "ainda não há campeão" e promovido.

- [ ] **Step 2: Ruído do treino (2 seeds a mais)**

Run (um por vez): `bin/run src/experiment.py ref_dois_estagios_s1 --model two_stage --seed 1 --nota "ruído: seed 1"` e o mesmo com `--seed 2` (`ref_dois_estagios_s2`).
Run: `bin/run src/compare.py <id_s1>` e `bin/run src/compare.py <id_s2>` (sem `--promover`).
Expected: vereditos "não comprovado"; anotar |ganho| e o `completo` das 3 seeds (desvio entre seeds = ruído do treino).

- [ ] **Step 3: Variante > 6 h**

Run: `bin/run src/experiment.py nm_retas_6h --model two_stage_nm --nm-min-ms 21600 --seed 0 --nota "diagnóstico v4: reta só em NM ausente com atraso > 6 h"`
Run: `bin/run src/compare.py <id_nm_retas_6h> --promover`
Expected (diagnóstico): ganho ≈ 53 s; sem top-10 ≈ +1,9 (< 10 %); jan IC baixo ≈ +0,2; veredito **FRÁGIL** → não promovido. Anotar todas as linhas.

- [ ] **Step 4: Re-teste da célula de Roma (> 2 h)**

Run: `bin/run src/experiment.py nm_retas_2h_s0 --model two_stage_nm --nm-split-ms --seed 0 --nota "re-teste com regra nova"`
Run: `bin/run src/compare.py <id> --promover` → anotar veredito.

- [ ] **Step 5: Teto e candidato v5 (sem enviar)**

Run: `bin/run src/teto.py submissions/outgoing-boat_v3.parquet submissions/outgoing-boat_v4.parquet --oficial-base 338.6742 --min-ms 21600 --salvar submissions/outgoing-boat_v5.parquet`
Expected: 96 linhas trocadas; teto ≈ 8,5 s; arquivo com 344.841 linhas, sem nulos. **Não enviar.** O controller leva ao usuário: veredito da Step 3, teto, e o critério do diagnóstico para aposta de cauda (teto ≥ 3 s e dano nos voos normais sem NM ≤ 1 s).

- [ ] **Step 6: Commit**

```bash
git add experiments.jsonl champion.json
git commit -m "Plano 3a: referência determinística, ruído de seeds e variante > 6 h"
git push
```

---

### Task 6: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1:** Atualizar:
- **Modelo atual / ciclo:** regra nova do `compare.py` (critérios e veredito `FRÁGIL`, `--aceitar-fragil`), `teto.py`, seeds determinísticas.
- **Roadmap / Plano 3:** marcar regra, seeds (com o ruído medido), variante > 6 h e re-teste 2 h com os números; manter sweep/XGBoost/ablação/mutmut como plano 3b.
- **Submissões:** linha v5 se o usuário aprovar o envio (senão, "candidato não enviado").
- **Uso:** `--seed`, `--nm-min-ms`, `compare.py --aceitar-fragil`, `teto.py`.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "README: plano 3a"
git push
```
