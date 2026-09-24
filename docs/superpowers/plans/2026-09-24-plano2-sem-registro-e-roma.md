# Plano 2 — voos sem registro NM e célula de Roma (itens 3–4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduzir o erro dos voos sem registro no Network Manager (1% dos voos, NM ausente 2.442 s na v3) com um modelo linear por aeroporto no atraso `ms = MVT − SCHED`, com variante segmentada que isola a célula LIRF ∧ sem NM ∧ ms > 2 h (44% do erro).

**Architecture:** Novo modelo `two_stage_nm` em `src/models.py`: herda `TwoStage` (inalterado para voos com NM) e troca a previsão dos voos sem NM por retas `y = a + b·ms` ajustadas por grupo (aeroporto, ou aeroporto × faixa de ms) com fallback global. Funções puras `fit_lines`/`apply_lines` testáveis. Antes, dois reparos da revisão final no cache e um teste do rótulo.

**Tech Stack:** Python 3.14, pandas 3, numpy, LightGBM 4.7, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-pipeline-melhoria-continua-design.md` (parte 2, itens 3–4; parte 3, ciclo).

## Global Constraints

- Todo processo pesado roda por `bin/run`, supervisionado (hub `start`), um por vez; LightGBM `num_threads=12`; RAM ≤ 7 GB.
- L2 no alvo bruto; sem log, Huber, corte de outliers ou teto de previsão (piso 0 permitido).
- `--promover` só sem base explícita (compara com o campeão).
- Nada é enviado ao placar sem ok explícito do usuário; o implementador nunca roda `s3.py submit`.
- `experiments.jsonl` e `champion.json` versionados; textos em português.
- Commit + push na branch de trabalho ao fim de cada tarefa.

## Fatos medidos que o plano usa

- v3 (campeão `20260924-185134-dois_estagios`): simulação 388,16; NM ausente 2.442,18; LIRF 957,1; oficial 338,7.
- Forense (docs/research/2026-09-24-forense-dados.md §3, §5, §8): no subgrupo NM ausente, constante 3.962 → LightGBM 2.626 → **linear por aeroporto 1.993**; em (LIRF, NM ausente) `y ~ a + b·ms` tem r = 0,896, inclinação 1,062, intercepto −2.818 s; na célula LIRF ∧ NM ausente ∧ ms > 2 h a regressão linear dedicada chega a ~4.700 (dois estágios: 9.956).

## Estrutura de arquivos

| Arquivo | Ação | Responsabilidade |
|---|---|---|
| `src/cache.py` | alterar | chave inclui `cache.py`; escrita atômica |
| `src/models.py` | alterar | `fit_lines`, `apply_lines`, `TwoStageNM`, `MODELS["two_stage_nm"]` |
| `src/experiment.py` | alterar | flag `--nm-split-ms`; `config()` do novo modelo |
| `tests/test_models.py` | alterar | testes de `copied_from_sched`, `fit_lines`, `apply_lines` |
| `README.md` | alterar | resultados e roadmap |

---

### Task 1: Reparos do cache e teste do rótulo

**Files:**
- Modify: `src/cache.py:51-55` (`cache_key`), `src/cache.py:83-86` (escrita)
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `cache_key(paths)` passa a depender também de `cache.py`; nenhuma mudança de assinatura.

- [ ] **Step 1: Teste do rótulo (falha se o limite de 60 s ou o NaT mudarem)**

Acrescentar a `tests/test_models.py`:

```python
import pandas as pd

from models import copied_from_sched


def test_rotulo_de_copia_usa_60_s_e_trata_nulos_como_normais():
    t = pd.Timestamp("2025-07-01 10:00", tz="UTC")
    s = pd.Timedelta(seconds=1)
    df = pd.DataFrame({
        "SCHED_TIME_UTC_mvt": [t, t, t, t, pd.NaT],
        "BLOCK_TIME_UTC_mvt": [t + 60 * s, t - 60 * s, t + 61 * s, pd.NaT, t],
    })
    assert copied_from_sched(df).tolist() == [True, True, False, False, False]
```

- [ ] **Step 2: Rodar**

Run: `.venv/bin/python -m pytest tests/test_models.py -q`
Expected: `3 passed` (o comportamento já existe; o teste trava o contrato).

- [ ] **Step 3: `cache_key` inclui `cache.py`**

Trocar `cache_key` por:

```python
def cache_key(paths: list[Path]) -> str:
    """Muda quando features.py, cache.py (montagem/filtros) ou os dados mudam."""
    h = hashlib.sha256(Path(F.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    for p in paths:
        h.update(f"{p.name}:{p.stat().st_size}".encode())
    return h.hexdigest()[:12]
```

- [ ] **Step 4: Escrita atômica**

Trocar as linhas de escrita no fim de `load_split` por:

```python
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    df.to_parquet(tmp, compression="zstd", index=False)
    tmp.replace(target)  # arquivo pela metade nunca vira cache válido
    for old in CACHE.glob(f"{name}-*.parquet"):
        if old != target:
            old.unlink()
    return df
```

- [ ] **Step 5: Refazer o cache (supervisionado, nome `prc-cache`)**

Run: `bin/run src/cache.py`
Expected: 4 splits refeitos (a chave mudou), mesmas contagens: train2025 1.740.628; holdout2025 344.419; full2025 2.085.047; ranking2026 344.841. Pico de RAM ≤ 5,5 GB. `data/cache/` com exatamente 4 `.parquet` e nenhum `.tmp`.

- [ ] **Step 6: Suíte e commit**

Run: `.venv/bin/python -m pytest -q` → Expected: 11 passed.
Acrescentar a `tests/test_models.py` (juntar os imports no topo do arquivo):
```bash
git add src/cache.py tests/test_models.py
git commit -m "cache: chave inclui cache.py e escrita atômica; teste do rótulo de cópia"
git push
```

---

### Task 2: Modelo `two_stage_nm` (retas para voos sem NM)

**Files:**
- Modify: `src/models.py` (acrescentar antes de `MODELS`; trocar `MODELS`)
- Modify: `src/experiment.py:50-63` (`config`, argumento novo)
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `TwoStage`, `SCHED_GAP`, `F.AIRPORT`, `F.TARGET`; coluna `nm_missing` (int8) das features.
- Produces:
  - `NM_MIN_ROWS = 50`, `NM_SPLIT_S = 7200`
  - `nm_groups(df, split_ms: bool) -> np.ndarray[str]` (chave `"<aeroporto>"` ou `"<aeroporto>|>2h"`/`"<aeroporto>|<=2h"`)
  - `fit_lines(ms, y, keys, min_rows=NM_MIN_ROWS) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]` (retas por grupo `(a, b)` e reta global)
  - `apply_lines(ms, keys, lines, fallback) -> np.ndarray` (piso 0; `ms` nulo → NaN)
  - `TwoStageNM(cfg)` com cfg `cls_rounds`, `reg_rounds`, `nm_split_ms: bool`; `best_iter = None`
  - `MODELS["two_stage_nm"]`; CLI `--model two_stage_nm [--nm-split-ms]`

- [ ] **Step 1: Testes que falham**

Acrescentar a `tests/test_models.py`:

```python
import numpy as np

from models import apply_lines, fit_lines


def test_retas_por_grupo_recuperam_a_relacao_e_usam_fallback():
    ms = np.array([0.0, 1000.0, 2000.0, 3000.0] * 20 + [0.0, 1000.0])
    keys = np.array(["LIRF"] * 80 + ["EDDF"] * 2)
    y = np.where(keys == "LIRF", -2818 + 1.062 * ms, 900.0)
    lines, fallback = fit_lines(ms, y, keys, min_rows=50)
    assert set(lines) == {"LIRF"}  # EDDF tem 2 linhas: usa a reta global
    a, b = lines["LIRF"]
    assert abs(a + 2818) < 1e-6 and abs(b - 1.062) < 1e-9
    out = apply_lines(np.array([3000.0, 1000.0, np.nan]), np.array(["LIRF", "EDDF", "LIRF"]),
                      lines, fallback)
    assert abs(out[0] - (-2818 + 1.062 * 3000)) < 1e-6
    assert out[1] == max(0.0, fallback[0] + fallback[1] * 1000.0)
    assert np.isnan(out[2])


def test_retas_nunca_preveem_negativo():
    lines = {"LIRF": (-2818.0, 1.062)}
    assert apply_lines(np.array([0.0]), np.array(["LIRF"]), lines, (0.0, 0.0)).tolist() == [0.0]
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/bin/python -m pytest tests/test_models.py -q`
Expected: FAIL (`cannot import name 'apply_lines'`).

- [ ] **Step 3: Implementar em `src/models.py`**

Acrescentar antes de `MODELS = ...` e trocar a linha `MODELS`:

```python
NM_MIN_ROWS = 50  # grupos menores usam a reta global
NM_SPLIT_S = 7200  # célula de Roma: atraso > 2 h


def nm_groups(df: pd.DataFrame, split_ms: bool) -> np.ndarray:
    """Chave da reta: aeroporto, ou aeroporto × (atraso > 2 h)."""
    keys = df[F.AIRPORT].astype(str).to_numpy()
    if not split_ms:
        return keys
    late = np.where(df[SCHED_GAP].to_numpy(float) > NM_SPLIT_S, ">2h", "<=2h")
    return np.char.add(np.char.add(keys.astype(str), "|"), late)


def _line(ms: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    b, a = np.polyfit(ms, y, 1)
    return float(a), float(b)


def fit_lines(ms, y, keys, min_rows: int = NM_MIN_ROWS):
    """Mínimos quadrados y = a + b·ms por grupo (L2, alinhado ao RMSE) e reta global."""
    ms, y, keys = np.asarray(ms, float), np.asarray(y, float), np.asarray(keys)
    ok = ~np.isnan(ms) & ~np.isnan(y)
    ms, y, keys = ms[ok], y[ok], keys[ok]
    lines = {
        str(k): _line(ms[keys == k], y[keys == k])
        for k in np.unique(keys)
        if (keys == k).sum() >= min_rows
    }
    return lines, _line(ms, y)


def apply_lines(ms, keys, lines: dict, fallback: tuple[float, float]) -> np.ndarray:
    ms, keys = np.asarray(ms, float), np.asarray(keys)
    a = np.array([lines.get(str(k), fallback)[0] for k in keys])
    b = np.array([lines.get(str(k), fallback)[1] for k in keys])
    return np.clip(a + b * ms, 0, None)  # NaN em ms continua NaN


class TwoStageNM(TwoStage):
    """Dois estágios para voos com NM; reta por aeroporto no atraso para voos sem NM."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.split_ms = bool(cfg.get("nm_split_ms", False))

    def fit(self, train, cols, run=None, valid=None) -> "TwoStageNM":
        super().fit(train, cols, run=run, valid=valid)
        nm = train[train["nm_missing"] == 1]
        self.lines, self.fallback = fit_lines(
            nm[SCHED_GAP], nm[F.TARGET], nm_groups(nm, self.split_ms)
        )
        if run:
            run.log(f"retas NM ausente: {len(self.lines)} grupos · global a={self.fallback[0]:.0f} b={self.fallback[1]:.3f}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = super().predict(df)
        nm = (df["nm_missing"] == 1).to_numpy()
        lines = apply_lines(df[SCHED_GAP], nm_groups(df, self.split_ms), self.lines, self.fallback)
        use = nm & ~np.isnan(lines)
        pred[use] = lines[use]
        return pred


MODELS = {"single": SingleLGBM, "two_stage": TwoStage, "two_stage_nm": TwoStageNM}
```

- [ ] **Step 4: CLI em `src/experiment.py`**

Trocar `config` por:

```python
def config(a: argparse.Namespace) -> dict:
    if a.model == "single":
        return {"model": a.model, "rounds": a.rounds}
    cfg = {"model": a.model, "cls_rounds": a.cls_rounds, "reg_rounds": a.reg_rounds}
    if a.model == "two_stage_nm":
        cfg["nm_split_ms"] = a.nm_split_ms
    return cfg
```

E acrescentar depois de `ap.add_argument("--reg-rounds", ...)`:

```python
    ap.add_argument("--nm-split-ms", action="store_true",
                    help="two_stage_nm: retas separadas para atraso > 2 h (célula de Roma)")
```

No docstring do módulo, acrescentar a linha:

```
    bin/run src/experiment.py <nome> --model two_stage_nm [--nm-split-ms]
```

- [ ] **Step 5: Rodar e ver passar**

Run: `.venv/bin/python -m pytest -q`
Expected: 13 passed.

- [ ] **Step 6: Commit**

```bash
git add src/models.py src/experiment.py tests/test_models.py
git commit -m "Modelo two_stage_nm: retas por aeroporto no atraso para voos sem NM"
git push
```

---

### Task 3: Experimentos, decisão e v4 (sem enviar)

**Files:**
- Modify (gerado): `experiments.jsonl`, `champion.json`

**Interfaces:**
- Consumes: `experiment.py --model two_stage_nm [--nm-split-ms]`, `compare.py`, `train.py submit`.

- [ ] **Step 1: Re-medir o campeão com o código atual (supervisionado, `prc-exp`)**

O código mudou desde a v3, então o campeão é re-medido para a comparação pareada ser justa:

Run: `bin/run src/experiment.py dois_estagios_r --model two_stage --cls-rounds 400 --reg-rounds 400 --nota "re-medida do campeão no código do plano 2"`
Expected: `completo` 388 ± 3 (mesma config e dados; bagging com seed fixa do LightGBM).

Run: `bin/run src/compare.py <id_dois_estagios_r>` → Expected: ganho ≈ 0, não comprovado (sem `--promover`).

- [ ] **Step 2: Variante A — reta por aeroporto**

Run: `bin/run src/experiment.py nm_retas --model two_stage_nm --nota "item 3: reta por aeroporto no atraso para NM ausente"`
Run: `bin/run src/compare.py <id_nm_retas> --promover`
Expected: `nm_ausente` < 2.442 (forense: ~2.000); registrar o veredito.

- [ ] **Step 3: Variante B — reta por aeroporto × atraso > 2 h**

Run: `bin/run src/experiment.py nm_retas_2h --model two_stage_nm --nm-split-ms --nota "item 4: retas separadas para atraso > 2 h (célula de Roma)"`
Run: `bin/run src/compare.py <id_nm_retas_2h> --promover`
Expected: compara contra o campeão vigente (a v3 ou a A, se promovida); registrar o veredito e o RMSE de LIRF.

Se nenhuma variante for `MELHOR`: não gerar v4; registrar os números no relatório e parar (o controller decide com o usuário).

- [ ] **Step 4: Gerar a v4 se houve promoção (supervisionado, `prc-submit`)**

Run: `bin/run src/train.py submit 4`
Expected: `submissions/outgoing-boat_v4.parquet`, 344.841 linhas, sem nulos, `preenchidas com a mediana: 0`. **Não enviar.** Se abortar por `src_hash` diferente do campeão, é bug do procedimento (o campeão foi medido neste código): parar e relatar.

- [ ] **Step 5: Commit**

```bash
git add experiments.jsonl champion.json
git commit -m "Plano 2: experimentos das retas para voos sem NM"
git push
```

---

### Task 4: Documentação

**Files:**
- Modify: `README.md` (Modelo atual, Roadmap, Submissões, Uso)

- [ ] **Step 1: README**

- **Modelo atual:** se promovido, descrever `two_stage_nm` (retas `y = a + b·(MVT − SCHED)` por aeroporto — ou aeroporto × atraso > 2 h — para voos sem registro NM; grupos com < 50 voos usam a reta global; piso 0).
- **Roadmap:** marcar itens 3 e 4 com os números medidos (simulação completa, NM ausente, LIRF, ganho ± IC, veredito); próximo = item 5 (alvo residual sobre `MVT − AOBT_3`).
- **Submissões:** linha da v4 se gerada, com a simulação e "aguardando aprovação/envio".
- **Uso:** acrescentar `bin/run src/experiment.py <nome> --model two_stage_nm [--nm-split-ms]`.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "README: resultados do plano 2"
git push
```
