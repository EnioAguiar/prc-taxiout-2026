# Plano 5 — v7: corretor empilhado com cross-fitting por mês

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tornar enviável o corretor do `stack.py` (317,57 na simulação contra 323,50 da campeã v6), com previsões da base fora do bloco no ano inteiro.

**Architecture:** Novo split `blind2025` (12 meses montados como o ranking). Novo `src/crossfit.py` gera previsões da base fora do bloco (blocos de 2 meses consecutivos, P10 refeita por bloco). `stack.py --crossfit` mede a rota na simulação e grava uma corrida `stack_cf`; `train.py submit` sabe gerar o envio quando a campeã é `stack_cf`.

**Tech Stack:** Python 3.14, pandas 3, numpy, LightGBM 4.7, pytest (`.venv/bin/python -m pytest -q`).

**Spec:** `docs/superpowers/specs/2026-09-24-pipeline-melhoria-continua-design.md`, Parte 5.

## Global Constraints

- Blocos: meses consecutivos 2 a 2, derivados dos meses presentes (ordenados); nenhum mês fixo no código. `train2025` (meses 2–6, 8–12) → {2,3} {4,5} {6,8} {9,10} {11,12}; `full2025` → {1,2} {3,4} {5,6} {7,8} {9,10} {11,12}. Número ímpar de meses: o último bloco fica com 1 mês.
- Previsões fora do bloco só em linhas do `blind2025` (BLOCK e alvo apagados nas DEP antes das features); o treino de cada bloco usa as linhas normais (`train2025`/`full2025`) dos outros meses.
- Em cada bloco: `prepare(treino_do_bloco, [cegas_do_bloco])` em cópias — P10 só com os meses de treino do bloco; `leaky_columns` contra o ranking como no `experiment.py`.
- Rodadas: base dos blocos com a config da base sem escala; só a base final do envio usa × 1,2 (`ROUNDS_SCALE`).
- Corretor: o mesmo do `stack.py` atual (`PARAMS`, `ROUNDS = 300`, entradas `pred`, aeroporto, `nm_missing`, `hour`, `to_takeoff_from_*`, `adsb_*`, `adsb_menos_pred`); saída `pred + correção` com piso 0.
- Nome da config: `{"model": "stack_cf", "base": <id>, "base_config": <config da base>, "adsb": true, "rounds": 300, "seed": 0}`.
- Processo pesado só por `bin/run`, um por vez; RSS ≤ 7 GB. Subagentes **não** rodam `bin/run` de treino/cache (quem roda é o controlador); testes com dados sintéticos.
- L2 no alvo bruto; piso 0. Nada é enviado ao placar sem ok explícito do usuário.
- Português (código, docstrings, commits); commit + push em `main` por tarefa. Testes só de comportamento (sem testar texto/fiação).

---

### Task 1: split `blind2025`

**Files:**
- Modify: `src/cache.py` (`SPLITS`, `split_paths`, `load_split`)
- Test: `tests/test_cache.py`

**Interfaces:**
- Produces: `load_split("blind2025")` → DataFrame de todos os meses de 2025 montados com `build_blind` (ARR + DEP como o `holdout2025`; `y_true` nas DEP), com as colunas `adsb_*` de `add_features`.

- [ ] Teste: `split_paths("blind2025")` devolve os 12 arquivos de `training_2025-*` (use `monkeypatch` em `cache.DATA` com 12 arquivos vazios de nome real, como o teste existente fizer; se não houver padrão, crie-os em `tmp_path`).
- [ ] Teste: `load_split` monta `blind2025` com `build_blind` (monkeypatch de `F.load`/`F.build` com um DataFrame sintético mínimo de 2 DEP + 1 ARR; conferir BLOCK nulo nas DEP antes do `build` e `y_true` igual ao alvo original). Ver o estilo de `tests/test_cache.py`.
- [ ] Implementar: `case "blind2025": return training` em `split_paths`; em `load_split`, `build_blind` para `name in ("holdout2025", "blind2025")`; `SPLITS` inclui `"blind2025"`; `__main__` do `cache.py` também o monta.
- [ ] `pytest -q` verde; commit `cache: split blind2025 (ano inteiro montado como o ranking)`; push.

**Acceptance:** testes novos passam; nenhuma mudança no comportamento dos outros splits.

### Task 2: `src/crossfit.py` — previsões fora do bloco

**Files:**
- Create: `src/crossfit.py`
- Test: `tests/test_crossfit.py`

**Interfaces:**
- Consumes: `models.MODELS`, `models.prepare`, `models.leaky_columns`, `features` (`F.ID`, `F.TARGET`), coluna de mês a partir de `MVT_TIME_UTC_mvt`.
- Produces:
  - `month_blocks(months) -> dict[int, int]`: mês → índice do bloco (meses únicos ordenados, agrupados 2 a 2).
  - `oof_base(cfg: dict, train: pd.DataFrame, blind: pd.DataFrame, ranking_cols_ref: pd.DataFrame, run=None) -> pd.DataFrame`: para cada bloco, treina `MODELS[cfg["model"]](cfg)` em `train` sem os meses do bloco e prevê as linhas **DEP com `y_true` não nulo** de `blind` nos meses do bloco. `ranking_cols_ref` é o `ranking2026` (só para `leaky_columns`). Devolve `DataFrame` com `MVT_ID_mvt`, `dia` (`%Y-%m-%d`), `mes`, `y_true`, `pred`, uma linha por DEP cega, ordem estável por `MVT_ID_mvt`.
  - Só considera meses presentes em `train` (a simulação passa `train2025` e usa só os meses dele de `blind`).

```python
def month_blocks(months) -> dict[int, int]:
    uniq = sorted({int(m) for m in months})
    return {m: i // 2 for i, m in enumerate(uniq)}
```

Núcleo de `oof_base` (por bloco, em cópias, porque `prepare` muta os frames):

```python
tr = train[~train_mes.isin(meses_k)].copy()
te = blind[blind_mes.isin(meses_k) & blind["y_true"].notna()].copy()
cols = prepare(tr, [te])
cols = [c for c in cols if c not in leaky_columns(tr, ranking_cols_ref, cols)]
model = MODELS[cfg["model"]](cfg).fit(tr, cols, run=run)
pred = model.predict(te)
```

Com `run`, uma fase/linha de log por bloco (`bloco k/K meses [...]`), `del tr, te, model` no fim de cada bloco (RSS).

- [ ] Teste `month_blocks`: `[2,3,4,5,6,8,9,10,11,12]` → {2,3}{4,5}{6,8}{9,10}{11,12}; 1..12 → 6 blocos; meses repetidos/desordenados dão o mesmo resultado; 3 meses → último bloco com 1.
- [ ] Teste sem vazamento: registrar em `MODELS` (monkeypatch) um modelo falso cujo `fit` guarda os meses vistos e cujo `predict` devolve a média do alvo do treino; com um `train`/`blind` sintético de 4 meses, conferir que o modelo que previu cada linha nunca viu o mês dela e que todo DEP cego com `y_true` recebeu exatamente uma previsão.
- [ ] Teste da P10 por bloco: com o modelo falso devolvendo `ref_p10` como previsão, alterar só o alvo dos meses de um bloco **não** altera as previsões das linhas cegas desse bloco.
- [ ] Implementar; `pytest -q` verde; commit `crossfit: previsões da base fora do bloco por meses`; push.

**Acceptance:** os três testes passam; nenhum treino real rodado.

### Task 3: `stack.py --crossfit` (simulação)

**Files:**
- Modify: `src/stack.py`
- Test: `tests/test_stack.py` (novo)

**Interfaces:**
- Consumes: `crossfit.oof_base`, `load_split("train2025"|"blind2025"|"holdout2025"|"ranking2026")`, `runs/<base>.parquet` (colunas `MVT_ID_mvt`, `dia`, `y_true`, `pred`), `experiments.jsonl` (config da base), `experiment.metrics`.
- Produces (usado pela Task 4):
  - `corrector_frame(df: pd.DataFrame, pred: np.ndarray, adsb: bool = True) -> pd.DataFrame` — as entradas do corretor (extraído do `main` atual, sem mudar colunas).
  - `fit_corrector(X: pd.DataFrame, y: np.ndarray, base: np.ndarray) -> lgb.Booster` — `lgb.train(PARAMS, Dataset(X, y − base), ROUNDS)`.
  - `apply_corrector(model, X, base) -> np.ndarray` — `clip(base + model.predict(X), 0)`.
  - `oof_correction` passa a usar `fit_corrector`/`apply_corrector` (mesmo resultado).
  - `base_config(base_id) -> dict`: config da corrida `base_id` no `experiments.jsonl`.

Fluxo `--crossfit`: config `stack_cf` (Global Constraints) → `oof_base(base_cfg, train2025, blind2025, ranking2026, run)` → gravar `runs/<run.id>_oof.parquet` → `X_oof = corrector_frame(blind indexado pelos IDs do oof, oof.pred)` → `fit_corrector` → holdout: `corrector_frame(hold na ordem de runs/<base>.parquet, base.pred)` → `apply_corrector` → `metrics` → `runs/<run.id>.parquet` e `run.metric`/`run.set(previsoes=…, oof=…)`. Sem `--crossfit`, o comportamento atual fica idêntico.

- [ ] Teste: `apply_corrector` com um corretor que devolve −10⁶ dá previsão 0 (piso), e com correção 0 devolve a base.
- [ ] Teste: `corrector_frame(..., adsb=False)` não tem nenhuma coluna `adsb_*`; com `adsb=True` tem `adsb_menos_pred = adsb_taxi_move − pred`.
- [ ] Refatorar e implementar `--crossfit`; atualizar a docstring do módulo; `pytest -q` verde.
- [ ] Commit `stack: modo --crossfit (corretor treinado fora do bloco no ano)`; push.

**Acceptance:** testes passam; `bin/run src/stack.py --help` mostra `--crossfit`. (O controlador roda a corrida real.)

### Task 4: `train.py submit` para campeã `stack_cf`

**Files:**
- Modify: `src/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `crossfit.oof_base`, `stack.corrector_frame`/`fit_corrector`/`apply_corrector`, `final_config`.
- Produces: `final_config(champ)` para `stack_cf` devolve a config com `base_config` escalado (rodadas da base final × 1,2) e o resto intocado; `submit(N)` com campeã `stack_cf`:
  1. `oof = oof_base(champ["config"]["base_config"] (sem escala), full2025, blind2025, ranking2026, run)`;
  2. corretor em `corrector_frame(blind nos IDs do oof, oof.pred)`;
  3. base final = `MODELS[...](base_config escalado).fit(full2025, cols)` → `pred_rk`;
  4. `apply_corrector(corretor, corrector_frame(rk, pred_rk), pred_rk)` → `build_submission`.
  Grava as previsões fora do bloco em `submissions/<TEAM>_vN_oof.parquet` (ignorado pelo git) para diagnóstico.

- [ ] Teste: `final_config` com campeã `stack_cf` (`base_config` com `cls_rounds 400`, `reg_rounds 400`) → `base_config` com 480/480, `rounds` do corretor (300) intocado; a config da campeã original não é mutada.
- [ ] Teste: `final_config` de campeã `two_stage_nm` continua igual ao de hoje (o teste existente cobre; manter verde).
- [ ] Implementar; `pytest -q` verde; commit `train: envio da campeã stack_cf`; push.

### Task 5 (controlador): medir, decidir e documentar

- [ ] `bin/run src/cache.py` (monta `blind2025`; conferir RSS no log).
- [ ] `bin/run src/stack.py v7_cf --crossfit` → `bin/run src/compare.py <id>`; também comparar contra `stack_adsb_sobre_adsb` (317,57) para ver o custo/ganho de treinar o corretor em 10 meses.
- [ ] Se MELHOR (ou FRÁGIL com teto e ok): `--promover`, `train.py submit 7`, `teto.py submissions/outgoing-boat_v6.parquet submissions/outgoing-boat_v7.parquet --oficial-base 314.76`. **Parar e mostrar ao usuário** simulação, ganho ± IC (completo, normais, sem loteria), teto e nota projetada.
- [ ] Docs: README (Modelo atual, Uso, Estrutura com `crossfit.py`, Roadmap), CONTEXTO "Retomar", `saltos.json` (`v7_stack` com ganho real), caixas deste plano; commit + push.

## Ordem

1 → 2 → 3 → 4 → 5. Prazo 11/10/2026 23:59:59 CET.
