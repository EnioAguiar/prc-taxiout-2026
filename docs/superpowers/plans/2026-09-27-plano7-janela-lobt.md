# Plano 7 — janela do LOBT (|BLOCK − LOBT| ≤ 3606 s)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Usar a regra exata do dado — em 100 % das 2.062.577 DEP de 2025 com `LOBT_flt`, |BLOCK − LOBT| ≤ 3606 s — para limitar previsões e o classificador de cópia do SCHED.

**Architecture:** `models.janela_lobt(df) -> (lo, hi)` dá o intervalo de `y`: `lo = MVT − LOBT − 3606`, `hi = MVT − LOBT + 3606` (NaN sem LOBT). Com `cfg["janela_lobt"]` verdadeiro: (a) `TwoStage.predict` zera `p` onde o SCHED cai fora da janela (a cópia é impossível); (b) toda previsão final de `build_model(cfg)` é projetada em `[lo, hi]` e depois recebe piso 0. O corretor do `stack.py` ganha `dist_lo = pred − lo` e `dist_hi = hi − pred` e sua saída é projetada na mesma janela quando a base usa `janela_lobt`.

**Tech Stack:** Python 3.14, pandas 3, numpy, LightGBM 4.7, pytest (`.venv/bin/python -m pytest -q`).

**Spec:** `docs/superpowers/specs/2026-09-24-pipeline-melhoria-continua-design.md`; evidência em `docs/research/2026-09-27-janela-lobt.md` (Task 1 grava).

## Global Constraints

- Constante `JANELA_LOBT_S = 3606`. Sem `LOBT_flt` (ou sem `MVT`), a janela não existe: a previsão fica como está.
- `janela_lobt` ausente ou falso: previsões **idênticas** às de hoje (campeã e corridas antigas reproduzem).
- Ordem na saída: projeção em `[lo, hi]`, depois piso 0 (`y` real pode ser −12 s; o piso 0 continua valendo).
- Zerar `p` só quando o SCHED existe e `|SCHED − LOBT| > 3606`; a combinação por esperança continua a mesma no resto.
- Config registrada inclui `janela_lobt: true` só quando ligada.
- Processo pesado só por `bin/run`, um por vez; subagentes só rodam pytest sintético; `runlog.Run` em testes/smoke recebe `registry=`/`logs=` (nunca gravar em `experiments.jsonl`, `logs/`, `runs/` reais).
- Português; commit + push em `main` por tarefa. Nada é enviado ao placar sem ok do usuário.

---

### Task 1: janela no modelo base + nota de pesquisa

**Files:**
- Modify: `src/models.py`, `src/experiment.py` (flag `--janela-lobt`, `config`)
- Create: `docs/research/2026-09-27-janela-lobt.md`
- Test: `tests/test_models.py`, `tests/test_experiment.py`

**Interfaces:**
- Produces: `JANELA_LOBT_S`; `janela_lobt(df) -> tuple[np.ndarray, np.ndarray]`; `limitar_janela(pred, df) -> np.ndarray` (projeção + piso 0); `build_model(cfg)` aplica `limitar_janela` em `predict` quando `cfg.get("janela_lobt")` (vale também para `SeedAvg`: limitar a média, uma vez).

- [x] Teste: `janela_lobt` com MVT 10:00, LOBT 09:40 → lo = 1200 − 3606, hi = 1200 + 3606; sem LOBT → NaN.
- [x] Teste: `limitar_janela` — previsão acima de `hi` vira `hi`; abaixo de `lo` vira `lo`; dentro fica; sem LOBT fica; negativa vira 0 depois da projeção.
- [x] Teste: `TwoStage` com `janela_lobt` e um voo com SCHED 2 h depois do LOBT e classificador que devolve p = 1 → a previsão não é `MVT − SCHED` (p zerado); com SCHED dentro da janela, p segue. Use modelos LightGBM minúsculos ou substitua `cls`/`reg` por stubs com `predict`.
- [x] Teste: `janela_lobt` ausente → `build_model(cfg).predict` idêntico ao de hoje (mesmo objeto de classe, sem projeção).
- [x] Nota de pesquisa com os números: 100 % de |BLOCK−LOBT| ≤ 3606 nas 2.062.577 DEP com LOBT (máx 3606); IOBT 99,997 %, EOBT_1 99,970 %, AOBT_3 99,914 %, SCHED 94,844 %; sem LOBT 22.470 DEP (22.424 sem NM), 78,6 % do Σy² da cauda y > 1 h; v6 no holdout 323,50 → 320,30 limitada (79 linhas); v7_cf 320,67 → 317,98 (53); ranking v6: 117 linhas fora, Σ(p−q)²/N = 16.125 → limite oficial ≤ √(314,76² − 16.125) = 288,0 se a regra valer em 2026; fonte: elegant-alligator (`scripts/queue_features.py:68,82`, assert no treino).
- [x] `pytest -q` verde; commit `models: janela do LOBT (projeção e p zerado fora da janela)`; push.

**Acceptance:** testes passam; `experiment.py --help` mostra `--janela-lobt`.

### Task 2: janela no corretor e no envio

**Files:**
- Modify: `src/stack.py`, `src/train.py`
- Test: `tests/test_stack.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: `models.janela_lobt`, `models.limitar_janela`.
- Produces: `corrector_frame(df, pred, adsb=True, janela=False)` — com `janela`, colunas `dist_lo = pred − lo`, `dist_hi = hi − pred` (NaN sem LOBT). `stack.py --crossfit` liga `janela` quando `base_config["janela_lobt"]`; a saída do corretor passa por `limitar_janela`. `train.py` (rota `stack_cf`) faz o mesmo, lendo do `base_config`.

- [x] Teste: `corrector_frame(..., janela=True)` tem `dist_lo`/`dist_hi` com os valores certos para um voo sintético; `janela=False` não tem.
- [x] Teste: saída do corretor com correção enorme é projetada em `[lo, hi]` quando a base tem `janela_lobt` (teste na função que monta a saída, não no `main`).
- [x] `pytest -q` verde; commit `stack/train: janela do LOBT no corretor`; push.

**Acceptance:** testes passam; sem `janela_lobt` na base, saídas idênticas às de hoje.

### Task 3 (controlador): medir, decidir, documentar

- [ ] `bin/run src/experiment.py janela --model two_stage_nm --nm-min-ms 21600 --seed 0 --janela-lobt` → `compare.py` contra a v6 (323,50).
- [ ] `stack.py v9_cf --crossfit --base <id janela>` (via `systemd-run --user`, dura ~16 min) → `compare.py` contra a v6 e a v7.
- [ ] Limite garantido no ranking: projeção da v6 enviada (≤ 288,0) e, se promovida, `train.py submit 9` + `teto.py` contra a v6. **Parar e mostrar ao usuário.**
- [ ] Docs: README (Modelo atual, Uso, Roadmap), CONTEXTO, `saltos.json`, caixas deste plano; commit + push.

## Ordem

1 → 2 → 3. Prazo 11/10/2026 23:59:59 CET.
