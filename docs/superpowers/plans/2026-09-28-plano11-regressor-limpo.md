# Plano 11 — regressor-base treinado só no que ele atende

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tirar do regressor L2 do `TwoStage` os voos que o distorcem (taxi-outs de horas e Roma sem NM), deixando a cauda para o classificador de cópia, as retas NM e a janela do LOBT.

**Evidência:** kind-mango (README:124, v51) −5,99 s oficial tirando LIRF e y > 80.000 do regressor-base; zestful (`arrival_boost_contest.py:62,73`) −7,11 s oficial com resíduo cortado em ±7200 só no treino (misturado com CatBoost). Pesquisa: `docs/research/2026-09-27-concorrentes.md`. Descarte antigo de "cortar outliers" (v1) era num modelo único, sem classificador nem retas.

**Architecture:** Duas chaves opcionais na config do `TwoStage` (herdadas por `TwoStageNM`): `reg_corte` (segundos; o alvo do regressor vira `min(y, reg_corte)` só no treino) e `reg_sem_lirf_nm` (bool; o regressor não treina com linhas `AIRPORT == "LIRF"` e `nm_missing == 1`). Flags `--reg-corte S` e `--reg-sem-lirf-nm` em `experiment.py`; a config registrada só inclui a chave quando ligada. O classificador, as retas NM, a janela e o corretor não mudam.

## Global Constraints

- Sem as chaves: previsões idênticas às de hoje.
- O corte vale só para o alvo do treino do regressor; previsão, métricas e validação usam o alvo bruto.
- Processo pesado só por `bin/run`; subagentes só pytest sintético; `runlog.Run` em testes com `registry=`/`logs=`.
- Português; commit + push em `main`. Envio só do que melhorar na simulação (sem envios de sondagem) e com ok do usuário.

### Task 1: chaves no `TwoStage` e flags no `experiment.py`

**Files:** `src/models.py`, `src/experiment.py`, `README.md` (linha de `experiment.py` em "Uso"); testes em `tests/test_models.py`, `tests/test_experiment.py`.

- [x] Teste: com `reg_corte=7200`, o regressor recebe alvo ≤ 7200 (capturar o `lgb.Dataset` via monkeypatch de `lgb.train` ou um stub) e o classificador recebe as mesmas linhas de hoje.
- [x] Teste: com `reg_sem_lirf_nm=True`, nenhuma linha LIRF sem NM entra no regressor; linhas LIRF com NM e sem NM de outros aeroportos continuam.
- [x] Teste: `experiment.config` inclui `reg_corte`/`reg_sem_lirf_nm` só quando ligados.
- [x] `pytest -q` verde; commit `models: regressor com alvo cortado e sem LIRF sem NM (opcional)`; push.

### Task 2 (controlador): medir e decidir

- [x] Três bases (≈4 min cada, uma por vez): `--reg-corte 7200`, `--reg-sem-lirf-nm`, as duas; todas com `--model two_stage_nm --nm-min-ms 21600 --seed 0 --janela-lobt`. Comparar com `20260927-133137-janela` (320,29) no `compare.py`; ler também jan e jul separados.
- [x] A melhor, se ganhar: `stack.py v15_cf --crossfit --conjunto --base <id>` → `compare.py` contra a v12 (309,78).
- [x] Se ganhar: `train.py submit 15`, 0 fora da janela, regra de Roma por cima (como a v13), **parar e mostrar ao usuário**.
- [x] Docs: README, CONTEXTO, `saltos.json`, caixas; commit + push.

**Resultado (28/09):** bases `reg_corte` 317,25, `reg_sem_lirf` 318,29, `reg_ambos` 318,00 (contra 320,29). `v15_cf` (v12 com `--reg-corte 7200`) = 309,15 contra 309,78 da v12: +0,6 s (IC −1,6 a 3,1; sem os 10 maiores −2,0) → não compensa gerar e enviar. O corretor já corrigia o que o corte melhora na base.
