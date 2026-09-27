# Plano 9 — conjunto de corretores (global, por aeroporto, CatBoost)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trocar o corretor único do `stack_cf` pela média de três corretores treinados nas mesmas entradas.

**Evidência (teste barato de 27/09, corretor fora do fold por dia, base `20260927-133137-janela` + `ctx_*`):** global 309,27 · global + por aeroporto 307,11 · global + CatBoost 307,21 · **os três 305,77 (−3,5 s)**; normais com NM 201,78 → 200,67; sem loteria 247,08 → 245,84. CatBoost sozinho 309,53 e por aeroporto sozinho 309,89: o ganho vem da média. Fontes: GREKI (Discord 27/09, "2ª família sobre a mesma entrada"), jubilant-vase (−3,8 s oficial com modelos por aeroporto 50/50).

**Architecture:** Em `src/stack.py`, `fit_corrector(X, y, base, conjunto=False)` devolve o booster de hoje com `conjunto=False` ou um objeto `Conjunto` com `.predict(X)` = média de: (1) LightGBM global (`PARAMS`, `ROUNDS`, o de hoje); (2) um LightGBM por aeroporto (`PARAMS` com `min_data_in_leaf=100`, `ROUNDS`; aeroporto sem modelo usa o global); (3) CatBoost (`iterations=800, depth=8, learning_rate=0.08, loss_function="RMSE", random_seed=0`, GPU se disponível, senão CPU com `thread_count=12`; `AIRPORT` como categórica em texto). Todos aprendem `y − base`. `apply_corrector`/`previsao_corrigida` não mudam de contrato. A chave `corretor: "conjunto"` na config `stack_cf` liga o conjunto; `stack.py --conjunto` a grava; `train.py` lê da config da campeã.

**Tech Stack:** Python 3.14, LightGBM 4.7, CatBoost 1.2.10 (já instalado no `.venv`; acrescentar `catboost` ao `requirements.txt`), pytest.

## Global Constraints

- Sem `corretor` na config (ou `--conjunto` ausente): saídas idênticas às de hoje.
- Média simples dos três (pesos iguais; nada ajustado no holdout).
- Ordem na saída continua: base + correção → janela do LOBT (se a base tem `janela_lobt`) → piso 0.
- CatBoost determinístico com `random_seed=0`; GPU via `task_type="GPU"` quando `catboost.utils.get_gpu_device_count() > 0`.
- Processo pesado só por `bin/run`; subagentes só pytest sintético (CatBoost em CPU com poucas iterações nos testes); `runlog.Run` com `registry=`/`logs=`.
- Português; commit + push em `main`. Nada é enviado ao placar sem ok do usuário.

---

### Task 1: `Conjunto` no corretor

**Files:** Modify `src/stack.py`, `src/train.py`, `requirements.txt`, `README.md` (só a linha de `stack.py` em "Uso": acrescentar `[--conjunto]`). Test: `tests/test_stack.py`, `tests/test_train.py`.

- [x] Teste: `Conjunto.predict` = média exata dos três submodelos (stubs com `predict` constante 10, 20, 60 → 30).
- [x] Teste: aeroporto visto só na previsão usa o modelo global na parte "por aeroporto".
- [x] Teste: sem `corretor` na config, `fit_corrector` devolve o mesmo tipo de hoje (`lgb.Booster`) e `train.py` segue o caminho antigo; com `"conjunto"`, `train.py` usa `Conjunto`.
- [x] Teste: config gravada por `stack.py --crossfit --conjunto` tem `corretor: "conjunto"`; sem a flag não tem a chave.
- [x] `pytest -q` verde; commit `stack: conjunto de corretores (global, por aeroporto, CatBoost)`; push.

### Task 2 (controlador): medir, enviar, documentar

- [x] `stack.py v12_cf --crossfit --conjunto --base 20260927-133137-janela` (systemd-run) → `compare.py` contra a v11 (311,09).
- [x] Se ganhar: promover, `train.py submit 12`, 0 linhas fora da janela, **parar e mostrar ao usuário**.
- [x] Docs: README, CONTEXTO, `saltos.json`, caixas; commit + push.

**Resultado (27/09):** `v12_cf` (`--conjunto`, 18 min, pico 7,2 GB) = 309,78 contra 311,09 da v11: ganho 1,3 s (IC 0,5 a 2,2; jan 0,5, jul 2,1; normais com NM 1,5). O teste barato prometia 3,5 s. Não promovida; fica como candidata para o envio final (custo do envio ~2× o da v11).
