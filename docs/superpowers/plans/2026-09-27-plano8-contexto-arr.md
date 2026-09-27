# Plano 8 — contexto das chegadas e vizinhos no corretor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Levar ao corretor da v9 o taxi-in das chegadas (linhas ARR, 100 % preenchidas também no ranking) e a média de `MVT − AOBT_3` das decolagens vizinhas, passadas e futuras.

**Evidência (teste barato de 27/09, `.superpowers/arr_viz_ref.py`, corretor fora do fold por dia sobre a base `20260927-133137-janela`):** base 314,20 → +ARR 311,26 → +vizinhos 313,92 → **+ambos 310,68 (−3,5 s)**; normais com NM 205,79 → 204,00; sem loteria 253,48 → 249,04. zestful-fountain reporta −2,6 s (ARR) e −2,8 s (vizinhos) oficiais.

**Architecture:** Novo `src/contexto.py` calcula, a partir dos movimentos brutos de um período (DEP + ARR), 20 colunas `ctx_*` por `MVT_ID_mvt` de DEP. `cache.load_split` junta essas colunas às DEP antes de gravar o cache (todas as divisões, inclusive `blind2025`, `holdout2025` e `ranking2026`). O prefixo `ctx_` fica fora de `features.feature_columns`, então a base não muda; só o corretor (`stack.corrector_frame`) passa a usá-las.

**Tech Stack:** Python 3.14, pandas 3, numpy, LightGBM 4.7, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-pipeline-melhoria-continua-design.md`; pesquisa `docs/research/2026-09-27-concorrentes.md`.

## Global Constraints

- Só dados do organizador; nada de alvo das DEP: `TAXITIME_SEC_mvt`/`BLOCK_TIME_UTC_mvt` das **DEP** nunca entram (nas divisões cegas já estão apagados; nas de treino, as funções não podem lê-los). Das ARR usa-se `TAXITIME_SEC_mvt` (existe no ranking).
- Tempos em segundos epoch a partir das colunas datetime já convertidas por `features.load`.
- Colunas (nomes exatos): `ctx_arr_tin_apt_15`, `ctx_arr_tin_apt_60`, `ctx_arr_n_apt_15`, `ctx_arr_n_apt_60`, `ctx_arr_tin_rwy_15`, `ctx_arr_tin_rwy_60`, `ctx_arr_n_rwy_15`, `ctx_arr_n_rwy_60`, `ctx_stand_ult_idade`, `ctx_stand_ult_tin`, `ctx_viz_pas_apt_15`, `ctx_viz_pas_apt_60`, `ctx_viz_fut_apt_15`, `ctx_viz_fut_apt_60`, `ctx_viz_pas_rwy_15`, `ctx_viz_pas_rwy_60`, `ctx_viz_fut_rwy_15`, `ctx_viz_fut_rwy_60`, `ctx_viz_proprio_menos_pas`, `ctx_viz_proprio_menos_fut`.
- Definições:
  - ARR: aeroporto = `ADES_mvt`; taxi-in = `TAXITIME_SEC_mvt` cortado em [0, 3600]; instante = `MVT_TIME_UTC_mvt` (pouso). `ctx_arr_tin_{apt,rwy}_W` = média do taxi-in das ARR do mesmo aeroporto (ou aeroporto × `RUNWAY_mvt`) com pouso em (MVT_dep − W min, MVT_dep]; `ctx_arr_n_*` = contagem (0 se nenhuma; a média fica NaN).
  - `ctx_stand_ult_idade` = MVT_dep − in-block (`BLOCK_TIME_UTC_mvt` da ARR) da última ARR no mesmo aeroporto × `STAND_mvt` com in-block ≤ MVT_dep; NaN se > 86400 s ou inexistente. `ctx_stand_ult_tin` = taxi-in dessa ARR.
  - Proxy da DEP = `MVT − AOBT_3` em segundos se em [0, 7200], senão NaN. `ctx_viz_pas_{apt,rwy}_W` = média do proxy das **outras** DEP do mesmo aeroporto (ou aeroporto × pista) com MVT em [MVT − W min, MVT); `ctx_viz_fut_*` = com MVT em (MVT, MVT + W min]. `ctx_viz_proprio_menos_{pas,fut}` = proxy próprio − `ctx_viz_{pas,fut}_apt_60`.
- Base inalterada: `features.feature_columns` não pode incluir `ctx_*` (teste).
- Processo pesado só por `bin/run`, um por vez; subagentes só pytest sintético; `runlog.Run` em testes recebe `registry=`/`logs=`.
- Português; commit + push em `main` por tarefa. Nada é enviado ao placar sem ok do usuário.

---

### Task 1: `src/contexto.py`, cache e corretor

**Files:**
- Create: `src/contexto.py`, `tests/test_contexto.py`
- Modify: `src/cache.py` (`load_split`: `df = df.merge(contexto(raw), on=F.ID, how="left")` antes do `del raw`), `src/stack.py` (`corrector_frame` inclui as colunas `ctx_*` presentes em `df`), `README.md` só na lista "Estrutura" (uma linha para `src/contexto.py`).

**Interfaces:**
- Produces: `contexto.contexto(raw: pd.DataFrame) -> pd.DataFrame` com `MVT_ID_mvt` + as 20 colunas, uma linha por DEP de `raw`; `contexto.COLS` (lista dos 20 nomes).
- Referência de implementação (vetorizada com `searchsorted`/`merge_asof`): `.superpowers/arr_viz_ref.py` (script do teste barato; adapte, não copie os prints).

- [x] Teste: ARR sintéticas no mesmo aeroporto com pousos 10, 20 e 70 min antes da DEP e taxi-in 300/600/900 → `ctx_arr_tin_apt_15` = 300 (só a de 10 min); `_60` = média de 300 e 600 = 450; `ctx_arr_n_apt_60` = 2; ARR de outro aeroporto não conta.
- [x] Teste: `ctx_stand_ult_*` pega a última ARR do mesmo stand com in-block ≤ MVT_dep e ignora a posterior.
- [x] Teste: vizinhos excluem a própria DEP; passado e futuro separados; proxy fora de [0, 7200] é ignorado.
- [x] Teste de vazamento: mudar `TAXITIME_SEC_mvt`/`BLOCK_TIME_UTC_mvt` de uma DEP não muda nenhuma coluna `ctx_*` de nenhuma DEP.
- [x] Teste: `features.feature_columns` de um frame com colunas `ctx_*` não as inclui; `stack.corrector_frame` inclui.
- [x] `pytest -q` verde; commit `contexto: taxi-in das chegadas e vizinhos de MVT−AOBT_3 no corretor`; push.

**Acceptance:** testes passam; nenhum treino real rodado.

### Task 2 (controlador): medir, enviar, documentar

- [x] `bin/run src/cache.py` (refaz os 5 splits com `ctx_*`).
- [x] `stack.py v11_cf --crossfit --base 20260927-133137-janela` (systemd-run) → `compare.py` contra a v9 (317,23).
- [x] Se ganhar: promover, `train.py submit 11`, conferir 0 linhas fora da janela, **parar e mostrar ao usuário**.
- [x] Docs: README, CONTEXTO, `saltos.json` (`arr_vizinhos`), caixas; commit + push.
