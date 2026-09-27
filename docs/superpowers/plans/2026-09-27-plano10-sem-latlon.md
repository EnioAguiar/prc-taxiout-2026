# Plano 10 — v14 = v12 sem `adsb_lat0/lon0` (deriva 2025 → 2026)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Gerar `submissions/outgoing-boat_v14.parquet` = mesma receita da campeã v12 (`20260927-160433-v12_cf`), sem as colunas `adsb_lat0` e `adsb_lon0` na base e no corretor, pronta para enviar às 00:00 UTC de 28/09 junto com a v13.

**Evidência:** `saltos.json` item `deriva_adsb`: KS 0,26 entre 2025 e 2026 para `adsb_lat0/lon0` (onde o avião foi ouvido pela primeira vez). GREKI (Discord 27/09): "new inputs that shift between 2025 and 2026 came in well under the estimate". O holdout é de 2025 e não tem a deriva: a simulação só confirma que tirar não piora além do ruído; o ganho só aparece no placar.

**Architecture:** Chave nova `sem_features` (lista de nomes) na config. Base: `models.prepare(train, others, sem=())` tira os nomes da lista de colunas; os três pontos que chamam `prepare` (`experiment.py`, `crossfit.oof_base`, `train.base_final`) passam `cfg.get("sem_features", ())`. Corretor: `stack.corrector_frame(..., sem=())` tira as mesmas colunas de `X`; `stack.py --sem-feature` grava a lista na config da corrida e na `base_config`. Envio: `train.py submit N --corrida <id>` gera o arquivo de qualquer corrida registrada em `experiments.jsonl`, sem mexer em `champion.json`.

**Tech Stack:** Python 3.14, LightGBM 4.7, CatBoost 1.2.10, pytest.

**Spec:** este plano (pedido do usuário em 27/09: adiantar a v14 antes do reset dos envios; nada é enviado sem ok).

## Global Constraints

- Sem `sem_features` na config: colunas, saídas e configs gravadas idênticas às de hoje (a chave só aparece quando a lista não é vazia).
- Nome inexistente em `--sem-feature` é erro (não ignora em silêncio).
- `champion.json` não muda neste plano. A v12 continua campeã.
- `train.py submit N` sem `--corrida` continua usando `champion.json`, como hoje.
- Processo pesado só por `bin/run` (systemd-run); subagentes rodam só `pytest` sintético.
- Português; commit + push em `main` (convenção do repositório). Nada é enviado ao placar sem ok do usuário.

---

### Task 1: `sem_features` na base, no corretor e no envio

**Files:** Modify `src/models.py` (`prepare`), `src/experiment.py` (flag `--sem-feature`, config, chamada de `prepare`), `src/crossfit.py` (`oof_base`), `src/stack.py` (`corrector_frame`, `previsao_corrigida`, `simulacao_crossfit`, `simulacao_folds`, `parser`, `config_da_corrida`, docstring "Uso"), `src/train.py` (`base_final`, `corretor_final`, `corrigir_ranking`, `submit`, `__main__`, docstring), `README.md` (linhas de uso de `experiment.py`, `stack.py`, `train.py`). Test: `tests/test_models.py`, `tests/test_stack.py`, `tests/test_train.py`, `tests/test_experiment.py`.

**Interfaces:**
- Produces: `prepare(train, others, sem: Iterable[str] = ()) -> list[str]` — levanta `ValueError` se algum nome de `sem` não está nas colunas candidatas.
- Produces: `corrector_frame(df, pred, adsb=True, janela=False, sem=())` e `previsao_corrigida(model, df, base, adsb, janela, sem=())`.
- Produces: config de `experiment.py --sem-feature X --sem-feature Y` com `"sem_features": ["X", "Y"]` (ordem da linha de comando); de `stack.py --crossfit --sem-feature X` com `"sem_features": [...]` no topo **e** dentro de `base_config`.
- Produces: `train.py submit N [--forcar] [--corrida <id>]`; com `--corrida`, `champ` = a última linha de `experiments.jsonl` com esse `id` (campos `id`, `config`, `src_hash`, `best_iter`); o `Run` grava `"campeao": <id>` como hoje.

- [x] Teste (`test_models.py`): `prepare` com `sem=("adsb_lat0",)` devolve a mesma lista de `sem=()` menos `adsb_lat0`; com nome inexistente levanta `ValueError`.
- [x] Teste (`test_stack.py`): `corrector_frame(..., sem=("adsb_lat0", "adsb_lon0"))` não tem as duas colunas e mantém as outras `adsb_*` e `adsb_menos_pred`; sem `sem`, colunas iguais às de hoje.
- [x] Teste (`test_stack.py`): `config_da_corrida` com `--crossfit --sem-feature adsb_lat0 --sem-feature adsb_lon0` tem `sem_features` no topo e em `base_config`; sem a flag, nenhuma das duas chaves.
- [x] Teste (`test_experiment.py`): config de `experiment.py` com `--sem-feature` tem `sem_features`; sem a flag, config idêntica à de hoje.
- [x] Teste (`test_train.py`): `submit` com `--corrida <id>` lê a linha do registro (fixture com `experiments.jsonl` sintético) e não lê `champion.json`; `id` inexistente sai com erro que cita o id.
- [x] Implementar; `pytest -q` verde (hoje: 81 passed).
- [x] Commit `sem_features: tirar colunas da base e do corretor; train.py submit --corrida`; push.

### Task 2 (controlador): medir, gerar a v14, documentar

- [x] Base: `bin/run src/experiment.py janela_sem_latlon --model two_stage_nm --nm-min-ms 21600 --janela-lobt --sem-feature adsb_lat0 --sem-feature adsb_lon0` (mesma config da base `20260927-133137-janela` + `sem_features`).
- [x] Corretor: `bin/run src/stack.py v14_cf --crossfit --conjunto --base <id da base acima> --sem-feature adsb_lat0 --sem-feature adsb_lon0` → `compare.py` contra `20260927-160433-v12_cf` (309,78). Critério: não piorar mais que o ruído (IC do `compare.py` incluindo zero ou ganho). Se piorar além do ruído, **parar e mostrar ao usuário** antes de gerar o arquivo.
- [x] `bin/run src/train.py submit 14 --corrida <id da v14_cf>`; conferir 0 linhas fora da janela do LOBT e mediana parecida com a v12.
- [x] Docs: README (Submissões/Roadmap), CONTEXTO.md ("Retomar"), `saltos.json` (`deriva_adsb`: simulação medida, status "pronto para envio"), caixas deste plano; commit + push.
- [x] **Parar e mostrar ao usuário.** Ordem proposta às 00:00 UTC: v13 (Roma), v14 (sem lat/lon); cada uma muda uma coisa só em relação à v12.

**Resultado (27/09 21:41 UTC):** base `20260927-181854-janela_sem_latlon` 321,23 (a da v12, 320,29); `20260927-182244-v14_cf` = 311,88 contra 309,78 da v12: **−2,1 s (IC −3,5 a −1,0)**, pior além do ruído no holdout de 2025, como o critério previa. Parado antes do `train.py submit 14 --corrida 20260927-182244-v14_cf` (cerca de 20 min): decisão do usuário, porque a deriva só aparece no placar.

**Arquivo (27/09 22:46 UTC):** `train.py submit 14 --corrida 20260927-182244-v14_cf` interrompido por queda de energia (19:13 local, base final a 10 %); relançado e concluído em 28m46s, pico 8,1 GB. `submissions/outgoing-boat_v14.parquet`: 344.841 linhas, 0 nulos, 0 negativos, 0 fora da janela do LOBT; diferença para a v12 49 s (RMS), média −0,2 s. Aguarda envio depois de 00:00 UTC junto com a v13.
