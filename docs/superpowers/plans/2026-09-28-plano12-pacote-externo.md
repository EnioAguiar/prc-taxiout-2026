# Plano 12 — pacote de ganhos pequenos no corretor (companhia, séries diárias, OPDI)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Somar no corretor três blocos que, sozinhos, deram ~1 s cada no teste barato: taxa de cópia do SCHED por companhia, séries diárias da EUROCONTROL e tempo em solo do OPDI.

**Evidência (corretor barato fora do fold, base `janela`, 309,27):** taxa de cópia por (aeroporto, companhia, NM) −1,0 s (sem loteria −1,3); séries diárias −0,8 s (LTFM −2,3); OPDI −0,8 s. Ver `docs/research/2026-09-28-estudo-cauda.md` e `docs/research/2026-09-27-concorrentes.md`.

**Architecture:** Novo `src/externos.py`:
- `baixar(raiz=DATA/"externo")`: baixa para `data/externo/` (ignorado pelo git) os CSV `atfm_slot_adherence_{2025,2026}`, `all_pre_departure_delays_{2025,2026}`, `atc_pre_departure_delays_{2025,2026}` de `https://www.eurocontrol.int/performance/data/download/csv/<nome>.csv` e as flight lists do OPDI `flight_list_{202501..202512,202601,202607}.parquet` de `https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/flight_list_<AAAAMM>.parquet`; pula o que já existe; CLI `bin/run src/externos.py baixar`.
- `diarias(df) -> DataFrame` (mesma ordem de `df`): colunas `ext_reg_frac = FLT_DEP_REG_1/FLT_DEP_1`, `ext_late_frac = FLT_DEP_OUT_LATE_1/FLT_DEP_REG_1`, `ext_reg_n = FLT_DEP_REG_1`, `ext_pre_min = DLY_ALL_PRE_2/FLT_DEP_IFR_2`, `ext_atc_min = DLY_ATC_PRE_3/FLT_DEP_3`, juntando por (data UTC de `MVT_TIME_UTC_mvt`, aeroporto). NaN sem dado.
- `opdi(df) -> DataFrame` (mesma ordem): `ext_solo_s` (first_seen do voo − last_seen do voo anterior da mesma `icao24`) e `ext_mesmo_apt` (o voo anterior chegou no mesmo aeroporto), casando cada DEP pelo aeroporto e `CALLSIGN_flt` com decolagem OPDI (`first_seen`) a ±600 s; senão só aeroporto a ±90 s. Carrega só os meses presentes em `df`.
- `CopiaCia`: taxa de cópia do SCHED (|BLOCK − SCHED| ≤ 60 s) nos voos com `MVT − SCHED` > 3600 s, por (aeroporto, companhia = regex `^[A-Z]{2,3}` de `FLIGHT_mvt`, com/sem NM), suavizada para a taxa do aeroporto com k = 20. `fit(raw_dep)` guarda contagens por (chave, mês); `transform(df, meses_treino)` usa só `meses_treino`, **excluindo o mês da própria linha** (sem vazamento); devolve `ext_taxa_cia` e `ext_taxa_cia_ms = ext_taxa_cia × (MVT − SCHED)`.

`stack.corrector_frame(df, pred, adsb, janela, externos=None)`: com `externos` (um dict pronto de colunas por linha, mesma ordem), acrescenta as colunas `ext_*`. `stack.py --crossfit --externos` monta os `ext_*` para as cegas do oof (meses de treino = meses do oof) e para o holdout (meses de treino = meses do oof); `train.py` faz o mesmo com o `full2025` e o ranking (meses de treino = 12 meses de 2025). Config `externos: true` só quando ligado.

## Global Constraints

- Sem `--externos`: saídas idênticas às de hoje.
- `CopiaCia` nunca usa o alvo/BLOCK de linhas de jan/jul nas simulações (só dos meses de treino) e nunca o do próprio mês da linha.
- Dados brutos de 2025 para `CopiaCia`: `features.load` dos `training_2025-*.parquet` (só DEP).
- Documentar as duas fontes novas na seção "Dados externos" do README (EUROCONTROL: dados públicos; OPDI: "open data, freely used … provided that the data source is attributed").
- Processo pesado só por `bin/run`; subagentes só pytest sintético (sem rede: testar `diarias`/`opdi`/`CopiaCia` com frames sintéticos e arquivos em `tmp_path`).
- Português; commit + push em `main`. Envio só com ok do usuário.

### Task 1: `src/externos.py`, `corrector_frame`, `--externos`, README

- [ ] Teste `CopiaCia`: com 3 meses sintéticos, a taxa de uma linha do mês 2 ignora o mês 2 e usa só `meses_treino`; grupo sem histórico cai na taxa do aeroporto.
- [ ] Teste `diarias`: junção por (dia UTC, aeroporto) com CSV sintético em `tmp_path`; dia ausente → NaN.
- [ ] Teste `opdi`: casamento por callsign ±600 s e fallback ±90 s; `ext_solo_s` = first_seen − last_seen anterior da mesma aeronave.
- [ ] Teste: `corrector_frame` sem `externos` igual ao de hoje; com, acrescenta só colunas `ext_*`.
- [ ] Teste: config de `stack.py --crossfit --externos` tem `externos: true`; sem a flag não tem.
- [ ] `pytest -q` verde; commit `externos: companhia, séries diárias e OPDI no corretor`; push.

### Task 2 (controlador)

- [ ] `bin/run src/externos.py baixar`; `stack.py v16_cf --crossfit --conjunto --externos --base 20260927-211418-reg_corte` → `compare.py` contra a v12 (olhar `sem_loteria` e normais).
- [ ] Se ganhar: `train.py submit 16`, 0 fora da janela, regra de Roma por cima, **parar e mostrar ao usuário**.
- [ ] Docs: README, CONTEXTO, `docs/mapa.md`, `saltos.json`, caixas; commit + push.
