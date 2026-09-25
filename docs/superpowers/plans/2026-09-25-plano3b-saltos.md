# Plano 3b — medir melhor, cortar alarmes falsos, mistura de cópias e voos normais

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sair do patamar da v5 (331,0 oficial, 332,86 na simulação) com saltos em áreas grandes do erro, medidos de forma que ganhos de ~5 s fiquem visíveis.

**Architecture:** `experiment.py` passa a relatar métricas sem a "loteria" e por fatia de erro; o `compare.py` ganha o ganho "sem loteria" como informação. Depois, experimentos em ordem de teto: alarmes falsos → mistura por horário copiado (com a classe "24 h + taxi") → voos normais (alvo residual, vizinhos, ensemble). Sweep, ablação e mutmut no fim.

**Tech Stack:** Python 3.14, pandas 3, numpy, LightGBM 4.7, XGBoost (CUDA), pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-pipeline-melhoria-continua-design.md`. Evidência: análise de 25/09 (seção abaixo), `docs/research/2026-09-24-forense-dados.md`, `docs/research/2026-09-24-competicoes-e-placar.md`.

## Global Constraints

- Processo pesado por `bin/run`, supervisionado, um por vez; `num_threads=12`; RAM ≤ 7 GB; no máximo um job de GPU.
- L2 no alvo bruto; combinar por esperança, nunca argmax; só piso 0.
- Todo experimento usa `--seed 0` e é comparado à campeã com `bin/run src/compare.py <id>` (sem base explícita).
- `--promover` só com veredito `MELHOR`. `FRÁGIL` → rodar `teto.py` e **parar para o ok do usuário**.
- **Nada é enviado ao placar sem ok explícito do usuário**; implementador nunca roda `s3.py submit`.
- Um experimento que perde é registrado no roadmap do README como "descartado: <motivo>" e o código dele sai (sem flags mortas).
- `experiments.jsonl` e `champion.json` versionados; textos em português.
- Commit + push em `main` ao fim de cada tarefa.

## Fatos que o plano usa (análise de 25/09 na campeã `20260924-215852-nm_retas_6h_r`)

Holdout jan+jul/2025, 344.419 voos, RMSE 332,86.

| Fatia | Voos | Parte do erro² | Cenário |
|---|---|---|---|
| Voos normais (y ≤ 1 h) com NM, RMSE 220,7 | 338.278 | 43,2 % | 210 → 326,0; 200 → 319,8; 190 → 313,7 |
| Alarmes falsos: y ≤ 1 h e previsão > 1 h (67 % LIRF, 56 % sem NM) | 204 | 10,0 % | previsão 900 → 317,4 |
| Cauda y > 1 h que é cópia (±5 min) de SCHED/EOBT/AOBT/LOBT | ~550 | ~10 % | oráculo → 319,5 |
| y > 3 h sem cópia ("loteria"); 2 voos LFPG sozinhos = 26,8 % | 11 | 32,0 % | imprevisível |

- Top 10 voos = 34,0 % do erro² (RMSE sem eles 270,3); top 100 = 43,9 %.
- 1 h < y ≤ 3 h: 921 voos; a ±5 min, a melhor referência é SCHED 395, EOBT 56, AOBT_3 39, LOBT 36.
- **Troca de data:** voos com `data(MVT) > data(SCHED)`, 2025 inteiro:

  | Grupo | n | y ≈ k·24 h + taxi | y ≈ MVT−SCHED | y normal |
  |---|---|---|---|---|
  | LIRF sem NM | 67 | 16,4 % | 79,1 % | 4,5 % |
  | LIRF com NM | 108 | 0,9 % | 21,3 % | 94,4 % |
  | outros sem NM | 150 | 0 % | 2,7 % | 67,3 % |
  | outros com NM | 4.257 | 0 % | 8,7 % | 98,1 % |

  (k = dias entre SCHED e MVT; as colunas se sobrepõem quando ms ≤ 1 h.) Ranking 2026: 25 voos LIRF sem NM com troca de data (jan+jul/2025: 20). y > 3 h em 2025: 202 voos = 172 cópias do SCHED + 14 "24 h + taxi" (13 LIRF) + 17 sem explicação.
- Ruído: duas seeds da mesma configuração dão IC 95% pareado de ≈ −10 a +14 s no RMSE completo, dominado pelos poucos voos da cauda.

## Estrutura de arquivos

| Arquivo | Ação | Responsabilidade |
|---|---|---|
| `src/experiment.py` | alterar | métricas por fatia (`fatias`) e `sem_loteria` |
| `src/compare.py` | alterar | ganho e IC nos voos normais e sem loteria (informativo) |
| `src/models.py` | alterar | `CopyMixture` (tarefa 3), alvo residual (tarefa 4), `Ensemble` (tarefa 6) |
| `src/features.py` | alterar | `days_shift` (tarefa 3), features de vizinhos (tarefa 5) |
| `src/sweep.py`, `src/ablation.py` | criar | tarefas 7 e 8 |
| `tests/…` | alterar/criar | contratos de cada tarefa |
| `README.md`, `../CONTEXTO.md` | alterar | resultado de cada tarefa |

---

### Task 1: Medir melhor (métricas sem loteria)

**Files:** Modify `src/experiment.py`, `src/compare.py`; Test `tests/test_compare.py` (e `tests/test_experiment.py` novo se preciso).

**Interfaces:**
- `experiment.metrics` passa a incluir `"normais_nm"` (y ≤ 1 h e NM presente), `"alarmes_falsos"` (dict `n`, `parte_erro2`: y ≤ 1 h e previsão > 3600), `"cauda_copia"` (RMSE na cauda y > 1 h que é cópia, a ≤ 5 min de algum horário de `PLAN_REFS`), `"sem_loteria"` (RMSE sem os voos com y > 3 h cuja distância a todos os horários em `PLAN_REFS` é > 300 s).
- `compare.py` imprime também ganho e IC 95% do bootstrap pareado restrito a `normais_nm` e a `sem_loteria`. **Não muda o veredito** (é informação para decidir, a regra de promoção continua a do plano 3a).

- [x] Step 1: teste que falha para a função `lottery_mask(df, y) -> np.ndarray[bool]` (em `experiment.py`): voo y = 84.240 com ms = 1.740 e demais horários nulos → True; voo y = 30.000 com ms = 29.900 → False; voo y = 900 → False.
- [x] Step 2: implementar `lottery_mask` e as métricas novas; `compare.py` monta as máscaras a partir do holdout (`cache.load_split("holdout2025")` juntado por `MVT_ID_mvt`).
- [x] Step 3: `.venv/bin/python -m pytest -q` verde (21 testes).
- [x] Step 4: re-medir a campeã (`bin/run src/experiment.py base_3b --model two_stage_nm --nm-min-ms 21600 --seed 0`). `20260925-105318-base_3b`: completo 332,86 (idêntico), `normais_nm` 220,68, `alarmes_falsos.n` = 204 (10,0 % do erro²), `cauda_copia` 2.287,20, `sem_loteria` **274,42** — a estimativa de ≈ 285 do plano estava errada: tirar 32,0 % do erro² dá 332,86 × √0,6797 = 274,4.
- [x] Step 5: `20260925-105705-base_3b_s1` (seed 1) contra `base_3b`: completo 0,0 s (IC −1,6 a 2,2); voos normais com NM −0,5 s (IC −0,9 a −0,2); sem loteria 0,2 s (IC −1,8 a 2,7). Anotado no README.
- [x] Step 6: commit + push (`experiment/compare: métricas por fatia e sem loteria`).

**Acceptance:** o README registra o ruído entre seeds nas métricas novas; a regra de promoção não mudou.

---

### Task 2: Cortar alarmes falsos

Teto medido: −15 s (332,9 → 317,4). São voos normais que receberam previsão de horas porque `p·ms` é grande.

**Files:** Modify `src/models.py`, `src/experiment.py`; Test `tests/test_models.py`.

Experimentos, um de cada vez, todos contra a campeã:

- [ ] **2a — calibração do classificador por célula.** Depois do treino, recalibrar `p` com regressão isotônica (ou Platt) ajustada **fora do fold** (5 folds por dia dentro de `train2025`) por grupo `(aeroporto == LIRF, nm_missing, faixa de ms: ≤2 h, 2–6 h, 6–12 h, >12 h)`. Flag `--calibrar`. Teste: `p` calibrado fica em [0, 1] e é monótono em `p` bruto dentro do grupo.
- [ ] **2b — híbrido para voos sem NM.** Em vez de trocar tudo pela reta acima de 6 h, `ŷ = p·reta + (1 − p)·ŷ_regressor` para `nm_missing` (backlog do plano 3a: a reta piora os voos normais sem NM 1.088 → 1.324 s). Flag `--nm-hibrido`.
- [ ] Para cada um: `experiment.py` → `compare.py`; anotar `alarmes_falsos` antes/depois. MELHOR → promover; FRÁGIL → `teto.py` contra o último envio e **parar para o ok**.
- [ ] Commit + push por experimento.

**Acceptance:** `alarmes_falsos.parte_erro2` cai sem subir o erro da cauda; veredito registrado no README.

---

### Task 3: Mistura por horário copiado, com a classe "24 h + taxi"

Teto do oráculo de cópias: −13 s. Generaliza o estágio 1 de binário para multiclasse.

**Files:** Modify `src/features.py` (`days_shift = data(MVT) − data(SCHED)` em dias), `src/models.py` (`CopyMixture`, registrado em `MODELS` como `"copy_mix"`), `src/experiment.py`; Test `tests/test_models.py`.

**Interfaces:**
- Rótulo `copy_class(df) -> np.ndarray[int]` no treino: 0 = normal; 1 = SCHED; 2 = EOBT_1; 3 = LOBT; 4 = AOBT_3 (|BLOCK − horário| ≤ 60 s, prioridade nessa ordem quando empata); 5 = "troca de data" (`days_shift ≥ 1` e |y − 86400·days_shift| < 3600).
- Componentes: `c_k = MVT − horário_k` para k = 1…4; `c_5 = 86400·days_shift + ŷ_normal`; `c_0 = ŷ_normal`.
- `ŷ = Σ p_k·c_k` (componente com horário nulo: `p_k` redistribuído para `c_0`), piso 0. Classificador LightGBM `multiclass` com as mesmas features + `days_shift`.
- Regressor normal treinado só na classe 0.

- [ ] Step 1: testes que falham para `copy_class` (um caso por classe, incluindo empate e horário nulo) e para a combinação (p = one-hot reproduz o componente; `p_k` de horário nulo vai para o normal).
- [ ] Step 2: implementar; a reta de voos sem NM > 6 h continua por cima (como na campeã) numa flag, para medir com e sem.
- [ ] Step 3: `experiment.py copy_mix --model copy_mix --seed 0` e variante sem reta; `compare.py`.
- [ ] Step 4: checagem no ranking antes de qualquer envio: `teto.py` com o último envio, e contagem dos 25 voos LIRF sem NM com troca de data (a previsão deles deve cair entre ms e 86400·k + taxi).
- [ ] Commit + push.

**Acceptance:** RMSE `y_gt_1h` e `alarmes_falsos` registrados; veredito no README. Se a classe 5 não tiver exemplos suficientes para o multiclasse (13–14 por ano), usar para ela a taxa empírica suavizada por (LIRF, nm_missing, days_shift ≥ 1) = 0,164 em vez do classificador; documentar a escolha.

---

### Task 4: Voos normais — alvo residual sobre AOBT_3 (item 5)

**Files:** `src/models.py`, `src/experiment.py`, testes.

- Referência `ref = MVT − AOBT_3` quando existe e fica em [0, 7200] s; senão `MVT − EOBT_1`, `MVT − LOBT`, `ref_p10`. O regressor normal aprende `y − ref` e prevê `ref + ŷ_res`. Flag `--residual`; vale para o regressor de qualquer modelo de dois estágios/mistura.
- [ ] Teste: `ref` segue a ordem de fallback e o intervalo [0, 7200].
- [ ] Experimento contra a campeã; olhar `normais_nm` (cenário: 220,7 → 200 vale −13 s).
- [ ] Commit + push.

### Task 5: Voos normais — features de vizinhos (item 6)

- Por decolagem: referência NM (`MVT − AOBT_3`) das decolagens vizinhas na mesma pista e no mesmo stand, janelas de 15 e 60 min antes e depois (média e contagem); taxi-in médio dos pousos da mesma pista nos últimos 15/30/60 min. Só colunas que existem no ranking (MVT, AOBT_3, pousos têm alvo no ranking).
- Uma família por experimento (vizinhos NM; pousos), cada um contra a campeã.
- [ ] Teste: `_counts_around`-style sem vazamento (a própria linha não entra na janela).
- [ ] Rodar `cache.py` de novo (sozinho) depois de mudar `features.py`.
- [ ] Commit + push.

### Task 6: Ensemble XGBoost CUDA + seeds LightGBM (item 7)

- `Ensemble`: média ponderada do regressor normal LightGBM (3 seeds) e XGBoost `device="cuda"` (1,4 GB VRAM, 37 s/300 rodadas); pesos por mínimos quadrados em previsões fora do fold de `train2025`. Aplicar só ao regressor normal; o classificador continua LightGBM.
- XGBoost sem suporte a categóricas grandes: usar `enable_categorical=True` com `tree_method="hist"`; se falhar, codificar por frequência.
- [ ] Experimento contra a campeã; commit + push.

### Task 7: `sweep.py` (polimento, não salto)

- Grade pequena na campeã: `learning_rate` {0,03, 0,05, 0,08} × `num_leaves` {127, 255, 511} × rodadas {×1, ×1,5}; LightGBM CPU e XGBoost GPU em paralelo dentro de metade do PC. Cada ponto vira linha no `experiments.jsonl`; o melhor passa pelo `compare.py`. Esperado: poucos segundos (benchmark: 1500 rodadas pioram 1,4 s).

### Task 8: `ablation.py` e mutmut (lixo)

- `ablation.py`: treina a campeã sem uma família de features por vez (`round_*`, `gap_*`, `apt_*`, `rwy_*`, `to_takeoff_from_*`, `ref_p10`, hora/dia); família cuja remoção não piora `normais_nm` (IC contém 0 ou ganho) sai do código. Suspeita principal: `round_*` (forense §1).
- `mutmut` em `src/compare.py`, `src/models.py`, `src/cache.py`: mutação sobrevivente = teste fraco → escrever o teste que a mata.

---

## Ordem e parada

1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Qualquer candidato a envio (MELHOR, ou FRÁGIL com `teto.py`) **para a execução e pede o ok do usuário**, mostrando: simulação, ganho ± IC (completo, normais, sem loteria), teto no ranking, nota oficial projetada. Envios: ≤ 3 por dia; nunca para sondar o placar.

Prazo: 11/10/2026 23:59:59 CET. Até 08/10: repositório público GPLv3 com README de reprodução (condição do prêmio).
