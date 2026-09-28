# PRC Data Challenge 2026: taxi-out time

Competição da EUROCONTROL Performance Review Commission (PRC) com a
OpenSky Network (OSN). Não é Kaggle.

- Site: <https://ansperformance.eu/study/data-challenge/dc2026/>
- Dados: <https://prc-data-challenge-2026.netlify.app/data.html>
- Termos/elegibilidade: <https://prc-data-challenge-2026.netlify.app/eligibility.html>
- Discord: servidor OpenSky (<https://discord.gg/RPh89jpVVz>), canal `#prc-data-competition`

Mapa curto do projeto: `docs/mapa.md`.

## Problema

Prever o taxi-out de cada decolagem: segundos entre sair do
portão (off-block) e decolar.

```
TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt   (PHASE_mvt == "DEP")
```

Regressão tabular. Sem LLM, sem GPU obrigatória.

## Regras-chave

| Item | Valor |
|---|---|
| Período | 01/09/2026 → **11/10/2026 23:59:59 CET** |
| Métrica | RMSE (segundos), menor é melhor |
| Conjunto de ranking | movimentos de **jan e jul de 2026** |
| Prêmio | 5.000 EUR somados entre os 3 primeiros |
| Condição do prêmio | repositório público no GitHub sob **GPLv3** |
| Opcional | artigo open access na JOAS (Journal of Open Aviation Science) |

## Inscrição

1. [x] Preencher o [formulário de criação de equipe](https://docs.google.com/forms/d/e/1FAIpQLScgRRk0j5Giot8puUAjzXC7ScR926Oupd62LbRVS1g8Y2p4hw/viewform) (aprovação manual). Enviado em 23/09/2026, equipe solo.
2. [x] Equipe aprovada em 24/09/2026: `outgoing-boat`, bucket de envio `prc-2026-outgoing-boat`.
3. [x] Chaves do MinIO geradas (`minio-credentials.json`, fora do git) e `.env` preenchido.
4. [ ] Entrar no Discord da OpenSky.

## Dados

Aeroportos (10): EDDF (FRA), EDDM (MUC), EGLL (LHR), EHAM (AMS),
LEBL (BCN), LEMD (MAD), LFPG (CDG), LIRF (FCO), LTFM (IST), LSZH (ZRH).

Distribuídos via bucket da OSN (cliente MinIO; instruções em
<https://ansperformance.eu/study/data-challenge/dc2024/data.html#using-minio-client>).

| Arquivo | Conteúdo | Tamanho |
|---|---|---|
| `training_2025-MM-01_....parquet` (12) | movimentos de 2025, com alvo | ~20–25 MB cada |
| `ranking.parquet` | jan + jul 2026; `BLOCK_TIME_UTC_mvt` e `TAXITIME_SEC_mvt` apagados nas DEP | 27 MB |
| `submitting.parquet` | template: `MVT_ID_mvt`, `TAXITIME_SEC_mvt` | 1.1 MB |

~4,17 milhões de movimentos (ARR + DEP). Linhas de movimento
(`*_mvt`) com left join dos dados de voo do Network Manager (`*_flt`).
Dado real, com inconsistências entre `_mvt` e `_flt`.

Colunas mais úteis: `ADEP_mvt`, `RUNWAY_mvt`, `STAND_mvt`,
`AIRCRAFT_TYPE_mvt`, `WK_TBL_CAT_flt`, `MARKET_SEGMENT_flt`,
`AIRCRAFT_OPERATOR_flt`, `SCHED_TIME_UTC_mvt`, `MVT_TIME_UTC_mvt`,
`EOBT_1_flt`, `AOBT_3_flt`.

Vazamento (verificado em 24/09): `AOBT_3_flt` (off-block real do NM)
está preenchido em 98,5% das DEP do ranking, então pode ser usado. Ele
bate com o off-block oficial (±2 min) em 38% dos voos; EOBT/LOBT em ~20%;
SCHED em 17%. O BLOCK oficial tem resolução de segundo e AOBT_3/EOBT/SCHED
de minuto, então a proximidade de 38% com AOBT_3 não é cópia. Cópia existe só na
cauda, e do SCHED: 84,7% dos y > 3 h têm |BLOCK − SCHED| ≤ 60 s
(`docs/research/2026-09-24-forense-dados.md`).

Outliers entram na nota: a verdade oficial mantém taxi-outs de horas
(até 36 h). Eles são ~37% do erro quadrático. Cortá-los do treino/validação
deixa a validação otimista e o modelo cego para eles (erro da v1).

## Modelo atual (`src/`)

Melhor nota oficial: **v21 = 252,34** = v20 (id `20260928-131704-v20_cf`, campeã em `champion.json`, 301,73 na simulação; v18 com a base usando também as colunas `ctx_*` via `--base-ctx`) + pós-regra de Roma. Antes: v19 = 253,95 = v18 (id `20260928-113219-v18_cf`, 303,82 na simulação; v16 + `--plano13`: METAR, rotação no stand, consistência NM e companhia no corretor) + pós-regra de Roma. Antes: v17 = 256,86 = v16 (id `20260927-223230-v16_cf`, 306,65 na simulação; base `--reg-corte 7200`, corretor `--conjunto --externos`) + pós-regra de Roma aplicada ao arquivo. Antes: v13 = 264,35; v12 (id `20260927-160433-v12_cf`, 309,78 na simulação, 264,74 oficial) = v11 com `--conjunto` (média de 3 corretores). A v11 (id `20260927-145005-v11_cf`, 311,09 na simulação, 266,81 oficial) = v9 + colunas `ctx_*` no corretor (`src/contexto.py`: taxi-in das chegadas e vizinhos de `MVT − AOBT_3`, plano 8). A v9 era `stack_cf` com janela do LOBT (id
`20260927-133718-v9_cf`, 317,23 na simulação, 275,90 oficial). A base é
`two_stage_nm` (`--nm-min-ms 21600 --janela-lobt`, features `adsb_*`) com corretor
treinado fora do bloco por meses (`src/crossfit.py`, `stack.py --crossfit`); toda previsão é
projetada em `MVT − LOBT ± 3606 s`. Promovida à mão com o ok do usuário ("não comprovado"
na regra: ganho 6,3 s). Envio: `train.py submit N` (~27 min, pico 6,94 GB). A v11 foi medida no código `5d75eef`; depois só entrou o `--conjunto` (plano 9), que não muda nada sem a flag (revisado), então `train.py submit` da v11 precisa de `--forcar`. Antes dela, a
v6 (id `20260926-223248-nm_retas_6h_adsb`, 314,76) foi promovida à mão com o ok do usuário
depois do oficial: no `compare.py` o veredito foi "não comprovado" (ganho 9,4 s, IC
7,5 a 11,7, abaixo dos 10 s; sem os 10 maiores 8,7; jan e jul > 0). As colunas
`adsb_*` entram por `cache.load_split` (merge com `events.parquet`, NaN sem evento),
fora do cache de features. `train.py submit N` precisa do `events.parquet` no SSD.

- Estágio 1: classificador LightGBM de `eq = |BLOCK − SCHED| ≤ 60 s`,
  a cópia que existe na cauda.
- Estágio 2: regressor L2 treinado só nos voos normais (`~eq`).
- Combinação pela esperança: `ŷ = p·(MVT − SCHED) + (1 − p)·ŷ_normal`,
  com piso 0; nunca argmax, porque errar a classe custa horas². Sem SCHED,
  só o regressor.
- Voos sem registro NM (`nm_missing`): a previsão acima é trocada por
  uma reta `y = a + b·(MVT − SCHED)` ajustada por aeroporto (com
  `--nm-split-ms`, por aeroporto × atraso > 2 h); grupos com < 50 voos usam
  a reta global; piso 0. Com `--nm-min-ms S`, a reta só troca os voos com
  `MVT − SCHED` > S (ex.: 21600 = 6 h); os demais ficam com o dois estágios.
- Janela do LOBT (`--janela-lobt`, plano 7): em 100 % das DEP com LOBT, |BLOCK −
  LOBT| ≤ 3606 s (regra do dado, vale em 2026). Toda previsão é projetada em
  `MVT − LOBT ± 3606` (depois o piso 0), e `p` da cópia é zerado quando o SCHED cai fora
  da janela. Sem LOBT (≈ 1 %, quase todos sem NM) a janela não existe.
- Corretor (`stack_cf`, plano 5): LightGBM que aprende `y − pred_base` com previsões
  da base fora do bloco (blocos de 2 meses, split `blind2025` montado como o ranking);
  entradas `pred`, aeroporto, `nm_missing`, hora, `to_takeoff_from_*`, `adsb_*`,
  `dist_lo`/`dist_hi` até as bordas da janela; saída projetada na janela.
- `--seeds N` (plano 6) faz a média de N seeds na base; medido e não usado (+0,3 s).
- Features: referência P10 por (aeroporto, stand, pista) com fallback,
  calculada só no fold de treino; tempos entre os horários planejados/NM
  (SCHED, LOBT, IOBT, EOBT, AOBT_3) e a decolagem; diferenças entre esses
  horários e flags de arredondamento; congestionamento (decolagens e pousos
  do aeroporto e da pista, janelas de 10/20/30/60 min); hora, dia da semana,
  categóricas; `nm_missing` (voo sem linha do Network Manager).
- LightGBM com `num_threads=12` (6 núcleos físicos × 2; 24 threads é 2 a 4×
  mais lento, ver `docs/research/2026-09-24-hardware-benchmark.md`).
- Seeds determinísticas: `--seed N` fixa as seeds do LightGBM com
  `deterministic` e `force_row_wise`; a mesma seed reproduz o mesmo número.
  Ruído do treino com 3 seeds da campeã de referência: ≤ 1,5 s no RMSE
  completo agregado, mas esse número é o piso errado para julgar ganhos.
  Pareado no `compare.py`, duas seeds da mesma configuração dão IC 95% de
  ≈−10 a ≈+14 s (s1: −10,5 a 13,9; s2: −7,9 a 14,1), e o ganho "sem os 10
  maiores voos" fica em ≈−6 s a efeito zero (−6,2 e −5,9): tirar os k voos
  que mais contribuem sempre favorece a base, então esse critério tem viés
  negativo embutido.
- Métricas por fatia (plano 3b): além do RMSE completo, `experiment.py`
  grava `normais_nm` (y ≤ 1 h com registro NM), `alarmes_falsos` (voo normal
  previsto > 1 h: `n` e parte do erro²), `cauda_copia` (y > 1 h a ≤ 5 min de
  algum horário planejado) e `sem_loteria` (RMSE sem os voos com y > 3 h que
  nenhum horário planejado explica, 11 no holdout, 32 % do erro²).
  `compare.py` imprime ganho e IC pareados nos voos normais com NM e sem
  loteria como informação; o veredito continua o do plano 3a.
- Ruído entre seeds nas fatias novas (`base_3b` seed 0 → `base_3b_s1`
  seed 1, mesma configuração da campeã): completo 332,86 → 332,83, ganho
  0,0 s (IC 95% −1,6 a 2,2); voos normais com NM −0,5 s (IC −0,9 a −0,2);
  sem loteria 0,2 s (IC −1,8 a 2,7). A fatia dos normais é ~5× mais precisa
  que o RMSE completo, mas o IC dela nem contém zero: só leia como melhoria
  real um ganho acima de ~1 s. Os alarmes falsos oscilam de 204 para 224 voos
  (10,0 % → 10,1 % do erro²) só por troca de seed.
- Treina com todos os voos (sem corte de outliers) e sem limitar a
  previsão.
- Validação: `src/experiment.py` simula o ranking (jan+jul/2025 com o alvo
  apagado como no oficial) e grava a corrida em `experiments.jsonl`;
  `src/compare.py` decide por bootstrap pareado por dia e atualiza
  `champion.json` (que também guarda `src_hash` e `git_commit` do código que
  mediu o campeão). Veredito MELHOR só se tudo valer: ganho ≥ 10 s e IC
  95% > 0; ganho sem os 10 voos de maior ganho > 0 e ≥ 10% do ganho cheio; IC
  > 0 em jan e em jul separados. Senão FRÁGIL, que não promove; só
  `--aceitar-fragil` promove, e ele só se usa depois do `teto.py` e com o ok do
  usuário. Sob essa regra v2→v3 promoveria e v3→v4 seria barrada. A razão
  oficial/simulação é pessimista, mas não comparável entre versões: 0,836 na
  v2 veio da simulação antiga (`sim_ranking.py`) e 0,873 na v3 do holdout
  novo; a v2 medida no holdout novo daria 0,846.
- `src/teto.py` estima, antes de enviar, o teto do ganho oficial: compara
  dois arquivos de envio nas linhas que diferem (opcionalmente só com
  `MVT − SCHED` > `--min-ms`) contra um oráculo otimista `y = MVT − SCHED`;
  ganho simulado > 2 × teto = a simulação mede folga que o modelo final não
  tem. `--salvar` grava um candidato (base + novo só nessas linhas).
- `src/train.py submit N` refaz o campeão no ano inteiro com o `best_iter`
  (ou as rodadas configuradas, se não houver) × 1,2 (full2025 tem 2,085 M
  linhas contra 1,741 M do treino) e só gera o arquivo; o envio é um comando
  à parte. Aborta se o código mudou desde a promoção do campeão; `--forcar`
  ignora a checagem.

## Submissões

| Versão | Data | Mudança | Simulação (completo / sem outliers) | Oficial |
|---|---|---|---|---|
| v1 | 24/09 | modelo base; treino sem y ≥ 3 h, previsão limitada a 3 h | 535,9 / 264,4 | 514,5 |
| v2 | 24/09 | treino com outliers, sem limite; bug de unidade das janelas de congestionamento corrigido; features de diferença e arredondamento | 460,4 / 290,2 | **384,7** |
| v3 | 24/09 | dois estágios (classificador da cópia do SCHED + regressor) e `nm_missing`; base de experimentos nova | 388,16 / 269,87 | **338,7** |
| v4 | 24/09 | `two_stage_nm`: retas por aeroporto em `MVT − SCHED` para os voos sem NM | 345,89 / 285,52 | **337,2** |
| v5 | 24/09 | v3 + retas só em NM ausente com atraso > 6 h (96 linhas) | 332,86 (com o novo código) / 269,66 | **331,0** (−7,7 s sobre a v3; teto calculado 8,5 s) |
| v6 | 26/09 | configuração da v5 + features `adsb_*` (adsb.lol, 2025 inteiro + jan/jul 2026; 57 % do ranking com evento) | 323,50 / 257,91 | **314,76** (−16,2 s sobre a v5; relação oficial/simulação 0,973) |
| v9 | 27/09 | v7 (corretor com cross-fitting por mês) + janela do LOBT (projeção em `MVT − LOBT ± 3606`, `p` zerado fora dela) | 317,23 / 254,24 | **275,90** (−38,9 s sobre a v6; relação 0,870) |
| v11 | 27/09 | v9 + taxi-in das ARR e vizinhos de `MVT − AOBT_3` no corretor (plano 8) | 311,09 / 245,81 | **266,81** (−9,1 s sobre a v9; relação 0,858) |
| v12 | 27/09 | v11 com a média de 3 corretores (`--conjunto`, plano 9) | 309,78 / 243,96 | **264,74** (−2,1 s sobre a v11) |
| v21 | 28/09 | v20 (base com `ctx_*`: `--base-ctx`; corretor da v18) + regra de Roma | 301,73 / 233,04 | **252,34** (−1,61 s sobre a v19) |
| v19 | 28/09 | v18 (plano 13: METAR, rotação no stand, consistência NM, companhia no corretor) + regra de Roma | 303,82 / 235,25 | **253,95** (−2,91 s sobre a v17) |
| v17 | 28/09 | v16 (plano 11 + 12: regressor com alvo cortado em 2 h; companhia, séries diárias EUROCONTROL e OPDI no corretor) + regra de Roma | 306,65 / 238,65 | **256,86** (−7,49 s sobre a v13) |
| v13 | 28/09 | v12 + pós-regra de Roma (4 voos) | — | **264,35** (−0,39 s sobre a v12) |
| v14 | 28/09 | v12 sem `adsb_lat0`/`adsb_lon0` (plano 10) | 311,88 / — | 266,28 (+1,54 s: a deriva não atrapalha) |
| v10 | 27/09 | diagnóstico: v6 + só as 117 linhas projetadas na janela (garantia ≤ 288,01) | 320,30 / — | 284,17 (a regra vale em 2026) |

A simulação da v3 e da v4 vem do holdout novo (`experiment.py`), mais rigoroso que o
`sim_ranking.py` que mediu a v1 e a v2. A da v5 é a de `nm_retas_6h` (seeds
determinísticas, código do plano 3a).

Candidato v5: `teto.py` v3→v4 com `ms` > 6 h acha 96 linhas e teto de
8,53 s. O candidato é a v3 com essas 96 linhas da v4
(`submissions/outgoing-boat_v5.parquet`); veredito FRÁGIL no `compare.py`,
enviado com o ok do usuário depois do `teto.py`. O oficial deu 331,0 s,
−7,7 s sobre a v3: dentro do teto de 8,53 s e a primeira vez que um ganho
simulado se confirmou no placar.

Alerta v4: a simulação previa −42 s e o oficial deu só −1,5 s (relação
oficial/simulação 0,975, fora de 0,84 ± 0,05). Diagnóstico em
`docs/research/2026-09-24-diagnostico-v4.md`: o ganho simulado vinha de 10 voos
do LIRF (120,9 % do ganho; sem eles a v4 é 9,7 s pior) e media folga que a v3
enviada não tinha (cobertura de `ms` nos extremos do LIRF: 0,905 no ranking
contra 0,679 no holdout); o teto do ganho oficial era −8,5 s. Recomendação:
aplicar as retas só em `nm_missing` com `MVT − SCHED` > 6 h (holdout 334,51 s) e
endurecer a regra de promoção (ganho fora dos 10 maiores voos, ganho em jan e em
jul separados, teto no ranking).

## Roadmap

Feito:

- [x] Inscrição, credenciais e download dos 14 arquivos (com retentativa, a conexão da OSN cai).
- [x] Primeira submissão (v1) e diagnóstico da diferença validação × oficial.
- [x] Correção: outliers no treino; bug de unidade de tempo nas janelas (`// 10**9` com timestamps em µs virava janela de ~7 dias).
- [x] Features de diferença entre horários e arredondamento (v2).
- [x] Base de experimentos: `bin/run` (metade do PC), cache de features, progresso com ETA e RAM, `experiments.jsonl` e `compare.py` com bootstrap pareado.
- [x] Item 0 da parte 2, linha de base na base nova (`base_v2`, 400 rodadas): **454,93 s**, 2m15s, pico de 2,38 GB, `best_iter` 300.
- [x] Item 2 da parte 2, `nm_missing` (`base_nm`): 454,49 s, ganho de 0,4 s (IC 95% −1,8 a 2,5) → não promovido sozinho; ficou no código por entrar sem custo.
- [x] Item 1 da parte 2, dois estágios (`dois_estagios`, 400+400 rodadas): **388,16 s**
  (sem outliers 269,87; NM presente 242,2; NM ausente 2442,18; LIRF 1280,8 → 957,1),
  ganho de 66,8 s (IC 95% 26,3 a 111,0) → novo campeão. Virou a v3
  (480+480 rodadas, 5m23s): **338,7 s** oficiais.
- [x] Re-medida da campeã no código novo (`dois_estagios_r`): 388,16 s, idêntica à v3.
- [x] Item 3 da parte 2, retas por aeroporto em `ms = MVT − SCHED` para os
  voos `nm_missing` (`nm_retas`, `two_stage_nm`): **345,89 s** (NM ausente
  2442 → 1993; LIRF 957 → 717), ganho de 42,3 s (IC 95% 2,0 a 86,5) →
  novo campeão. Virou a v4 (480+480 rodadas): **337,2 s** oficiais, com ganho
  oficial de só 1,5 s (ver alerta em Submissões).
- [x] Item 4 da parte 2, célula aeroporto × `ms` > 2 h (`nm_retas_2h`,
  `--nm-split-ms`): 341,49 s (NM ausente 1944; LIRF 687), ganho de 4,4 s
  sobre `nm_retas` (IC 95% 1,5 a 7,7) → não comprovado (abaixo de 10 s);
  re-testado com seed no plano 3a: FRÁGIL.

Próximo (29/09): plano 13, mais sinais no corretor (METAR, prefixo da companhia nos voos com NM,
rotação no stand, consistência NM); depois validação jan↔jul. Ver `docs/mapa.md`.

Achados de 28/09 (scripts descartáveis): (1) outras janelas não são exatas como a do LOBT:
|BLOCK − IOBT| passa de 3606 s em 0,003 % das DEP (máx. 10.737), EOBT_1 e AOBT_3 bem mais;
projetar a v13 em `MVT − IOBT ± 3606` mexe em 10 voos e vale no máximo −0,2 s. (2) Voos sem NM
parecem ser os em que o casamento com o NM falhou por |BLOCK − LOBT| > 3606: nos 918 (4 %) cuja
chegada em outro dos 10 aeroportos tem NM, BLOCK − LOBT da chegada tem mediana +4.384 s e só
2,5 % cabem na janela. Esses 918 cobrem 3,8 % do Σy² dos sem NM: recuperar o AOBT_3 pela chegada
(|y − proxy| mediano 184 s) não é salto grande. A v12 (`--conjunto`) fica guardada para o envio final. Evidência em
`docs/research/2026-09-27-concorrentes.md`. Plano 3b segue pausado (itens 5 a 8).

Antes de 11/10 (abrir entre 08 e 10/10, decisão de 27/09): repositório público
GPLv3 (condição do prêmio).

Plano 3a (feito), regra robusta, seeds e variante > 6 h:

- [x] `compare.py` com a regra nova (ganho ≥ 10 s, IC > 0, ganho sem os 10
  maiores voos, IC > 0 em jan e em jul; senão FRÁGIL; `--aceitar-fragil`).
- [x] `teto.py`: teto do ganho oficial e gravação de candidato.
- [x] Seeds determinísticas (`--seed`). Campeã de referência
  `ref_dois_estagios_s0`: 383,09 s; seeds 1 e 2: 382,24 e 381,61 → ruído do
  treino ≤ 1,5 s.
- [x] Variante > 6 h (`nm_retas_6h`, `--nm-min-ms 21600`): **332,86 s**, ganho
  de 50,2 s (IC 95% 11,5 a 92,6); sem os 10 maiores voos só +1,5 s (3%); IC
  jan 0,2 a 33,2, jul 19,2 a 145,4 → FRÁGIL (reprova no critério dos 10
  maiores). Gerou o candidato v5 (ver Submissões).
- [x] Promoção da v5 a campeã: re-medida no código final (`nm_retas_6h_r`,
  332,86 s, ganho 0,0 s contra `nm_retas_6h`) e promovida com
  `--aceitar-fragil`, com teto de 8,53 s, ok do usuário e oficial de −7,7 s já
  confirmados.
- [x] Re-teste de `nm_retas_2h` com seed (`nm_retas_2h_s0`): 339,79 s; sem os
  10 maiores voos −6,3 s; IC de jan com limite inferior −7,0 → FRÁGIL.

Plano 3b (pausado em 25/09 depois da tarefa 4; 1 feita, 2 a 4 descartadas), saltos, em ordem de teto (análise de 25/09 na campeã):

Onde está o erro (holdout, 332,86): voos normais com NM = 43 % do erro²
(RMSE 220,7); 204 alarmes falsos (voo normal previsto > 1 h) = 10 %; cauda
que é cópia de um horário = ~10 %; 11 voos "loteria" (sem cópia) = 32 %, dos
quais 2 voos do LFPG = 27 %. Achado: voos LIRF sem NM que decolam no dia
seguinte ao programado são 79 % cópia do SCHED, 16 % "24 h + taxi" (off-block
gravado na data do SCHED) e 4,5 % normais; 25 desses no ranking 2026.

- [x] 1. Medir melhor: `normais_nm`, `alarmes_falsos`, `cauda_copia` e
  `sem_loteria` no `experiment.py`; ganho e IC das duas fatias úteis no
  `compare.py` (informativo, veredito intocado). Campeã re-medida
  (`base_3b`): completo 332,86 (idêntico), normais com NM **220,68**,
  alarmes falsos **204 voos = 10,0 % do erro²**, cauda que é cópia 2.287,20
  (576 voos), **sem loteria 274,42** (as 11 loterias são 32,0 % do erro², daí
  a queda de 332,86 → 274,42). Ruído entre seeds nas fatias novas: ver
  "Modelo atual".
- [x] 2. Alarmes falsos (teto −15 s): os dois experimentos foram descartados e a
  campeã não mudou. Nenhum dos dois mecanismos consegue o que a tarefa pedia
  (baixar `alarmes_falsos.parte_erro2` sem piorar a cauda).
  - 2a, calibração de `p` por célula (`cal_celula`, `--calibrar`): descartado, com
    ganho 0,1 s (IC 95% −0,1 a 0,3), "não comprovado". Isotônica fora do fold
    (5 folds por dia) por célula (LIRF × `nm_missing` × faixa de `ms`) em cima da
    campeã: 332,79 (contra 332,86), `normais_nm` 220,81 (contra 220,68), alarmes
    falsos 207 voos = 10,2 % do erro² (contra 204 = 10,0 %). Motivo: o
    classificador já é calibrado, com `p` médio fora do fold 0,0949 contra taxa
    real de cópia 0,0972, e só 5 das 20 células têm cópias suficientes (≥ 20)
    para ajustar um calibrador; os alarmes falsos não vêm de `p` enviesado na
    média da célula, vêm de voos isolados com `p·ms` grande. Código removido.
  - 2b, híbrido `ŷ = p·reta + (1 − p)·ŷ_regressor` nos voos sem NM
    (`nm_hibrido`, `nm_hibrido_6h`, `--nm-hibrido`): descartado, com −46,5 s sem
    limiar (379,37) e −48,0 s só acima de 6 h (380,90). Sem limiar ele até faz o
    que prometia na fatia-alvo (alarmes falsos 204 → 173 voos, 10,0 % → 5,6 % do
    erro², voos normais com NM intactos), mas paga caro na cauda: cauda que é
    cópia 2.287 → 3.540, LIRF 634,5 → 912,1. Medido nas 52 linhas trocadas pela
    variante de 6 h: cobertura mediana `híbrido/reta` = 0,83, e nas 16 cópias
    verdadeiras (|y − ms| ≤ 60 s) o RMSE vai de 816 para 15.101 s: cortar 17 %
    de um `ms` de 10 h custa milhares de segundos ao quadrado. Nos 30 voos normais
    dessas linhas o híbrido também piora (565 → 734), porque `p` ≈ 0,83 é alto
    demais para proteger normal e baixo demais para não estragar cópia. Código
    removido. Achado para o item 3: o que falta não é escala em `p`, é saber qual
    horário foi copiado, que é exatamente a mistura multiclasse do item 3.
- [x] 3. Mistura por horário copiado (teto −13 s): descartada, perde 4,0 s com a
  reta da campeã (336,87) e 37,8 s sem ela (370,65); a campeã não mudou.
  `copy_mix` (`CopyMixture`): classificador LightGBM `multiclass` de qual horário o
  BLOCK copiou (0 normal 1.167.071 · SCHED 169.206 · EOBT 87.506 · LOBT 12.028 ·
  AOBT_3 304.811 voos do treino), regressor só na classe 0, `ŷ = Σ p_k·(MVT −
  horário_k)` com a massa de horário nulo voltando ao normal, mais `days_shift`
  como feature. A classe "24 h + taxi" tem 6 exemplos em train2025, longe do
  que o multiclasse precisa, então usou-se a taxa empírica de 2025 na única célula
  onde ela existe (LIRF × sem NM × `days_shift` ≥ 1) = 0,164, como manda o plano.

  | métrica | campeã | copy_mix | sem reta |
  |---|---|---|---|
  | completo | 332,86 | 336,87 | 370,65 |
  | normais_nm | 220,68 | 220,80 | 220,80 |
  | y_gt_1h | 4.164,31 | 4.337,89 | 5.216,81 |
  | cauda_copia | 2.287,20 | 2.779,10 | 3.316,82 |
  | alarmes_falsos | 204 voos = 10,0 % | **188 = 8,7 %** | 188 = 7,2 % |

  Ganho pareado: −4,0 s (IC 95% −9,4 a 0,9), sem os 10 maiores −6,5 s, jan −1,3 e
  jul −6,4 → não comprovado. A parte multiclasse em si não é o problema: ela
  baixa os alarmes falsos (10,0 % → 8,7 % do erro²) sem mexer nos voos normais com
  NM (−0,1 s, IC −0,6 a 0,4). Quem custa é a taxa fixa de 0,164 da classe 5:
  ela soma 0,164 × 86.400 ≈ 14.170 s a *todo* voo da célula, e nos 20 voos LIRF sem
  NM com troca de data do holdout (14 cópias puras do SCHED, 5 "24 h + taxi", 1
  normal) o RMSE vai de 9.311 para 12.417, o que sozinho vale ≈ +5,8 s no RMSE
  completo, mais do que os 4,0 s perdidos. Nas cópias de `ms` pequeno (ex.: y =
  5.950 s) a previsão pula de 4.854 para 18.876 s; nos 5 voos que são mesmo "24 h +
  taxi" o componente do SCHED já entregava a ordem certa (ms 58 a 93 k contra y ≈
  87 k), então a classe 5 quase não tem o que ganhar. Código removido (`copy_mix`,
  `copy_class`, `days_shift` e os testes); `experiments.jsonl` guarda as duas corridas
  e o cache foi refeito com as 58 features da campeã.
- [x] 4. Alvo residual sobre `MVT − AOBT_3` (teto do cenário: −13 s): descartado, as
  duas formas empatam com a campeã e a campeã não mudou. `ref = MVT − AOBT_3` quando
  cai em [0, 7200] s, senão `MVT − EOBT_1`, `MVT − LOBT`, `ref_p10`, senão 0 (cobre
  98,4 % dos voos, igual no holdout e no ranking).
  - 4a, alvo residual (`residual_aobt`, `--residual alvo`): o regressor normal aprende
    `y − ref` e prevê `ref + ŷ_res`. Completo 333,06 (contra 332,86), `normais_nm`
    221,02 (contra 220,68), alarmes falsos 229 voos = 10,4 % (contra 204 = 10,0 %);
    ganho −0,2 s (IC 95% −1,1 a 0,8), nos voos normais com NM −0,3 s (IC −0,9 a 0,1) →
    "não comprovado".
  - 4b, `ref` como feature extra do regressor (`residual_feat`, `--residual feature`):
    completo 333,43, `normais_nm` 221,08; ganho −0,6 s (IC 95% −1,2 a 0,1), normais com NM
    −0,4 s (IC −0,7 a −0,1) → "não comprovado".
  - Motivo: `ref` já é feature do regressor (`to_takeoff_from_AOBT_3_flt`) e ele já a
    usa. A correlação entre `ŷ − ref` da campeã e `y − ref` é 0,806, e `ref` sozinha
    erra 372,7 s contra 220,7 da campeã nessa fatia. Reescrever o alvo em torno dela só
    troca a parametrização (e tira do LightGBM a liberdade de ignorar `ref` onde ela é
    ruim, daí os alarmes falsos subirem em 4a). Código removido (`residual_ref`, a flag
    `--residual` e os testes); `experiments.jsonl` guarda as duas corridas.
- [ ] A. adsb.lol → plano 4 (prioridade 1). Discord do desafio: o 3º colocado
  (SoK) usa adsb.lol + clima + stands do X-Plane; GREKI "subiu muito" com jan+jul
  completos; o organizador confirmou que dado aberto declarado vale.
  Teste de 1 dia (EDDM, 15/01/2025): 357 de 361 decolagens casadas por callsign +
  decolagem (mediana 19 s do MVT); off-block = 1º ponto no chão do rastro: |erro|
  mediana 82 s, 41 % a ±60 s, 75 % a ±300 s, contra `MVT − AOBT_3` nos mesmos
  voos: mediana 356 s, 14 % a ±60 s.
  - [x] `src/adsb.py`: recorte diário (caixa ~11 km, chão ou ≤ 3.000 ft, parquet
    zstd, retomável). Jan+jul 2025/2026: 124 dias, ~1,7 GB, ~7 h com 5 processos
    (download ~1,5 a 3 min/dia, leitura 14 a 30 min/dia). Sobreviveu a uma queda de
    energia (dias prontos íntegros; retomar pula os feitos). 29/01/2026 tinha um
    rastro corrompido (`zlib.error`): agora o rastro é pulado e contado.
  - [x] Resto de 2025 (serviço `prc-adsb`, 10 processos): 427 dias completos
    (2025 inteiro + jan/jul 2026), 5,7 GB, 0 rastros corrompidos; sobreviveu a 3
    quedas de energia (espera a rede, reinicia sozinho). 2025-12-31 está no
    repositório de 2026.
  - [x] Plano 4, tarefas 1 a 3 (`src/adsb_events.py`;
    `docs/research/2026-09-26-adsb-cobertura.md`): cobertura boa em EHAM, LEBL,
    EDDF, LSZH, EDDM; nula em LTFM; fraca em LEMD, LFPG, EGLL. Avião visto parado:
    erro mediano 20 a 50 s. Troca direta não ganha; empilhamento fora do fold
    332,86 → 317,44 (−15,4 s), normais 248,2 → 223,9, jan e jul melhoram.
  - [x] Plano 4, tarefa 4: controle do empilhamento sem `adsb_*` 327,25 (a antena
    vale ~10 s); eventos do ano inteiro (1,26 M decolagens); `adsb_*` no treino da
    campeã → 323,50 (**v6, 314,76 oficial**). Empilhamento sobre ela
    (`src/stack.py`) 317,57, MELHOR, mas ainda não enviável (o corretor só existe para
    jan/jul; exige base treinada sem esses meses).
  - [x] Plano 5 (`docs/superpowers/plans/2026-09-27-plano5-v7-crossfit.md`), v7 com
    cross-fitting por mês: split `blind2025` (ano montado como o ranking),
    `src/crossfit.py` (base fora do bloco, meses 2 a 2, P10 por bloco),
    `stack.py --crossfit`, `train.py submit` para campeã `stack_cf`.
    `v7_cf` (corretor treinado em 10 meses fora do bloco; 16 min, pico 6,29 GB):
    **320,67**, normais com NM 205,57 (contra 220,7), sem loteria 260,12. Contra a
    campeã: ganho 2,8 s (IC 95% 1,5 a 4,5), sem os 10 maiores 1,3, jan 3,1
    (1,7 a 5,7), jul 2,6 (0,5 a 4,7) → não comprovado (< 10 s), mas positivo em
    todos os critérios. Contra o corretor só no holdout (317,57): −3,1 s; o ganho
    extra dele estava na cauda de jan/jul (folds por dia dentro dos mesmos meses);
    nos normais o cross-fitting é melhor (+0,7 s). Envio da v7 pede
    `--aceitar-fragil`, `teto.py` e ok do usuário; `train.py submit 7 --forcar`
    (a `v7_cf` foi medida em `4822e01`; depois só `train.py` mudou).
  - [x] Plano 6 (`docs/superpowers/plans/2026-09-27-plano6-seeds.md`), média de seeds
    (`--seeds N`, `models.build_model`/`SeedAvg`): base com 5 seeds `seeds5` 323,29
    (ganho 0,2 s, IC −1,9 a 1,7; normais com NM +1,3 s), 15 min. `v8_cf` (v7 + 5
    seeds, 65 min, pico 6,46 GB): **320,36**, contra a v6 ganho 3,1 s (IC 1,3 a 4,9;
    jul −0,5 a 5,4), contra a v7 só 0,3 s (IC −1,8 a 1,8; normais +0,9) → as
    seeds não pagam 5× o custo; a v7 (1 seed) fica como candidata.
  - [x] Detector (tarefa 5), teste barato de 27/09: descartado (0,8 s < portão de 3 s).
    Coordenadas dos stands aprendidas dos voos vistos parados (mediana lat/lon por
    aeroporto × `STAND_mvt`, ≥ 3 voos: 1.440 stands, 77 % dos voos). Nos voos vistos
    andando do holdout (109 mil), `move + a + b·dist` por aeroporto: |erro| mediano
    311 → 157 s, RMSE 746 → 614 (a constante `a` sozinha já leva a 172/655). Corretor
    barato (5 folds por dia) com `adsb_dist_stand` e `adsb_visto_parado`: 317,57 →
    316,81, normais 206,27 → 205,41; o corretor já aprende o atraso do "visto
    andando" com `adsb_gs0`/`adsb_lat0`/`adsb_lon0`. Scripts descartados.
  - [x] Fila vista pelo ADS-B (`fila_adsb`), teste barato de 27/09: descartado (0,2 s).
    Aviões distintos no chão e andando (gs > 1 kt) por aeroporto × minuto nos recortes,
    lidos no off-block estimado (`MVT − pred`), no meio do táxi e 1 min antes do MVT
    (66,5 % dos voos com contagem). Corretor barato: 317,57 → 317,36, normais 206,27 →
    206,07. As janelas de congestionamento do NM (decolagens e pousos de 10 a 60 min) já
    carregam essa informação. Script descartado.
  - [x] Plano 7 (`docs/superpowers/plans/2026-09-27-plano7-janela-lobt.md`), janela do
    LOBT: em 100 % das 2.062.577 DEP de 2025 com LOBT, |BLOCK − LOBT| ≤ 3606 s
    (`docs/research/2026-09-27-janela-lobt.md`). `--janela-lobt`: previsão projetada em
    `MVT − LOBT ± 3606` e `p` da cópia zerado com o SCHED fora da janela; o corretor
    ganha `dist_lo`/`dist_hi`. Base `janela` 320,29 (ganho 3,2 s, IC 0,4 a 8,7);
    `v9_cf` (v7 + janela, 17 min, pico 6,58 GB) **317,23**: contra a v6 ganho 6,3 s (IC
    2,6 a 12,5; jan 3,3, jul 9,0) e contra a v7 3,4 s → não comprovado (< 10 s). O holdout
    de 2025 quase não tem previsões fora da janela; no ranking a v6 tem 117, e só
    projetá-las garante v6 ≤ 288,0 oficial (se a regra valer em 2026).
    **Oficial (27/09): v9 = 275,90** (−38,9 s sobre a v6; relação oficial/simulação 0,870)
    e v10 (diagnóstico: v6 só com as 117 linhas projetadas) = 284,17 ≤ 288,01 → a regra
    vale em 2026; a janela sozinha valeu −30,6 s e o corretor + base nova −8,3 s.
  - [x] Plano 8 (`docs/superpowers/plans/2026-09-27-plano8-contexto-arr.md`), contexto
    no corretor (`src/contexto.py`, colunas `ctx_*`): taxi-in das chegadas por aeroporto e
    pista (15/60 min), última chegada no mesmo stand, média de `MVT − AOBT_3` das DEP
    vizinhas antes e depois. Teste barato −3,5 s; `v11_cf` 311,09 (ganho 6,1 s sobre a v9,
    IC 4,0 a 8,7, normais +4,9). **Oficial: v11 = 266,81** (−9,1 s sobre a v9).
  - [x] CatBoost como corretor (teste barato 27/09): sozinho 309,53 contra 309,27 do
    LightGBM; corte do alvo em ±7200 s só ajuda o LightGBM (−0,4). Descartado sozinho.
  - [x] Plano 9 (`docs/superpowers/plans/2026-09-27-plano9-conjunto-corretor.md`), média de
    três corretores (LightGBM global, LightGBM por aeroporto, CatBoost; `--conjunto`).
    Teste barato −3,5 s; `v12_cf` 309,78, só 1,3 s sobre a v11 (IC 0,5 a 2,2; jan com IC
    incluindo zero). Não promovida; guardada para o envio final.
  - [x] Média mensal oficial (ansperformance, 27/09): descartada. A média exclui voos sem
    referência (degelo) e a fração válida caiu em jan/2026 (EDDM 0,77 → 0,67); corrigir o viés
    por ela piora o holdout (311,09 → 311,56). A v11 já não tem viés por aeroporto × mês.
  - [x] Pós-regra de Roma sem NM com `MVT − SCHED` em (15 h, 30 h] ("24 h + táxi" × cópia,
    q = 0,62 ajustado fora de jan/jul): holdout 311,09 → 307,21; no ranking são 4 voos.
    Arquivo pronto: `submissions/outgoing-boat_v13.parquet` (v12 + regra). Rejeitado em 27/09 pelo
    limite diário (5 envios por dia UTC, a v6 da madrugada contou); reenviado em 28/09 00:00 UTC: **264,35** (−0,39 s sobre a v12; esperado −6 s).
  - [x] Plano 10 (`docs/superpowers/plans/2026-09-27-plano10-sem-latlon.md`), v14 = v12 sem
    `adsb_lat0`/`adsb_lon0` (deriva 2025 → 2026, dica do GREKI): `20260927-182244-v14_cf` 311,88,
    −2,1 s no holdout (IC −3,5 a −1,0), o que o holdout não consegue medir. Arquivo pronto
    (`submissions/outgoing-boat_v14.parquet`). Oficial (28/09): v14 = 266,28, **1,54 s pior** que a v12 → descartada; v13 (regra de Roma) =
    264,35, −0,39 s. v15 não se justifica.
  - [x] README "Dados externos" e "Reprodução" (tarefa 6, 27/09); `PRC_ADSB_RAIZ` configurável.
- [ ] 5. Features de vizinhos (item 6).
- [ ] 6. Ensemble XGBoost CUDA + seeds LightGBM (item 7), baixa prioridade:
  no Discord, XGBoost ganhou peso zero e pesos de blend ajustados perderam 4/4.
- [ ] 7. `sweep.py` (polimento), baixa prioridade: tuning não significativo
  em LightGBM/CatBoost/XGBoost (relato no Discord).
- [ ] 8. `ablation.py` (poda de features) e `mutmut` (teste do teste).

Pesquisa de 25/09 (Discord do desafio), para não repetir:

- Âncora `MVT − AOBT_3` foi o único passo grande de um time; o resto < 1 s cada.
- Não funcionou para outros times no placar: tuning de hiperparâmetros, pesos de
  blend ajustados no holdout (pesos iguais ganharam 4/4), XGBoost como 3º modelo
  (peso zero), tirar colunas sazonais, pesar linhas por (aeroporto, mês).
- Funcionou: clima com temperatura e spread de ponto de orvalho (degelo em manhãs
  limpas) > flag de neve; um time ganhou 3,4 s consertando fuso no join do clima.
  Sequência de esteira: +0,25 s.
- Regras de validação sugeridas: jan e jul melhorando separados (já temos) e
  rejeitar ganho concentrado em < 100 voos.
- Organização pode criar fase 2 se houver "engenharia reversa do placar".
- Atualização de 26/09: o organizador reafirmou que dado aberto vale ("open data
  sources can be usable to devise a better model"), sem vetar posições de chão
  depois do pushback. Pode haver uma etapa final oculta ("possibly a 1 final
  submission"), com formato (arquivo novo ou código rodado por eles) e período não
  decididos. Consequência: o pipeline inteiro (download do adsb.lol →
  `adsb_events.py` → features → modelo) precisa rodar em outro período/aeroportos
  com um comando, e o ganho tem que vir de generalização, não do placar atual.
- Atualização de 27/09 (Discord e repositórios; detalhes em
  `docs/research/2026-09-27-concorrentes.md`): o organizador liberou usar a média mensal
  publicada de taxi-out por aeroporto (ansperformance.eu), inclusive jan/jul 2026; `_mvt`
  vem do APDF. GREKI (topo): validar treinando em jan e testando em jul (e vice-versa);
  checar deriva 2025 → 2026 de cada entrada (a rede do adsb.lol mudou: features de *onde*
  o avião foi ouvido não transferem, as de *movimento* sim); corrigir uma base forte com
  uma 2ª família de modelos. Janela do LOBT veio do código do elegant-alligator.

## Dados externos

Condição do prêmio (`eligibility.html`): todo dado externo aberto e documentado. Usamos quatro:

| Fonte | O que é | Licença | Como obter |
|---|---|---|---|
| adsb.lol `globe_history_2025` e `globe_history_2026` (<https://github.com/adsblol/globe_history_2025>, <https://github.com/adsblol/globe_history_2026>) | rastros ADS-B/MLAT diários de todo o mundo, um release por dia (2–4 GB) | **ODbL 1.0** | `bin/run src/adsb.py baixar --dias 2025-01,…,2026-07` escolhe a réplica em `PREFERRED_RELEASES.txt`, lê o tar em fluxo e guarda só os pontos no chão ou ≤ 3.000 ft a ±0,10° dos 10 aeroportos (`PRC_ADSB_RAIZ/cut/AAAA-MM-DD.parquet`, ~7–19 MB/dia; 427 dias = 5,7 GB, ~1 dia de download com 10 processos) |
| Séries diárias da EUROCONTROL (<https://ansperformance.eu/csv/>): `atfm_slot_adherence`, `all_pre_departure_delays` e `atc_pre_departure_delays` de 2025 e 2026 | por aeroporto e dia: voos regulados, saídas fora do slot, atraso pré-partida total e de ATC por voo | dados públicos da EUROCONTROL (uso livre com atribuição) | `bin/run src/externos.py baixar` → `data/externo/*.csv` (~33 MB) |
| OPDI v0.0.2 (EUROCONTROL/OpenSky, <https://www.opdi.aero/>), flight lists de 2025-01…2025-12, 2026-01 e 2026-07 | um voo por linha: `icao24`, `adep`, `ades`, `first_seen`, `last_seen` (ADS-B tratado) | open data, "freely used … provided that the data source is attributed" | `bin/run src/externos.py baixar` → `data/externo/opdi/flight_list_AAAAMM.parquet` (~30–55 MB/mês) |
| METAR do IEM ASOS (Iowa State University, <https://mesonet.agron.iastate.edu/request/download.phtml>), 2025-01-01…2026-08-01 dos 10 aeroportos | observação de superfície a cada 30 min: temperatura, ponto de orvalho, vento, rajada, visibilidade, fenômenos (`wxcodes`) e teto | dados públicos do IEM/NOAA (uso livre com atribuição) | `bin/run src/plano13.py baixar` → `data/externo/metar/<ICAO>.csv` (~1,5 MB/aeroporto) |

Do adsb.lol só saem features derivadas por voo (`adsb_*`, `src/adsb_events.py` →
`PRC_ADSB_RAIZ/events.parquet`): off-block e decolagem observados, velocidade no 1º ponto,
pontos e lacunas no chão. Das duas fontes da EUROCONTROL sai parte das colunas `ext_*` do
corretor (`src/externos.py`, só com `--externos`; as outras, `ext_taxa_cia*`, saem só dos
dados do organizador): das séries diárias, a fração de voos
regulados do dia, a fração deles que saiu fora do slot e os minutos de atraso pré-partida
total e de ATC; do OPDI, quanto tempo a aeronave ficou em solo desde o pouso anterior e se
esse pouso foi no mesmo aeroporto (casando callsign e horário a ±600 s, ou só aeroporto a
±90 s). Do METAR saem as colunas `met_*` do corretor (`src/plano13.py`, só com `--plano13`):
a observação mais recente do aeroporto até 2 h antes do movimento e `met_degelo` (frio com
ar úmido ou precipitação). Nenhum dado de 2026 do organizador (verdade) é usado; os meses
do ranking entram só como entrada (rastros de jan/jul 2026 e METAR), como qualquer feature.

Não usamos layout de aeroporto nem dados de placar.

## Reprodução (da v12, 264,74)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                      # chaves MinIO da OSN e TEAM_NAME
export PRC_ADSB_RAIZ=/caminho/com/6GB     # recortes do adsb.lol e events.parquet
.venv/bin/python src/s3.py download       # dados do organizador em data/
bin/run src/adsb.py baixar --dias 2025-01,2025-02,2025-03,2025-04,2025-05,2025-06,2025-07,2025-08,2025-09,2025-10,2025-11,2025-12,2026-01,2026-07 --procs 10
bin/run src/adsb_events.py                # eventos por voo → $PRC_ADSB_RAIZ/events.parquet
bin/run src/cache.py                      # features dos 5 splits (inclui ctx_* de src/contexto.py)
bin/run src/experiment.py janela --model two_stage_nm --nm-min-ms 21600 --seed 0 --janela-lobt
bin/run src/stack.py v12_cf --crossfit --conjunto --base <id da corrida janela>
bin/run src/compare.py <id da v12_cf> --promover   # ou champion.json já versionado
bin/run src/train.py submit 12            # submissions/<TEAM>_v12.parquet (~32 min, pico 7,5 GB)
```

Seeds fixas (`deterministic`, `force_row_wise`): a mesma máquina reproduz o mesmo número no LightGBM; o CatBoost do `--conjunto` roda na GPU e pode variar na última casa.
Hardware usado: Xeon E5-2670 v3 (6 núcleos físicos via `bin/run`), 15 GB de RAM.

## Uso

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                              # chaves e TEAM_NAME
.venv/bin/python src/s3.py download               # dados em data/
bin/run src/cache.py                              # features em cache (uma vez, sozinho)
bin/run src/experiment.py <nome> --model two_stage [--seed N] [--seeds N]
bin/run src/experiment.py <nome> --model two_stage_nm [--nm-split-ms] [--nm-min-ms 21600] [--janela-lobt] [--reg-corte 7200] [--reg-sem-lirf-nm] [--seed N] [--seeds N] [--sem-feature COLUNA]
bin/run src/compare.py <id> --promover            # decide contra o campeão (FRÁGIL não promove)
bin/run src/compare.py <id> --promover --aceitar-fragil   # só após teto.py e ok do usuário
bin/run src/teto.py <base.parquet> <novo.parquet> --oficial-base <RMSE> [--min-ms 21600] [--salvar submissions/<TEAM>_vN.parquet]
bin/run src/train.py submit N [--forcar] [--corrida <id>]   # gera a vN (não envia); --corrida usa a receita daquela corrida do experiments.jsonl, sem mexer no champion.json
.venv/bin/python src/s3.py submit submissions/<TEAM>_vN.parquet   # só após aprovação
bin/run src/adsb.py baixar [--dias 2025-01,2025-07] [--dia AAAA-MM-DD] [--procs 5]   # recortes adsb.lol no SSD
bin/run src/adsb_events.py                        # eventos por voo → <SSD>/events.parquet
bin/run src/stack.py <nome> [--base <id>] [--sem-adsb] [--crossfit [--seeds N] [--conjunto] [--externos] [--plano13] [--reusar-oof <id>]] [--sem-feature COLUNA]   # corretor fora do fold (teste barato) ou fora do bloco no ano (enviável; ~25 min, rodar via systemd-run --user); --conjunto = média de global, por aeroporto e CatBoost; --externos = colunas ext_*; --plano13 = METAR, rotação no stand, consistência NM e a companhia; --reusar-oof = lê o oof daquela corrida em vez de recalcular a base fora do bloco (só com base e base_config idênticas; ~7 min)
bin/run src/externos.py baixar                    # séries diárias da EUROCONTROL e flight lists do OPDI → data/externo/ (pula o que já existe)
bin/run src/plano13.py baixar                     # METAR dos 10 aeroportos → data/externo/metar/ (pula o que já existe)
.venv/bin/python ferramentas/projecao.py          # placar do dia + docs/projecao.md
.venv/bin/python ferramentas/auditoria.py         # docs/auditoria/AAAA-MM-DD.md
.venv/bin/python -m pytest -q
```

Todo comando pesado passa pelo `bin/run`, que limita a 6 núcleos físicos e
prioridade baixa. Limite do placar: 5 envios por dia UTC (zera às 00:00 UTC, 21h em Brasília), 1 GB por bucket. Conta
a melhor submissão. A organização monitora quem tenta "aprender com o
placar": testar localmente e enviar só o que melhorou.

## Rotina diária (auditoria e projeção)

O timer `prc-auditoria` (systemd do usuário, 09:00 local, `Persistent=true`: roda ao
ligar se o PC estava desligado) executa `ferramentas/auditoria.py`, que:

- baixa a foto do placar (`placar/AAAA-MM-DD.json`) e regenera `docs/projecao.md`;
- confere bucket × `submissions.jsonl` × placar, campeã × `experiments.jsonl` ×
  código (`src_hash`), README/CONTEXTO citando campeã e melhor nota, `src/` e
  `ferramentas/` listados no README, idade do `saltos.json`, pytest, git limpo e
  enviado nos dois repositórios, `events.parquet` em dia, serviços `prc-*` sem falha;
- grava `docs/auditoria/AAAA-MM-DD.md` com ✅/⚠️ (não faz commit).

Na primeira conversa do dia: ler a auditoria, corrigir as ⚠️ e o texto velho que a
máquina não pega (roadmap, "Retomar" do CONTEXTO, caixas do plano), atualizar
`saltos.json` com o que foi medido e fazer commit. Projeção: cortes do 1º/3º/10º/50º
no prazo em três cenários (parado, desacelerando com meia-vida de 7 dias, ritmo
atual) e Monte Carlo da nossa nota final sobre a fila de `saltos.json`.

## Estrutura

```
prc-taxiout-2026/
  bin/run             # orçamento de hardware (taskset + nice) para tudo que é pesado
  src/s3.py           # listar, baixar e enviar (MinIO)
  src/runlog.py       # progresso por fase, ETA, RAM/CPU/GPU e registro das corridas
  src/cache.py        # splits com features prontas (train2025, holdout2025, full2025, ranking)
  src/features.py     # features e referência P10
  src/models.py       # SingleLGBM, TwoStage e a combinação por esperança
  src/experiment.py   # experimento na simulação calibrada
  src/compare.py      # bootstrap pareado por dia, regra robusta (MELHOR/FRÁGIL) e campeão
  src/teto.py         # teto do ganho oficial antes de enviar; grava candidato
  src/train.py        # versão final a partir do campeão (só gera o arquivo)
  src/adsb.py         # recorte diário do adsb.lol (ODbL) perto dos 10 aeroportos
  src/adsb_events.py  # decolagens no ADS-B, off-block observado, casamento, features adsb_*
  src/stack.py        # corretor LightGBM sobre uma corrida base (fora do fold no holdout, ou --crossfit)
  src/crossfit.py     # previsões da base fora do bloco (meses 2 a 2) para o corretor
  src/contexto.py     # taxi-in das chegadas e vizinhos de MVT − AOBT_3 (colunas ctx_*, só no corretor)
  src/externos.py     # dados abertos: taxa de cópia por companhia, séries diárias e OPDI (colunas ext_*)
  src/plano13.py      # METAR, rotação no stand, consistência NM e companhia (só no corretor, --plano13)
  ferramentas/projecao.py   # placar do dia e projeção até o prazo (docs/projecao.md)
  ferramentas/auditoria.py  # auditoria diária (docs/auditoria/)
  submissions.jsonl   # nossos envios com a nota oficial (versionado)
  saltos.json         # fila de saltos candidatos: ganho estimado, chance, dias (versionado)
  placar/             # fotos diárias do placar público (versionado)
  tests/              # pytest
  experiments.jsonl   # uma linha por corrida (versionado)
  champion.json       # config campeã (versionado)
  docs/               # specs, planos e pesquisa
  data/               # parquet baixados (ignorado pelo git)
  submissions/        # arquivos gerados (ignorado pelo git)
  LICENSE             # GPLv3 (exigido para prêmio)
```

## Leaderboard

<https://prc-challenge-2026.vercel.app/>. Em 27/09/2026: 186 equipes, 1º 224,50, 3º
228,59, 10º 242,81, 50º 278,39. Nós: **252,34 s** (v21, 28/09; antes 253,95, 256,86, 264,35, 264,74, 266,81, 275,90, 314,76, 331,0,
338,7 e 384,7). Fotos diárias em `placar/`.

## Referências

- Indicador oficial: <https://www.eurocontrol.int/prudata/dashboard/metadata/additional-taxi-out-time/>
- Repositórios de equipes 2026 (públicos), ex.:
  <https://github.com/ahmetabdullahgultekin/prc-taxiout-2026>
- Edições anteriores: <https://github.com/prc-data-challenge-2024>,
  <https://github.com/prc-data-challenge-2025>
