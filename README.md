# PRC Data Challenge 2026 — Taxi-out time

Competição da EUROCONTROL Performance Review Commission (PRC) com a
OpenSky Network (OSN). Não é Kaggle.

- Site: <https://ansperformance.eu/study/data-challenge/dc2026/>
- Dados: <https://prc-data-challenge-2026.netlify.app/data.html>
- Termos/elegibilidade: <https://prc-data-challenge-2026.netlify.app/eligibility.html>
- Discord: servidor OpenSky (<https://discord.gg/RPh89jpVVz>), canal `#prc-data-competition`

## Problema

Prever o **taxi-out** de cada decolagem: segundos entre sair do
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

**Vazamento (verificado em 24/09):** `AOBT_3_flt` (off-block real do NM)
está preenchido em 98,5% das DEP do ranking, então pode ser usado. Ele
bate com o off-block oficial (±2 min) em 38% dos voos; EOBT/LOBT em ~20%;
SCHED em 17%. O BLOCK oficial tem resolução de segundo e AOBT_3/EOBT/SCHED
de minuto: a proximidade de 38% com AOBT_3 não é cópia. Cópia existe só na
cauda, e do SCHED: 84,7% dos y > 3 h têm |BLOCK − SCHED| ≤ 60 s
(`docs/research/2026-09-24-forense-dados.md`).

**Outliers entram na nota:** a verdade oficial mantém taxi-outs de horas
(até 36 h). Eles são ~37% do erro quadrático. Cortá-los do treino/validação
deixa a validação otimista e o modelo cego para eles (erro da v1).

## Modelo atual (`src/`)

Campeão em `champion.json`: **dois estágios com retas para voos sem NM**
(`two_stage_nm`), 345,89 s na simulação.

- **Estágio 1:** classificador LightGBM de `eq = |BLOCK − SCHED| ≤ 60 s`,
  a cópia que existe na cauda.
- **Estágio 2:** regressor L2 treinado só nos voos normais (`~eq`).
- **Combinação pela esperança:** `ŷ = p·(MVT − SCHED) + (1 − p)·ŷ_normal`,
  com piso 0; nunca argmax, porque errar a classe custa horas². Sem SCHED,
  só o regressor.
- **Voos sem registro NM (`nm_missing`):** a previsão acima é trocada por
  uma reta `y = a + b·(MVT − SCHED)` ajustada por aeroporto (com
  `--nm-split-ms`, por aeroporto × atraso > 2 h); grupos com < 50 voos usam
  a reta global; piso 0. Com `--nm-min-ms S`, a reta só troca os voos com
  `MVT − SCHED` > S (ex.: 21600 = 6 h); os demais ficam com o dois estágios.
- Features: referência P10 por (aeroporto, stand, pista) com fallback,
  calculada só no fold de treino; tempos entre os horários planejados/NM
  (SCHED, LOBT, IOBT, EOBT, AOBT_3) e a decolagem; diferenças entre esses
  horários e flags de arredondamento; congestionamento (decolagens e pousos
  do aeroporto e da pista, janelas de 10/20/30/60 min); hora, dia da semana,
  categóricas; `nm_missing` (voo sem linha do Network Manager).
- LightGBM com `num_threads=12` (6 núcleos físicos × 2; 24 threads é 2–4×
  mais lento — `docs/research/2026-09-24-hardware-benchmark.md`).
- **Seeds determinísticas:** `--seed N` fixa as seeds do LightGBM com
  `deterministic` e `force_row_wise`; a mesma seed reproduz o mesmo número.
  Ruído do treino medido com 3 seeds da campeã de referência: ≤ 1,5 s.
- Treina com **todos** os voos (sem corte de outliers) e sem limitar a
  previsão.
- Validação: `src/experiment.py` simula o ranking (jan+jul/2025 com o alvo
  apagado como no oficial) e grava a corrida em `experiments.jsonl`;
  `src/compare.py` decide por bootstrap pareado por dia e atualiza
  `champion.json` (que também guarda `src_hash` e `git_commit` do código que
  mediu o campeão). Veredito **MELHOR** só se tudo valer: ganho ≥ 10 s e IC
  95% > 0; ganho sem os 10 voos de maior ganho > 0 e ≥ 10% do ganho cheio; IC
  > 0 em jan e em jul separados. Senão **FRÁGIL**, que não promove; só
  `--aceitar-fragil` promove, e ele só se usa depois do `teto.py` e com o ok do
  usuário. Sob essa regra v2→v3 promoveria e v3→v4 seria barrada. A razão
  oficial/simulação é pessimista, mas não comparável entre versões: 0,836 na
  v2 veio da simulação antiga (`sim_ranking.py`) e 0,873 na v3 do holdout
  novo — a v2 medida no holdout novo daria 0,846.
- `src/teto.py` estima, antes de enviar, o teto do ganho oficial: compara
  dois arquivos de envio nas linhas que diferem (opcionalmente só com
  `MVT − SCHED` > `--min-ms`) contra um oráculo otimista `y = MVT − SCHED`;
  ganho simulado > 2 × teto = a simulação mede folga que o modelo final não
  tem. `--salvar` grava um candidato (base + novo só nessas linhas).
- `src/train.py submit N` refaz o campeão no ano inteiro com o `best_iter`
  (ou as rodadas configuradas, se não houver) × 1,2 (full2025 tem 2,085 M
  linhas contra 1,741 M do treino) e só gera o arquivo — o envio é um comando
  à parte. Aborta se o código mudou desde a promoção do campeão; `--forcar`
  ignora a checagem.

## Submissões

| Versão | Data | Mudança | Simulação (completo / sem outliers) | Oficial |
|---|---|---|---|---|
| v1 | 24/09 | modelo base; treino sem y ≥ 3 h, previsão limitada a 3 h | 535,9 / 264,4 | 514,5 |
| v2 | 24/09 | treino com outliers, sem limite; bug de unidade das janelas de congestionamento corrigido; features de diferença e arredondamento | 460,4 / 290,2 | **384,7** |
| v3 | 24/09 | dois estágios (classificador da cópia do SCHED + regressor) e `nm_missing`; base de experimentos nova | 388,16 / 269,87 | **338,7** |
| v4 | 24/09 | `two_stage_nm`: retas por aeroporto em `MVT − SCHED` para os voos sem NM | 345,89 / 285,52 | **337,2** |
| v5 | 24/09 | v3 + retas só em NM ausente com atraso > 6 h (96 linhas) | 332,86 (com o novo código) / 269,66 | candidato não enviado |

A simulação da v3 e da v4 vem do holdout novo (`experiment.py`), mais rigoroso que o
`sim_ranking.py` que mediu a v1 e a v2. A da v5 é a de `nm_retas_6h` (seeds
determinísticas, código do plano 3a).

**Candidato v5:** `teto.py` v3→v4 com `ms` > 6 h acha 96 linhas e teto de
8,53 s. O candidato é a v3 com essas 96 linhas da v4
(`submissions/outgoing-boat_v5.parquet`); veredito FRÁGIL no `compare.py`,
não enviado, aguardando decisão do usuário.

**Alerta v4:** a simulação previa −42 s e o oficial deu só −1,5 s (relação
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
- [x] Item 0 da parte 2 — linha de base na base nova (`base_v2`, 400 rodadas): **454,93 s**, 2m15s, pico de 2,38 GB, `best_iter` 300.
- [x] Item 2 da parte 2 — `nm_missing` (`base_nm`): 454,49 s, ganho de 0,4 s (IC 95% −1,8 a 2,5) → **não promovido sozinho**; ficou no código por entrar sem custo.
- [x] Item 1 da parte 2 — dois estágios (`dois_estagios`, 400+400 rodadas): **388,16 s**
  (sem outliers 269,87; NM presente 242,2; NM ausente 2442,18; LIRF 1280,8 → 957,1),
  ganho de 66,8 s (IC 95% 26,3 a 111,0) → **novo campeão**. Virou a v3
  (480+480 rodadas, 5m23s): **338,7 s** oficiais.
- [x] Re-medida da campeã no código novo (`dois_estagios_r`): 388,16 s, idêntica à v3.
- [x] Item 3 da parte 2 — retas por aeroporto em `ms = MVT − SCHED` para os
  voos `nm_missing` (`nm_retas`, `two_stage_nm`): **345,89 s** (NM ausente
  2442 → 1993; LIRF 957 → 717), ganho de 42,3 s (IC 95% 2,0 a 86,5) →
  **novo campeão**. Virou a v4 (480+480 rodadas): **337,2 s** oficiais — ganho
  oficial de só 1,5 s (ver alerta em Submissões).
- [x] Item 4 da parte 2 — célula aeroporto × `ms` > 2 h (`nm_retas_2h`,
  `--nm-split-ms`): 341,49 s (NM ausente 1944; LIRF 687), ganho de 4,4 s
  sobre `nm_retas` (IC 95% 1,5 a 7,7) → **não comprovado** (abaixo de 10 s);
  re-testado com seed no plano 3a: FRÁGIL.

Próximo: item 5 — alvo residual sobre `MVT − AOBT_3`.

Depois: features de vizinhos, ensemble
(XGBoost CUDA + seeds LightGBM), METAR e poda de features. E, antes de
11/10, repositório público GPLv3 (condição do prêmio).

Plano 3a (feito) — regra robusta, seeds e variante > 6 h:

- [x] `compare.py` com a regra nova (ganho ≥ 10 s, IC > 0, ganho sem os 10
  maiores voos, IC > 0 em jan e em jul; senão FRÁGIL; `--aceitar-fragil`).
- [x] `teto.py`: teto do ganho oficial e gravação de candidato.
- [x] Seeds determinísticas (`--seed`). Campeã de referência
  `ref_dois_estagios_s0`: 383,09 s; seeds 1 e 2: 382,24 e 381,61 → ruído do
  treino ≤ 1,5 s.
- [x] Variante > 6 h (`nm_retas_6h`, `--nm-min-ms 21600`): **332,86 s**, ganho
  de 50,2 s (IC 95% 11,5 a 92,6); sem os 10 maiores voos só +1,5 s (3%); IC
  jan 0,2 a 33,2, jul 19,2 a 145,4 → **FRÁGIL** (reprova no critério dos 10
  maiores). Gerou o candidato v5 (ver Submissões).
- [x] Re-teste de `nm_retas_2h` com seed (`nm_retas_2h_s0`): 339,79 s; sem os
  10 maiores voos −6,3 s; IC de jan com limite inferior −7,0 → **FRÁGIL**.

Plano 3b (pendente) — melhoria contínua e caça a "ouro falso" e lixo:

- [ ] `sweep.py`: grade de variantes da campeã (lr, folhas, rodadas), CPU e
  XGBoost GPU em paralelo dentro de metade do PC; tudo passa pelo `compare.py`.
- [ ] `ablation.py`: tirar um grupo de features por vez; o que não faz falta é
  lixo e sai.
- [ ] Teste do teste: `mutmut` periódico em `compare`, `train` e `cache`
  (mutação que não derruba nenhum teste = teste fraco).
- [ ] Híbrido para voos sem NM: a reta piora os voos normais sem NM
  (1.088 → 1.324 s) e ganha nos 56 extremos; combinar reta e regressor
  (ex.: pela probabilidade de cópia) em vez de trocar tudo pela reta.

## Uso

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                              # chaves e TEAM_NAME
.venv/bin/python src/s3.py download               # dados em data/
bin/run src/cache.py                              # features em cache (uma vez, sozinho)
bin/run src/experiment.py <nome> --model two_stage [--seed N]
bin/run src/experiment.py <nome> --model two_stage_nm [--nm-split-ms] [--nm-min-ms 21600] [--seed N]
bin/run src/compare.py <id> --promover            # decide contra o campeão (FRÁGIL não promove)
bin/run src/compare.py <id> --promover --aceitar-fragil   # só após teto.py e ok do usuário
bin/run src/teto.py <base.parquet> <novo.parquet> --oficial-base <RMSE> [--min-ms 21600] [--salvar submissions/<TEAM>_vN.parquet]
bin/run src/train.py submit N [--forcar]          # gera a vN (não envia)
.venv/bin/python src/s3.py submit submissions/<TEAM>_vN.parquet   # só após aprovação
.venv/bin/python -m pytest -q
```

Todo comando pesado passa pelo `bin/run`, que limita a 6 núcleos físicos e
prioridade baixa. Limite do placar: 5 envios por dia, 1 GB por bucket. Conta
a melhor submissão. A organização monitora quem tenta "aprender com o
placar": testar localmente e enviar só o que melhorou.

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
  tests/              # pytest
  experiments.jsonl   # uma linha por corrida (versionado)
  champion.json       # config campeã (versionado)
  docs/               # specs, planos e pesquisa
  data/               # parquet baixados (ignorado pelo git)
  submissions/        # arquivos gerados (ignorado pelo git)
  LICENSE             # GPLv3 (exigido para prêmio)
```

## Leaderboard

<https://prc-challenge-2026.vercel.app/>. Em 24/09/2026: 188 equipes,
melhor RMSE 234,1 s, mediana ~305 s. Nós: 338,7 s (v3; antes 384,7).

## Referências

- Indicador oficial: <https://www.eurocontrol.int/prudata/dashboard/metadata/additional-taxi-out-time/>
- Repositórios de equipes 2026 (públicos), ex.:
  <https://github.com/ahmetabdullahgultekin/prc-taxiout-2026>
- Edições anteriores: <https://github.com/prc-data-challenge-2024>,
  <https://github.com/prc-data-challenge-2025>
