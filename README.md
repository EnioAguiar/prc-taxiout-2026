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
SCHED em 17%. O off-block oficial costuma ser *cópia* de um desses horários.

**Outliers entram na nota:** a verdade oficial mantém taxi-outs de horas
(até 36 h). Eles são ~37% do erro quadrático. Cortá-los do treino/validação
deixa a validação otimista e o modelo cego para eles (erro da v1).

## Modelo atual (`src/`)

- Referência P10 por (aeroporto, stand, pista) com fallback para
  (aeroporto, pista) e aeroporto; calculada só no fold de treino.
- Tempo entre horários planejados/NM (SCHED, LOBT, IOBT, EOBT, AOBT_3)
  e a decolagem.
- Diferenças entre esses horários (pares) e flag de arredondamento
  (segundos zerados, minuto múltiplo de 5), para indicar qual foi copiado
  como off-block.
- Congestionamento: decolagens do aeroporto, pousos do aeroporto e
  decolagens da mesma pista, antes e depois da decolagem, em janelas
  de 10/20/30/60 min.
- Hora do dia, dia da semana e as categóricas (aeroporto, pista,
  stand, avião, companhia...).
- LightGBM, objetivo L2 (alinhado ao RMSE), 1500 rodadas, todos os
  núcleos da CPU; mostra progresso a cada 100 rodadas.
- Treina com **todos** os voos (sem corte de outliers) e sem limitar a
  previsão.
- Validação: `sim_ranking.py` monta jan+jul/2025 à parte, com o alvo
  apagado como no ranking, e mede RMSE com e sem outliers. É a que mais
  se aproxima da nota oficial (pessimista: 460 local vs 385 oficial na v2).

## Submissões

| Versão | Data | Mudança | Simulação (completo / sem outliers) | Oficial |
|---|---|---|---|---|
| v1 | 24/09 | modelo base; treino sem y ≥ 3 h, previsão limitada a 3 h | 535,9 / 264,4 | 514,5 |
| v2 | 24/09 | treino com outliers, sem limite; bug de unidade das janelas de congestionamento corrigido; features de diferença e arredondamento | 460,4 / 290,2 | **384,7** (~132º de 188) |

## Roadmap

Feito:

- [x] Inscrição, credenciais e download dos 14 arquivos (com retentativa, a conexão da OSN cai).
- [x] Primeira submissão (v1) e diagnóstico da diferença validação × oficial.
- [x] Correção: outliers no treino; bug de unidade de tempo nas janelas (`// 10**9` com timestamps em µs virava janela de ~7 dias).
- [x] Features de diferença entre horários e arredondamento (v2).
- [x] Progresso do treino visível.

Próximo:

1. [ ] Separar em duas partes: classificador "voo extremo / off-block copiado de qual horário" + regressão por caso; combinar pela esperança (minimiza RMSE). Hoje o modelo único piora os voos normais (264 → 290 s).
2. [ ] Treinos mais rápidos: cache das features em parquet, parada antecipada, configuração leve para experimentos.
3. [ ] Investigar LIRF (Roma): 1,2% dos voos > 1 h, pior aeroporto.
4. [ ] Ler repositórios públicos de outras equipes e o Discord sobre os outliers.
5. [ ] Depois: tuning, meteorologia (METAR), ensemble com CatBoost/XGBoost em GPU.
6. [ ] Antes de 11/10: repositório público GPLv3 (condição do prêmio).

## Uso

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env              # preencher chaves e TEAM_NAME

.venv/bin/python src/s3.py ls        # conferir acesso
.venv/bin/python src/s3.py download  # baixa para data/
.venv/bin/python src/train.py validate
.venv/bin/python src/train.py submit 1           # gera submissions/<TEAM>_v1.parquet
.venv/bin/python src/s3.py submit submissions/<TEAM>_v1.parquet
```

Limite: 5 envios por dia, 1 GB por bucket. Conta a melhor submissão.
A organização monitora quem tenta "aprender com o placar": testar
localmente e enviar só o que melhorou.

## Estrutura

```
prc-taxiout-2026/
  src/s3.py        # listar, baixar e enviar (MinIO)
  src/features.py  # features e referência P10
  src/train.py     # validação e geração da submissão
  sim_ranking.py   # simula o ranking em jan+jul/2025 (validação principal)
  data/            # parquet baixados (ignorado pelo git)
  submissions/     # arquivos gerados (ignorado pelo git)
  LICENSE          # GPLv3 (exigido para prêmio)
```

## Leaderboard

<https://prc-challenge-2026.vercel.app/>. Em 24/09/2026: 188 equipes,
melhor RMSE 234,1 s, mediana ~305 s. Nós: 384,7 s (~132º).

## Referências

- Indicador oficial: <https://www.eurocontrol.int/prudata/dashboard/metadata/additional-taxi-out-time/>
- Repositórios de equipes 2026 (públicos), ex.:
  <https://github.com/ahmetabdullahgultekin/prc-taxiout-2026>
- Edições anteriores: <https://github.com/prc-data-challenge-2024>,
  <https://github.com/prc-data-challenge-2025>
