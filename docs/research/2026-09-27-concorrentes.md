# Repositórios de outros times e Discord — 27/09/2026

Leitura de código público (GPLv3, condição do prêmio) e da conversa do Discord de 27/09.
Nenhum código deles foi executado; números de placar/validação são os que eles publicam.

## Placar × validação

| Time (repositório) | Oficial | Validação própria | Razão |
|---|---|---|---|
| elegant-alligator (`javidmardanov/PRC-Data-Challenge-2026`) | 268,56 | 257,13 (jul+nov/2025) | — |
| zestful-fountain (`Phoenix-Ops-LTD/prc2026-taxiout`) | 273,56 | 312,97 (jan+jul/2025) | 0,87 |
| knowledgeable-helicopter (`satam2/knowledgeable-helicopter`) | 278,01 | 331,18 (sazonal) | 0,84 |
| jubilant-vase (`sergiuv11/prc-data-challenge-2026`) | 286,66 | — | — |
| vibrant-lollipop (`ahmetabdullahgultekin/prc-taxiout-2026`) | 331,23 | — | — |
| nós, v6 → v9 | 314,76 → 275,90 | 323,50 → 317,23 | 0,97 → 0,87 |

A nossa razão oficial/simulação era a mais alta; com a janela do LOBT (v9) ficou igual à
dos outros. Hipótese para o resto da diferença: 2026 tem mais cobertura ADS-B em LEMD/EGLL
que jan/jul 2025 e a cauda de 2026 é diferente (ganhos de cauda da v3–v5 transferiram mal).

## O que usamos deles

- **Janela do LOBT** (elegant-alligator, `scripts/queue_features.py:68,82`, com `assert`
  no treino): limita toda previsão a `MVT − LOBT ± 3606`. Conferido: 100 % das 2.062.577
  DEP de 2025 com LOBT. Virou o plano 7 (v9 = 275,90; v10 = 284,17).

## Candidatos (ganho que eles reportam)

- **Resíduo sobre relógio NM contínuo** (zestful `traffic_features.py:77-114`; helicopter
  `models/residual.py`): alvo `y − offset`, offset = `MVT − AOBT_3` (se em [−7200, 172800]
  e `ADEP_flt == ADEP_mvt`), senão `MVT − LOBT`, `MVT − EOBT_1`, senão 1000 s; LIRF sem IOBT
  → `MVT − SCHED`. Helicopter aplica a correção também a proxies > 2 h (−30,6 s na validação
  deles). Nosso teste do plano 3b (alvo residual só com proxy em [0, 7200]) empatou.
- **Resíduo cortado em ±7200 s só no treino + CatBoost fundo** (zestful
  `arrival_boost_contest.py:73`): −7,11 s oficial, o maior salto isolado deles.
- **Taxi-in das linhas ARR do ranking** (zestful `arrival_features.py:15-78`; alligator
  `scripts/arrival_context.py`): média/desvio do taxi-in em 15/30/60 min por
  aeroporto/pista e última chegada no mesmo stand. −2,62 s oficial.
- **Vizinhos de `MVT − AOBT_3` passados e futuros** (zestful `nm_neighbor_features.py`,
  `nm_following_features.py`): contagem, média, desvio, próprio − média, último/próximo por
  aeroporto/pista/stand em 15/60 min. −0,88, −1,42 e −1,42 s oficiais.
- **Especialista para LIRF sem IOBT** (zestful, CatBoost profundidade 5, 5 seeds; helicopter,
  especialista para todo voo sem AOBT_3: 491 → 369 na validação deles).
- **Duração planejada × real do voo NM** (zestful `duration_contest.py:46-54`): −1,42 s.
- **Um modelo por aeroporto 50/50 com o global** (zestful v9; jubilant-vase −3,8 s).
- **Roma sem NM por faixa de `MVT − SCHED`**, inclusive "24 h + táxi" (alligator
  `scripts/missing_specialist.py:23-70`).

## Discord 27/09

- Organizador (espinielli): pode usar a média mensal publicada de taxi-out por aeroporto
  (e por pista, dashboard AIU) de jan/jul 2026 como entrada. `_mvt` vem do APDF.
  Dataset: <https://ansperformance.eu/reference/dataset/taxi-out-additional-time/>.
- GREKI (topo, usa adsb.lol): validar em jan → jul e jul → jan (CV de 12 meses ou por dia
  enganou); checar deriva 2025 → 2026 de cada entrada (rede do adsb.lol mudou: *onde* foi
  ouvido não transfere, *movimento* sim); manter linhas em que registro e sensor discordam;
  corrigir base forte com 2ª família de modelos.
- piyush7911 (237–240): 1 % das linhas = 2/3 do erro² (off-block gravado como programado,
  antecipado ou padrão; LTFM e Roma em julho sem ADS-B); regras de 2025 não passaram a 2026.

## Deriva do ADS-B (medida em 27/09)

Cobertura (voos com evento) jan/jul 2025 → jan/jul 2026: EGLL 14 % → 79 %, LEMD 2 % → 67 %,
EDDM 62 % → 96 %, EDDF 73 % → 56 %, LIRF 45 % → 24 %. O ano de 2025 inteiro já contém o regime
novo (LEMD ~95 % em mar–jun e set–dez; EGLL até 39 % em dez), então o treino o vê; o holdout
jan/jul 2025 é que subestima o ADS-B. Maior deriva de distribuição: `adsb_lon0` (KS 0,26) e
`adsb_lat0` (0,13) — candidatas a sair (só mensurável no placar).

## Releitura de 28/09 (ideias ainda não testadas)

Nenhuma equipe do top 11 tem código público. Espelho do placar: <https://prc-leaderboard.fly.dev/data.json>.
kind-mango (265,27, logo atrás de nós): `skylinkapi/prc-data-challenge-2026-kind-mango`.

| Ideia | Onde | Ganho que reportam |
|---|---|---|
| Regressor-base sem as linhas que ele não serve (LIRF inteiro, y > 80.000) / resíduo cortado em ±7200 só no treino | kind-mango README:124 (v51); zestful `arrival_boost_contest.py:62,73` | −5,99 oficial; −7,11 oficial (misturado com CatBoost fundo) |
| Rotação no stand: tipo e prefixo da companhia da última ARR iguais ao da DEP | zestful `arrival_features.py:29-30,64-68` | parte do bloco ARR (−2,62 oficial) |
| Consistência NM: duração planejada EOBT_1→ARVT_1 × real AOBT_3→ARVT_3, campos mvt × flt iguais | zestful `duration_contest.py:36-55`; kind-mango v65 `plan_nm_taxi` | −1,42; −1,22 oficiais |
| Prefixo da companhia (regex em `FLIGHT_mvt`) e serviço recorrente como categorias | zestful `traffic_features.py:124-125`, `arrival_contest.py:32-34` | −3,9 local (misturado) |
| Especialista de resíduo sobre `MVT − LOBT`, α por aeroporto encolhido | alligator `scripts/autoresearch_known_lobt.py`, `_regional.py` | −1,97 e −0,52 oficiais |
| Teto por aeroporto fora do LIRF (5 linhas) | kind-mango README:121 (v48) | −2,33 oficial |
| Isotônica fora da amostra no classificador | kind-mango v64 | −1,14 oficial |

## Fontes de dados abertas (levantamento de 28/09)

Nenhuma fonte aberta traz o off-block real por voo em LTFM, LFPG ou EGLL.

| Fonte | O que traz | Cobertura jan/jul 2026 | Potencial |
|---|---|---|---|
| OPDI v0.0.2 (EUROCONTROL/OpenSky; <https://www.opdi.aero/>) | flight list mensal (icao24, adep, ades, first_seen, last_seen) e eventos por voo; **sem off-block** (`exit-parking_position` quase vazio: 0 em FCO, 0,2 % em CDG) | sim | tempo de solo da mesma aeronave antes do voo (`first_seen` − `last_seen` do voo anterior do mesmo `icao24`); likable-eagle: LB 278,84 → 276,99 só em LIRF sem NM. Eventos v4 com AOBT em ~90 % existem em pesquisa (PR `euctrl-pru/OPDI#8`), não publicados |
| Séries diárias EUROCONTROL (<https://ansperformance.eu/csv/>): slot adherence, atraso pré-partida (total e ATC), atraso ATFM de chegada por causa | aeroporto × dia; LTFM incluído | sim | dias de disrupção (greve, tempestade, espera no stand por sequenciamento) |
| METAR do IEM (domínio público) | a cada 30 min; RMK com vento por pista em LTFM | sim | degelo e tempestade na cauda de jan |
| ERA5 via Open-Meteo (CC-BY 4.0) | neve, precipitação, rajada por hora | sim | complementa o METAR |
| OSM `aeroway` (ODbL), OurAirports | stands, taxiways, pistas | estático | distância/caminho stand → pista |
| Média mensal de taxi-out | mensal por aeroporto | sim | já testada e descartada (27/09) |
