# Literatura de previsão de taxi-out: o que os artigos acham importante e o que falta no nosso modelo

Revisão feita em 01/10/2026. Foco: features que os artigos medem como importantes,
com tamanho de efeito, e quais delas **não temos** e dão para construir com dado
aberto ou com os campos do organizador.

Fontes primárias lidas por inteiro ou em trechos: Wang et al. 2021 (TRC, aceito,
White Rose), Jia/Wu et al. 2026 (Sci. Rep., open access), Lee/Coupe/Jung 2019 (NASA
NTRS), Lee/Malik 2016 (NASA NTRS), Simaiakis & Balakrishnan 2009 (MIT), Holländer &
Kallén 2026 (Chalmers, open), especificação A-CDM da EUROCONTROL, planilhas de taxi
time da EUROCONTROL.

---

## Resumo (o que vale testar)

Ordenado por ganho esperado ÷ esforço **no nosso caso** (erro concentrado na cauda e
nos aeroportos sem cobertura ADS-B de solo).

### 1. Eventos de superfície do OPDI v0.0.2 (`entry-runway`, `exit-parking_position`, `entry/exit-taxiway`)

- **O quê:** o OPDI publica, além da flight list que já usamos, uma tabela `EVENT` com
  marcos 4D por voo. Na v0.0.2 ela inclui entrada/saída de **pista**, **taxiway** e
  **posição de estacionamento**, casados por grade H3 resolução 12 sobre polígonos do
  OpenStreetMap (projeto HexAero).
  Fonte: <https://www.opdi.aero/methodology.html> (seção "Flight Events", v0.0.2).
- **Por que pode baixar o RMSE:** dá três coisas que hoje só temos pelo adsb.lol (ou
  não temos): (a) off-block observado independente do `AOBT_3`/`BLOCK`; (b) instante de
  entrada na pista, ou seja, o táxi puro separado do tempo de fila na cabeceira —
  exatamente a decomposição que Simaiakis & Balakrishnan usam (τ = τ_unimpeded +
  τ_taxiway + τ_dep.queue, <https://web.mit.edu/Hamsa/www/pubs/SimaiakisBalakrishnanGNC2009.pdf>);
  (c) a rota de taxiway realmente percorrida, que é a feature "distance/route" que os
  artigos põem em 2º–3º lugar e que a nossa distância reta do apt.dat não captura.
  O ponto decisivo é a **cobertura**: a fonte de state vectors é a OpenSky Network, cuja
  rede de receptores é diferente da do adsb.lol. Se o OPDI enxergar solo em LTFM/LFPG/LEMD,
  é informação nova onde está o nosso erro. **Testar a cobertura antes de qualquer feature.**
- **Dado/licença:** OPDI (EUROCONTROL/PRC + OpenSky), mesmo dado aberto que já citamos
  no README (atribuição). Downloads em parquet de 10 em 10 dias:
  `https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_{AAAAMMDD}_{AAAAMMDD+10}.parquet`
  (<https://www.opdi.aero/flight-event-data.html>). Verificado por HEAD em 01/10:
  `20260110_20260120` = 272 MB, `20260709_20260719` = 548 MB. Cobertura anunciada:
  01/2022 a 31/07/2026 — **inclui jan e jul/2026**.
- **Esforço:** médio-alto. ~37 arquivos por ano, 10–20 GB para 2025 + jan/jul 2026, mas
  dá para filtrar por aeroporto na leitura (parquet colunar) e guardar só os eventos dos
  10 aeroportos. Atenção à RAM: ler por fragmento.

### 2. Configuração de pista em uso (conjunto de pistas DEP e ARR ativas), não só a pista do voo

- **O quê:** categórica com o conjunto ordenado de pistas usadas por partidas e por
  chegadas nos últimos 30/60 min do aeroporto, mais a interação `pista do voo × configuração`
  e um sinalizador de troca de configuração na última hora.
- **Por que:** é o fator causal nº 1 na lista de Idris et al. (configuração de pista,
  companhia/terminal, restrições a jusante, tamanho da fila de decolagem;
  <https://arc.aiaa.org/doi/10.2514/atcq.10.1.1>), repetido por Simaiakis & Balakrishnan
  que definem o "segmento" = (configuração, condição meteorológica) e calibram tudo por
  segmento. Tamanhos de efeito: no CLT, o táxi no pátio é 4,02 min em fluxo norte contra
  6,91 min em fluxo sul (NASA, <https://ntrs.nasa.gov/api/citations/20190027660/downloads/20190027660.pdf>,
  seção IV.B); em ZRH, `runway info` é a 5ª feature mais importante de 33 em floresta
  aleatória, por causa das pistas que se cruzam (Wang et al. 2021, Tabela 14,
  <https://eprints.whiterose.ac.uk/id/eprint/169934/1/Taxi_Time_Prediction%20as%20finally%20submitted.pdf>);
  na tese da Chalmers a pista R3 é a feature de maior SHAP médio, 0,96 min, e usar R3 sobe
  a previsão em até 4 min (<https://odr.chalmers.se/items/ca3f1c99-49a2-4020-a149-2ab9a96dc018>, Fig. 4.7–4.8).
- **Dado:** só campos do organizador (`RUNWAY_mvt`, `MVT_TIME_UTC_mvt`, `PHASE_mvt`).
  Funciona em **todos** os aeroportos, inclusive LTFM e LFPG, onde não temos ADS-B de solo.
- **Esforço:** baixo (uma passada ordenada por aeroporto; o `_counts_around` já faz busca
  binária parecida). Hoje temos contagens por `(aeroporto, pista)` e a pista como
  categórica, mas **não** a configuração como estado do aeroporto.

### 3. Fila de decolagem ponderada por esteira e taxa de decolagem realizada contra a saturação

- **O quê:** além da contagem `sup_dep_decolam_durante`, acrescentar (a) quantas dessas
  decolagens são H/J (esteira pesada/super) na mesma pista; (b) a taxa de decolagem
  observada nos 20 min anteriores, em decolagens/min, e a razão dela para o máximo
  histórico daquele `(aeroporto, configuração)` — o "N*" de Simaiakis.
- **Por que:** a fila de decolagem é o preditor mais forte em toda a literatura
  (β = 0,68, p < 0,01 em Idris; 35% da importância em florestas aleatórias; SHAP 0,18 em
  Ding, ambos citados em Jia et al. 2026, <https://www.nature.com/articles/s41598-026-40898-5>),
  mas o **serviço** da pista não é homogêneo: a separação por esteira muda o tempo entre
  decolagens. A EUROCONTROL publica a diferença, por aeroporto: no EGLL o taxi-out médio
  é 25,4 min para H e 18,3 min para M; em LIRF 21,8 contra 15,9; em LFPG 20,8 contra 16,2
  (planilha `eurocontrol-taxi-out-times-wake-turbulence-category-winter-2025-2026.xlsx`,
  <https://www.eurocontrol.int/publication/taxi-times-winter-2025-2026>). Simaiakis mostra
  que a vazão satura em N* (16–21 aeronaves em BOS, 0,73 decolagem/min) e que acima disso o
  tempo extra vai todo para a fila; Jia et al. mostram transição não linear acima de 7
  chegadas na fila e de 5 pushbacks simultâneos.
- **Dado:** só campos do organizador (`WK_TBL_CAT_flt`, `RUNWAY_mvt`, `MVT_TIME_UTC_mvt`).
- **Esforço:** baixo-médio.

### 4. Velocidade média de solo e taxi-out observado das últimas 5/10 partidas (ADS-B)

- **O quê:** `AvgSpdLast5Dep`, `AvgSpdLast10Dep` (velocidade média de solo das últimas N
  partidas no aeroporto) e a média móvel do taxi-out **observado por ADS-B** dos vizinhos
  em 30/60 min.
- **Por que:** Wang et al. 2021 introduziram as features de velocidade e elas entram no
  top-4 em MAN e HKG (Tabelas 14 e 15); a ideia é que a velocidade dos outros aviões é um
  proxy barato de tudo o que não se mede (congestionamento de taxiway, LVP, degelo,
  pista molhada). Jia et al. 2026 colocam a média de taxi-out dos 30 min anteriores ao
  pushback entre as features de correlação moderada, com efeito não linear acima de 1200 s.
- **Diferença para o que já temos:** o `ctx_viz_pas_*` usa `MVT − AOBT_3` como proxy do
  taxi-out dos vizinhos. Essa versão herda o ruído do `AOBT_3` justamente onde ele é pior
  (LTFM, LIRF). A versão ADS-B é independente dele. A velocidade de solo não existe hoje
  em nenhuma forma.
- **Dado:** recortes do adsb.lol que já baixamos (ODbL 1.0); `gs` já está nos pontos.
- **Esforço:** baixo (é um agregado sobre `events.parquet` + os pontos recortados).
  **Limite honesto:** só vale onde há cobertura de solo, ou seja, não ataca LTFM.

### 5. Prior sazonal por aeroporto × categoria de esteira da EUROCONTROL

- **O quê:** juntar média, desvio, P10, mediana e P90 do taxi-out por `(aeroporto, WTC)` e
  por aeroporto, das planilhas sazonais da EUROCONTROL, como feature externa.
- **Por que:** é uma medida independente, feita com os horários reportados pelas próprias
  companhias, que cobre **inverno 2025-2026** (publicada em 20/05/2026, portanto já contém
  jan/2026) e os verões anteriores. Serve de âncora de nível por aeroporto em meses em que
  o nosso treino de 2025 pode ter derivado, e cobre LTFM, onde não temos ADS-B
  (média 18,6 min; H 20,3 contra M 18,1).
- **Dado/licença:** EUROCONTROL, download público em xlsx
  (<https://www.eurocontrol.int/publication/taxi-times-winter-2025-2026> e
  <https://www.eurocontrol.int/publication/taxi-times-summer-2024>). Uso livre com atribuição,
  mesma condição das séries diárias que já usamos.
- **Esforço:** muito baixo (2 arquivos de ~50 KB; já extraí as linhas dos 10 aeroportos, abaixo).
- **Ressalva:** a planilha de verão 2026 só sai em dezembro/2026, depois do prazo. Para
  jul/2026 teríamos de usar verão 2024/2025 como prior, o que é aceitável para nível mas
  não captura mudança do ano.

---

## Achados, com fontes

### 2.1 O que cada artigo mede como mais importante

| Artigo | Aeroporto / dado | Features no topo | Tamanho de efeito reportado |
|---|---|---|---|
| Idris, Clarke, Bhuva & Kang 2002, *Queuing Model for Taxi-Out Time Estimation* (<https://arc.aiaa.org/doi/10.2514/atcq.10.1.1>) | BOS, ASQP/PRAS | configuração de pista, companhia/terminal, **restrições a jusante**, tamanho da fila de decolagem | fila de decolagem é o fator dominante; β = 0,68 (p < 0,01) e 72% dos voos em ±3 min, conforme resumido por Jia et al. 2026 |
| Simaiakis & Balakrishnan 2009 (<https://web.mit.edu/Hamsa/www/pubs/SimaiakisBalakrishnanGNC2009.pdf>) | BOS | τ = unimpeded + taxiway + fila; N(t), Q(t), N*; segmento = (configuração, VMC/IMC) | τ(i) contra fila de decolagem NQ(i): R² = 0,51; contra número de aviões no solo N(t): R² = 0,10. Saturação N* = 16–21 por configuração; vazão satura em 0,73 decolagem/min |
| Wang, Brownlee, Woodward, Weiszer, Mahfouf & Chen 2021, TRC 124:102892 (<https://eprints.whiterose.ac.uk/id/eprint/169934/1/Taxi_Time_Prediction%20as%20finally%20submitted.pdf>) | MAN, ZRH, HKG, 33 features, FlightRadar24 + METAR | `depArr`, `distance`, `NDepDep` (partidas a caminho da pista no pushback), `angle_sum`, `AvgSpdLast5Dep`, `distance_long`, `runway_info` (só ZRH), `aircraft_weight` (só MAN) | importância de `depArr` chega a 0,70 em HKG; com o subconjunto pequeno a queda é < 1 ponto percentual nos acertos em 1, 3 e 5 min. **Features de tempo (chuva, neve, névoa, granizo, vento, visibilidade) são as primeiras eliminadas nos três aeroportos** |
| Jia, Wu, Mao, Huang, Zeng & Ma 2026, *Sci. Rep.* 16:10066 (<https://www.nature.com/articles/s41598-026-40898-5>) | Shenzhen (ZGSZ), A-CDM, 12 645 voos | fila de chegadas (x4) e fila de partidas (x3) = 36% do SHAP; features ligadas ao táxi dinâmico = 88% do SHAP | fila de chegadas vira positiva acima de **7 aeronaves**; pushbacks simultâneos acima de **5**; média de táxi de 30 min acima de **1500 s** vira não linear; companhia 3º lugar, distância de táxi 6,2% de SHAP com correlação de apenas 0,18; tipo de aeronave 2% (descartada) |
| Lee, Coupe & Jung 2019, NASA ATD-2/CLT (<https://ntrs.nasa.gov/api/citations/20190027660/downloads/20190027660.pdf>) | CLT, dados de torre de pátio | área de pátio (gate), tipo de aeronave, companhia, configuração de pista, distância gate→spot, conflito de portão, APREQ/EDCT | push-back médio 4,72 min (σ 2,27); por esteira: B 5,30 / D 4,67 / E 4,50 min; APREQ 5,30 contra 4,67 min; EDCT 5,16 contra 4,70; conflito de portão: média igual, **σ 4,38 contra 2,22 min**; táxi de pátio 4,02 (norte) contra 6,91 min (sul); modelo só com distância e velocidade mediana (6,6 kt) já empata com regressão linear |
| Lee & Malik 2016, NASA/CLT (<https://ntrs.nasa.gov/api/citations/20160004946/downloads/20160004946.pdf>) | CLT, HITL | gate, spot, pista, modelo, distância, nº de partidas e chegadas taxiando | RMSE: RF 1,15 min, kNN 1,22, SVM 2,29, LR 2,83; "dead reckoning" com taxi sem impedimento erra 5–7 min para menos |
| Holländer & Kallén 2026, MSc Chalmers/Saab (<https://odr.chalmers.se/items/ca3f1c99-49a2-4020-a149-2ab9a96dc018>) | um grande aeroporto, A-SMGCS + ASTERIX | pista R3 (SHAP médio 0,96 min), fila de pista (0,70), partidas taxiando, pista R2, stand F, direção do vento, hora e MTOW (0,18 cada) | fila de pista: efeito quase linear até +4 min; stand F: −4 min; MTOW muito baixo reduz, pesado aumenta ~1 min |
| Herrema, Curran, Visser, Huet & Lacote 2018, CDG (<https://research.tudelft.nl/en/publications/taxi-out-time-prediction-model-at-charles-de-gaulle-airport>) | LFPG, ~1 milhão de voos, 42 features | top 10 inclui taxi-out sem impedimento, nível de congestionamento e **nº de partidas nos últimos 20 min** (lista reproduzida por Wang et al. 2021, §2) | árvore de regressão: erro médio de 1,6 min por dia |

### 2.2 A lista oficial de insumos do "variable taxi time"

A especificação A-CDM da EUROCONTROL, requisito A-CDM-[OPS]-[270], lista o que o
cálculo do tempo de táxi variável *deve* usar: layout do aeroporto, disponibilidade de
infraestrutura, **pistas em uso**, stands e posições de estacionamento, tipo de aeronave,
operador, **método de pushback**, hora de entrega da autorização de pushback, ALDT/AIBT/
EOBT/TOBT/TSAT/AOBT, **degelo remoto necessário**, densidade de tráfego, demanda de
tráfego e procedimentos locais; os requisitos [290]/[300] acrescentam a rota de táxi
planejada e a autorizada. Fonte:
<https://www.eurocontrol.int/sites/default/files/2025-01/eurocontrol-specification-for-acdm.pdf>
(pág. ~70). Serve como checklist: dos 13 itens, só nos faltam, sem substituto aberto,
o método de pushback, o horário da autorização e TOBT/TSAT.

### 2.3 Taxi-out de referência da EUROCONTROL, nossos 10 aeroportos (inverno 2025-2026)

Extraído de `eurocontrol-taxi-out-times-winter-2025-2026.xlsx` e da versão por esteira
(<https://www.eurocontrol.int/publication/taxi-times-winter-2025-2026>), minutos:

| Aeroporto | Média | Desvio | P10 | Mediana | P90 | Média H | Média M |
|---|---|---|---|---|---|---|---|
| EDDF | 14,59 | 7,03 | 8 | 13 | 21 | 18,82 | 13,82 |
| EDDM | 15,01 | 6,95 | 9 | 13 | 23 | 18,70 | 14,56 |
| EGLL | 19,56 | 7,03 | 12 | 19 | 28 | 25,44 | 18,34 |
| EHAM | 14,22 | 7,17 | 9 | 13 | 20 | 17,59 | 13,77 |
| LEBL | 17,51 | 6,09 | 11 | 17 | 24 | 19,13 | 17,48 |
| LEMD | 18,80 | 6,52 | 13 | 18 | 25 | 20,61 | 18,57 |
| LFPG | 17,05 | 7,15 | 11 | 15 | 24 | 20,77 | 16,16 |
| LIRF | 16,22 | 5,60 | 11 | 15 | 23 | 21,82 | 15,91 |
| LSZH | 14,04 | 6,59 | 8 | 13 | 22 | 18,32 | 13,46 |
| LTFM | 18,63 | 7,25 | 12 | 17 | 27 | 20,31 | 18,14 |

(EDDM também tem categoria J: média 23,96 min, P90 40.)

### 2.4 O que falta no nosso modelo, item por item

| Feature citada na literatura | Artigo que mede | Temos? | Dá para construir com o quê |
|---|---|---|---|
| Fila de decolagem NQ (decolagens entre o off-block e a decolagem do voo) | Idris 2002; Simaiakis 2009 (R² 0,51 contra 0,10 de N(t)) | **Sim** (`sup_dep_decolam_durante`) | — |
| Partidas/chegadas nos últimos 20–30 min | Herrema 2018 (top 10); Jia 2026 (x5, x7) | **Sim** (`apt_dep_prev_*`, `apt_arr_prev_*`, `ctx_*`) | — |
| **Configuração de pista em uso (conjunto DEP/ARR) e troca de configuração** | Idris 2002; Simaiakis 2009 (segmento); Wang 2021 (ZRH 5º); NASA 2019 (4,02 contra 6,91 min) | **Não** (só a pista do voo e contagens por pista) | campos do organizador |
| **Mix de esteira na fila (H/J à frente) e taxa de decolagem contra saturação N\*** | Simaiakis 2009; planilha WTC da EUROCONTROL (H−M de 4–7 min) | **Não** (WTC só como categórica do próprio voo) | campos do organizador |
| **Velocidade média de solo das últimas 5/10 partidas** | Wang 2021 (top 4 em MAN e HKG) | **Não** | adsb.lol (ODbL), já baixado |
| **Taxi-out observado por ADS-B dos vizinhos (média móvel)** | Jia 2026 (x5/x7); Balakrishna, via Wang 2021 Tabela 1 | **Parcial**: temos a versão `MVT − AOBT_3`, que herda o ruído do AOBT_3 | adsb.lol / OPDI events |
| **Tempo até entrar na pista separado do tempo de fila** | Simaiakis 2009 (decomposição τ) | **Não** | OPDI `entry-runway` + `exit-parking_position` |
| **Rota/distância de taxiway realmente percorrida e soma de curvas (`angle_sum`)** | Ravizza 2013/2014 e Wang 2021 (top 3 em MAN) | **Não** (só distância reta stand→cabeceira, ganho ~0) | OPDI events (OSM/HexAero) |
| Degelo | A-CDM [OPS]-[270]; nosso proxy METAR | **Sim** (proxy `met_degelo`) | — |
| TOBT/TSAT, método de pushback, hora da autorização | A-CDM [OPS]-[270]; NASA 2019 (APREQ/EDCT +0,5 a +1,2 min) | **Não** e **sem fonte aberta** | — |
| Conflito de portão / congestionamento de pátio | NASA 2019 (σ 4,38 contra 2,22 min) | **Parcial** (rotação de stand) | campos do organizador |
| Companhia como proxy de terminal/handler | Idris 2002; Jia 2026 (3º no SHAP, clusters de 824 a 1399 s) | **Sim** (`cia`, operador categórico) | — |
| Peso/MTOW e categoria de esteira | Chalmers 2026 (SHAP 0,18 min); NASA 2019 (pushback por WTC) | **Sim** (categórica) | — |

---

## Descartado / irrelevante

- **Meteorologia detalhada como grupo grande de features.** Wang et al. 2021 eliminam
  chuva, neve, névoa, granizo, visibilidade e vento primeiro nos três aeroportos, com
  perda < 1 ponto percentual; Lee et al. (CLT) tentaram clima e **não** melhoraram a
  previsão. Bate com o nosso ganho de 1,8% no grupo meteorologia. Vale manter só o degelo
  e a interação com aeroporto.
- **Distância de táxi como feature isolada.** Jia et al. medem correlação de apenas 0,18
  com o alvo (SHAP 6,2%) e a classificação de pátio fica abaixo do limiar (3,9%, descartada).
  No nosso caso a `ref_p10` por `(aeroporto, stand, pista)` já absorve a geometria — é a
  explicação mais provável do ganho ~0 do apt.dat. Só vale repescar via rota real (item 1).
- **Decompor o alvo em "sem impedimento" + "dinâmico" e prever cada parte.** Jia et al.
  testaram e a versão em dois estágios ficou **pior** (MAPE 12,0% contra 10,6%; RMSE
  156,7 s contra 140,5 s), porque o erro do primeiro estágio propaga. Serve para
  interpretar, não para a nota.
- **Simulação de tempo acelerado (LINOS, node-link).** Precisa do layout e de calibração
  por aeroporto, e em CLT empatou com kNN/RF (RMSE 2,67 contra 1,15 min do RF). Custo alto,
  sem ganho.
- **TOBT/TSAT e mensagens DPI por voo.** Nenhuma fonte aberta; o B2B do NM exige contrato.
  Fora pela regra do prêmio.
- **Trino histórico da OpenSky.** Proibido pelo regulamento; o OPDI é a via aberta
  equivalente para marcos de voo.
- **Modelos de pushback só com gate e tipo de aeronave.** NASA mostra que a árvore de
  decisão com esses dois critérios empata com LR/RF/NN (RMSE ~2,2 min): o teto dessa
  informação é baixo, já estamos nele com stand + tipo.
