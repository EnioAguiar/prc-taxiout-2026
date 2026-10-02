# Vencedores de competições parecidas — o que dá para roubar (2026-10-01)

Fontes: soluções vencedoras do NASA/DrivenData "Pushback to the Future" (2023) e "Run-way
Functions" (2022), vencedores do PRC Data Challenge 2024 (ATOW) e 2025 (fuel), documentação
oficial do indicador de taxi-out da PRU/EUROCONTROL e o dataset OPDI. Tudo com link.
Marquei `[INFERÊNCIA]` o que não consegui verificar direto na fonte.

---

## Resumo (o que vale testar)

### 1. OPDI `flight_events` v0.0.2 — eventos ADS-B até 31/07/2026 (cobre Jan e Jul 2026)

**O que é.** A PRC/EUROCONTROL publica, junto com a OpenSky Network, uma tabela de *milestones*
por voo derivada de ADS-B: `off-block` (T04), `fim de pushback` (T05), `entrada na pista` (T06),
`lift-off/take-off` (T08), `runway vacated` (T19), `on-block` (T21). Cada evento tem
tempo UTC, lat/lon/alt e um campo `info` que carrega **a posição de estacionamento no off-block
e o ID da pista no take-off**. Cobertura declarada: **01/01/2022 – 31/07/2026** — ou seja, inclui
exatamente os dois meses do ranking.
Download: parquets de 10 dias, com os intervalos ancorados em 01/01/2022 (**não** alinham com o
início do mês), em
`https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_{YYYYMMDD}_{YYYYMMDD+10}.parquet`.
Verifiquei (HEAD, sem baixar) que os arquivos que cobrem os meses do ranking existem e têm ~270 MB
cada: Jan/2026 = `20251231_20260110`, `20260110_20260120`, `20260120_20260130`, `20260130_20260209`;
Jul/2026 = `20260629_20260709`, `20260709_20260719`, `20260719_20260729`, `20260729_20260808`.

**Por que pode derrubar o RMSE para nós.** Três usos, em ordem de valor:
- **Fila real na cabeceira**: com `entrada na pista` (T06) e `lift-off` (T08) dos *outros* voos
  dá para medir quantas aeronaves estavam na fila de decolagem e o tempo de ocupação de pista —
  exatamente a variável que a metodologia oficial da PRU diz ser a origem do "additional taxi-out
  time" (ver item 4). Isso é informação nova mesmo nos aeroportos onde já temos ADS-B.
- **Off-block independente** (T04) e **take-off independente** (T08) do próprio voo: segunda
  medida contra `AOBT_3`/`MVT`, útil para o classificador de "cópia/loteria".
- **Taxi-out observado dos voos vizinhos** nos próprios meses de teste (T04→T08), que é a feature
  nº 1 dos vencedores do pushback (ver item 2) e que não conseguimos tirar do dataset oficial
  porque `BLOCK_TIME_UTC_mvt` está apagado nas partidas do `ranking.parquet`.

**Risco.** A própria OPDI avisa: "a disponibilidade de eventos de pista, taxiway e posição de
estacionamento depende da cobertura ADS-B da OSN e da disponibilidade de dados geográficos do
OpenStreetMap" (<https://www.opdi.aero/data-preview>). Então em LTFM/LFPG os eventos de superfície
(T04, T05, T06) podem vir vazios; o `lift-off` (T08), que é derivado do voo no ar, deve existir em
quase todo lugar `[INFERÊNCIA]`. Vale medir a taxa de preenchimento por aeroporto antes de
investir.

**Fonte/licença.** <https://www.opdi.aero/> ; tabelas FLIGHT/EVENT/MEASUREMENT publicadas pela
OPDI em colaboração com a OSN (<https://www.opdi.aero/data>). A OPDI se define como *open data*:
"livre de custo e acessível… pode ser livremente usado, reusado e distribuído desde que a fonte
seja atribuída", com foco em uso **não comercial** (<https://www.opdi.aero/about>). Precisa citar
a fonte no README. **Atenção**: isto é a OPDI, não é a Trino histórica da OpenSky (essa é proibida).

**Esforço.** Médio. 4 arquivos de ~270 MB por mês de ranking + 2025 inteiro (~37 arquivos) para
treino; filtrar pelos 10 aeroportos (via `FLIGHT.ADEP`) antes de carregar na RAM.

---

### 2. Fila e ocupação de solo construídas só com o dataset oficial (sem ADS-B)

**O que é.** O conjunto de features que apareceu no top-4 do NASA Pushback, portado para as
colunas que o `ranking.parquet` **não** apaga. Lembrando a regra do desafio: só
`BLOCK_TIME_UTC_mvt` e `TAXITIME_SEC_mvt` **das partidas** foram apagados
(<https://prc-data-challenge-2026.netlify.app/data.html>). Logo, em Jan/Jul 2026 continuamos
tendo, para todos os movimentos: `MVT_TIME_UTC_mvt` (decolagem/pouso), `RUNWAY_mvt`, `STAND_mvt`,
`AOBT_3_flt`, `EOBT_1_flt`, `LOBT_flt`, e ainda o block-time e o taxi-in das chegadas.

Features concretas dos vencedores, traduzidas:
- `deps_taxiing` / `arrs_taxiing` (FLHuskies, 4º): número de aeronaves **taxiando naquele
  instante**. Versão nossa: partidas com `AOBT_3 < t < MVT` e chegadas com `MVT < t < BLOCK`.
  Isso é um contador de ocupação de solo que funciona **em LTFM, LFPG, LEMD, EGLL e LIRF**, onde
  o contador via adsb.lol falha — e é exatamente onde o nosso erro se concentra.
  (<https://github.com/drivendataorg/nasa-airport-pushback/blob/main/4th%20Place/Phase%201/src/table_scripts/add_traffic.py>)
- `dep20_time` (Moles, 2º): tempo decorrido entre a decolagem atual e a 20ª decolagem anterior →
  throughput instantâneo da pista. Com `MVT` + `RUNWAY` isso sai direto, inclusive no teste.
  (<https://github.com/drivendataorg/nasa-airport-pushback/blob/main/2nd%20Place/Phase%201/src/make_dataset.py>)
- `exp_deps_15min` / `exp_deps_30min` (FLHuskies): quantos voos têm decolagem prevista em ±15/±30
  min da nossa. Versão nossa: contar `MVT` dos vizinhos na janela ±15 min (demanda realizada no
  nosso slot) e, separadamente, contar `EOBT_1 + referência` (demanda planejada).
- `dep_taxi_30hr`, `arr_taxi_30hr`, `standtime_30hr` (FLHuskies): médias móveis de 30 h de
  taxi-out realizado, taxi-in realizado e tempo de parada. O taxi-in já usamos; a média móvel de
  **taxi-out realizado** no próprio mês de teste só sai via OPDI (item 1) ou via ADS-B.
- **Ultrapassagens** (não aparece nos repos, derivei da estrutura do dataset): nº de voos com
  `AOBT_3` maior que o nosso mas `MVT` menor que o nosso = voos que nos ultrapassaram na fila.
  Correlação direta com holding longo, e é calculável no ranking.
- **Mudança de configuração de pista**, inspirado no 3º lugar do Run-way Functions (AZ--KA, que
  modelava "distribuição das configurações passadas e duração de cada configuração ativa",
  <https://github.com/drivendataorg/nasa-airport-config>): como `RUNWAY_mvt` vem preenchido no
  ranking, dá para montar a série de configuração do aeroporto em Jan/Jul 2026 e marcar
  "configuração mudou nos últimos 30/60 min", "fração de partidas na pista X na última hora",
  "pista sumiu do dia" (= fechamento/obra). Não precisamos *prever* configuração, só usá-la.

**Por que pode derrubar o RMSE.** Ataca a parte do erro que está nos aeroportos sem ADS-B de solo,
sem depender de nenhuma fonte externa e sem risco de licença.

**Fonte/licença.** Dados do próprio desafio. Esforço: **baixo a médio** (é tudo `merge_asof` e
contagem por janela).

> Verificar antes: parte disso pode já estar coberto por "queue counts". O que eu apostaria que
> ainda não existe é (a) ocupação por intervalo `AOBT_3→MVT`, (b) `dep20_time`, (c) ultrapassagens,
> (d) dinâmica de mudança de configuração.

---

### 3. Média de várias sementes do LightGBM (1º lugar do PRC 2024)

**O que é.** Richard Alligier (team_likable_jelly, 1º do PRC 2024, mesma organização, mesma métrica
RMSE) publicou a tabela exata do ganho de treinar o **mesmo modelo, mesmos hiperparâmetros, mesmos
dados, só mudando a seed** e tirar a média:

| modelos na média | RMSE final |
|---|---|
| 1 | 1 612,41 kg |
| 10 | 1 564,09 kg |
| 20 | 1 561,63 kg |

Ou seja **−3,0 % de RMSE com 10 seeds** e −3,2 % com 20, sem nenhuma informação nova.
(<https://github.com/richardalligier/prc>, seção "The Results")

**Por que pode derrubar o RMSE para nós.** 3 % de 244,89 s ≈ **7,3 s**. Não são os 20 s, mas é o
item de melhor relação ganho/esforço da lista e é cumulativo com os outros. Ele usa modelos bem
grandes (50 000 árvores), ou seja, alta variância — o ganho em modelos mais regularizados tende a
ser menor `[INFERÊNCIA]`, mas o sinal é claro.

**Bônus do mesmo repo**: ele documenta que precisou de `deterministic=True` no LightGBM para
reproduzir resultados, e que teve de **arquivar os METARs** porque o Iowa Mesonet reescreve o
histórico (uma estação mudou de posição em duas semanas). Como o PRC exige reprodutibilidade e
pode rodar o pipeline em outro período, arquivar METAR/externos é obrigatório.

**Esforço.** Baixo (só tempo de CPU).

---

### 4. Resíduo sobre o tempo de referência oficial (P10 por STAND×RWY) + reescalonamento por grupo

**O que é.** Duas ideias que se encaixam:

(a) A PRU define formalmente o *Reference Taxi-Out Time* como o **percentil 10 dos taxi-outs
observados no grupo (STAND, pista de partida)** numa janela móvel de 12 meses, válido só se
houver **pelo menos 10 voos com taxi-out ≤ P10** naquele grupo; o "additional taxi-out time" é
actual − referência. O mesmo documento manda filtrar da amostra de referência: voos sem
pista/stand/off-block, **taxi-out > 120 min**, helicópteros e **voos com de-icing depois do
off-block**. (<https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf>, §3.2–3.5)
Note que a PRU diz explicitamente que *classe de aeronave não entra no agrupamento* porque fatia
demais a amostra sem ganho relevante — e que o agrupamento STAND×RWY já absorve a geometria do
aeroporto (o que explica o nosso ganho ~0 com distância de taxiamento do apt.dat).

(b) Alligier reescalou o alvo por grupo — `(TOW − EOW)/(MTOW − EOW)` — e treinou com
**peso = (MTOW − EOW)²** para que a perda continuasse sendo o RMSE na unidade original. Em árvores
"vanilla" transformações monótonas globais não mudam nada, mas **escalas diferentes por grupo
mudam a escolha dos splits** — é o argumento dele, e é o que faz essa transformação valer algo.

**Combinação proposta.** Alvo = `taxi-out − ref_P10(STAND, RWY)` (ou dividido pelo IQR do grupo),
com peso = quadrado do fator de escala para manter o RMSE em segundos. Ganhos esperados: o modelo
deixa de gastar splits para acertar o nível de cada combinação stand/pista e passa a gastar em
congestionamento; e os grupos com poucos voos herdam um nível razoável em vez de um ruído.

**Fonte/licença.** Documento metodológico público da EUROCONTROL/PRU (o cálculo é feito sobre o
nosso próprio dataset de treino). Esforço: **médio**.

---

### 5. METAR de estações vizinhas, vários raios espaço-temporais (trovoada e nevoeiro)

**O que é.** Alligier não usou só o METAR do aeroporto: montou um `NearestNeighbors` no domínio
espaço-tempo e, para trovoada/nevoeiro, **agregou todos os METARs dentro de um raio espaço-temporal
em torno do aeroporto e do horário**, repetindo para **vários raios** e deixando o LightGBM escolher
qual raio importa. Arquivo `feature_thunder_from_metars.py` no repo. Fonte dos METARs: Iowa State
Mesonet ASOS (<https://mesonet.agron.iastate.edu/ASOS/>), que ele classifica como domínio público.

**Por que pode derrubar o RMSE para nós.** Nosso erro de Jul 2026 se concentra em LFPG, LEMD e LIRF
— verão europeu, isto é, convecção. Tempestade a 30–80 km do aeroporto fecha rotas de saída (SID) e
gera *ground stop* e fila longa **com o METAR do próprio aeroporto limpo**. Usamos METAR só da
estação do aeroporto, então essa informação está literalmente ausente hoje. Mesma lógica em Jan para
nevoeiro/neve na região.

**Esforço.** Baixo — já baixamos METAR; é acrescentar estações num raio e agregar flags
(`TS`, `FG`, `SN`, `FZ`) em janelas de ±1 h/±3 h e raios de 25/50/100 km.

---

## Achados, por fonte

### NASA/DrivenData — "Pushback to the Future" (2023), 10 aeroportos dos EUA, métrica MAE

Repo oficial dos vencedores: <https://github.com/drivendataorg/nasa-airport-pushback>
Blog com entrevistas: <https://drivendata.co/blog/airport-pushback-finalists>

| # | Equipe | MAE (min) | Modelo |
|---|---|---|---|
| 1 | Team CDS (NYU) | 10,673 | ensemble de vários CatBoost **por aeroporto** |
| 2 | Moles (Caltech) | 10,728 | um CatBoost por aeroporto |
| 3 | Oracle 2 | 11,054 | pipeline multi-estágio: XGB regressor + XGB classificador de sub/superestimação |
| 4 | FLHuskies (UW Tacoma) | 11,105 | um LightGBM por aeroporto |
| 5 | Cuong_Syr | 11,946 | um XGBoost por aeroporto |

- **Todos os cinco treinaram um modelo por aeroporto.** FLHuskies testou modelo global e o local
  venceu. CDS manteve os dois e **ensemblou**: quatro famílias de modelo — V0 prevendo
  `pushback − ETD` (alvo em resíduo), V1 prevendo o alvo cru, V2 = V0 com outros parâmetros, e um
  modelo global com o aeroporto como variável categórica — com a arquitetura de ensemble escolhida
  para minimizar o MAE numa validação held-out
  (<https://github.com/drivendataorg/nasa-airport-pushback/blob/main/1st%20Place/Phase%201/README.md>).
  Transferível: manter em paralelo a versão "alvo cru" e a versão "alvo − âncora (MVT − AOBT_3)" e
  pesar as duas, em vez de escolher uma.
- **Features mais importantes (FLHuskies, gráfico de importância em KMEM)**: aeroporto de destino,
  `exp_deps_15min`, `dep_taxi_30hr`, `arr_taxi_30hr`, `standtime_30hr`. Isto é: *demanda na janela
  de decolagem* + *taxi realizado recente*. Nenhuma feature meteorológica no top-5.
- **Oracle 2 (3º) — tratamento de outliers e cauda** (o mais próximo do nosso problema):
  1. treina um `XGBRegressor` "interno" num subconjunto; 2. calcula o erro nos demais; 3. marca
  como outlier quem tem erro absoluto > 20–30 min (varia por aeroporto); 4. treina o modelo
  principal **só nos não-outliers**; 5. treina um classificador de "vai superestimar / vai
  subestimar" (AUC 0,64) e, na inferência, **soma ou subtrai a mediana do resíduo** do grupo
  previsto. Eles chegaram nisso olhando o histograma de MAE por aeroporto e vendo que KMEM (cargueiro,
  mix de voos atípico) tinha densidade alta de erros grandes
  (<https://github.com/drivendataorg/nasa-airport-pushback/blob/main/3rd%20Place/Phase%201/reports/DrivenData-Competition-Winner-Documentation.pdf>).
  **Cuidado**: descartar outliers do treino é legítimo para MAE; para RMSE remove justamente o que
  mais pesa. O que transfere é (a) usar resíduos *cross-fitted* de um modelo interno para
  **rotular** linhas contaminadas, e (b) corrigir a previsão por mediana de resíduo condicionada a
  um classificador — nossa correção hoje é por mistura de dois ramos, não por deslocamento de
  mediana.
- **Moles (2º) — features de revisão da estimativa**: `etd_slope` e `etd_slope_cumsum`, isto é, a
  *derivada* da sequência de atualizações do ETD por voo (quanto a estimativa de decolagem
  escorregou por segundo desde a estimativa anterior / desde a primeira). No nosso dataset existe o
  par `IOBT_flt` (off-block inicial) vs `EOBT_1_flt` (off-block estimado do M1) vs `LOBT_flt`
  (último off-block conhecido) vs `AOBT_3_flt` — ou seja, dá para montar a mesma ideia como
  "deriva do plano": `LOBT − IOBT`, `EOBT_1 − IOBT`, `AOBT_3 − LOBT`, e o sinal/magnitude dessas
  diferenças. Voo que foi repetidamente adiado tende a sair na janela congestionada.
- **O que eles testaram e não funcionou** (útil para não gastarmos tempo): features TBFM/TFM e
  validação cruzada temporal (CDS); fila baseada em *scheduled* e taxas de chegada/partida
  anteriores (Moles — note que o 4º lugar usou features parecidas **com sucesso**, a diferença é
  usar tempos realizados em vez de previstos); tráfego, meteorologia e "plane initialization"
  (Oracle); previsões meteorológicas para as próximas n horas em vez do tempo atual (FLHuskies);
  estatísticas de `arrival_runway_estimated_time` (Cuong_Syr — melhorou na arena aberta e piorou na
  fechada, isto é, overfit de leaderboard).

### NASA/DrivenData — "Run-way Functions" (2022), previsão de reconfiguração de pista

Repo: <https://github.com/drivendataorg/nasa-airport-config> ·
Blog: <https://drivendata.co/blog/airport-configuration-winners>

- 1º (Stuytown2): CatBoost, **um modelo por aeroporto e por horizonte de previsão**, com features de
  tráfego e meteorologia amostradas a cada 15 min.
- 3º (AZ--KA): XGBoost com features puramente de **histórico de configuração**: distribuição das
  configurações passadas e **duração de cada configuração ativa**.
- Para nós o problema de *prever* configuração não existe (o `RUNWAY_mvt` vem preenchido no
  ranking), mas as features de **estado e duração da configuração** são aproveitáveis como está
  no item 2.

### PRC Data Challenge 2024 (ATOW) — mesma organização, mesma métrica RMSE

- Editorial dos organizadores (Spinielli et al., JOAS 2025):
  <https://journals.open.tudelft.nl/joas/article/view/8252>. Descreve o desenho da avaliação: parte
  A = treino, parte B = ranking intermediário, **B+C = ranking final**, com C (52 190 voos) só
  entrando na última fase, split aleatório, distribuição de tipos de aeronave verificada para ser
  consistente entre treino e teste. Reprodutibilidade: todos os repos ficam em
  <https://github.com/PRC-Data-Challenge-2024/>.
- **1º lugar, Alligier & Gianazza (ENAC)** — <https://github.com/richardalligier/prc>. Além da média
  de seeds e do reescalonamento por grupo (itens 3 e 4): correção dos *timestamps* oficiais usando a
  trajetória (pegam o ponto de maior timestamp dentro de 10 NM do aeroporto de partida e extrapolam
  de volta assumindo 1 000 ft/min; se não há ponto dentro de 10 NM, mantêm o valor oficial) — mesma
  família de ideia que a nossa reconstrução de off-block por ADS-B, mas com *fallback* explícito e
  raio generoso, escolhido porque raios menores deixavam voos demais sem correção. Dados externos
  dele: `airports.csv` do OurAirports e METARs do Iowa Mesonet, ambos declarados domínio público;
  massas por tipo de aeronave via OpenAP (LGPLv3) + Wikipédia.
- **2º lugar, ITA (Murça et al.)**, artigo completo: <https://journals.open.tudelft.nl/joas/article/view/7963>.
  Dois pontos transferíveis: (i) **ensemble por *stacking* com pesos resolvidos por programação
  quadrática**, com restrições `w ≥ 0` e `Σw = 1`, ajustados num holdout de 20 % que não viu o
  treino dos modelos-base, e depois os modelos-base são retreinados em 100 % dos dados mantendo os
  pesos; (ii) **binning de variáveis contínuas** (quintis de temperatura/umidade) justamente "para
  controlar o efeito de outliers e capturar não-linearidades". Eles também relatam que imputar
  faltantes pela **mediana** bateu kNN e regressão linear, consistentemente para todos os
  algoritmos.

### PRC Data Challenge 2025 (fuel burn) — a edição imediatamente anterior

Resultado e **review dos repositórios feito pelos organizadores**:
<https://prc-data-challenge-2025.netlify.app/outcome.html>

- 53 equipes submeteram, 2 127 submissões, melhor RMSE final 200,83.
- O que os organizadores elogiaram no 1º (resourceful-quiver, <https://github.com/PRC-Data-Challenge-2025/resourceful-quiver>):
  "metodologia fortemente informada por conhecimento de domínio e não apenas por técnicas de ML",
  reaproveitamento explícito do estimador de massa da edição anterior, e um **relatório em formato
  de artigo JOAS** no repo. O que criticaram no 3º: abordagem "predominantemente ML", features só de
  estatística de trajetória, "pouca evidência de validação informada por domínio, como checagem de
  outliers ou valores fisicamente implausíveis".
- A premiação **não é só RMSE**: "o comitê de seleção considera o ranking de RMSE final, a
  originalidade e a abertura do código/documentação". Pelo review, documentar *por que* cada feature
  existe e mostrar limpeza de dados com argumento operacional pesa.
- O ranking final foi recalculado por intervalos (`Sep`, `Oct`, `Sep+Oct`) com **viradas grandes de
  posição na fase final** (bump chart na página) — reforça que a fase final do PRC 2026 em Jan+Jul
  2026 pode reordenar tudo e que vale olhar a estabilidade do nosso modelo entre os dois meses
  separadamente, não só no agregado.
- A página de ranking de 2026 avisa explicitamente: "vamos monitorar submissões que tentem aprender
  com ou explorar o processo de ranking… consideramos isso injusto"
  (<https://prc-data-challenge-2026.netlify.app/ranking.html>). Ou seja, sondar o leaderboard para
  inferir rótulos está fora.

---

## Descartado / irrelevante

- **Fase 2 do NASA Pushback (federated learning)** — todo o esforço foi em transformar os modelos em
  federados; nenhum ganho de acurácia (os scores da fase 2 são *piores*: 10,67 → 16,33 no 1º lugar).
  Nada aproveitável.
- **Redes neurais e sequenciais**: Oracle 2 testou RNN+LSTM (bom na arena aberta, não generalizou) e
  CNN-LSTM (caro, sem ganho); o ITA testou o transformer SAINT e foi **o pior dos modelos testados**
  no problema de regressão com variáveis majoritariamente contínuas. Os cinco vencedores do pushback
  e os três do PRC 2025 são todos GBDT. Confirma que o caminho é feature/dado, não arquitetura.
- **Distância de taxiamento stand→pista** (nosso teste deu ~0): consistente com a metodologia da PRU,
  que já absorve a geometria ao agrupar por STAND×RWY e que descarta a classe de aeronave como fator
  "sem impacto maior nos resultados". Não insistir.
- **Prever a configuração de pista** (Run-way Functions): desnecessário, `RUNWAY_mvt` vem preenchido
  no `ranking.parquet`.
- **Previsão meteorológica à frente (TAF/LAMP)**: FLHuskies testou previsão para as próximas n horas
  em vez do tempo atual e não entrou na solução final. Priorizar METAR observado em vizinhança
  (item 5) em vez de TAF.
- **Descartar os outliers do treino** à la Oracle 2 (|erro| > 20–30 min): feito para MAE. Com RMSE,
  as "loterias" são 38 % do nosso erro quadrático — removê-las do treino tira justamente o alvo.
  Usar o modelo interno só para *rotular*, não para excluir.
- **OpenSky Trino histórico**: proibido pelas regras; a OPDI (item 1) é a via legítima para o mesmo
  tipo de informação derivada de ADS-B.
- **Datasets de "flight delay" do Kaggle** (vários, EUA, nível de voo/dia): granularidade e geografia
  erradas, sem stand/pista, sem valor aqui.
