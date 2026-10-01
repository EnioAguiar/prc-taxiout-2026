# Repositórios de outros times — 01/10/2026

Varredura nova de código público (GitHub `gh search repos`/`gh search code`, API de repositório,
busca pelos nomes dos 21 primeiros do placar). Continuação de
`docs/research/2026-09-27-concorrentes.md`: aqui só o que é **novo ou mudou desde 27/09**.
Nenhum código deles foi executado; todo número é o que eles publicam.

Placar no momento da leitura (espelho <https://prc-leaderboard.fly.dev/data.json>, 01/10 14h44 UTC):
nós somos **23º com 244,89**; 3º lugar = jolly-lobster 222,91; 1º = strong-crane 219,64.

## O que mudou desde 27/09

- **Nenhum time do top 20 publicou código.** Confirmado por busca de repositório e de código
  com os 21 primeiros nomes do placar, por `TAXITIME_SEC_mvt`, `AOBT_3_flt`, `MVT_TIME_UTC_mvt`
  e por varredura de repositórios recentes com "taxi-out". A mesma conclusão está no
  `docs/WINNING_PLAN.md:278-318` do kind-mango, que leu a API oficial com 2.412 envios.
- **elegant-alligator, zestful-fountain, knowledgeable-helicopter, jubilant-vase e
  vibrant-lollipop estão parados** (últimos commits 15/09, 13/09, 12/09, 11/09 e 01/09).
  Nada novo neles desde a leitura de 27/09.
- **kind-mango teve 1 commit novo** (28/09, "Add NM clocks track B: v73 to v89, best 264.34 s"):
  a trilha B inteira, de 281,87 → 264,34. Hoje o placar dele é 249,39 (29º).
- **Quatro repositórios que não estavam na nossa lista**: `ChandanHegde07/OpenAir`
  (likable-eagle, 276,99 — o mais rico em medições), `col3name/air-data` (kind-earthquake,
  268,52), `kagervig/PRC-Challenge-2026` (unique-umbrella, 293,86; 108 KB de diário de
  ablações) e `RavindraTarunokusumo/prc-data-challenge-2026` (genuine-cabbage, commit de
  hoje, sem nenhum envio ainda).

### Mapa dos repositórios lidos

| Repositório | Time (posição, melhor) | Último commit | Licença | Estado |
|---|---|---|---|---|
| [`skylinkapi/prc-data-challenge-2026-kind-mango`](https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango) | kind-mango (29º, 249,39) | 28/09 | GPLv3 | ativo, diário completo (`README.md`, `RECAP.md`, `docs/WINNING_PLAN.md`) |
| [`ChandanHegde07/OpenAir`](https://github.com/ChandanHegde07/OpenAir) | likable-eagle (58º, 276,99) | 15/09 | GPLv3 | ~120 experimentos com RMSE interno **e** oficial em `status.md` (1.897 linhas) |
| [`col3name/air-data`](https://github.com/col3name/air-data) | kind-earthquake (39º, 268,52) | 04/09 | sem licença | 5 gerações de script CatBoost, sem números publicados |
| [`kagervig/PRC-Challenge-2026`](https://github.com/kagervig/PRC-Challenge-2026) | unique-umbrella (98º, 293,86) | 30/09 | GPLv3 | `learnings.md` (108 KB) + `model.py` + 22 JSONs do avaliador oficial |
| [`RavindraTarunokusumo/prc-data-challenge-2026`](https://github.com/RavindraTarunokusumo/prc-data-challenge-2026) | genuine-cabbage (sem envio) | 01/10 | sem licença | 24 experimentos, só dado oficial, protocolo de validação muito forte |
| [`victoralcadi/prc-data-challenge-2026`](https://github.com/victoralcadi/prc-data-challenge-2026) | eager-jungle (fora do espelho, 689,69 segundo o kind-mango `WINNING_PLAN.md:290`) | 17/09 | GPLv3 | pipeline LightGBM limpo, sem resultados próprios |
| [`MykhailoHalchenko/Asterion-5G`](https://github.com/MykhailoHalchenko/Asterion-5G) | — (projeto 5G/ASTERIX) | 30/09 | sem licença | não é competidor de RMSE; só o roteador de taxiway interessa |
| [`elna4os/prc_2026`](https://github.com/elna4os/prc_2026) | não identificado | 01/10 | sem licença | só um notebook de EDA |
| `rocoroza11/nimble-barn-…`, `fenil264/prc-2026-taxi-out`, `Necklacesaura/…`, `ruchir321/…`, `Enzoferrariz/…`, `georgejhanlon/prc-taxiout-2026` | nimble-barn, joyful-lemon, zestful-jewel, … | 17/09 a 01/09 | varia | esqueleto ou vazio, nada a aproveitar |

## Achados ordenados por impacto provável no nosso RMSE

### 1. Decompor `T = D − G` nos voos sem registro NM (OpenAir)

Identidade `TAXITIME = D − G`, com `D = MVT − SCHED` (observado, está no ranking) e
`G = BLOCK − SCHED` (o atraso de portão latente, que é o que falta). Em vez de prever o táxi
nessas linhas, eles treinam um CatBoost para prever **G** nas partidas do LIRF e reconstroem
`T = max(D − Ĝ, 0)` só nas linhas LIRF sem casamento NM. Entradas do modelo de G: `log(D)`,
posto de `D` dentro do LIRF, `D` cruzado com contagens de chegada/partida, hora em seno/cosseno
e janelas de superfície de 5–30 min.
Ganho deles: `LIRF_u` RMSE 6033 → 4717 → 3937; placar **316,97 → 292,99 → 288,90**
(`status.md:1291-1324`, `run_e33_delay_decomposition.py`, `run_e34_lirf_improve.py`).

Refinamentos que vieram depois, todos com transferência ~1:1 para o placar:

- **Mistura de dois modelos de G** com λ pequeno: `G = (1−λ)·G_todo_LIRF + λ·G_só_unmatched`,
  λ = 0,20–0,25. λ = 1 é veneno (placar 300,95), λ = 0 perde; a mistura ganha nos dois splits.
  `LIRF_u` 3732 → 3625; placar 280,18 → **278,84** (`status.md:1779-1800`, E88).
  Variante E93: categórica `prefixo_companhia|flag_unmatched` para o CatBoost global aprender
  o G de unmatched sem ajuste separado (`status.md:1805-1816`).
- **Regra de long-hold**: avião parado ≥ 2 h no solo (`icao_turn = first_seen − last_seen`
  anterior do mesmo `icao24`, OPDI) tem táxi real mediano ~1.132 s, enquanto a mediana do G
  dessas linhas é 4.679 s. Aplicam `T = 0,5·T₀ + 0,5·típico` em 146 linhas do ranking:
  `LIRF_u` 3543 → 3450, placar 278,36 → **276,99** (`status.md:1855-1865`, E110).
  Falha em dezembro (regime de inverno) e eles aceitam porque o análogo do ranking é jan+jul.
- **OPDI para recuperar operador e tipo nas linhas sem NM** (`icao_operator`, `typecode`,
  `icao_aircraft_class` por `icao24`, inclusive com callsign vazio): cobertura de `icao_turn`
  43 % no treino unmatched e 219/383 (57 %) no ranking; `LIRF_u` 3625 → 3558 → 3543
  (`status.md:1821-1834`, `status.md:1844-1854`).

Por que é o primeiro da lista: é exatamente a fatia em que estamos fracos (11 "loterias" = 32 %
do erro², mais as retas por aeroporto dos voos sem NM), é a maior transferência placar-por-placar
documentada em qualquer repo público, e já temos as três peças (OPDI, `icao24`, modelos por
regime). **Aviso medido por eles**: a decomposição só vale no LIRF — `corr(T, D)` é 0,90 no LIRF
e ≤ 0,44 nos demais (0,02 no LFPG), e aplicar fora do LIRF piora (`status.md:1717-1726`).

### 2. Espelho das chegadas: calibrar a deriva de 2026 com rótulo verdadeiro (kind-mango)

As linhas ARR do `ranking.parquet` **mantêm o taxi-in verdadeiro de 2026** — é o único rótulo
de 2026 que existe. Eles pontuam um modelo de taxi-in treinado em 2025 sobre as chegadas de
2026 e leem o resíduo por aeroporto/mês (`models/v3/arrival_mirror.json`,
`docs/WINNING_PLAN.md:163-175`): EHAM dá **405,5 s em janeiro contra 129,4 s em julho**; LFPG
250,7 e 189,1. Conclusão deles: existe um regime em jan/2026 no EHAM que não está em nenhum mês
de treino, e um modelo que lê relógio do voo acompanha esse regime linha a linha, enquanto um
modelo de planos e contagens não acompanha — é um dos mecanismos da vantagem dos usuários de
`AOBT_3`. O lever correspondente é o L7 (`day-level artefact share from ranking arrivals`,
−0 a −1.500 MSE).

Nós temos `ctx_arr_tin_*` (média do taxi-in observado em janelas), que é o **nível** observado;
o que falta é o **resíduo contra o esperado** (observado − previsto por um modelo de taxi-in de
2025), que separa deriva de regime de mudança de composição (mix de stands, tipos e pistas).
Candidato barato: uma coluna por (aeroporto, dia) e outra por (aeroporto, hora) com esse
resíduo, entrando no corretor.

### 3. Bloco de fila que não temos: retenção no portão e push real dos vizinhos

Três medidas independentes, duas delas com ganho oficial confirmado:

- **`overdue_runway_queue`** (unique-umbrella, `model.py:437-481`): no push do voo `i`, conta
  voos da mesma (aeroporto, pista) com `EOBT_j ≤ t < AOBT_j` — ou seja, quem já passou do
  horário estimado de calço e **ainda não empurrou**. Reduz a duas contagens acumuladas
  (`|{EOBT ≤ t}| − |{AOBT ≤ t}|`). Guardas: só intervalos com `EOBT ≤ AOBT` e duração < 7200 s
  (derruba artefatos de virada de data) e fora quem nunca empurra. Por pista −1,30 s no harness;
  por aeroporto −1,05; as três variantes juntas −1,32 (a parcimoniosa ganha). No placar:
  307,01 → **304,95** (−2,06 s, maior salto de versão deles desde a v28), `learnings.md:1511-1546`.
  É fila de **portão**, não de taxiway: ortogonal ao nosso `pista_fila_peso` e ao `sup_*`.
- **Push real dos vizinhos em vez do push estimado**: o `sup_*` usa `o_j = MVT_j − pred_j` para
  os vizinhos (`src/superficie.py:19-24`). unique-umbrella e OpenAir usam o `AOBT_3` dos
  vizinhos, que existe em 98,5 % das DEP do ranking. Trocar a estimativa pelo relógio real nas
  mesmas contagens (`active_departures_queue = |{AOBT_j ≤ t}| − |{MVT_j ≤ t}|`, −0,79 s no
  harness, placar 310,08 → **308,98**) é mudança de uma linha de código no bloco que já temos.
- **Atraso de push recente dos outros voos** (`recent_delay`, unique-umbrella `model.py:353-387`;
  `dis_state_30m`, OpenAir `status.md:757-782`): média de `AOBT_3 − SCHED` das partidas do mesmo
  aeroporto na janela anterior (15 min é o ótimo medido deles; 30/45/60/90/120 todas piores),
  com a própria linha fora. Mecanismo: em colapso de capacidade o avião empurra cedo para liberar
  portão mas segura o off-block. OpenAir mede `corr` 0,45 com o próprio `AOBT − EOBT` (não é
  redundante) e mostra que no oitavo octil a taxa de táxi > 30 min sobe de 2,37 % para 12,34 %;
  ganho interno jan+jul 378,28 → 376,04. unique-umbrella: −0,55 s e o pacote com ele valeu
  −2,92 s no placar.

Complemento barato do mesmo bloco: trocar a média móvel de 60 min do sinal de congestionamento
por **EWMA com meia-vida de 10 min** (−0,57 s; varredura 10/20/30/45/60/90 = 313,86/313,89/313,98/
314,67/314,41/314,36, `learnings.md:1412-1425`). A varredura de janela do boxcar (15 a 120 min)
não sai do ruído — o que importa é o decaimento suave, não o tamanho da janela.

### 4. Tirar do **treino** as linhas que já desviamos para regra/modelo especial (genuine-cabbage)

Eles já roteavam "LIRF sem NM" para um modelo próprio. O achado do dia 4 é que **deixar essas
linhas no treino do GBM contamina as outras**: o GBM aprende a convenção "bloco = horário
programado" e exporta previsões absurdas para as linhas sem NM dos outros nove aeroportos.
Excluindo-as do treino (`route_train_exclude`, `src/prc/models/routed.py:12-15`): −2,04 s em
todas as linhas, previsões fora de faixa na banda de atraso > 3 h caem de **81 para 5**, e o
grosso das linhas `NM_missing_other` melhora 105 a 230 s por fold
(`research/EXPERIMENT_JOURNAL.md:278-287`).

Nós sobrescrevemos a previsão dessas linhas (retas por aeroporto, pós-regra de Roma) mas elas
continuam no treino da base. É um teste de uma flag. kind-mango mediu a mesma ideia por outro
caminho (MB3, base treinada só nas linhas servidas, LIRF e `y > 80.000` fora): **−5,99 s
oficiais**, o maior salto isolado da temporada deles (`README.md` v51).

### 5. Membro ancorado no `AOBT_3`, misturado (não substituindo o alvo) — kind-mango L5

Um membro do conjunto treinado em `y − clip(MVT − AOBT_3, 0, 7200)` que serve
`âncora + offset`, e usa o membro normal onde `AOBT_3` é nulo. Não é troca de alvo: é **mistura
0,5 ancorado + 0,5 normal** dentro de cada termo (LightGBM e CatBoost).
Holdout −818 MSE em 9/9 aeroportos; placar 270,58 → **266,84** (v74) e → **266,15** com o
CatBoost ancorado (v75). Isso explica por que o nosso teste do plano 3b (alvo residual puro)
empatou: eles também perdem com o alvo residual sozinho; o ganho está em ter os dois membros.

### 6. Árvores lineares com `linear_lambda` alto (kind-mango MB5)

A base deles é LightGBM com **folhas lineares**; um estudo Optuna de 24 tentativas na era dos
relógios escolhe `linear_lambda` 15,1 (contra 0,0055 antes), 613 folhas, `learning_rate` 0,015,
2.284 rodadas. Holdout −965 MSE em 9/9 aeroportos; placar 266,15 → **265,27**
(`README.md` v76). Nós usamos árvores constantes (`src/models.py:32-36`, `src/stack.py:88`):
`linear_tree` nunca foi testado aqui, e é justamente o que ajuda a extrapolar em entradas
contínuas como `MVT − SCHED` e `MVT − AOBT_3`.

### 7. Combinação de membros: mediana quando discordam, e pesos NNLS

- **MS3** (kind-mango): quando dois membros discordam em mais de 3.600 s, serve a **mediana**
  dos membros em vez da média (L13, grade A, 0 a −250 MSE; está dentro da receita servida desde
  a v73). Protege exatamente contra o caso em que um membro acerta a cauda e o outro não.
- **Pesos NNLS/ridge ≥ 0 no holdout** em vez de média simples (OpenAir, `status.md:1033-1070`):
  com cinco especialistas o otimizador zera o modelo base e fica com CatBoost 0,456,
  XGBoost só-numérico 0,053 e LightGBM por aeroporto 0,491 — jan+jul 372,36 → **368,03**
  (−4,33 s). A correlação de resíduos é 0,96–0,98: o ganho vem de os membros serem melhores,
  não de diversidade. Hoje tiramos **média simples** de dois corretores (`src/campeao.py`).
- **Corretor com porta por magnitude** (OpenAir, `experiments/matched_submit.py:126-131`):
  aplicam `pred = base + λ·(rec − base)` só se `base > P90` (1.462 s) **ou** a correção passa de
  200 s, com λ = 0,5; depois somam um segundo resíduo "leftover" com λ = 0,35 e só a parte
  positiva. Nosso corretor é aplicado em todas as linhas sem porta.

### 8. Peças menores, já com número publicado

| Ideia | Onde | Ganho que reportam |
|---|---|---|
| Janelas **para a frente** do relógio NM dos vizinhos (média e contagem de `MVT − AOBT_3` de quem decola em (t, t+15] e (t, t+30], aeroporto e pista) na **base** | kind-mango v85, `src_v3/build_nm_forward_v85.py` | −604 MSE no membro, placar −0,72 s (temos isso só no corretor) |
| Um LightGBM **por aeroporto** misturado 50/50 com o global em cada termo | kind-mango v89 | sozinho +1.112 MSE, misturado −657 MSE em 9/9; placar −0,22 s (temos `--base-por-apt`, mas não a mistura 50/50 dentro de cada termo) |
| Dispersão entre os relógios de off-block (`clock_std`, `clock_range`, `n_clocks` sobre `[AOBT_3, EOBT_1, IOBT, LOBT]`) | OpenAir `experiments/matched_submit.py:33-36`; E66 | linhas de resíduo alto têm `clock_range` mediano 3.240 s contra 480 s; matched 250,98 → 243,51 com todos os relógios |
| Esteira do voo **imediatamente anterior na mesma pista** (`WK_TBL_CAT_flt.shift(1)` por aeroporto × pista) | unique-umbrella `model.py:556-569` | sem número isolado (temos peso de esteira agregado em `pista_fila_peso`, não o anterior imediato) |
| Razão do dia: média do táxi do aeroporto desde a meia-noite ÷ média de longo prazo, só depois de 5 voos | unique-umbrella `model.py:287-350` | −2,5 a −2,8 s na época; `MIN_DAY_FLIGHTS=3` é pior (+0,39 s) |
| Estatísticas de alvo com **decaimento exponencial** (meias-vidas 15/60/180/1440/4320/10080 min, só voos estritamente anteriores) por aeroporto, tipo, aeroporto×tipo, aeroporto×hora, aeroporto×pista | col3name `train4.py:319-408` | sem número publicado |
| Hora e dia da semana **locais** (fuso do aeroporto, com horário de verão) do horário programado | victoralcadi `features.py:110-123`; genuine-cabbage `features.py:44-47`; unique-umbrella `model.py:54-66` | parte de blocos de −9,1 s (NM presente) |
| `is_diverted = ADES_flt != ADES_FILED_flt` e fila líquida de 3 h (programadas − decoladas) | victoralcadi `features.py:153-154,197-205` | sem número |
| Rota **roteada** stand → pista pelo grafo de taxiways do OSM (`osmnx`, `aeroway=taxiway`, snap ≤ 250 m, `shortest_path` por comprimento) | Asterion-5G `yezhik-asterix/taxiway_tracks.py:127-193` | sem número (nosso `map_*` do apt.dat é distância, não caminho) |
| Destino (`ADES`) como categórica | unique-umbrella `model.py:71` | −2,3 s na validação limpa |

## Negativos caros que eles já pagaram (não repetir)

- **Séries de atraso ATFM pré-partida por dia não explicam o resíduo.** unique-umbrella juntou
  o arquivo diário por aeroporto (cobertura 91,2 %) ao resíduo real de 2025:
  `corr(atraso_por_partida, resíduo com sinal) = +0,020` e com `|resíduo|` = +0,053; do lado de
  2026 o taxi-in real das chegadas é plano contra o atraso diário (490 → 504 s entre decis,
  corr +0,010). Veredito deles: caçada encerrada (`learnings.md:1548-1582`,
  `analyze_error_vs_predeparture.py`). **Isso é exatamente a origem das nossas colunas `ext_*`
  das séries da EUROCONTROL** — vale uma ablação antes de carregá-las até o fim.
- **Transformar o alvo piora**: `log1p` +7,31 s e `sqrt` +0,90 s no número honesto; comprime o
  corpo e estraga a cauda do LIRF, que governa o RMSE (`learnings.md:1647-1664`).
- **Perda assimétrica (punir subprevisão) piora monotonicamente** (α 1,5/2,0/2,5 → +2,47/+5,74/
  +9,09) e piora **dentro** do LIRF: a cauda é variância, não viés — é lacuna de feature, não de
  perda (`learnings.md:1666-1689`). Peso amostral maior no LIRF: zero (`learnings.md:1691-1697`).
- **Peso com decaimento temporal piora** (meia-vida 270/180/90/60 dias → +1,4/+2,0/+4,2/+6,1 s).
- **Interações aeroporto × {pista, stand, prefixo de stand} e ID único de portão**: +0,07 a
  +2,04 s, todas revertidas, mesmo com 24 % dos códigos de stand repetidos entre aeroportos.
- **Features de reconfiguração de pista** (contagem de pistas ativas em 30/60 min, parte da
  pista minoritária): +0,43 a +0,65 s, apesar da evidência diária forte (no LIRF o táxi triplica
  na hora em que 16R/34L entram na mistura). Correlação em nível de hora não virou sinal por voo
  (`learnings.md:1439-1498`).
- **METAR**: na ablação de unique-umbrella só a **temperatura** trabalha (+9,6 s ao desligar);
  vento +0,1, visibilidade 0,0, precipitação −0,1, código +0,8. OpenAir chegou ao mesmo:
  METAR ajuda dezembro (−9 s) e piora a cauda em jan+jul. kind-mango: METAR no `EOBT_1`
  rejeitado (+0,33 s). Depois que a temperatura entrou, a categórica de **mês** passou a
  atrapalhar.
- **Modelos por aeroporto puros** (sem mistura com o global): +4,0 s (unique-umbrella);
  kind-mango mede o mesmo sozinho (+1.112 MSE) e só ganha misturando 50/50.
- **Roteamento explícito por regime** (classificador de disrupção + dois regressores + mistura):
  +1,1 s, e a fatia de SSE dos piores 5 % sobe — o modelo único já faz detecção implícita.
- **XGBoost nas mesmas colunas**: 12,4 % pior que LightGBM (genuine-cabbage E011).
- **Âncora crua `MVT − AOBT_3` como previsão direta**: +2,22 s (genuine-cabbage E004).
- **Encoding de alvo sofisticado (LOMO + m-estimate hierárquico) por cima de stand × pista**:
  +0,189 de R² na EDA linear virou −0,29 s dentro do GBM com âncora NM (piso pré-registrado era
  −3,0 s) → mecanismo falsificado. Confirma que nossa referência P10 por célula já pega o sinal.
- **`T ≤ since_arr`** (limitar o táxi pelo tempo desde a chegada no stand): destrói
  (3.896 → 11.579), porque o stand é reusado (OpenAir E71-E72).
- **Fila exata de superfície no `AOBT` por sobreposição de intervalos** e **prior de identidade
  por callsign/rota com shrinkage empírico-Bayes**: ambos rejeitados (E41, E42: matched
  244,76 → 249,64). **TCN sequencial** de 64 passos sobre o histórico do aeroporto: placar
  317,47 contra 316,97 do modelo tabular.

## Riscos de 2026 que eles mediram e que valem para nós

- **Excesso de voos sem NM muito atrasados em jan/2026** (genuine-cabbage,
  `research/day-04/eda/forward_support.json`): 435 linhas sem NM fora do LIRF com
  `d_sched > 3 h` e 92 com `> 5 h`, contra máximo mensal de 2025 de 222 e 36. É exatamente a
  população que as nossas retas por aeroporto (`--nm-min-ms 21600`) atendem, super-representada
  no mês que vale 44,3 % das linhas pontuadas, e nenhum holdout de 2025 mede isso.
- **Ruptura de configuração de pista no LFPG** (`docs/methodology/DATASET_AUDIT.md:121-127`):
  27L e 09R fazem 30,2 % e 13,6 % das partidas em jun/2025, caem a **0 %** de ago a nov, e
  **voltam** em jan e jul/2026 (27L com 29,5 % e 20,4 %). A mediana do táxi do LFPG sobe de
  895 s para 1.012–1.015 s nesses meses. Referências por pista do LFPG calibradas em ago–nov
  descrevem um regime que não existe nos meses pontuados.
- **O RMSE oficial é dominado por contagem de artefato, não por habilidade** (unique-umbrella,
  `learnings.md:1045-1050`): com o mesmo modelo, o RMSE honesto por mês vai de 275 (outubro,
  1 artefato) a 617 (julho, 17 artefatos); 8 linhas de artefato (0,002 % da validação) fizeram
  47 % de todo o SSE. Uma linha de rótulo ~86.000 s prevista em ~900 vale ~92.000 linhas normais.
- **Validação de inverno subestima features de congestionamento em 2–3×** (n = 2: v28 previu
  "neutro" e deu −2,92 s; v29 previu −0,79 e deu −2,08). Regra que eles adotaram: multiplicar o
  delta de validação dessas features por 2–3 e enviar mesmo ganhos pequenos.
- **A métrica é dominada pela cauda**: o mesmo bloco de features vale −9,1 s nas linhas com NM e
  −1,72 s (q95 +9,39, indistinguível de zero) em todas as linhas (genuine-cabbage, regra 11).
  Ganho medido só no miolo pode não mover o placar — é o nosso `sem_loteria` visto de fora.

## Protocolo de validação deles (comparação com o nosso)

- **genuine-cabbage**: bootstrap pareado com **cluster (aeroporto × dia UTC)**, 2.000
  reamostragens; WIN se q0,90 < 0, LOSS se q0,10 > 0. Promover exige média ≤ −1,0 s, q0,95 < 0,
  ≥ 3 WIN em 5 folds, fold sazonal obrigatoriamente WIN e nenhum LOSS. Além disso **gêmeos
  causais** de fold (treino estritamente anterior à validação) que anulam vitórias vindas de
  treinar com meses posteriores — pegou um caso real (E012). Nosso bootstrap é pareado por dia;
  o cluster por aeroporto × dia é mais conservador.
- **Controle nulo na triagem de features** (genuine-cabbage `scripts/eda_day3.py:22-24`): medem a
  redução de RMSE de uma feature **constante** e subtraem. O nulo sozinho "reduz" 6,85 s por puro
  grau de liberdade — sem isso, metade das features de congestionamento pareceria útil.
- **Invariância de mascaramento**: toda contagem "conhecida no push" subtrai analiticamente a
  própria linha, com teste que varia o `MVT` da linha e exige valor idêntico (verificado em
  1.919.370 linhas). Fecha um vazamento sutil: janelas que terminam na decolagem incluem a
  própria linha e viram proxy do alvo.
- **Verificação de faixa por banda de `d_sched`** (`< 1 h`, 1–3 h, 3–5 h, `> 5 h`) × (NM
  presente/ausente) × (LIRF/outros): contam previsões `< 0` ou `> 3600 s` em linhas cujo alvo é
  `< 3600 s`. A banda `< 1 h` tem 60–83 casos em qualquer variante (ruído genérico); só a banda
  `> 3 h` indica dano de feature (81 no campeão contra 13–18 sem o bloco novo). A nossa projeção
  na janela `MVT − LOBT ± 3606` **mascara** esse diagnóstico em vez de revelá-lo.
- **unique-umbrella, validação "honesta"**: treinar só em linhas limpas mas **manter os
  artefatos na validação** e aplicar o mesmo override das previsões, reportando dois números
  (honesto e só-limpo). Antes disso a validação deles dizia 250 enquanto o placar dizia 518.
- **OpenAir**: holdout jan+jul (os meses do ranking) como split primário, dezembro só como
  estresse, e mantêm o **offset interno → placar** (~44–52 s) para estimar a nota antes de gastar
  um envio; exigem ganho nos dois splits. Regra de código: `common.py` recusa carregar parquet
  que não comece com `training_`.

## Sobre o override da cauda (nossa pós-regra de Roma)

unique-umbrella pagou caro para calibrar overrides e publicou a tabela toda:

- Override por **padrão** (voo cruzando a meia-noite: `AOBT` hora 23 e `MVT` hora 0) custou
  **+104 a +124 s oficiais** (v15 573,85 → v17 697,73), porque a maioria desses voos é partida
  normal. Regra que ficou: só sobrescrever o que é fisicamente impossível, nunca um padrão que
  dispara em voos normais (`learnings.md:1011-1019`).
- Precisão do override **por subgrupo**, nunca agregada: com LIRF+LFPG+LSZH a precisão era 83 %;
  quebrando, LIRF 44/0 (100 %), LFPG 0/8, LSZH 0/1. Restringir ao LIRF: 349,83 → **313,00**.
- Varredura do limiar de `MVT − SCHED` no LIRF sem plano de voo: 25.000 s → −6,89; 21.600 →
  −10,19 (zero falso positivo); 18.000 → −13,40 (1 FP); 15.000 → −14,25 (6 FP); 12.000 → −10,94
  (27 FP, já pior). Faixa 6–10 h tem 38 % de precisão, faixa 3–6 h tem **0 %** (441 FP).
  Escolheram o corte com zero FP porque cada FP vale ~20.000 s de erro numa linha real.
- Fora do LIRF o mesmo formato (sem plano, `MVT − SCHED > 10 h`) é **0 de 51** artefatos reais:
  beco sem saída verificado, não suposto (`learnings.md:1620-1624`).

## Fontes

- kind-mango: <https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango> (`README.md`,
  `docs/WINNING_PLAN.md` seções 4.3, 5.3, 6, 9, 10)
- OpenAir / likable-eagle: <https://github.com/ChandanHegde07/OpenAir> (`status.md`,
  `experiments/matched_submit.py`)
- unique-umbrella: <https://github.com/kagervig/PRC-Challenge-2026> (`learnings.md`, `model.py`,
  `results/*.json` do avaliador oficial)
- genuine-cabbage: <https://github.com/RavindraTarunokusumo/prc-data-challenge-2026>
  (`research/EXPERIMENT_JOURNAL.md`, `docs/methodology/DATASET_AUDIT.md`, `config/splits.yaml`)
- kind-earthquake: <https://github.com/col3name/air-data> (`train3.py`, `train4.py`)
- victoralcadi: <https://github.com/victoralcadi/prc-data-challenge-2026> (`src/prc2026/`)
- Asterion-5G: <https://github.com/MykhailoHalchenko/Asterion-5G>
  (`yezhik-asterix/taxiway_tracks.py`)
- Placar: <https://prc-leaderboard.fly.dev/data.json> e
  `https://datacomp.opensky-network.org/api/competitions/<id>/leaderboard` (paginado por
  `nextCursor`, campos `teamName`/`filename`/`score`/`usedPairs`; leitura sem credencial,
  implementação em victoralcadi `src/prc2026/leaderboard.py:16-60`)
