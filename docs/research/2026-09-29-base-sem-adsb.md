# Voos normais nos aeroportos sem ADS-B (item 4e) — 29/09/2026

Alvo: os voos **normais** (`y ≤ 1 h`, com registro NM) de **LTFM, LFPG, LEMD e EGLL**, onde o
estudo da cauda de 28/09 colocou ~31 % do erro² e a distância para o topo.

Base do diagnóstico: previsões da campeã **v28** (`runs/20260929-093849-v28_cf.parquet`,
299,41 na simulação) no holdout jan+jul/2025 (344.419 voos), cruzadas com `cache.load_split`.
Scripts descartáveis (`/tmp/diag1.py` … `/tmp/diag11.py`), nada deles entrou em `src/`.

## 1. Onde está o erro

Voos normais com NM: 338.278 voos, RMSE 193,45, **41,0 % do erro² total**.

| Aeroporto | n | RMSE | viés (resíduo médio) | % do erro² total | % do erro² dos normais |
|---|---|---|---|---|---|
| LTFM | 45.855 | 232,48 | −1,1 | 8,0 | 19,6 |
| LIRF | 25.819 | 307,39 | −19,6 | 7,9 | 19,3 |
| LFPG | 38.605 | 225,28 | −15,1 | 6,4 | 15,5 |
| EGLL | 39.526 | 206,74 | −11,3 | 5,5 | 13,4 |
| LEMD | 35.019 | 175,74 | −10,1 | 3,5 | 8,5 |
| LEBL | 28.492 | 191,99 | +12,5 | 3,4 | 8,3 |
| EDDF | 36.208 | 143,00 | −3,9 | 2,4 | 5,9 |
| EHAM | 40.098 | 102,90 | +0,1 | 1,4 | 3,3 |
| LSZH | 21.813 | 136,22 | −0,7 | 1,3 | 3,2 |
| EDDM | 26.843 | 120,77 | −1,7 | 1,3 | 3,1 |

Os quatro sem ADS-B somam 159.005 voos, RMSE 212,92, **23,3 % do erro² total** (56,9 % do
erro² dos normais). O viés negativo é em parte artefato do corte `y ≤ 1 h`, que apara a
cauda de cima; a inclinação de `y` sobre `pred` sem esse corte (voos com NM e `y ≤ 3 h`) é
0,974 a 1,031 em todos os dez aeroportos, isto é, **a previsão já está calibrada**.

Tamanho do prêmio: um corte de 5 % no RMSE da fatia de um aeroporto vale, no RMSE completo,
LTFM −1,17 s · LFPG −0,92 s · LEMD −0,51 s · EGLL −0,80 s. Os quatro juntos a −5 % ≈ −3,4 s.

## 2. O que já não explica nada (medido)

**R² leave-one-out da média do resíduo por grupo** (positivo = o grupo carrega sinal que a
v28 não usou; negativo = só ruído), nos quatro aeroportos:

| grupo | LTFM | LFPG | LEMD | EGLL |
|---|---|---|---|---|
| stand | −0,6 % | −0,5 % | +0,8 % | +0,1 % |
| stand × pista | −2,8 % | −5,4 % | −0,8 % | −0,6 % |
| companhia | −0,2 % | 0,0 % | +0,3 % | +0,4 % |
| companhia × stand | −7,6 % | −4,4 % | −4,7 % | −1,8 % |
| destino | −0,7 % | −0,3 % | −0,5 % | −0,1 % |
| tipo de aeronave | −0,1 % | 0,0 % | +0,8 % | +0,3 % |
| pista × hora | +1,1 % | +1,0 % | +1,0 % | +0,8 % |
| dia | +1,4 % | +1,4 % | +0,9 % | +0,8 % |
| **dia × hora** | **+9,0 %** | **+3,1 %** | **+3,4 %** | **+2,6 %** |

Ou seja: **nenhum agrupamento estático sobrou**. Os campos do NM estão 100 % preenchidos nessa
fatia (SCHED, LOBT, IOBT, EOBT, AOBT_3, operador, segmento, esteira, stand, pista), então não
há fatia "sem informação" para tratar à parte. Calibrar a previsão por aeroporto (50 faixas de
`pred`, ajuste **dentro da própria amostra**) piora: 212,92 → 214,35. Tirar o viés por
(aeroporto, stand) também dentro da amostra só leva a 210,94.

### Candidatas novas testadas e reprovadas

Cada uma foi montada e medida contra o resíduo da v28 (LightGBM fora do fold, 5 folds por dia,
só as candidatas como entrada):

| bloco | colunas | correlação com o resíduo | RMSE do resíduo |
|---|---|---|---|
| — | (nada) | — | 212,92 |
| **fila de pushback pelo AOBT_3** | aviões que empurraram e ainda não decolaram no meu push (aeroporto e pista), quantos decolam entre o meu push e a minha decolagem, quantos me ultrapassam, quantos empurram durante o meu táxi | −0,027 a +0,007 | 213,44 |
| **configuração de pista** | nº de pistas de partida ativas na última hora, fatia da minha pista, chegadas na minha pista (modo misto), chegadas no aeroporto | −0,008 a +0,014 | 213,89 |
| as duas juntas | | | 213,42 |

O `AOBT_3` está preenchido em 98,4 % das DEP, então não é falta de cobertura: as janelas de
congestionamento do NM que a base já tem (`apt_dep_*`, `rwy_dep_*`, 10 a 60 min) e as colunas
`ctx_*` do corretor já carregam esse sinal. É o mesmo veredito que `--superficie` (ATD2)
recebeu no laço fiel 2 (−0,24 s).

### OPDI `first_seen` não serve de off-block nesses aeroportos

O casamento atual (`externos.opdi`) exige `|MVT − first_seen| ≤ 600 s`. Refazendo o casamento
por (aeroporto, callsign) sem essa restrição, nos voos normais com NM:

| Aeroporto | casados | mediana `MVT − first_seen` | mediana de `y` | erro mediano | ±60 s |
|---|---|---|---|---|---|
| LSZH | 95,3 % | 718 s | 719 s | −34 s | 36,1 % |
| EDDM | 19,6 % | 325 s | 834 s | −473 s | 5,0 % |
| **LEMD** | 3,2 % | 9 s | 985 s | −840 s | 0,5 % |
| **EGLL** | 2,1 % | 7 s | 1.262 s | −1.178 s | 0,2 % |
| **LFPG** | 1,5 % | 15 s | 1.014 s | −902 s | 1,2 % |
| **LTFM** | 0,0 % (22 voos) | 323 s | 1.324 s | −1.020 s | 4,5 % |

Nos quatro aeroportos-alvo o OPDI casa com 0 a 3 % dos voos e, quando casa, o `first_seen` é
praticamente a decolagem: **não há cobertura de solo do OpenSky ali**, o mesmo muro do
adsb.lol. (Achado lateral: em LSZH o `first_seen` é um off-block muito bom — e é exatamente o
que a tolerância de 600 s de hoje descarta. LSZH vale 1,3 % do erro² dos normais.)

## 3. O que sobrou: o estado do aeroporto em blocos de 30–60 min

Oráculo (tira de cada voo a média dos resíduos *dos outros* voos do mesmo bloco de tempo):

| janela | chave | LTFM | LFPG | LEMD | EGLL |
|---|---|---|---|---|---|
| — | — | 232,5 | 225,3 | 175,7 | 206,7 |
| 15 min | aeroporto | 224,9 | 226,9 | 175,5 | 208,4 |
| 30 min | aeroporto | 221,0 | 222,1 | 172,5 | 204,3 |
| 30 min | aeroporto × pista | **219,0** | 224,2 | 174,5 | 204,3 |
| 60 min | aeroporto × pista | 218,5 | 222,4 | 173,2 | 204,1 |
| 120 min | aeroporto × pista | 221,3 | **221,4** | **172,5** | **204,3** |

Componente de variância em blocos de 30 min (sinal, já descontado o ruído do tamanho do
bloco): desvio de **57,9 s (EGLL), 59,9 s (LEMD), 70,5 s (LFPG) e 91,6 s (LTFM)**. A
autocorrelação do resíduo médio entre blocos vizinhos é 0,40 (LTFM) e 0,23 (LFPG).

É real, mas **não é observável**: nas linhas do ranking o taxi-out das partidas está apagado, e
os dois substitutos que existem no ranking falham.

- Média do "resíduo-proxy" dos vizinhos, `MVT − AOBT_3 − pred` (98,4 % de cobertura): a
  correlação por voo com o resíduo é **negativa** (−0,036 a −0,013) e aplicar a média do bloco,
  com qualquer peso de 0,2 a 1,0, piora (LTFM 232,5 → 243,5).
- Resíduo do taxi-in das chegadas no mesmo bloco (taxi-in das ARR existe no ranking):
  correlação de −0,02 a +0,046; aplicar também piora.

Os piores blocos são episódios (LFPG 21/01/2025: resíduo médio −118,5 s, o modelo prevendo
2.735 s contra 2.124 s reais), isto é, o modelo *reage* ao mau tempo e exagera, não que ignore
o dia.

## 4. Conclusão do diagnóstico

Um segundo corretor com **todo o arsenal** (105 colunas: base + `ctx_*` + METAR + rotação +
consistência NM + fila + pátio), fora do fold por dia, treinado só nos quatro aeroportos, não
acha nada: 212,92 → 216,85 (piora em três dos quatro). Com as features de hoje a v28 já está no
limite de informação dessa fatia; o que falta é **informação nova**, e as três fontes novas que
existiam estão fechadas (adsb.lol sem cobertura, OpenSky/Trino vetado, OPDI sem solo).

Restam duas alavancas que não são informação nova e sim **forma do modelo**, e são as duas
hipóteses levadas ao corretor de verdade.

## 5. Hipóteses medidas

Portão (laço fiel): ganho > 0 no `completo`, IC 95 % baixo > −0,5, ganho sem os 10 maiores
voos > 0, ganho > 0 em jan e em jul. Referência: campeã `20260929-093849-v28_cf` (299,41).

### H1 — estatísticas da célula no corretor (`--corretor-ref`, `src/refcel.py`)

Motivo: nos quatro aeroportos a célula (aeroporto × stand × pista) é quase toda a informação
que existe, e o modelo só vê o **P10** dela (`features.ref_p10`, na base) — nunca o nível, a
dispersão nem o tamanho. O corretor recebe hoje o pátio (`stand_p`, que valeu +0,90 s no laço
fiel 2) mas nenhum número da célula. As colunas novas são `cel_p50`, `cel_p90`, `cel_dp`,
`cel_n`, `cel_nivel` e `cel_pred_menos_p50`, ajustadas sempre fora do bloco de dois meses da
linha (mesma regra do `crossfit.oof_base`), com o alvo cortado em 7.200 s e queda para chaves
mais gerais abaixo de 10 voos. No holdout, 99,1 % das linhas casam no nível mais específico.

Corrida `20260929-124349-h1_ref_cel` (9m38s, pico 5,85 GB; corretor da v28 com
`--corretor-ref`, `--reusar-oof 20260928-131704-v20_cf`), contra a v28:

```
20260929-093849-v28_cf → 20260929-124349-h1_ref_cel
  RMSE simulação 299.41 → 299.23
  ganho 0.2 s (IC 95% -0.7 a 1.1)
  sem os 10 maiores voos: -0.3 s
  mês 01: ganho 0.5 s (IC 95% 0.1 a 1.0)
  mês 07: ganho -0.1 s (IC 95% -1.9 a 1.5)
  informativo (fora do veredito):
    voos normais com NM (338,278 voos): ganho 0.6 s (IC 95% 0.4 a 0.8)
    sem loteria (344,408 voos): ganho 0.5 s (IC 95% 0.0 a 1.0)
  veredito: não comprovado
```

**Portão: reprovado** — IC baixo −0,7 (< −0,5), sem os 10 maiores −0,3 (< 0) e julho −0,1
(< 0). Mas é o **único bloco desde 28/09 que mexe na fatia-alvo**: `normais_nm` 193,45 →
192,85 (+0,60 s, IC +0,4 a +0,8), e dentro dos quatro aeroportos sem ADS-B

| Aeroporto | v28 | H1 | |
|---|---|---|---|
| LFPG | 225,28 | 223,51 | **−1,76** |
| LEMD | 175,74 | 174,56 | −1,18 |
| EGLL | 206,74 | 206,29 | −0,45 |
| LTFM | 232,48 | 232,26 | −0,22 |
| os quatro | 212,92 | 212,08 | −0,84 |

O ganho da fatia não chega ao `completo` porque o RMSE completo é dominado pela cauda: o
bloco piora os 10 maiores voos e julho. Código pronto e **desligado** (`src/refcel.py`,
`stack.py --corretor-ref`, `train.py` já sabe reproduzi-lo no envio).

### H2 — regressor por aeroporto na base (`--base-por-apt`, `src/models.py`)

Motivo: os aeroportos não dividem stand nem pista, então no regressor global toda árvore gasta
os primeiros cortes separando aeroporto antes de chegar ao stand; e 46 % das linhas do holdout
são dos quatro aeroportos em que todas as colunas `adsb_*` são nulas, que no modelo global
caem sempre no mesmo lado padrão dos cortes feitos para os outros seis. A flag treina, ao lado
do regressor global, um regressor por aeroporto com ≥ 20.000 linhas normais e prevê pela média
dos dois. O classificador da cópia não muda.

**A base sozinha melhora muito.** `20260929-125421-base_ctx_por_apt` (8m59s) contra
`20260928-131214-base_ctx`:

```
20260928-131214-base_ctx → 20260929-125421-base_ctx_por_apt
  RMSE simulação 311.18 → 308.75
  ganho 2.4 s (IC 95% 1.8 a 3.3)
  sem os 10 maiores voos: 2.2 s
  mês 01: ganho 2.2 s (IC 95% 1.3 a 4.2)
  mês 07: ganho 2.7 s (IC 95% 1.8 a 3.4)
  informativo (fora do veredito):
    voos normais com NM (338,278 voos): ganho 3.9 s (IC 95% 3.5 a 4.3)
    sem loteria (344,408 voos): ganho 3.1 s (IC 95% 2.6 a 3.8)
  veredito: não comprovado
```

É o maior ganho de base desde o `--base-ctx` (317,25 → 311,18) e passa o portão do laço fiel
em todos os critérios. RMSE por aeroporto da base: EGLL 261,7 → 259,2 · LTFM 253,5 → 250,9 ·
LEMD 182,9 → 178,7 · LFPG 568,8 → 568,0 · EDDF 160,7 → 155,5 · EHAM 150,4 → 144,3 ·
EDDM 145,7 → 139,7 · LSZH 178,2 → 173,0 · LEBL 199,1 → 194,9.

**Com o corretor da v28 em cima, quase tudo é absorvido.** `20260929-130343-v31_por_apt`
(48m35s, pico 7,55 GB; `--crossfit --conjunto --externos --plano13 --dist-plano
--stand-prefixo --corretor-rounds 500`, sem reaproveitar oof porque a base mudou):

```
20260929-093849-v28_cf → 20260929-130343-v31_por_apt
  RMSE simulação 299.41 → 299.23
  ganho 0.2 s (IC 95% -0.8 a 1.1)
  sem os 10 maiores voos: -0.2 s
  mês 01: ganho 0.5 s (IC 95% -0.0 a 1.4)
  mês 07: ganho -0.2 s (IC 95% -2.1 a 1.5)
  informativo (fora do veredito):
    voos normais com NM (338,278 voos): ganho 1.6 s (IC 95% 1.3 a 1.8)
    sem loteria (344,408 voos): ganho 0.5 s (IC 95% -0.4 a 1.3)
  veredito: não comprovado
```

**Portão: reprovado** — IC baixo −0,8, sem os 10 maiores −0,2, julho −0,2. Na fatia-alvo:

| Aeroporto | v28 | H2 | |
|---|---|---|---|
| LEMD | 175,74 | 173,37 | **−2,36** |
| LTFM | 232,48 | 231,20 | −1,29 |
| EGLL | 206,74 | 205,55 | −1,20 |
| LFPG | 225,28 | 225,20 | −0,08 |
| os quatro | 212,92 | 211,78 | −1,14 |

`normais_nm` 193,45 → 191,90 (+1,55 s), o maior ganho já medido nessa fatia. Código pronto e
**desligado** (`models.MIN_LINHAS_APT`, `experiment.py --base-por-apt`; `crossfit` e
`train.py submit` herdam pela config, sem mudança).

## 6. Veredito

Nenhuma das duas passa o portão do `completo` — as duas ganham 0,2 s ali, dentro do ruído, e
perdem nos 10 maiores voos e em julho. As duas ganham de verdade onde o item 4e mandou olhar:

| | completo | normais NM | os 4 sem ADS-B |
|---|---|---|---|
| v28 | 299,41 | 193,45 | 212,92 |
| H1 (célula no corretor) | 299,23 | 192,85 | 212,08 |
| H2 (regressor por aeroporto) | 299,23 | 191,90 | 211,78 |

A conclusão prática é a mesma do estudo da cauda: **a fatia dos normais sem ADS-B rende ~1–2 s,
não os 25 s que faltam para o 3º lugar**, porque 59 % do erro² está fora dela e o corretor da
v28 já recupera a maior parte do que uma base melhor entrega. A distância para o topo não está
aqui.

Próximo passo recomendado: **H1 + H2 juntas** (`--base-por-apt` na base e `--corretor-ref` no
corretor). Elas atacam aeroportos diferentes — H1 é LFPG (−1,76) e LEMD (−1,18), H2 é LEMD
(−2,36), LTFM (−1,29) e EGLL (−1,20) — então a soma na fatia pode chegar a −2 s e empurrar o
`completo` acima do ruído. Custo: ~50 min (a base muda, não dá para reaproveitar o oof).

### H1 + H2 juntas (29/09)

`20260929-135844-h1h2_por_apt_ref` (9m11s, pico 6,79 GB; base `20260929-125421-base_ctx_por_apt`,
corretor da v28 + `--corretor-ref`, `--reusar-oof 20260929-130343-v31_por_apt`) contra a v28:
simulação 299,41 → 299,10, ganho 0,3 s (IC −0,7 a 1,3), sem os 10 maiores −0,0, janeiro +0,5, julho +0,1;
`normais_nm` 193,45 → 191,48 (+2,0 s, IC 1,7 a 2,3), sem loteria +1,2 s (IC 0,7 a 1,8).
**Portão: reprovado** (IC baixo −0,7, sem top 10 = 0). Os ganhos somam na fatia, mas o `completo`
continua dentro do ruído. Enviada mesmo assim com o ok do usuário, pelo ganho sem loteria:
**v29 = 247,83 oficial (−1,33 s sobre a v28)**; promovida a campeã à mão. As duas flags agora fazem
parte da campeã.
