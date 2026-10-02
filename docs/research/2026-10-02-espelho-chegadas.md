# Espelho das chegadas — 02/10/2026

Hipótese: as linhas ARR do `ranking.parquet` guardam o `TAXITIME_SEC_mvt` verdadeiro — **o
único rótulo de 2026 que existe**. Pontuar um modelo de taxi-in treinado em 2025 sobre elas e
ler o **resíduo** (observado − esperado) mede a deriva 2025 → 2026 com rótulo de verdade, e
separa deriva de regime de mudança de composição (o nível observado já está no corretor, em
`ctx_arr_tin_*`).

Scripts descartáveis (fora do git, como os demais estudos de uma noite): `espelho.py` (modelo
de taxi-in, quadro de resíduos em cache e as colunas `esp_*` que seriam a flag) e `medir.py`
(todas as tabelas abaixo, em uma rodada).
Holdout: jan+jul de 2025, campeã v36 (`champion.json`, três corretores sobre `e76_base`), **sem**
as pós-regras. Linha de base: `completo` **295,63** · `sem_loteria` **228,69**.

**Veredito: o espelho mede a deriva muito bem e não vira coluna do corretor.** Ele não virou
flag; não há corrida `--crossfit` para comparar, porque a triagem de viabilidade (passo 1 do
plano) reprovou — a correlação com o nosso erro é ~0 em todos os aeroportos.

## O modelo de taxi-in

Uma linha por chegada de 2025 (2.082.749) e de 2026 (344.693). Entradas, todas presentes nas ARR
do ranking exatamente como no treino: aeroporto, pista, stand, tipo de aeronave, categoria de
esteira, companhia, segmento, aeroporto de origem, hora e dia da semana do pouso e quatro
contagens de tráfego em volta do pouso (chegadas e decolagens do aeroporto em ±30 min, chegadas
em ±60 min, chegadas da mesma pista em ±30 min). O taxi-in da própria linha não entra em feature
nenhuma, e **o mês fica fora de propósito**: com mês o modelo descreveria cada mês em separado e
o resíduo de 2026 não teria contra o que ser lido.

2025 é cruzado por mês (12 folds: o mês da linha nunca está no treino dela); 2026 usa o modelo
de 2025 inteiro. O alvo é cortado em 7.200 s antes de treinar e de formar o resíduo. Custo:
13 ajustes de LightGBM, ~5 min, em cache em `data/cache/espelho-*.parquet`.

## 1. O que o espelho mede: a deriva existe e é grande

Resíduo médio (observado − esperado, s) por aeroporto × mês:

| aeroporto | 2025-01 | 2025-07 | 2026-01 | 2026-07 | Δjan | Δjul |
|---|---|---|---|---|---|---|
| EDDF | −3,5 | 5,3 | 17,0 | −52,9 | +20,5 | −58,2 |
| EDDM | −4,5 | 13,8 | 7,1 | 4,2 | +11,6 | −9,6 |
| EGLL | 13,5 | 30,8 | 12,2 | −3,8 | −1,3 | −34,5 |
| **EHAM** | 20,1 | 2,0 | **98,7** | −10,5 | **+78,6** | −12,5 |
| LEBL | −8,6 | 13,0 | −4,3 | −5,7 | +4,2 | −18,7 |
| LEMD | −25,3 | 11,7 | −25,8 | −29,0 | −0,5 | −40,7 |
| LFPG | 12,9 | 5,9 | 18,7 | −16,0 | +5,8 | −21,9 |
| LIRF | −14,6 | −2,7 | −30,4 | 15,1 | −15,7 | +17,8 |
| LSZH | 6,5 | −0,5 | 23,1 | −1,3 | +16,6 | −0,9 |
| LTFM | 4,9 | 18,7 | 12,3 | −3,2 | +7,3 | −21,9 |

Em RMSE do resíduo:

| aeroporto | 2025-01 | 2025-07 | 2026-01 | 2026-07 |
|---|---|---|---|---|
| **EHAM** | 154,7 | 155,9 | **436,5** | 129,6 |
| **LFPG** | 196,8 | 206,0 | **264,9** | 191,0 |
| EGLL | 306,5 | 341,0 | 277,0 | 283,1 |
| LIRF | 285,6 | 330,8 | 294,5 | 355,5 |
| EDDF | 159,7 | 175,9 | 164,0 | 151,9 |
| os outros cinco | 81–165 | 109–194 | 74–170 | 77–183 |

Julho de 2026 é mais **fácil** que julho de 2025 em quase toda parte (a média do resíduo cai em
9 dos 10 aeroportos, só o LIRF sobe); janeiro de 2026 é mais difícil em 7 dos 10.

## 2. A deriva de janeiro é concentrada em poucos dias de ruptura

EHAM, jan/2026 (`obs` e `esperado` são as médias do taxi-in em segundos):

| dia | chegadas | resíduo médio | mediana | obs | esperado | parte acima de 30 min |
|---|---|---|---|---|---|---|
| 2026-01-05 | 201 | **1.578,9** | 1.029,7 | 1.967,1 | 388,2 | **40 %** |
| 2026-01-04 | 355 | 522,1 | 148,8 | 886,1 | 364,1 | 10 % |
| 2026-01-03 | 431 | 359,3 | 72,4 | 727,4 | 368,1 | 10 % |
| 2026-01-02 | 468 | 295,0 | 16,8 | 664,9 | 369,9 | 10 % |
| 2026-01-07 | 199 | 239,3 | 117,7 | 683,6 | 444,3 | 0 % |
| 2026-01-06 | 287 | 172,1 | 95,6 | 590,7 | 418,6 | 0 % |

No dia 05 o aeroporto recebeu **201 chegadas** (o normal de janeiro é 520–620) e o taxi-in médio
foi **5× o esperado** — é fechamento de aeroporto, não congestionamento. O pior dia de jan/2025
no EHAM tem resíduo médio de **87,7 s**: nenhum mês de treino contém nada parecido. LFPG tem um
dia do mesmo tipo, 2026-01-07 (540,3 s de resíduo, taxi-in médio 1.120 s), e depois nada acima de
78 s. **A deriva de janeiro de 2026 é um punhado de dias de ruptura, não um nível novo.**

## 3. O que o espelho **não** faz: prever o nosso erro

Correlação entre o resíduo de taxi-out da campeã no holdout jan+jul/2025 (344.419 decolagens) e
as cinco colunas candidatas do espelho, por aeroporto:

| aeroporto | ADS-B | `esp_res_dia` | `esp_res_3h` | `esp_res_1h` | `esp_n_1h` | `esp_res_1h_menos_dia` |
|---|---|---|---|---|---|---|
| todos | — | 0,005 | 0,007 | 0,010 | 0,010 | 0,009 |
| EDDF | sim | −0,032 | −0,042 | −0,024 | 0,009 | −0,010 |
| EDDM | sim | 0,030 | 0,016 | 0,015 | 0,027 | −0,001 |
| EGLL | **não** | 0,019 | 0,021 | 0,033 | 0,001 | 0,028 |
| EHAM | sim | 0,011 | 0,030 | 0,022 | −0,008 | 0,019 |
| LEBL | sim | 0,030 | 0,016 | 0,013 | 0,027 | −0,004 |
| LEMD | **não** | 0,009 | 0,025 | 0,035 | −0,017 | 0,036 |
| LFPG | **não** | −0,030 | −0,024 | −0,001 | 0,023 | 0,017 |
| LIRF | **não** | −0,012 | −0,005 | 0,000 | 0,014 | 0,004 |
| LSZH | sim | −0,005 | 0,051 | 0,049 | 0,015 | 0,055 |
| LTFM | **não** | 0,020 | 0,032 | 0,031 | 0,004 | 0,026 |

Nenhum |r| passa de 0,055, inclusive nos cinco aeroportos sem cobertura ADS-B — a fatia que vale
27,3 % do erro² e que motivou a busca. Com o resíduo cru (sem corte do alvo) e com Spearman dá o
mesmo: ±0,014.

Agregando por **dia-aeroporto** (620 dias), onde a hipótese de "regime do dia" deveria aparecer
mais forte, o sinal continua ausente e **os sinais discordam entre aeroportos**:

```
pooled +0,056 | EDDF −0,408 · EDDM +0,180 · EGLL +0,204 · EHAM +0,135 · LEBL +0,233
               LEMD +0,098 · LFPG −0,372 · LIRF −0,156 · LSZH −0,065 · LTFM +0,159
```

Com n = 62 dias por aeroporto o erro-padrão é ±0,13: EDDF −0,41 e LFPG −0,37 contra LEBL +0,23 e
EGLL +0,20 não são um mecanismo, são dez amostras em volta de zero com duas caudas. Por hora-
aeroporto (11.648 células com ≥ 5 voos): pearson +0,050, spearman +0,028.

### Por que: o espelho acerta a **magnitude**, não a **direção**

Quintis do resíduo de chegada do dia contra o erro da campeã naquele dia-aeroporto:

| quintil | `esp_res_dia` médio | RMSE da campeã no dia | viés da campeã no dia | dias |
|---|---|---|---|---|
| 0 | −25,3 | 249,8 | −13,6 | 124 |
| 1 | −8,1 | 207,0 | −7,4 | 124 |
| 2 | 1,5 | **198,8** | −6,5 | 124 |
| 3 | 12,8 | 268,5 | −10,2 | 124 |
| 4 | 44,2 | **283,4** | −4,9 | 124 |

O RMSE do dia faz um **U**: dias em que as chegadas desviam (para cima ou para baixo) são dias em
que erramos mais — 283 contra 199 no miolo. Mas o **viés** é plano (−4,9 contra −6,5 s): o espelho
não diz para que lado. Isso bate com as correlações: com `|resíduo|` a correlação sobe para
+0,038 (pearson) e +0,073 (spearman), e com o resíduo com sinal fica em +0,005.

Um corretor que minimiza RMSE prevê a **média condicional**. Saber que a variância do dia é maior,
sem saber a direção, não muda a média condicional — não há ganho a tirar daí. Seria um sinal para
um modelo de incerteza (intervalo, quantil), que a competição não pontua.

E não é redundância com o que já temos: a correlação das colunas novas com as `ctx_arr_tin_*` do
corretor é só **+0,16 a +0,30**. O resíduo é informação **nova** — ela simplesmente não aponta
para lado nenhum do nosso erro.

### Teto de qualquer uso linear

OLS por aeroporto (as 5 colunas + intercepto, 60 parâmetros), ajustado **na própria amostra** do
holdout — portanto um teto otimista, não uma estimativa:

| | antes | depois (dentro da amostra) | ganho |
|---|---|---|---|
| `completo` | 295,63 | 295,43 | **+0,20 s** |
| `sem_loteria` | 228,69 | 228,42 | **+0,28 s** |

O controle nulo de 60 parâmetros em 344.419 linhas vale ~0,03 s de redução só por graus de
liberdade. Mesmo tomando os +0,28 s como reais, estão **abaixo do `GANHO_A = 0,3`** da régua
(`src/regua.py`), e isso é o teto medido dentro da amostra, com os coeficientes escolhidos
depois de ver as respostas. Fora da amostra o valor esperado é ~0.

## 4. Mesmo se o sinal existisse, o corretor não poderia aprendê-lo

O corretor treina em 2025 (cegas) e é aplicado em 2026. As colunas do espelho em 2026 caem
largamente **fora da faixa que 2025 mostra**:

| coluna | 2025 (p1 / p99 / máx) | 2026 (p1 / p99 / máx) | 2026 acima do máximo de 2025 |
|---|---|---|---|
| `esp_res_dia` | −50 / 81 / **256** | −66 / 160 / **1.579** | 0,55 % |
| `esp_res_1h` | −95 / 165 / 2.921 | −101 / 220 / 3.921 | 0,003 % |

No EHAM, onde o espelho mais grita, é pior: `esp_res_dia` médio de **98 s** em jan/2026 contra um
**máximo de 88 s** no EHAM em 2025 inteiro — **25,6 % das decolagens do EHAM em jan/2026 estão
acima de tudo que 2025 mostrou**. Árvore não extrapola: nessas linhas a resposta satura no último
bin aprendido em 2025, e o que foi aprendido em 2025 é ~0 (a tabela de correlações). A coluna
entraria como peso morto exatamente nas linhas que a motivaram.

## Decisão

**Não adotar, e não gastar um `--crossfit`.** O plano previa construir a flag `--espelho` e medir
o A/B; o passo 1 (triagem de viabilidade) reprovou em três medições independentes — correlação
~0 por voo, por dia e por hora; teto linear dentro da amostra abaixo do limiar da régua; e falta
de suporte em 2026 para a única região onde o sinal é grande. A esteira **não foi pausada**:
nenhuma corrida pesada foi disparada por este estudo.

Vale remedir se a base ou o corretor mudarem de forma a **perder** o relógio do voo
(`AOBT_3`, ADS-B): a explicação mais provável do zero é que a campeã já acompanha o regime do
dia linha a linha por esses relógios, então o resumo do dia vindo das chegadas não acrescenta
nada — um modelo que lê relógio do voo acompanha o regime, um modelo de planos e contagens não.

## O que fica deste estudo

1. **A medição de deriva** (`espelho.py` → `data/cache/espelho-*.parquet`): é a única régua com
   rótulo verdadeiro de 2026 que temos. Serve para auditar qualquer hipótese sobre 2026 sem
   gastar envio (~5 min na primeira vez, ~1 min com o cache).
2. **Os dias de ruptura de jan/2026 estão identificados**: EHAM 02 a 07/01 (05/01 com 201
   chegadas e taxi-in 5× o esperado) e LFPG 07/01. Janeiro vale 44,3 % das linhas pontuadas.
   Qualquer regra que mexa nesses dias deve ser julgada por esta lista, não por holdout de 2025
   — 2025 não tem um dia sequer dessa magnitude.
3. **Julho de 2026 parece mais fácil que julho de 2025** no taxi-in de 9 dos 10 aeroportos
   (EDDF −58 s, LEMD −41, EGLL −35). Se isso valer para o taxi-out, a nota oficial de julho
   deve sair melhor que a simulação de jan+jul/2025 sugere.

## Negativos a não repetir

- **Resíduo do taxi-in das chegadas como coluna do corretor** (dia, ±3 h, ±1 h, contagem e
  desvio da hora contra o dia): correlação ≤ 0,055 por voo em todos os aeroportos, sinais
  discordantes por aeroporto no nível de dia, teto linear dentro da amostra +0,20 s
  (`completo`) / +0,28 s (`sem_loteria`).
- **Nível observado do taxi-in por dia** (sem modelo, só a média do dia menos a média do
  aeroporto no mês), que é o que `ctx_arr_tin_*` já traz agregado: correlação com o resíduo do
  dia da campeã de −0,47 a +0,18, pooled −0,04. Mesma conclusão por caminho mais barato.
- **Procurar o sinal nos dias extremos de 2025**: nos 10 dias-aeroporto de maior resíduo de
  chegada (EGLL 06/07 com 256 s, EGLL 11/01 com 182 s, EDDF 05/01 com 165 s) o viés da campeã
  vai de −56,6 a +56,0 s, sem sinal estável. Dia ruim para as chegadas não é dia em que
  erramos para cima nas partidas.
