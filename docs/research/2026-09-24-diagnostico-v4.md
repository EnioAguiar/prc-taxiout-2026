# Diagnóstico da divergência da v4 (simulação −42 s, oficial −1,5 s)

24/09/2026. Pergunta: por que as retas por aeroporto nos voos sem registro NM
(`two_stage_nm`) ganharam 42,3 s na simulação (388,16 → 345,89) e só 1,5 s no
placar oficial (338,67 → 337,18)?

Restrição dura usada o tempo todo: a v4 só difere da v3 nas 5.282 linhas
`nm_missing` do ranking, então toda a variação oficial vem delas:

```
ΔSSE_oficial = 344.841 × (337,18² − 338,67²) = −347,3 milhões de s²
             = −65.694 s² por linha alterada
ΔSSE_simulado (holdout jan+jul/2025)          = −10.687 milhões de s²
```

**A simulação prometeu 31× mais redução de erro quadrático do que o oficial
entregou.** As três causas, em ordem de tamanho, estão na seção
[Diagnóstico](#diagnóstico-o-que-de-fato-aconteceu).

## Resumo

1. O ganho simulado é uma **loteria de 10 voos**: as 10 maiores contribuições
   valem 120,9 % do ganho total (todas LIRF, `ms` > 2 h, cópia do SCHED).
   Tirando essas 10 linhas de 344.419, a v4 **piora** a v3 em 9,7 s.
2. O ganho simulado media **folga que a v3 enviada não tinha**. Nos extremos do
   LIRF (`ms` > 6 h) a v3 da simulação (10 meses, 400 rodadas) prevê 67,9 % de
   `ms`; a v3 enviada (12 meses, 480 rodadas) prevê **90,5 %**. Sobrava 14.096 s
   por voo na simulação contra 3.951 s no ranking.
3. Teto calculável **antes de enviar**: mesmo que todos os 96 extremos de 2026
   fossem cópia (`y = ms`), as retas só podiam render **−8,5 s** oficiais. A
   simulação prometia −42,3 s. O oficial (−1,5 s) é o teto menos o dano
   sistemático nos voos normais sem NM.
4. Recomendação: **encolher** — manter as retas só onde a v3 deixa folga
   (`nm_missing` e `ms` > 6 h, 96 linhas do ranking em vez de 5.282). No holdout
   isso dá 334,51 s (contra 345,89 da v4 e 388,16 da v3) e zera o dano nos voos
   normais (RMSE 1.081,9 contra 1.083,2 da v3 e 1.318,9 da v4).

## 1. Distribuição 2025 × 2026 nos voos sem NM

`ms = MVT − SCHED` (segundos), só DEP, só `nm_missing`.

| Conjunto | linhas | nm | % | ms p50 | ms p90 | ms p99 | > 2 h | > 6 h |
|---|---|---|---|---|---|---|---|---|
| ranking 2026 (jan+jul) | 344.841 | 5.283 | 1,53 | 5.838 | 12.002 | 28.329 | 32,5 % (1.719) | 1,82 % (96) |
| holdout 2025 (jan+jul) | 344.419 | 5.366 | 1,56 | 5.700 | 9.725 | 20.953 | 26,5 % (1.422) | 0,97 % (52) |
| jan/2026 | — | 2.449 | — | 6.478 | 13.919 | 29.954 | 42,5 % (1.040) | 2,33 % (57) |
| jul/2026 | — | 2.834 | — | 5.524 | 9.779 | 26.639 | 24,0 % (679) | 1,38 % (39) |
| jan/2025 | — | 1.432 | — | 5.238 | 9.153 | 23.162 | 22,2 % (318) | 1,12 % (16) |
| jul/2025 | — | 3.934 | — | 5.814 | 9.904 | 20.820 | 28,1 % (1.104) | 0,92 % (36) |

Onde 2026 difere:

- **A cauda de `ms` é maior, não menor**: 32,5 % acima de 2 h contra 26,5 %, e 96
  extremos (> 6 h) contra 52. Ou seja, **não** é o caso de "sumiram os voos onde
  a reta ganha". A oportunidade bruta existe.
- **A composição por aeroporto virou.** Ranking: EHAM 1.206 (22,8 %), LFPG 732,
  LTFM 719, LSZH 659, LEBL 441, EDDF 421, LIRF 383, EGLL 248, EDDM 238, LEMD 236.
  Holdout: LFPG 879, EHAM 808, LTFM 653, EDDF 599, LSZH 524, LEBL 494, EGLL 466,
  LIRF 397, LEMD 309, EDDM 237. Em jan/2026 o EHAM sozinho é 33 % dos voos sem NM
  (805 de 2.449).
- **Nos extremos a virada é o problema**: dos 96 voos com `ms` > 6 h de 2026,
  41 são EHAM e só 26 são LIRF. Em 2025 inteiro os 207 extremos eram LIRF 63,
  LFPG 36, EHAM 33, EDDF 29, … Isso importa porque a "cópia do SCHED"
  (`|y − ms| ≤ 60 s`) na cauda **só existe em Roma**: dos 397 voos sem NM com
  `ms` > 2 h e cópia em 2025, **396 são LIRF** (1 é EGLL). Entre os extremos
  (`ms` > 6 h) de 2025: LIRF 51 cópias em 63 voos (81 %), demais aeroportos
  **0 cópias em 144 voos** (`y` mediano 1.024 s).
- A distribuição do alvo não mudou de forma dramática: no holdout os voos sem NM
  têm `y` mediano 976 s, p99 11.053 s, 111 acima de 2 h e 23 acima de 6 h.

Taxa de cópia por faixa de `ms` (2025 inteiro, voos sem NM):

| faixa de ms | n | cópias | taxa | y mediano |
|---|---|---|---|---|
| ≤ 2 h | 17.484 | 1.033 | 5,9 % | 898 |
| 2–4 h | 4.401 | 296 | 6,7 % | 1.122 |
| 4–6 h | 332 | 50 | 15,1 % | 1.144 |
| 6–10 h | 112 | 18 | 16,1 % | 1.073 |
| > 10 h | 95 | 33 | 34,7 % | 1.846 |

## 2. O que as retas mudam no ranking de 2026

5.282 das 5.283 linhas `nm_missing` mudaram (uma coincidiu). Quatro linhas com NM
diferem em < 1e-6 s (ruído de threading do LightGBM) e não contam.

| Estatística de `d = v4 − v3` | valor |
|---|---|
| média | −11,6 s |
| mediana | −0,6 s |
| p10 / p90 | −637 / +467 s |
| média de \|d\| | 459,8 s |
| Σ\|d\| | 2.430.448 s |
| sobe / desce | 2.640 / 2.646 |

Por faixa de `ms`:

| faixa | n | d médio | d mediano | v3 médio | v4 médio |
|---|---|---|---|---|---|
| ms ≤ 0 | 109 | +110,6 | +208,7 | 391 | 502 |
| 0–30 min | 893 | +29,9 | +133,2 | 796 | 826 |
| 30 min–2 h | 2.565 | −42,1 | −7,1 | 1.201 | 1.159 |
| 2–6 h | 1.623 | −96,3 | −100,1 | 1.854 | 1.757 |
| > 6 h | 96 | **+1.709,3** | +357,7 | 11.698 | 13.407 |

Por aeroporto (só os extremos, `ms` > 6 h, que é onde está o dinheiro):

| aeroporto | n | ms mediano | v3 mediano | v4 mediano | d mediano |
|---|---|---|---|---|---|
| EHAM | 41 | 27.258 | 1.193 | 1.226 | +225 |
| **LIRF** | **26** | **40.560** | **34.948** | **40.258** | **+3.949** |
| EDDF | 6 | 48.350 | 2.028 | 1.377 | −488 |
| outros | 23 | ~27.000 | ~1.800 | ~1.500 | ±1.000 |

**Nos extremos do LIRF a v3 enviada já previa 90,5 % de `ms`** (mediana de
`v3/ms`; v3 mediano 34.948 s contra `ms` mediano 40.560 s). A reta só completou
os últimos 9,5 %. No holdout a v3 da simulação previa 67,9 % (faltavam 14.096 s
por voo) — quatro vezes mais folga.

### Conta com a restrição oficial

O placar dá um único escalar (−347,3 M s²), então a separação entre cauda e
normais é estimativa. Usando o dano por linha medido no holdout (as 5.314 linhas
sem NM com `ms` ≤ 6 h perderam 2,666 G s², ou +501,7 k s² por linha) e aplicando
às 5.187 linhas equivalentes de 2026:

| Parcela | ΔSSE estimado | em RMSE |
|---|---|---|
| dano nos voos normais sem NM (5.187 linhas) | **+2,60 G** | +11,2 s |
| ganho na cauda (96 linhas), por diferença | **−2,95 G** | −12,7 s |
| total observado no placar | −0,347 G | −1,49 s |
| *teto* se todos os 96 extremos fossem cópia (`y = ms`) | −1,97 G | −8,5 s |

Leitura: em 2026 as retas **continuaram ganhando na cauda e perdendo nos
normais**, exatamente como na simulação — só que o ganho da cauda encolheu de
−13,35 G (52 voos do holdout) para ≈ −2,95 G (96 voos do ranking), isto é, de
−256,8 M por voo extremo para ≈ −30,7 M. Não é "a reta estourou em 2026": é "a v3
enviada já tinha feito o trabalho". O dano nos normais, esse sim, transferiu
inteiro.

## 3. Estabilidade dentro de 2025 (leave-one-month-out, só retas)

Retas ajustadas por aeroporto em 11 meses, previstas no mês retirado, só em
linhas `nm_missing` de `full2025`. Base: previsão da v3 onde ela existe
(jan e jul, das corridas do holdout) e mediana do treino nos demais meses.
Convenção: ΔSSE negativo = reta melhor.

| mês | n | base | RMSE base | RMSE reta | ΔSSE | ΔSSE sem os 10 maiores | top-10 (% do ganho) |
|---|---|---|---|---|---|---|---|
| 1 | 1.432 | v3 | 3.111,1 | 2.946,7 | −1,43 G | **+0,58 G** | 140,5 % |
| 2 | 1.109 | mediana | 4.830,8 | 2.066,2 | −21,15 G | −1,39 G | 93,4 % |
| 3 | 1.081 | mediana | 3.190,1 | 1.062,2 | −9,78 G | −1,27 G | 87,0 % |
| 4 | 1.083 | mediana | 2.969,4 | 891,7 | −8,69 G | −0,92 G | 89,4 % |
| 5 | 1.603 | mediana | 2.859,8 | 2.477,3 | −3,27 G | −0,93 G | 71,6 % |
| 6 | 2.499 | mediana | 3.259,7 | 1.941,9 | −17,13 G | −1,73 G | 89,9 % |
| 7 | 3.934 | v3 | 2.147,5 | 1.500,1 | −9,29 G | **+2,04 G** | 121,9 % |
| 8 | 2.323 | mediana | 3.170,6 | 1.340,9 | −19,18 G | −2,42 G | 87,4 % |
| 9 | 2.458 | mediana | 3.379,0 | 1.322,0 | −23,77 G | −1,35 G | 94,3 % |
| 10 | 1.910 | mediana | 1.571,4 | 898,4 | −3,17 G | −0,90 G | 71,7 % |
| 11 | 1.349 | mediana | 3.774,7 | 1.437,7 | −16,43 G | −0,48 G | 97,1 % |
| 12 | 1.643 | mediana | 3.177,2 | 888,7 | −15,29 G | −1,73 G | 88,7 % |

- A reta ganha em 12/12 meses, mas **nunca com menos de 71 % do ganho vindo de 10
  voos**; em 6 dos 12 meses um único voo vale mais de 40 % do ganho (jan 112 %,
  fev 80 %, mar 72 %, dez 49 %).
- Nos dois únicos meses em que a base é o modelo de verdade (v3, jan e jul), o
  ganho **inverte de sinal** ao tirar os 10 maiores: +0,58 G e +2,04 G. Contra a
  mediana o sinal se mantém, mas a mediana é uma base boba — ela não tenta prever
  cauda nenhuma.
- O coeficiente do LIRF é estável e é literalmente a cópia (`b` entre 0,99 e 1,29
  em 9 dos 12 meses; `b` = 1,068 no ajuste dos 10 meses de treino); os outros
  nove aeroportos têm `b` entre 0,010 e 0,170 — lá a "reta" é uma constante.
- O número de cópias extremas do LIRF por mês é o que manda no ganho: 4, 3, 0, 3,
  1, 5, **12**, 9, 8, 1, 2, 3. Julho de 2025 (12 cópias) é o mês mais rico do ano
  — e é metade do holdout.

## 4. Concentração no holdout (jan+jul/2025)

ΔSSE total da v3 → v4: −10,687 G s² (100 % vindo das linhas `nm_missing`).

| Corte | ganho acumulado | % do ganho | composição |
|---|---|---|---|
| top 10 voos | 12,92 G | **120,9 %** | 10/10 LIRF com `ms` > 2 h |
| top 56 voos | 14,28 G | **133,6 %** | 48 LIRF (40 com `ms` > 2 h), 4 LSZH, 2 LFPG, 2 EGLL |
| top 100 voos | 14,58 G | 136,4 % | — |
| 5.366 linhas nm | −10,69 G | 100 % | 2.093 melhoram, 3.273 pioram |

- Ganho por regime: cauda (`y` ≥ 3 h) −13,69 G; normais (`y` < 3 h) **+3,01 G**
  (a reta piora).
- Excluindo apenas as 52 linhas com `ms` > 6 h: v3 = 327,31 s, v4 = 338,93 s — a
  v4 é **11,6 s pior** em 344.367 dos 344.419 voos.
- Em RMSE: ganho cheio 42,27 s; **sem os 10 maiores voos, −9,71 s**; sem os 56
  maiores, −18,51 s.
- Por mês: jan 12,87 s (IC 95 % −9,34 a +48,14), jul 65,38 s (IC −0,78 a
  +132,86). Nenhum dos dois meses sustenta o ganho sozinho com IC acima de zero.

## Diagnóstico: o que de fato aconteceu

1. **O efeito real é pequeno e caro.** A reta só agrega onde há cópia do SCHED na
   cauda, o que em 2025 é um fenômeno de um aeroporto (LIRF, 51 das 51 cópias
   extremas) e de poucas dezenas de voos por ano. Nos outros 9 aeroportos a reta
   é uma constante que substitui um LightGBM — dano garantido (+3,01 G no
   holdout, ≈ +2,6 G estimados em 2026).
2. **A simulação mediu a folga errada.** O incremento foi medido contra uma v3
   treinada em 10 meses com 400 rodadas; o que foi ao ar foi uma v3 treinada em
   12 meses (inclusive jan e jul, que contêm 16 das 51 cópias extremas do LIRF)
   com 480 rodadas. O estágio 1 enviado já classifica as cópias de Roma quase
   perfeitamente: cobertura `v3/ms` mediana de **0,905** no ranking contra
   **0,679** no holdout. Teste direto: elevando a v3 da simulação para a mesma
   cobertura nas 21 linhas LIRF `ms` > 6 h do holdout, o RMSE da v3 cai de 388,16
   para 340,59 e as retas passam a **perder 5,3 s** em vez de ganhar 42,3 s.
3. **O teto de 2026 era conhecido antes do envio.** Com as previsões da v3 e da
   v4 no ranking e o oráculo otimista `y = ms` nos 96 extremos, o ganho máximo
   possível era −8,5 s (LIRF sozinho: −8,85 s). Uma simulação que promete −42,3 s
   sobre um teto de −8,5 s está medindo outra coisa.

Não é um problema de vazamento nem de `ms` ausente (0 % de `ms` NaN nos dois
conjuntos), nem de "2026 não tem cauda" (tem mais cauda que 2025).

## 5. Recomendação

**(b) Encolher: manter as retas só nos voos sem NM com `ms` > 6 h.**

Varredura do limiar no holdout (abaixo do limiar fica a previsão da v3):

| limiar | linhas trocadas | RMSE | jan | jul | RMSE nm (y<3 h) | ganho | ganho sem top-10 | IC 95 % |
|---|---|---|---|---|---|---|---|---|
| v3 (nenhuma) | 0 | 388,16 | 367,68 | 403,91 | 1.083,2 | — | — | — |
| 0 h (**v4**) | 5.223 | 345,89 | 354,81 | 338,53 | 1.318,9 | 42,33 | **−9,65** | 2,0 a 86,6 |
| 1 h | 4.230 | 344,58 | 354,80 | 336,12 | 1.291,0 | 43,58 | −8,38 | 2,6 a 88,8 |
| 2 h | 1.422 | 342,85 | 353,41 | 334,11 | 1.260,8 | 45,31 | −6,61 | 4,8 a 91,4 |
| 3 h | 383 | 339,42 | 350,21 | 330,47 | 1.199,7 | 48,74 | −3,11 | 8,5 a 93,8 |
| 4 h | 153 | 333,87 | 349,56 | 320,66 | 1.091,5 | 54,29 | +2,56 | 13,7 a 98,9 |
| **6 h** | **52** | **334,51** | **349,84** | **321,63** | **1.081,9** | **53,65** | **+1,90** | **12,7 a 98,2** |
| 8 h | 30 | 335,25 | 349,93 | 322,94 | 1.082,6 | 52,91 | +1,15 | 12,5 a 96,8 |

Outras variantes testadas no holdout (todas piores): blend 0,5 com a v3 = 354,58;
`max(v3, reta)` = 346,20; só LIRF com `ms` > 2 h = 342,86; teto de 3 h ou 6 h na
reta = 397,3 (destrói a cauda). Blend não resolve porque o problema não é escala:
é aplicar reta onde não há cópia.

Por que 6 h e não 2 h: a cópia só aparece de verdade acima de 4 h (15–35 % contra
6 % abaixo de 2 h) e é só do LIRF; abaixo disso a reta do LIRF (`b` = 1,07)
prevê horas para voos que saem em 20 minutos. Por que não 4 h (RMSE 0,6 s melhor
no holdout): 4 h troca 3× mais linhas do ranking (321 contra 96) com taxa de
cópia de 15 %, e a diferença está dentro do ruído. 6 h é o corte conservador.

Efeito esperado em 2026 (96 linhas trocadas em vez de 5.282): o dano nos voos
normais some por construção, e o ganho da cauda é o mesmo que a v4 já obteve —
ou seja, a variante é **dominante sobre a v4 no placar**, com ganho estimado
entre −0,5 s (se o dano nos normais de 2026 tiver sido zero) e −12,8 s (se tiver
sido igual ao do holdout), contra os −1,5 s da v4. Teto otimista: −8,5 s.

### Mudança concreta na regra de promoção

Hoje (`src/compare.py`): `ganho ≥ 10 s` e `ic_baixo > 0` no bootstrap pareado por
dia. A v4 passou com 42,27 s e IC 1,97 a 86,54. Acrescentar três exigências,
todas baratas (só previsões já gravadas em `runs/`):

1. **Robustez a poucos voos**: recalcular o ganho descartando as 10 linhas de
   maior contribuição para o ΔSSE; exigir `ganho_sem_top10 > 0` e
   `≥ 10 % do ganho cheio`.
2. **Ganho nos dois meses**: bootstrap por dia separado em jan e em jul; exigir
   `ganho > 0` e `ic_baixo > 0` **em cada mês**.
3. **Teto no ranking** (para mudanças que mexem num subgrupo, antes de enviar):
   aplicar a mudança às previsões do ranking e avaliar sob o oráculo otimista do
   mecanismo alegado (aqui, `y = ms` nas linhas empurradas para `ms`); se
   `ganho_simulado > 2 × teto`, não enviar — a simulação está medindo folga que o
   modelo final não tem.

Como as mudanças já feitas pontuariam nessa régua:

| Mudança | ganho | sem top-10 | jan (IC baixo) | jul (IC baixo) | veredito novo | oficial |
|---|---|---|---|---|---|---|
| v2 → v3 (dois estágios) | 66,77 | **+10,18** (15,2 %) | 21,25 (+2,98) | 97,93 (+32,80) | **promove** | −46,0 s ✅ |
| v3 → v4 (retas em todo nm) | 42,27 | **−9,71** | 12,87 (**−9,34**) | 65,38 (**−0,78**) | **barra** (3 critérios) | −1,5 s ❌ |
| v4 → `nm_retas_2h` | 4,40 | +2,00 | 1,99 | 6,45 | barra (< 10 s, já barrado) | — |
| v3 → retas só `ms` > 6 h | 53,65 | +1,90 (3,5 %) | 17,84 (+0,19) | 82,28 (+18,04) | barra no critério 1 (3,5 % < 10 %) | estimado −0,5 a −12,8 s |

A regra separa exatamente o caso que transferiu (v2 → v3) do que não transferiu
(v3 → v4). A variante recomendada passa em 2 e 3 mas fica no limite do critério 1
(3,5 % do ganho fora dos 10 maiores) — o que é honesto: ela **é** uma aposta na
cauda, só que uma aposta barata (96 linhas, dano zero nos demais voos) e com teto
conhecido. Sugestão prática: promover mudanças "de cauda" que falhem só o
critério 1 apenas se o teto no ranking (critério 3) for ≥ 3 s e o dano fora do
subgrupo for ≤ 1 s — a variante `ms` > 6 h cumpre os dois (teto −8,5 s; RMSE dos
voos sem NM normais 1.081,9 contra 1.083,2 da v3).

## Como foi medido

- `data/cache/{ranking2026,holdout2025,full2025}-*.parquet` (features prontas),
  `runs/20260924-194237-dois_estagios_r.parquet` (v3 simulada),
  `runs/20260924-194654-nm_retas.parquet` (v4 simulada),
  `submissions/outgoing-boat_v{3,4}.parquet` (o que foi ao placar).
- Ganho por voo = `(v3 − y)² − (v4 − y)²`; ΔSSE = soma. Bootstrap pareado por dia
  com `src/compare.py:paired_bootstrap` (1.000 reamostragens, seed 0).
- LOMO: `models.fit_lines`/`apply_lines` nos 11 meses restantes, previsão no mês
  retirado, só linhas `nm_missing` com `ms` e alvo válidos.
- Teto de 2026: `Σ[(v4 − ms)² − (v3 − ms)²]` nas 96 linhas com `ms` > 6 h, que é o
  melhor caso possível se toda a cauda fosse cópia do SCHED.
- Scripts descartáveis em `/tmp/prc-V4Diag/` (não versionados); nada de `src/`
  foi alterado e nada foi enviado.
