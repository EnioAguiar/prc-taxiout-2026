# Diagnóstico de Roma (LIRF) — 28/09/2026

Alvo: entender por que os voos normais de LIRF (y ≤ 1 h) têm RMSE 424 s no holdout da
campeã `20260928-113219-v18_cf` (RMSE completo 303,82) e medir consertos baratos.
Scripts descartáveis em `.superpowers/roma/` (`diag.py`, `testes.py`); tabelas cruas em
`.superpowers/roma/out/`. Nada em `src/` foi tocado.

## Resumo em uma linha

Os 424 s são **duas coisas coladas**: 114 voos em que o modelo aposta na cópia do SCHED
(46,2 % do erro² dos normais de Roma, e a aposta **paga** — retirá-la piora tudo) e 25.954
voos comuns com RMSE 311, cuja parte explicável já foi extraída. **Não há conserto barato
dentro dos normais de Roma.** O único ganho reproduzível encontrado é genérico e não é de
Roma: dar ao corretor a categórica `aeroporto|prefixo de stand` (com as contagens de fila e
`aeroporto|pista` junto), **−1,4 s no completo e −1,85 s sem loteria** na simulação barata,
em três sementes — o efeito aparece no LTFM e na cauda de Roma, não nos normais dela.

## 1. Onde está o erro (holdout jan+jul/2025, 344.419 voos, erro² 31.791 M s²)

LIRF sozinho: 26.528 voos, erro² 8.577 M = **27,0 %** do total, RMSE 568,6.

| Fatia de LIRF | n | erro² (M) | % do total | RMSE |
|---|---|---|---|---|
| normais (y ≤ 1 h) | 26.068 | 4.683 | 14,7 % | 424 |
| — dos quais com `pred` > 1 h (a aposta) | 114 | 2.166 | 6,8 % | 4.359 |
| — dos quais o resto | 25.954 | 2.517 | 7,9 % | 311 |
| cauda y > 1 h (sem loteria) | 455 | 2.448 | 7,7 % | 2.319 |
| loteria (y > 3 h sem plano que explique) | 5 | 1.446 | 4,5 % | 17.007 |

RMSE dos normais por aeroporto (a mesma tabela do `compare.py`, mas somando as linhas sem NM):

| Aeroporto | n | RMSE | erro² (M) | % do total |
|---|---|---|---|---|
| LIRF | 26.068 | 424 | 4.683 | 14,7 % |
| LTFM | 46.499 | 237 | 2.623 | 8,2 % |
| LFPG | 39.474 | 232 | 2.129 | 6,7 % |
| EGLL | 39.970 | 212 | 1.797 | 5,7 % |
| LEMD | 35.328 | 178 | 1.119 | 3,5 % |
| LEBL | 28.986 | 195 | 1.102 | 3,5 % |
| EDDF | 36.806 | 148 | 803 | 2,5 % |
| EHAM | 40.895 | 110 | 494 | 1,6 % |
| LSZH | 22.334 | 143 | 455 | 1,4 % |
| EDDM | 27.078 | 125 | 421 | 1,3 % |

### A fatia que manda: linhas sem NM

| Recorte dos normais de LIRF | n | viés | RMSE | erro² (M) | % do erro² dos normais |
|---|---|---|---|---|---|
| com NM | 25.819 | +20 | 308 | 2.442 | 52,1 % |
| **sem NM** | **249** | **+1.897** | **3.000** | **2.241** | **47,9 %** |

Ou seja: **0,96 % dos voos normais de Roma carregam metade do erro² deles**. Com NM
presente, LIRF é um aeroporto ruim mas comum (RMSE 308, contra 237 do LTFM e 232 do LFPG).
Os cortes por segmento, operador e tipo de aeronave repetem exatamente esta linha (a
categoria vazia = linha sem NM); não são eixos independentes.

### Mês, pista, hora, stand (só os normais)

| mês × pista | n | viés | RMSE | % do erro² dos normais |
|---|---|---|---|---|
| jul · 25 | 13.479 | +51 | 485 | 67,6 % |
| jan · 25 | 8.897 | +23 | 284 | 15,3 % |
| jul · 16R | 1.167 | +44 | 630 | 9,9 % |
| jan · 16R | 1.230 | +17 | 323 | 2,7 % |
| jan · 34L | 1.079 | +3 | 327 | 2,5 % |
| jul · 34L | 117 | +191 | 794 | 1,6 % |

Julho = 79,2 % do erro² dos normais (RMSE 500 contra 294 de janeiro). Pista 25 = 82,9 %,
mas ela também é 86 % dos voos: o problema é julho, não a pista em si.

Por hora, o pico aparente das 21 h (RMSE 702, 18,3 %) **é a aposta, não o horário**: com
`pred` ≤ 1 h os blocos de 3 h ficam entre 265 e 353 s, e os 22 voos apostados das 21 h
respondem por 655 M dos 857 M do bloco.

| bloco 3 h | n (pred ≤ 1 h) | viés | RMSE | n apostado | erro² apostado (M) |
|---|---|---|---|---|---|
| 0 | 72 | +102 | 448 | 1 | 61 |
| 3 | 2.158 | +19 | 265 | 3 | 31 |
| 6 | 4.508 | +28 | 284 | 10 | 216 |
| 9 | 4.337 | +7 | 353 | 16 | 141 |
| 12 | 4.884 | +27 | 325 | 31 | 285 |
| 15 | 4.090 | +15 | 295 | 20 | 354 |
| 18 | 4.190 | +23 | 299 | 11 | 424 |
| 21 | 1.715 | +50 | 343 | 22 | 655 |

Prefixo de stand: 3 (38,2 % do erro², RMSE 466), 8 (13,0 %, RMSE 611), 9 (5,6 %, RMSE 855,
só 358 voos). Já entre os voos com `pred` ≤ 1 h de julho/pista 25 o RMSE por prefixo fica
entre 269 e 484 — o stand separa distância de táxi, não erro anômalo.

ADS-B presente (44 % das linhas) tem RMSE **maior** (501 contra 350): é confundimento com
julho e com as linhas sem NM, não sinal de que o ADS-B atrapalhe.

### Atraso na largada é o eixo real

| MVT − SCHED (min) | n | viés | RMSE | % do erro² dos normais |
|---|---|---|---|---|
| ≤ 10 | 2.552 | ~0 | 166 | 1,6 % |
| 10–20 | 6.792 | −2 | 161 | 3,7 % |
| 20–30 | 5.560 | +13 | 239 | 6,8 % |
| 30–45 | 4.917 | +14 | 334 | 11,7 % |
| 45–60 | 2.521 | −7 | 469 | 11,8 % |
| 60–90 | 2.085 | +156 | 516 | 11,9 % |
| 90–120 | 845 | +212 | 688 | 8,5 % |
| > 120 | 796 | +480 | 1.608 | 43,9 % |

O erro cresce monotonicamente com o atraso entre o horário programado e a decolagem, e
explode acima de 2 h — exatamente onde o modelo tem de decidir se o BLOCK registrado é a
cópia do SCHED (y enorme) ou um pushback de verdade (y normal).

### Cópia do SCHED também acontece dentro dos normais

`y` a ≤ 60 s de `MVT − SCHED` (definição de cópia) nos voos normais:

| Aeroporto | p(cópia) | RMSE cópias | RMSE resto |
|---|---|---|---|
| **LIRF** | **18,2 %** | 330 | 442 |
| LTFM | 10,3 % | 189 | 242 |
| LEBL | 10,2 % | 331 | 173 |
| EDDM | 9,3 % | 91 | 128 |
| EGLL | 8,6 % | 165 | 216 |
| LEMD | 8,0 % | 157 | 180 |
| LFPG | 7,0 % | 170 | 236 |
| EDDF | 6,6 % | 114 | 150 |
| EHAM | 5,8 % | 95 | 111 |
| LSZH | 5,5 % | 128 | 144 |

Roma copia o dobro dos outros. O desvio-padrão do próprio alvo em 2025 é 1.207 s em LIRF
contra 291–433 s nos demais: é um aeroporto com outra física de registro, não um aeroporto
com táxi mais difícil. Não há sinal de arredondamento de horário fora do comum (fração de
BLOCK com segundo zero: 8,4 % em LIRF, 1,7–8,4 % nos demais).

## 2. O que muda em 2026 (`data/ranking.parquet`, sem alvo)

| LIRF | jan/2025 | jan/2026 | jul/2025 | jul/2026 |
|---|---|---|---|---|
| pista 25 | 79,1 % | **91,9 %** | 90,8 % | 88,6 % |
| pista 16R | 10,9 % | 7,4 % | 8,0 % | 9,7 % |
| pista 34L | 9,6 % | **0,4 %** | 0,9 % | 1,0 % |
| sem NM | 0,5 % | 1,0 % | 2,2 % | 1,7 % |
| voos | 11.317 | 11.061 | 15.211 | 15.838 |

- Em 2026 há **pista 25 em julho e em janeiro**, com janeiro ainda mais concentrado nela;
  a 34L praticamente sumiu de janeiro. Como 34L e 25 têm RMSE parecido em janeiro (327 e
  284), o efeito é levemente favorável.
- A taxa de linhas sem NM em Roma é a mesma ordem (1,0 % e 1,7 % contra 0,5 % e 2,2 %):
  **o problema das linhas sem NM vai aparecer em 2026 com o mesmo peso** — ~383 voos.
- A distribuição de `MVT − SCHED` e `MVT − LOBT` é praticamente idêntica entre 2025 e 2026
  (diferenças ≤ 1,5 pp em todas as faixas): o diagnóstico transfere.
- O **uso dos pátios mudou**: prefixo 5 (stands 501–509) cai de 7,6 % para 0,1 %, o prefixo
  8 sobe de 6,6 % para 10,3 % e o 3 de 31,5 % para 35,7 %. Não é renumeração: 99,7 % dos
  voos de LIRF em 2026 usam um código de stand já visto em 2025 (LFPG 99,9 %, EGLL 100 %,
  LTFM 99,5 %), e o pátio 50x simplesmente saiu de uso. Categóricas de stand continuam
  significando a mesma coisa em 2026.

## 3. Os 114 voos com `pred` > 1 h — a aposta é certa, não um bug

Em LIRF o modelo prevê > 1 h em 387 voos:

| grupo | n | y médio | pred médio | erro² (M) |
|---|---|---|---|---|
| falso alarme (y ≤ 1 h) | 114 | 1.712 | 5.372 | 2.166 |
| acerto (y > 1 h) | 273 | 10.670 | 9.660 | 2.667 |
| perdido (y > 1 h, pred ≤ 1 h) | 187 | 4.609 | 2.308 | 1.227 |

O perfil dos falsos e dos acertos é **o mesmo**: `MVT − SCHED` 8.485 s contra 10.716 s,
`MVT − LOBT` 4.661 contra 5.077, 60 % contra 50 % sem NM, ADS-B com táxi curto nos dois
(550 e 664 s — o sensor vê o avião parado no pátio nos dois casos). Não existe eixo que
separe os dois grupos entre as features disponíveis.

Nos 10 meses de treino, em LIRF sem NM, a taxa real de cópia por faixa de atraso:

| MVT − SCHED | n | p(cópia) | p(y > 1 h) | y médio |
|---|---|---|---|---|
| 1–1,5 h | 216 | 0,32 | 0,35 | 2.375 |
| 1,5–2 h | 338 | 0,49 | 0,49 | 3.671 |
| 2–3 h | 286 | 0,69 | 0,70 | 6.503 |
| 3–6 h | 118 | 0,70 | 0,72 | 11.006 |
| > 6 h | 42 | 0,83 | 0,98 | 47.678 |

É uma moeda viciada, não um sinal: com p ≈ 0,5 a previsão ótima em RMSE é a esperança
`p·ms + (1−p)·táxi normal`, que é exatamente o que o `combine` do `two_stage_nm` faz. E, de
fato, **toda tentativa de desfazer ou calibrar a aposta piora**:

| Variante (LIRF, fora da amostra) | RMSE LIRF | ganho erro² (M) |
|---|---|---|
| v18 (referência) | 568,6 | 0 |
| sem NM ← média da faixa aprendida nos 10 meses | 904,4 | −13.120 |
| sem NM ← 50 % v18 + 50 % média da faixa | 679,4 | −3.669 |
| sem NM ← teto de 2 h | 1.482,9 | −49.759 |
| teto de 2 h em todo o LIRF | 1.561,9 | −56.142 |
| recalibração linear `y ≈ a + b·pred` (jan↔jul) nos 387 apostados | 3.581 (no grupo) | −130 |
| encolher a aposta: `pred × 0,9` | 3.892 (no grupo) | −1.030 |
| `pred × 1,1` | 3.743 (no grupo) | −590 |

A regressão fora de fold dá **b = 1,03 (treinando em julho) e 1,10 (em janeiro)**: se
alguma coisa, a aposta está *baixa*, não alta. O λ ótimo é 1,0.

Um classificador dedicado de cópia (LightGBM nos 10 meses, com companhia do `FLIGHT_mvt`,
pista, prefixo de stand, atraso, fila, ADS-B) chega a **AUC 0,861** nas linhas de LIRF sem
NM do holdout — e mesmo assim não vira ganho: trocar `pred` por `p·ms + (1−p)·táxi normal`
custa −8.324 M; usar o `p` só para limitar quando `p` < 0,3 rende +38 M (0,7 % do erro²
daquelas linhas, 163 voos) e vira −6.736 M com o limiar em 0,4. É ruído de limiar.

**Conclusão da seção:** a aposta da cópia está calibrada. Os 114 falsos alarmes são o preço
dos 273 acertos, e o preço é menor que o prêmio.

## 4. Julho, pista 25, `pred` ≤ 1 h — variância, não viés

13.402 voos, RMSE 333, viés +29 s → o viés explica **0,8 %** do erro² do grupo.

| corte | resultado |
|---|---|
| por hora (blocos de 2 h) | viés entre +7 e +108; RMSE 232–496, sem padrão de configuração |
| por dia | pior dia 04/07 (RMSE 420, viés +92); os 31 dias variam 300–420 |
| por prefixo de stand | RMSE 269–484, viés +5 a +70 |
| por segmento | Mainline 302, Lowcost 333 |
| sem NM dentro do grupo | 136 voos, viés +544, RMSE 1.019 (10 % do erro² do grupo) |
| dias com mistura de pistas | dia de pista única RMSE 262; dia com < 60 % na pista dominante RMSE 436 |

Correlação do resíduo com as features de fila e contexto: o máximo é `MVT − SCHED` com
**0,117**; todas as `apt_dep_*`, `rwy_dep_*` e `ctx_*` ficam abaixo de 0,04. Não existe fila
não capturada esperando para ser codificada.

Dois testes de teto:

1. **Corretor só de LIRF, 5 folds por dia dentro do próprio holdout** (otimista: vê dias
   vizinhos do mesmo mês, com todas as ~90 colunas, stand exato e companhia):

| recorte | RMSE v18 | RMSE com o corretor LIRF |
|---|---|---|
| normais `pred` ≤ 1 h | 311 | **334** (piora) |
| julho pista 25 `pred` ≤ 1 h | 333 | **365** (piora) |
| normais sem NM | 3.000 | 2.443 (melhora) |
| cauda y > 1 h | 2.910 | 3.130 (piora) |

   Mesmo com informação que não teríamos em 2026, não há o que extrair dos normais: o
   modelo específico só faz overfit. O único lugar onde ele melhora é o grupo sem NM — e
   lá o ganho vem de ver janeiro e julho ao mesmo tempo, que em produção não existe.

2. **Piso de ruído por vizinho**: para cada voo, o voo mais próximo no tempo (≤ 30 min) com
   mesma pista e mesmo prefixo de stand. `rms(Δy)/√2` estima o que nenhum modelo pode
   acertar nos dois:

| recorte | pares | piso estimado | RMSE v18 | RMSE / piso |
|---|---|---|---|---|
| LIRF jul pista 25 | 13.392 | 393 | 333 | 0,85 |
| LIRF jan pista 25 | 8.878 | 298 | 256 | 0,86 |
| LFPG normais | 39.175 | 292 | 231 | 0,79 |
| EHAM normais | 40.723 | 231 | 109 | **0,47** |

   O estimador é grosseiro (não fixa companhia nem tipo), mas a comparação entre aeroportos
   é clara: em EHAM o modelo fica a menos da metade da variação entre voos vizinhos; em Roma
   ele já está em 0,85 dela. **[inferência]** o que sobra em Roma é ruído de registro
   (quando o BLOCK foi anotado), não física de aeroporto.

## 5. Testes baratos no corretor (padrão `.superpowers/noite/noite.py`)

Corretor OOF de 5 folds por dia sobre a base `20260927-211418-reg_corte` (mesma base da
v18). O valor absoluto não é o da v18 (aqui não há conjunto, externos nem plano 13); o que
vale é a diferença para a referência. Três sementes por variante nas duas linhas principais.

Variantes (o que entra no quadro do corretor, além do que ele já tem):
`fila` = contagens `apt_dep/arr_*`, `rwy_dep_*` + categóricas `aeroporto|pista` e
`aeroporto|prefixo de stand`; `stand_p` = só `aeroporto|prefixo de stand`; `stand_full` =
`aeroporto|stand` exato; `sinais_copia` = `MVT − SCHED`, `pred − ms`, razão `pred/ms`,
bandeira `pred > 1 h`, atraso só nas linhas sem NM, bandeira LIRF sem NM;
`iso_aero`/`linear_aero` = pós-correção isotônica/linear por aeroporto, fora de fold, nos
voos com `pred` ≤ 1 h.

| teste | completo | sem loteria | normais NM | LIRF | LIRF normais | LTFM | LFPG | ganho completo | ganho sem loteria |
|---|---|---|---|---|---|---|---|---|---|
| ref | 309,96 | 248,63 | 201,24 | 588,3 | 412,7 | 259,1 | 565,3 | −0,12 | −0,25 |
| ref semente 1 | 309,96 | 248,40 | 201,14 | 587,1 | 410,3 | 259,0 | 565,9 | −0,12 | −0,02 |
| ref semente 2 | 309,61 | 248,11 | 201,17 | 586,3 | 409,1 | 258,9 | 565,1 | +0,23 | +0,27 |
| **fila** | **308,41** | **246,60** | 201,07 | 577,3 | 410,7 | 256,2 | 568,9 | **+1,43** | **+1,78** |
| fila semente 1 | 308,51 | 246,60 | 201,13 | 576,4 | 410,2 | 256,4 | 569,3 | +1,33 | +1,78 |
| fila semente 2 | 308,34 | 246,39 | 201,00 | 577,0 | 409,7 | 256,1 | 569,0 | +1,50 | +1,99 |
| stand_p | 308,51 | 247,06 | 201,40 | 577,2 | 408,8 | 258,9 | 568,1 | +1,33 | +1,32 |
| stand_p semente 1 | 308,65 | 247,03 | 201,36 | 577,5 | 408,2 | 258,9 | 568,3 | +1,19 | +1,35 |
| stand_full | 313,40 | 251,89 | 204,71 | 594,1 | 427,7 | 262,0 | 573,4 | −3,56 | −3,51 |
| stand_full semente 1 | 313,69 | 252,16 | 205,60 | 593,8 | 426,8 | 261,7 | 575,2 | −3,85 | −3,78 |
| apt_rwy | 310,56 | 248,48 | 201,43 | 586,9 | 413,0 | 258,4 | 570,0 | −0,72 | −0,10 |
| fila_contagens | 309,95 | 248,25 | 200,94 | 587,6 | 410,7 | 258,1 | 565,9 | −0,11 | +0,13 |
| fila_cats | 308,63 | 246,92 | 201,24 | 578,4 | 410,5 | 257,1 | 568,8 | +1,21 | +1,46 |
| fila_sem_stand | 310,26 | 247,84 | 201,06 | 586,4 | 412,5 | 257,2 | 569,7 | −0,42 | +0,54 |
| sinais_copia | 309,12 | 247,27 | 200,62 | 582,2 | 410,2 | 258,3 | 566,3 | +0,72 | +1,11 |
| fila+sinais | 308,77 | 246,58 | 200,49 | 582,0 | 408,9 | 256,3 | 567,1 | +1,07 | +1,80 |
| ref+iso_aero | 311,48 | 250,54 | 201,35 | 594,4 | 408,5 | 260,0 | 565,9 | −1,64 | −2,16 |
| ref+linear_aero | 310,61 | 249,38 | 201,33 | 591,3 | 411,0 | 259,3 | 565,7 | −0,77 | −1,00 |
| fila+iso | 310,20 | 248,82 | 201,23 | 585,0 | 405,8 | 257,0 | 569,5 | −0,36 | −0,44 |
| fila+sinais+iso | 310,67 | 248,97 | 200,74 | 590,6 | 404,6 | 256,9 | 567,7 | −0,83 | −0,59 |

Ganho = referência (média das três sementes: completo 309,84, sem loteria 248,38) menos a
variante. O espalhamento entre sementes da referência é 0,35 s no completo e 0,52 s sem
loteria: só diferenças acima disso contam.

Leitura:

- **O único ganho reproduzível é o `prefixo de stand` como categórica do corretor.**
  `stand_p` sozinho dá +1,33 e +1,19 (duas sementes); o pacote `fila` dá +1,43/+1,33/+1,50
  no completo e +1,78/+1,78/+1,99 sem loteria — as contagens e `aeroporto|pista` só ajudam
  **junto** com o prefixo (sozinhas: contagens +0,13, `apt_rwy` −0,10, e as duas juntas sem
  o prefixo, `fila_sem_stand`, +0,54 sem loteria mas −0,42 no completo).
- **Stand exato é veneno:** `stand_full` perde 3,6 s (LIRF normais 412,7 → 427,7). São 134
  códigos em Roma e 377 em LFPG; o corretor decora o stand e leva o erro junto. A base já
  usa `STAND_mvt` como categórica — repetir isso no corretor é que quebra.
- **E nada disso é de Roma:** o `LIRF normais` fica em 410,2 (contra 410,7 da referência);
  quem melhora é o LTFM (259,1 → 256,2) e a cauda de Roma (LIRF 587,2 → 576,9, que é o
  `nm_ausente` 1.791 → 1.775).
- **`sinais_copia`** (atraso cru, `pred − ms`, razão, bandeiras) dá +0,72 no completo e
  +1,11 sem loteria, abaixo do ruído entre sementes e sem efeito nos normais de Roma: o
  `pred` já carrega essa informação.
- **Pós-correção isotônica ou linear por aeroporto piora** (−1,64 e −0,77) mesmo baixando
  `LIRF normais` para 408,5: ela ganha centavos nos normais e paga caro na cauda. Fora.

## 6. Recomendação

1. **Não há conserto barato dentro dos normais de Roma.** As três hipóteses da tarefa
   morreram com medida: a aposta da cópia já está calibrada (λ ótimo = 1,0; regressão fora
   de fold com b = 1,03 e 1,10), o erro de julho na pista 25 é variância e não viés
   (viés² = 0,8 % do erro² do grupo, correlação máxima do resíduo 0,117), e o corretor por
   aeroporto — isotônico, linear ou LightGBM com todas as colunas — **piora** os normais.
2. **O que vale colocar em código é genérico:** somar ao `corrector_frame` (`src/stack.py`)
   a categórica `aeroporto|prefixo de stand` (o primeiro caractere de `STAND_mvt`), e junto
   com ela as contagens `apt_*`/`rwy_*` e `aeroporto|pista`, que só valem acompanhadas.
   Ganho na simulação barata, média de três sementes: **−1,4 s no completo e −1,85 s sem
   loteria** (LIRF 587,2 → 576,9, LTFM 259,1 → 256,2, LFPG piora 565,4 → 569,1).
   Expectativa na simulação completa (crossfit + conjunto + externos + plano 13):
   **[inferência]** menos que isso, porque o conjunto já tem um corretor por aeroporto que
   captura parte do mesmo efeito — projetar −0,5 a −1,5 s e medir antes de promover.
   - **Nunca o stand exato** (−3,6 s) e nunca a pista sozinha (−0,7 s).
   - Risco de 2026 conferido: 99,7 % dos voos de LIRF em 2026 usam stands já vistos em 2025
     (LFPG 99,9 %, EGLL 100 %, LTFM 99,5 %). O pátio 50x de Roma saiu de uso (7,6 % → 0,1 %)
     e o tráfego foi para o 80x (6,6 % → 10,3 %), mas os códigos não foram renumerados: a
     categórica continua significando a mesma coisa.
3. **Onde ainda há dinheiro em Roma (mas não barato):** as 249 linhas normais sem NM
   (2.241 M = 7,0 % do erro² total) e as 460 da cauda (3.894 M = 12,2 %) são o mesmo
   fenômeno — decidir se o BLOCK é cópia do SCHED. Como o teto realista é pequeno (zerar
   **todo** o erro dos normais de Roma vale +23,3 s; cortar 25 % dele vale +5,6 s) e o
   classificador dedicado já chega a AUC 0,861 sem virar ganho, a alavanca aqui é dado
   externo que diga quando o pátio de Roma realmente parou (fluxo real de partidas), não
   mais uma feature derivada do mesmo registro.

### Tetos de ganho (RMSE total ao zerar um pedaço)

| pedaço | erro² (M) | RMSE se zerado | ganho (s) | ganho se cair 25 % |
|---|---|---|---|---|
| normais de LIRF inteiros | 4.683 | 280,5 | +23,3 | +5,6 |
| cauda de LIRF (y > 1 h) | 3.894 | 284,6 | +19,2 | +4,7 |
| normais sem NM de LIRF | 2.241 | 292,9 | +10,9 | +2,7 |
| os 114 apostados | 2.166 | 293,3 | +10,5 | +2,6 |
| julho pista 25, `pred` ≤ 1 h | 1.488 | 296,6 | +7,2 | +1,8 |
