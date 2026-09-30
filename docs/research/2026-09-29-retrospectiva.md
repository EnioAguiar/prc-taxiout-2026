# Retrospectiva — 29/09/2026 (campeã v30 = 247,76)

Pergunta: o que deixamos passar? Quatro leituras (scouts) de todo `docs/`, README, `saltos.json`
e laços de 29/09, mais os números de `submissions.jsonl`. Relatórios completos nas transcrições
`RetroPesquisaInicial`, `RetroPesquisaRecente`, `RetroLacos`, `RetroPlanos`.

## O que os envios mostram

| Envio | Δ simulação | Δ oficial |
|---|---|---|
| v9 janela do LOBT | −6,3 | −38,9 |
| v11 taxi-in e vizinhos | −9,2 | −17,4 |
| v17 dados externos | −5,2 | −9,4 |
| v19 · v21 · v27 · v28 | −2,8 · −2,1 · −1,1 · −1,2 | −2,9 · −1,6 · −1,2 · −1,9 |
| v29 · v30 | −0,3 · −1,1 | −1,3 · −0,07 |

Os saltos vieram de descobertas no dado; os últimos seis envios são colunas no corretor e estão
no piso de ruído da simulação (~1 s). 29/09: ~8 h de máquina para −3,3 s.

## Falha de método encontrada

Três réguas diferentes decidiram os candidatos: o laço barato (sem_loteria + guarda jan↔jul), os
laços fiéis 1 e 2 (só `completo` e sem top 10) e o `compare.py` (≥ 10 s). Candidatos aceitos no
barato morreram no fiel pelo `completo`, que a v29 mostrou ser a régua errada (sem_loteria +1,2 →
oficial −1,33). A guarda jan→jul dá +8 a +11 para qualquer peso ou pós-processamento: é dominada
pelas loterias e não serve de aval.

## Deixados para trás, por custo

### Baratos (laço fiel sobre a v30, ~4–10 min cada com `--reusar-oof`)

| Item | Evidência | Fonte |
|---|---|---|
| `peso_loteria_0.3` no corretor (nunca foi ao fiel; `stack.py` não tem peso de amostra) | barato: sem_loteria +1,70, combo +1,93 | laco/0-RESUMO.md:23-25,96-98 |
| `superficie` | fiel 2: completo −0,24, normais +0,81 | laco-fiel-2/0-RESUMO.md:22-23 |
| `fila` + `rounds_500` (combinação nunca feita) | fila sozinha: completo +0,54, sem_loteria +0,87 | laco-fiel-2/0-RESUMO.md:17-20 |
| `p_copia`, `peso_y3h_07` | fiel 2: +0,36 / sem_loteria +0,43 | laco-fiel-2/0-RESUMO.md:21 |
| parâmetros do corretor (lr 0,03, 127 folhas, l2 50) | busca: sem_loteria +1,79 | noite/6-busca-parametros-corretor.md:7-9 |
| `pos_shrink_global` | a guarda jan↔jul mais forte dos 61 (+11,3/+2,8) | laco/0-RESUMO.md:39 |
| teto da previsão por aeroporto fora de LIRF (kind-mango v48) | −2,33 oficial deles | concorrentes.md:81 |
| isotônica global do classificador (kind-mango v64) | −1,14 oficial deles; nós só testamos a célula de LIRF | concorrentes.md:82 |

### Médios (mudam a base; ~1 h com a base crossfit)

| Item | Evidência | Fonte |
|---|---|---|
| regressor sem LIRF inteiro e sem y > 80.000 (kind-mango v51) | −5,99 oficial deles; nós só tiramos LIRF sem NM (+2,0 s na base, nunca levado ao corretor) | concorrentes.md:76; plano11 |
| `--base-mapa`, `--base-p13` | descartados só pelo `completo`, sem sem_loteria | mapa.md:93-97 |
| média de seeds na base | normais +1,3 s, descartado pelo `completo` | README (plano 6) |

### Grandes (cauda, onde está a distância para o topo)

| Item | Evidência | Fonte |
|---|---|---|
| mistura multiclasse (`copy_mix`) sem a classe fixa "24 h + táxi" (hoje coberta pela regra de Roma) | cortava alarmes falsos 10,0 → 8,7 % sem mexer nos normais; o −4,0 s veio só da classe 5 (+5,8 s) | README:314-344 |
| próximo ocupante do stand, separando o próprio avião por `icao24` (OPDI, 91 % de casamento) ou `reg` do ADS-B | oráculo 299,38 → 289,27 (−10 s); parou por falta de matrícula | cacador-de-regras-2.md:6-33 |
| folha `dist_lo ≤ 0` da árvore de erro | 558 voos = 42,4 % do erro², sem viés (erro de dispersão), companhias AAL, EJU, FIN, ITY, PGT, RYR | laco/0-arvore-de-erro.md:11-14 |
| registro × sensor: MVT − LOBT > 2 h com táxi ADS-B curto | 28 de 32 voos; GREKI (topo) diz usar | estudo-cauda.md:22-25 |
| loterias do LFPG sem NM (y 84.240 e 58.206 s): BLOCK nunca comparado com horários de outras linhas | 31 % do erro² da v12 | estudo-cauda.md:8; forense-dados.md §6 |

## Dúvidas de medição

- EGLL e LEMD têm ADS-B em 79 % e 67 % do ranking 2026 contra 14 % e 2 % no holdout
  (concorrentes.md:63-67): a simulação não tem a mistura do oficial, e o item 4e mirou dois
  aeroportos que em 2026 não são "sem ADS-B".
- `saltos.json` parado em 28/09 com itens já feitos como pendentes; a projeção sai dele.

## Mudança de método adotada

1. Uma régua só nos laços: `sem_loteria` com IC > 0 (normais_nm como segunda leitura);
   `completo` vira informação.
2. Teto antes de treinar: ideia nova só treina se a conta com dados prontos der ≥ 3 s, ou se o
   custo for de um laço fiel (~10 min).
3. Envio só muda o corretor → ~21 min (cache da base, 29/09).

## Resultado dos baratos (29/09 à noite, `.superpowers/noite2/`)

Laço fiel sobre a v30 (base, oof e receita da v30; ~10 min cada). A primeira linha é a própria
v30 rodada de novo: o CatBoost na GPU não repete o número, então **ganhos abaixo de ~0,3–0,5 s
são ruído**, mesmo com IC "positivo".

| Candidato | completo | sem top 10 | sem loteria | normais NM | Leitura |
|---|---|---|---|---|---|
| v30 repetida | −0,3 (IC −0,5 a −0,1) | −0,4 | −0,2 | 0,0 | ruído |
| `superficie` | −0,6 (IC −1,9 a 0,8) | −1,4 | +0,2 | **+0,9** (IC 0,6 a 1,2) | só normais |
| `fila` no lugar de `stand-prefixo` | −0,3 | −0,7 | +0,1 | +0,6 (IC 0,4 a 0,7) | só normais |
| `--peso-loteria 0.3` | −1,0 (IC −2,8 a 0,2) | −1,5 | 0,0 | 0,0 | descartado; flag removida |
| 700 rodadas | −0,2 | −0,3 | −0,1 | +0,2 | ruído |
| teto por aeroporto fora de LIRF (pós, sem treino) | 300,5 a 313 contra 298,05 | | 235,6 a 250,8 contra 233,3 | | descartado |

Nenhum candidato barato move o `sem_loteria` acima do ruído. `superficie` e `fila` ganham só nos
normais (+0,6 a +0,9) e perdem nos maiores voos. Confirma a retrospectiva: o corretor está
esgotado; o próximo passo é a base (regressor sem LIRF inteiro e sem y > 80.000) e a cauda.

### Regressor global sem LIRF (kind-mango v51) — descartado

`--reg-sem-lirf` (global sem LIRF; LIRF só com o regressor próprio do `--base-por-apt`). Base
`20260929-182302-base_ctx_por_apt_semlirf`: 308,75 → 309,08 (−0,3; sem loteria −0,4, IC −0,7 a
−0,0). Com o corretor da v30 (`20260929-183120-v31_sem_lirf_cf`): 298,05 → 299,05 (−1,0, IC −1,9
a −0,3; sem loteria −0,8, IC −1,6 a −0,1; julho −1,8). O −5,99 s deles não se repete aqui: o
regressor por aeroporto e o corte do alvo em 2 h já separam Roma [inferência]. Flag removida.

### Mistura multiclasse sem a classe "24 h + táxi" — descartada

`--copia-multi`: classificador de 5 classes (normal, SCHED, LOBT, EOBT_1, AOBT_3; treino 2025:
1.167.077 · 169.206 · 38.871 · 60.663 · 304.811), previsão `Σ p_k·(MVT − horário_k) + p_0·reg`,
massa de horário nulo ou fora da janela do LOBT de volta ao normal. Só a base, contra 308,75:

| Variante | completo | sem loteria | normais NM | alarmes falsos |
|---|---|---|---|---|
| regressor só na classe 0 (`..._multi`) | 309,41 (−0,7) | −0,9 | −1,9 (IC −2,3 a −1,6) | 7,4 → 6,7 % |
| regressor nas linhas de antes (`..._multi_r`) | 309,89 (−1,1) | −1,4 (IC −2,4 a −0,4) | −2,4 (IC −2,7 a −2,1) | 6,6 % |

O corte de alarmes falsos se repete, mas a mistura piora os normais: nos voos em que o AOBT_3
fica perto do BLOCK sem bater, `p_AOBT_3·(MVT − AOBT_3)` troca a previsão do regressor por um
valor exato e errado [inferência]. A segunda variante mostra que não é falta de linhas no
regressor. Código removido; o corretor não foi rodado por cima.

### Coortes da árvore de erro, revistas na v30 — sem regra nova

- A folha "558 voos, 42 % do erro²" (`dist_lo ≤ 1e-35`) não é a previsão presa no piso da
  janela: na v30 só 11 voos ficam no piso (0,0 % do erro²). O corte pega `dist_lo` nulo, isto é,
  voos sem LOBT (sem NM), onde estão as loterias e Roma sem NM: fatia já estudada.
- A folha "ADS-B vê táxi bem menor que a previsão" (`pred − adsb_taxi > 2218`, sem loteria):
  616 voos, 14,6 % do erro² sem loteria. 150 deles são LIRF sem NM (12,4 %: a moeda viciada do
  diagnóstico de Roma). Dos 616, 237 têm y = cópia de horário (198 do SCHED), 13 têm y ≈ táxi do
  ADS-B, e 366 não batem com nada: táxi ADS-B mediano 280 s contra y mediano 2.717 s, BLOCK com
  segundos quebrados (só 7 % no segundo 0). É off-block real seguido de ~40 min parado no stand
  antes de andar [inferência]; nenhum horário ou sinal do quadro separa esses voos.

### OPDI `icao24`: cobertura no holdout (passo 1 do "próximo ocupante do stand")

Casando cada DEP com a flight list do OPDI como `externos.opdi` faz (callsign ±600 s; sem
callsign, só aeroporto ±90 s), guardando o `icao24`:

| Recorte | voos | erro² sem loteria | casado por callsign | só por hora |
|---|---|---|---|---|
| todos | 344.419 | 100 % | 81 % | 10 % |
| sem NM | 5.366 | 24,1 % | 0 % | 90 % |
| y > 1 h (sem loteria) | 970 | 20,4 % | 76 % | 23 % |
| LIRF | 26.528 | 35,1 % | 95 % | 2 % |
| LIRF sem NM | 397 | 17,3 % | 0 % | 99 % |
| LTFM | 46.539 | 15,2 % | 51 % | 22 % |

O "0–3 %" do item 4e era dos eventos de solo do OPDI, não da flight list: a aeronave é
identificável em quase toda a cauda. Voos sem NM só casam por hora (±90 s); a taxa de erro
desse casamento não foi medida. Próximo passo: ligar cada DEP à chegada da mesma aeronave
(`icao24`) e ao próximo ocupante do stand, e refazer o oráculo de 299,38 → 289,27 só com o que
é observável no ranking.

### Próximo ocupante do stand pelo `icao24` — descartado

Script descartável `/tmp/ocupante.py` (v30, holdout). DEP casadas com o OPDI: 91 %; pousos
(ARR, por `ades` + `last_seen` ≈ pouso): 69 %. Para cada DEP, o último pouso no mesmo stand
nas 24 h antes do MVT; se o `icao24` difere, seria o próximo ocupante (piso y ≥ MVT − in-block).

| Último pouso no stand | voos | viola a regra |
|---|---|---|
| outro `icao24` | 81.522 | 50.699 (y < piso) |
| o próprio `icao24` | 214.508 | 96 (y > teto) |

Violação do piso por distância do pouso ao MVT: ≤ 5 min 0 %, 5–15 min 1 %, 15–30 min 5 %,
30–60 min 73 %, > 1 h 99–100 %. Só pousos recentes identificam o próximo ocupante, e aí o piso
quase não prende: com gap ≤ 10/15/20/30 min ele mexe em 36/104/186/372 voos e dá 298,06 /
298,11 / 298,17 / 298,80 contra 298,05. O teto do próprio avião mexe em 70 voos e piora (386).
O oráculo de −10 s vinha dos gaps longos, onde "outro `icao24`" não é o próximo ocupante
(reboque, stand compartilhado ou casamento errado) [inferência]. Frente fechada.

### Simulação repesada para a mistura de 2026 — não explica o oficial melhor

Cobertura ADS-B por aeroporto, holdout → ranking: EGLL 0,14 → 0,79; LEMD 0,02 → 0,67; EDDM
0,62 → 0,96; LIRF 0,45 → 0,24; EDDF 0,73 → 0,56; LTFM 0 → 0. Peso de cada voo do holdout =
fração da célula (aeroporto × ADS-B sim/não × NM sim/não) no ranking ÷ no holdout.

| Δ para a versão anterior | completo | repesado | sem loteria | repesado sem loteria | oficial |
|---|---|---|---|---|---|
| v28 | −1,23 | −1,84 | −1,25 | −1,54 | −1,94 |
| v29 | −0,31 | +0,61 | −1,22 | −0,96 | −1,33 |
| v30 | −1,06 | −1,43 | −0,68 | −0,67 | −0,07 |

Nenhuma régua acerta a v30; `sem_loteria` segue a melhor nas três, e repesar não ajuda. Com
três pontos, o ruído do próprio oficial (~0,5–1 s por envio) ainda não se separa [inferência].

### Média de corretores sobre a mesma base (29/09 à noite)

Previsões de corretores da mesma base (v30), sem treinar nada de novo, média simples, bootstrap
pareado contra a v30. Corretores com colunas diferentes somam; repetições da mesma receita
(`ref`, 700 rodadas, v29 sem mapa) diluem.

| Média | completo | sem loteria | sem top 10 |
|---|---|---|---|
| v30 + `fila` + `superficie` (v31, arquivo pronto, não enviado) | +0,23 | +0,75 (IC 0,12 a 1,38) | −0,11 |
| + v29 | +0,03 | +0,55 | |
| v30 + `e2_fila_sup` (`--fila --superficie` numa corrida) | **+0,38** | **+0,95** (IC baixo +0,21) | −0,02 |

`e2_fila_sup` sozinho: 298,42, sem loteria 232,82, normais 189,96. `--corretor-xgb` quebra com
categoria nova no holdout (`stand_p` = `EDDM|C`). A melhor de 98 combinações tem viés de
seleção; é também a mais simples (2 membros).

### Força bruta sobre o resíduo da v30 (29/09, 23h) — nada sobra nas colunas

Scripts em `.superpowers/noite2/`. Nenhum treina o modelo principal.

1. **Janelas por grupo** (`cacador3.py`): BLOCK − X para 8 horários × 8 recortes (aeroporto,
   NM, companhia, pista, tipo de voo e cruzamentos) × cortes 0 / 1e-4 / 1e-3, aprendidas em 10
   meses e aplicadas à v30 no holdout. Melhor: +0,007 % do erro² sem loteria (LOBT global). Só a
   janela do LOBT existe.
2. **Deslocamentos exatos** (`cacador4.py`): pico de BLOCK = X + c (passos de 1 min) por grupo.
   Os picos fortes são todos c = 0 em LIRF (a cópia do SCHED, já tratada); os 43 com c ≠ 0 somam
   < 0,1 % do erro² sem loteria.
3. **Detetive do resíduo** (`detetive2.py`): LightGBM no y − pred da v30 com todas as colunas do
   holdout, 5 dobras por dia dentro de jan+jul (otimista). Com `FLIGHT`/`CALLSIGN`: 233,31 →
   235,63 sem loteria (overfit); sem eles: 233,31 → 234,07, normais 191,31 → 193,07. Nem o teto
   otimista reduz o erro.

Conclusão: o erro que sobra não é explicável pelas colunas que temos. Ganho novo exige
informação nova (fonte externa) ou a cauda/loterias, não mais busca no mesmo quadro.

### Risco da regra do ADS-B (30/09, pergunta do arnavhm13 no Discord)

Campeã v32 refeita sem o rastro do próprio voo (as 9 colunas `adsb_*` fora da base e dos dois
corretores; `superficie` e contexto ficam, porque vêm de outros aviões): base
`base_sem_adsb_proprio`, corretores `20260930-120329-sem_adsb_m0` e `20260930-124715-sem_adsb_m1`.
Simulação 297,49 → **306,36 (−8,9 s)**; sem loteria, metade A −10,9 s (IC −12,8 a −8,9).
Em 2026 o ADS-B cobre mais voos (EGLL 79 %, LEMD 67 %), então a perda no oficial tende a ser
maior que na simulação [inferência]. Se a regra proibir, a versão sem rastro próprio já está
treinada (os dois corretores acima). Esteira: candidato 75 (`--base-mapa`) reprovado (−0,12).
