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
