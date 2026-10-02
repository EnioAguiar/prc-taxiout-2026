# 2026-10-02 — Descoberta bruta: ablações do corretor, deriva 2025→2026 e fatias do resíduo

A esteira tinha esgotado o gerador de vizinhos (162 candidatos, 5 aprovados) e estava
ociosa. Este documento registra o que foi colocado na fila e o que as três varreduras
brutas acharam. Tudo medido contra a campeã de 02/10 (`20261001-185827-e122` +
`20260930-140011-e76_m1` + `20261002-015703-e140`, completo 295,63 · sem loteria 228,69).

## 0. Ferramenta nova: `--corretor-sem-feature`

O `--sem-feature` que já existia tira a coluna **da base e do corretor** e grava
`sem_features` dentro da `base_config`. Isso invalida o `--reusar-oof` (a base da corrida
deixa de ser a mesma), ou seja: cada ablação viraria uma base nova de ~75 min, e a medida
misturaria "o corretor perdeu a coluna" com "a base foi refeita".

Entrou `--corretor-sem-feature COLUNA` (pode repetir, só com `--crossfit`): tira a coluna
**só das entradas do corretor**, grava `corretor_sem_features` no topo da config e deixa a
`base_config` intacta. Com isso a ablação reaproveita o oof da campeã e custa ~11 min.
Um nome que o corretor não tem **erra alto** (`SystemExit`) em vez de medir a própria
campeã e ser lido como "tirar essa coluna não muda nada".

Também foi preciso ensinar o `argv_corretor` da esteira a repetir uma flag quando o valor
da receita é lista (`{"--corretor-sem-feature": ["met_temp", "met_vis"]}`), e o
`train.py` a somar `corretor_sem_features` ao `sem` do corretor no envio — senão uma
ablação promovida não se reproduziria no arquivo final.

Commit `8acd346`; testes em `tests/test_stack.py` e `tests/test_esteira.py`.

## 1. Fila enfileirada (candidatos 163–196)

Todas as receitas partem do membro `20261002-015703-e140`:

```
{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500,
 "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true,
 "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}
```

### 1a. Ablações por grupo de colunas (`--corretor-sem-feature`)

Grupos escolhidos por *sub-bloco*, não por flag: desligar `--mapa`, `--pista`,
`--retencao`, `--dist-plano` ou `--corretor-ref` inteiros é o que o gerador de vizinhos já
fazia (e reprovou). O que ninguém tinha medido é o pedaço de dentro de cada bloco.

| id | grupo | colunas |
|---|---|---|
| 163 | `met` | 12 colunas METAR (`met_temp` … `met_degelo`) |
| 164 | `ctx_viz` | 10 colunas `ctx_viz_*` |
| 165 | `nm_p13` | 8 colunas `nm_*` do plano 13 |
| 166 | `pista_cfg` | `pista_cfg`, `pista_cfg_mudou/dep/arr`, `pista_dominante`, `pista_parte_rwy` |
| 167 | `ctx_arr` | 8 colunas `ctx_arr_*` |
| 168 | `dist_ref` | `dist_sched/lobt/iobt/eobt/aobt` |
| 169 | `adsb_qual` | `adsb_gs0`, `adsb_takeoff_err`, `adsb_n_chao`, `adsb_gap_max`, `adsb_frac_mlat` |
| 170 | `ext_diarias` | `ext_reg_frac`, `ext_late_frac`, `ext_reg_n`, `ext_pre_min`, `ext_atc_min` |
| 171 | `cel_disp` | `cel_p90`, `cel_dp`, `cel_n`, `cel_nivel` |
| 172 | `pista_fila` | `pista_fila_peso`, `pista_fila_pesadas`, `pista_span20`, `pista_dep_h30` |
| 173 | `ret_fluxo` | `ret_ativos_rwy`, `ret_atraso_15m`, `ret_ewma_dep` |
| 174 | `mapa` | `map_dist`, `map_dmin`, `map_extra` |
| 175 | `rot` | `rot_mesma_cia`, `rot_mesmo_tipo`, `rot_idade` |
| 176 | `dist_agg` | `dist_min`, `n_planos`, `n_planos_longos` |
| 177 | `cel_p50` | `cel_p50`, `cel_pred_menos_p50` |
| 178 | `ret_over` | `ret_overdue_rwy`, `ret_overdue_apt` |
| 179 | `ext_cia` | `ext_taxa_cia`, `ext_taxa_cia_ms` |
| 180 | `ext_opdi` | `ext_solo_s`, `ext_mesmo_apt` |
| 181 | `ctx_stand` | `ctx_stand_ult_idade`, `ctx_stand_ult_tin` |
| 182 | `adsb_geo` | `adsb_lat0`, `adsb_lon0` |
| 183 | `janela` | `dist_lo`, `dist_hi` |
| 184 | `cia` | `cia` (categórica do plano 13) |
| 194 | `adsb_taxi` | `adsb_taxi` (mantém `adsb_taxi_move`) |
| 195 | `adsb_move` | `adsb_taxi_move`, `adsb_menos_pred` (mantém `adsb_taxi`) |
| 196 | `to_takeoff_iobt` | `to_takeoff_from_IOBT_flt` |

### 1b. Ablações dirigidas pela deriva (seção 2)

| id | grupo | motivo |
|---|---|---|
| 189 | `ctx_60` | as 8 colunas `ctx_*_60`: a janela de 60 min é a que mais separa os anos |
| 190 | `ctx_rwy` | as 8 colunas `ctx_*_rwy_*`: deriva + 83–89 % de ausentes |
| 191 | `adsb_cob` | `adsb_n_chao`, `adsb_gap_max`, `adsb_frac_mlat`: medem a **cobertura do receptor**, não o voo |
| 192 | `adsb_n_chao` | sozinha; 7,9 % do ganho do classificador adversário |
| 193 | `adsb_top4` | `adsb_lat0/lon0/n_chao/gs0`: as quatro de maior deriva juntas |

### 1c. Diversidade para a média (membros novos)

| id | receita extra |
|---|---|
| 185 | `--corretor-xgb` (quarto modelo do conjunto, XGBoost na GPU) |
| 186 | `--corretor-params {"feature_fraction": 0.7, "seed": 7}` |
| 187 | `--corretor-params {"bagging_fraction": 0.8, "bagging_freq": 1, "seed": 11}` |
| 188 | `--corretor-params {"seed": 3, "bagging_seed": 3, "feature_fraction_seed": 3}` |

Total: 34 candidatos, ~11 min cada ≈ 6 h de esteira.

## 2. Validação adversária 2025 → 2026

`.superpowers/bruta/adversarial.py`: LightGBM separando as 344.419 partidas do holdout
2025 (jan+jul) das 344.841 do ranking 2026 (jan+jul — os mesmos meses, então não é
sazonalidade), com as colunas que a base e o corretor compartilham (`ctx_*`, `adsb_*`,
`to_takeoff_*`, `gap_*`, contagens `apt_*`/`rwy_*`, `round_*`, `hour`, `dow`, `AIRPORT`).

**AUC = 0,880.** Os anos são facilmente distinguíveis, e quase tudo vem do ADS-B:

| coluna | % do ganho | AUC univariada |
|---|---|---|
| `adsb_lon0` | 22,7 % | 0,365 |
| `AIRPORT` | 13,2 % | — |
| `adsb_lat0` | 8,9 % | 0,454 |
| `adsb_n_chao` | 7,9 % | 0,531 |
| `ctx_arr_tin_apt_60` | 5,2 % | 0,475 |
| `hour` | 4,5 % | — |
| `ctx_viz_fut_apt_60` | 3,4 % | 0,524 |
| `ctx_viz_pas_apt_60` | 3,1 % | 0,523 |

(`adsb_taxi_move` 0,567 e `adsb_taxi` 0,561 na univariada; as `ctx_*_rwy_*` têm 83–89 %
de ausentes e ainda assim derivam.)

### Cobertura ADS-B por aeroporto (fração de voos com rastro)

| apt | jan/25 | jul/25 | jan/26 | jul/26 |
|---|---|---|---|---|
| EDDF | 70,7 | 74,8 | 50,9 | 59,0 |
| EDDM | 96,7 | 36,5 | 95,4 | 96,8 |
| **EGLL** | **20,9** | **7,5** | **59,7** | **97,7** |
| EHAM | 97,3 | 97,5 | 95,9 | 97,5 |
| LEBL | 96,5 | 79,3 | 86,4 | 91,0 |
| **LEMD** | **0,0** | **3,2** | **94,8** | **43,5** |
| LFPG | 0,0 | 13,5 | 10,7 | 11,6 |
| **LIRF** | **0,0** | **78,1** | **57,7** | **0,1** |
| LSZH | 74,6 | 70,9 | 83,8 | 86,7 |
| LTFM | 0,0 | 0,0 | 0,0 | 0,0 |

O mix de voos por aeroporto é estável (≤ 0,8 pp de diferença entre os anos): a deriva é
**só de cobertura**, não de composição.

### A distribuição do rastro muda junto com a cobertura

Mediana de `adsb_taxi_move` onde há rastro:

| apt | jan/25 | jul/25 | jan/26 | jul/26 |
|---|---|---|---|---|
| EGLL | 386 | 252 | 741 | **1086** |
| EDDF | 416 | 478 | 374 | 301 |
| LIRF | — | 376 | 233 | 267 |

E a mediana de `adsb_n_chao` (pontos no solo) em EGLL vai de 12/6 para 38/92.

Erro do próprio rastro contra o alvo, no holdout 2025 e só onde há rastro
(`.superpowers/bruta/adsb_qualidade.py`):

| apt/mês | n | |erro| mediano do rastro | viés do rastro | |erro| mediano do modelo |
|---|---|---|---|---|
| EHAM jul | 21.931 | 21,5 | +5,8 | 17,9 |
| EDDM jan | 10.890 | 46,5 | −44,5 | 22,6 |
| LSZH jan | 7.258 | 192,6 | −136,2 | 27,9 |
| LEBL jan | 11.801 | 363,3 | −362,3 | 39,0 |
| EDDF jan | 10.952 | 383,3 | −382,9 | 43,7 |
| LIRF jul | 11.876 | 639,4 | −638,4 | 140,7 |
| LFPG jul | 2.921 | 755,9 | −755,9 | 113,5 |
| **EGLL jul** | **1.552** | **1076,2** | **−1075,6** | **105,5** |

Leitura: o rastro quase nunca é o táxi — ele tem um viés negativo grande e **específico do
aeroporto** (o movimento é detectado tarde). O modelo aprende esse deslocamento por
aeroporto. O risco é que em EGLL o deslocamento foi aprendido numa amostra de 7,5–20,9 %
dos voos e em 2026 ele vale para 97,7 % deles — população diferente. Daí os candidatos
191–193 (tirar do corretor o que mede a cobertura do receptor em vez do voo).

## 3. Mineração de fatias no resíduo da campeã

`.superpowers/bruta/fatias.py`: resíduo `y − previsão` da campeã no holdout (RMSE completo
295,63), 6.583 fatias de 1 ou 2 colunas (aeroporto, hora, bloco de hora, companhia, tipo,
pista, prefixo do stand, stand completo, segmento, esteira, NM presente, ADS-B presente,
dia da semana, faixa de `MVT − AOBT_3`), suporte mínimo de 120 voos **em cada mês**.
O ganho é sempre **fora do mês**: o viés de janeiro corrige julho e vice-versa.

### As fatias com mais viés consistente

| fatia | n jan | n jul | viés jan | viés jul | ganho fora do mês |
|---|---|---|---|---|---|
| sem NM × 14–17 h | 308 | 1.283 | +110,8 | +89,1 | +0,06 s |
| A320 × sem NM | 194 | 510 | +247,5 | +84,0 | +0,04 s |
| LEMD × stand `T` | 4.939 | 5.693 | −33,2 | −23,5 | +0,04 s |
| LEMD (todo) | 16.510 | 18.818 | −14,5 | −14,3 | +0,04 s |
| LEMD × pushback 20–40 min | 2.367 | 2.437 | −34,7 | −40,8 | +0,03 s |
| LFPG × 15 h | 926 | 1.226 | +40,9 | +43,1 | +0,02 s |

O viés existe e é estável no sinal — mas é pequeno demais perto de um RMSE de 295 s.

### Corrigir a partição inteira **piora**

Aplicando o viés estimado no outro mês a *todos* os níveis de uma chave (encolhimento
`n/(n+200)`, suporte ≥ 120):

| chave | níveis | ganho completo |
|---|---|---|
| AIRPORT | 10 | **−0,05 s** |
| AIRPORT × bloco de hora | 60 | **−0,18 s** |
| AIRPORT × hora | 219 | **−0,29 s** |
| AIRPORT × prefixo do stand | 84 | **−0,09 s** |
| AIRPORT × companhia | 1.200 | **−0,07 s** |
| AIRPORT × NM | 20 | **−0,28 s** |
| stand completo | 2.158 | **−0,08 s** |

Com suporte ≥ 300 e encolhimento maior continua negativo; com seleção por limiar
(|viés| ≥ 30/60/120 s no mês que estima) também (de −0,87 s a 0,00 s).

### O teto é baixo mesmo com oráculo

Tirando o viés **do próprio mês** (impossível, é olhar a resposta):

| chave | oráculo |
|---|---|
| AIRPORT × hora | +0,60 s |
| AIRPORT × NM | +0,54 s |
| stand completo | +0,50 s |
| AIRPORT × tipo | +0,33 s |
| AIRPORT | +0,11 s |

**Conclusão:** nenhuma correção de viés por fatia chega aos 0,2 s pedidos — o melhor
oráculo dá 0,60 s e toda estimativa honesta (fora do mês) é negativa. Nenhuma pós-regra
foi implementada. Isso bate com a retrospectiva de 29/09 ("o erro que sobra não é
explicável pelas colunas que temos") e com o diagnóstico de 01/10 (20 voos = 41 % do
erro²): o resíduo da campeã é cauda, não viés.

## 4. O que fica para depois

- Ler o resultado de 163–196 em `docs/esteira.md`; o que ganhar vira membro ou ablação
  permanente.
- Se 191/192/193 ganharem, a pista é "tirar do corretor o que mede o receptor": vale
  construir uma versão **normalizada por cobertura** do rastro (deslocamento do aeroporto
  estimado fora do mês) em vez de só apagar colunas.
- Fatias do resíduo estão descartadas como fonte de pós-regra.

Scripts em `.superpowers/bruta/` (fora do índice do git), saídas em
`.superpowers/bruta/{auc_univariada,gain_adversarial,fatias}.csv`.
