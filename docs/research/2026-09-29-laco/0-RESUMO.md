# 0 — RESUMO do laço noturno (29/09)

Base congelada `20260928-131214-base_ctx`; corretor barato (um LightGBM, 300 rodadas) sobre o quadro da campeã (`corrector_frame(janela=True)` + externos + plano 13). 61 candidatos avaliados em 1.3 h.

**Protocolos.** (a) 5 folds por dia no holdout jan+jul; (b) guarda jan↔jul: treina em um mês e prevê o outro, nos dois sentidos.

**Portão.** aceito = ganho ≥ 0.5 s em `sem_loteria` em (a), piora de `normais_nm` ≤ 0.2 s e ganho > 0 nos dois sentidos da guarda. FRÁGIL = passa em (a) e falha na guarda (provável decoreba do holdout).

## Referência

| protocolo | completo | sem loteria | normais NM | nm ausente | y > 1 h |
|---|---|---|---|---|---|
| cv | 304.2 | 243.33 | 197.98 | 1755.93 | 4059.01 |
| jan2jul | 301.69 | 292.2 | 215.79 | 1336.21 | 2649.18 |
| jul2jan | 339.23 | 208.96 | 190.7 | 2851.36 | 6056.51 |

Normais × aeroporto na referência (P1.1), piores: LIRF 311.5, LTFM 234.8, LFPG 230.9, EGLL 211.9, LEBL 195.5.

## Todos os candidatos (ordenados pelo ganho em `sem_loteria`, protocolo a)

| candidato | tipo | sem loteria (a) | ganho (a) | normais NM (a) | jan→jul | jul→jan | veredito | min |
|---|---|---|---|---|---|---|---|---|
| r2_r2_peso_loteria_0.3+add_p_copia_dist+add_fila | combo | 241.4 | +1.93 | 196.82 (+1.16) | +8.71 | +0.96 | aceito | 3.8 |
| peso_loteria_0.3 | peso | 241.63 | +1.70 | 197.55 (+0.43) | +9.77 | +0.06 | aceito | 1.1 |
| r2_peso_loteria_0.3+add_p_copia_dist | combo | 241.7 | +1.63 | 197.33 (+0.65) | +9.91 | +1.00 | aceito | 3.4 |
| peso_loteria_0.1 | peso | 241.9 | +1.43 | 197.46 (+0.52) | +11.00 | -0.26 | fragil | 1.1 |
| peso_y3h_0.7 | peso | 242.28 | +1.05 | 197.73 (+0.25) | +5.67 | +0.19 | aceito | 1.1 |
| peso_y3h_0.5 | peso | 242.3 | +1.03 | 197.45 (+0.53) | +8.06 | -0.57 | fragil | 1.1 |
| peso_y3h_0.3 | peso | 242.46 | +0.87 | 197.16 (+0.82) | +10.40 | -1.25 | fragil | 1.1 |
| peso_y1h_0.7 | peso | 242.47 | +0.86 | 196.9 (+1.08) | +6.55 | -0.75 | fragil | 1.1 |
| add_p_copia_dist | features | 242.52 | +0.81 | 197.53 (+0.45) | +1.25 | +1.14 | aceito | 3.5 |
| add_fila | features | 242.71 | +0.62 | 197.62 (+0.36) | +1.41 | +0.84 | aceito | 1.4 |
| add_fila_gaps | features | 242.71 | +0.62 | 197.64 (+0.34) | +0.35 | +0.11 | aceito | 1.4 |
| peso_y1h_0.5 | peso | 242.76 | +0.57 | 196.1 (+1.88) | +8.63 | -1.91 | fragil | 1.1 |
| add_stand_p | features | 242.77 | +0.56 | 197.97 (+0.01) | +4.66 | -0.81 | fragil | 1.1 |
| add_dist_plano | features | 242.77 | +0.56 | 197.6 (+0.38) | +0.43 | +0.32 | aceito | 1.1 |
| sem_ctx | ablacao | 242.81 | +0.52 | 197.7 (+0.28) | +1.03 | +1.20 | aceito | 0.8 |
| par_folhas95 | params | 242.89 | +0.44 | 197.66 (+0.32) | -0.02 | +0.09 | descartado | 1.3 |
| par_folhas127 | params | 242.95 | +0.38 | 197.53 (+0.45) | +0.03 | +0.40 | descartado | 1.5 |
| add_gaps_plano | features | 242.96 | +0.37 | 197.74 (+0.24) | -0.36 | -0.11 | descartado | 1.2 |
| add_viz_pred | features | 242.97 | +0.36 | 197.73 (+0.25) | +0.78 | -0.75 | descartado | 1.1 |
| pos_shrink_global | pos | 242.98 | +0.35 | 197.62 (+0.36) | +11.33 | +2.79 | descartado | 1.9 |
| add_viz_contagens | features | 243.04 | +0.29 | 197.47 (+0.51) | +0.18 | -1.11 | descartado | 1.4 |
| par_ff08_bag08 | params | 243.12 | +0.21 | 197.6 (+0.38) | +2.81 | +0.95 | descartado | 1.0 |
| peso_suave_0.0 | peso | 243.14 | +0.19 | 196.18 (+1.80) | +11.10 | -0.82 | descartado | 1.1 |
| add_cs_pref | features | 243.15 | +0.18 | 197.93 (+0.05) | +0.54 | +0.02 | descartado | 1.1 |
| add_p_copia | features | 243.18 | +0.15 | 197.77 (+0.21) | +0.16 | +0.80 | descartado | 3.4 |
| add_apt_rwy | features | 243.19 | +0.14 | 197.84 (+0.14) | -2.46 | +0.38 | descartado | 1.1 |
| par_lr004_r400 | params | 243.2 | +0.13 | 197.84 (+0.14) | -0.20 | +0.57 | descartado | 1.3 |
| add_hora_ciclica | features | 243.23 | +0.10 | 197.82 (+0.16) | +0.26 | -1.44 | descartado | 1.1 |
| add_lobt_seg | features | 243.24 | +0.09 | 197.98 (+0.00) | +0.21 | +0.25 | descartado | 1.2 |
| add_tipo_aer | features | 243.25 | +0.08 | 197.79 (+0.19) | +0.89 | -0.80 | descartado | 1.1 |
| sem_ext_diarias | ablacao | 243.33 | +0.00 | 198.28 (-0.30) | -0.48 | -0.19 | descartado | 1.0 |
| par_l2_5 | params | 243.33 | +0.00 | 197.93 (+0.05) | +0.70 | +0.49 | descartado | 1.1 |
| add_dow | features | 243.34 | -0.01 | 197.88 (+0.10) | +0.12 | +0.21 | descartado | 1.0 |
| pos_shrink_apt | pos | 243.34 | -0.01 | 197.9 (+0.08) | +10.02 | +2.72 | descartado | 1.9 |
| sem_nm_cons | ablacao | 243.35 | -0.02 | 198.07 (-0.09) | +0.84 | +0.72 | descartado | 1.1 |
| par_lr003_r600 | params | 243.41 | -0.08 | 197.79 (+0.19) | -1.37 | +0.13 | descartado | 1.9 |
| sem_hour | ablacao | 243.47 | -0.14 | 197.92 (+0.06) | +0.56 | +1.05 | descartado | 1.0 |
| sem_ext_copia | ablacao | 243.48 | -0.15 | 197.62 (+0.36) | +0.45 | +0.10 | descartado | 1.0 |
| sem_adsb | ablacao | 243.5 | -0.17 | 198.72 (-0.74) | +0.84 | -2.41 | descartado | 0.9 |
| par_min400 | params | 243.5 | -0.17 | 197.58 (+0.40) | +4.84 | +1.14 | descartado | 1.1 |
| par_lr006_r250 | params | 243.54 | -0.21 | 198.03 (-0.05) | +0.26 | +0.71 | descartado | 0.9 |
| sem_rot | ablacao | 243.56 | -0.23 | 197.9 (+0.08) | +1.16 | +1.10 | descartado | 1.0 |
| pos_shrink_apt_livre | pos | 243.63 | -0.30 | 198.09 (-0.11) | +10.02 | +2.72 | descartado | 1.9 |
| add_contagens | features | 243.65 | -0.32 | 197.87 (+0.11) | +0.13 | +0.30 | descartado | 1.3 |
| sem_janela | ablacao | 243.65 | -0.32 | 198.43 (-0.45) | +0.23 | -0.62 | descartado | 1.0 |
| sem_met | ablacao | 243.67 | -0.34 | 198.28 (-0.30) | +0.47 | +1.84 | descartado | 1.0 |
| sem_ext_opdi | ablacao | 243.8 | -0.47 | 198.42 (-0.44) | +0.06 | -0.16 | descartado | 1.0 |
| par_r450 | params | 243.85 | -0.52 | 198.03 (-0.05) | -5.78 | -1.00 | descartado | 1.4 |
| sem_cia | ablacao | 244.0 | -0.67 | 197.6 (+0.38) | +7.82 | -0.41 | descartado | 1.0 |
| par_extra_trees | params | 244.0 | -0.67 | 197.69 (+0.29) | +3.55 | +1.23 | descartado | 1.0 |
| par_folhas31 | params | 244.06 | -0.73 | 198.33 (-0.35) | +1.30 | +0.28 | descartado | 0.8 |
| sem_p13 | ablacao | 244.28 | -0.95 | 198.17 (-0.19) | +7.15 | +0.79 | descartado | 0.8 |
| sem_to_takeoff | ablacao | 244.37 | -1.04 | 197.9 (+0.08) | +1.68 | -0.52 | descartado | 1.0 |
| sem_ext | ablacao | 244.53 | -1.20 | 198.41 (-0.43) | -1.18 | -1.03 | descartado | 0.9 |
| par_min100 | params | 244.54 | -1.21 | 198.18 (-0.20) | -6.66 | -0.81 | descartado | 1.0 |
| add_ades_reg | features | 245.62 | -2.29 | 198.7 (-0.72) | -4.27 | -3.93 | descartado | 1.2 |
| par_min50 | params | 246.43 | -3.10 | 198.76 (-0.78) | -10.45 | -0.73 | descartado | 0.9 |
| clip_6h | peso | 264.99 | -21.66 | 199.36 (-1.38) | +6.70 | -10.77 | descartado | 1.1 |
| clip_3h | peso | 285.63 | -42.30 | 200.5 (-2.52) | -2.63 | -24.75 | descartado | 1.0 |
| clip_2h | peso | 296.82 | -53.49 | 201.06 (-3.08) | -6.73 | -32.48 | descartado | 1.1 |
| clip_1.5h | peso | 303.97 | -60.64 | 200.69 (-2.71) | -10.83 | -37.68 | descartado | 1.1 |

## Veredito

- **aceitos (9)**: r2_r2_peso_loteria_0.3+add_p_copia_dist+add_fila, peso_loteria_0.3, r2_peso_loteria_0.3+add_p_copia_dist, peso_y3h_0.7, add_p_copia_dist, add_fila, add_fila_gaps, add_dist_plano, sem_ctx
- **frágeis (6)**: peso_loteria_0.1, peso_y3h_0.5, peso_y3h_0.3, peso_y1h_0.7, peso_y1h_0.5, add_stand_p

## Combinação gulosa (2ª rodada)

Melhor combinação: `r2_r2_peso_loteria_0.3+add_p_copia_dist+add_fila` — ganho +1.93 s em (a), +8.71 / +0.96 na guarda, veredito **aceito**.

## Recomendação para amanhã

1. `r2_r2_peso_loteria_0.3+add_p_copia_dist+add_fila` (+1.93 s em (a); guarda +8.71/+0.96) — receita: `{"peso": {"cond": "loteria", "w": 0.3}, "add": ["dist_plano", "fila", "p_copia"]}`.
1. `peso_loteria_0.3` (+1.70 s em (a); guarda +9.77/+0.06) — receita: `{"peso": {"cond": "loteria", "w": 0.3}}`.
1. `r2_peso_loteria_0.3+add_p_copia_dist` (+1.63 s em (a); guarda +9.91/+1.00) — receita: `{"peso": {"cond": "loteria", "w": 0.3}, "add": ["dist_plano", "p_copia"]}`.
1. `peso_y3h_0.7` (+1.05 s em (a); guarda +5.67/+0.19) — receita: `{"peso": {"cond": "y3h", "w": 0.7}}`.
1. `add_p_copia_dist` (+0.81 s em (a); guarda +1.25/+1.14) — receita: `{"add": ["p_copia", "dist_plano"]}`.
1. Rodada cheia com a combinação `r2_r2_peso_loteria_0.3+add_p_copia_dist+add_fila` (+1.93 s em (a)).
Não levar para o envio os FRÁGEIS (peso_loteria_0.1, peso_y3h_0.5, peso_y3h_0.3, peso_y1h_0.7, peso_y1h_0.5, add_stand_p): ganham em (a) e perdem na guarda jan↔jul.

Lembrete: o laço só produz candidatos. Promoção exige uma corrida completa `stack.py --crossfit` e o portão do `src/compare.py` (ganho ≥ 10 s, IC > 0, sobrevive sem os 10 maiores voos, IC > 0 em jan e em jul).
