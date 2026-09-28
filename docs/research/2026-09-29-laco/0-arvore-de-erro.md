# 0 — Árvore de erro sobre o resíduo da campeã

Campeã `20260928-131704-v20_cf` (RMSE completo 301.73 s) no holdout jan+jul 2025 (344.419 voos). Uma árvore rasa (LightGBM, 1 árvore, profundidade 4, ≥ 400 voos por folha) prevê `(y − pred)²`; cada folha é uma coorte com regra explícita. As colunas são as do corretor da campeã (`corrector_frame` + externos + plano 13) mais `gap_*`/`round_*`, `dow`, minuto dos horários, contagens e a própria previsão (`pred_campea`).

**Ordenado por cobertura** (Σ erro² da folha), nunca por taxa: uma folha com RMSE enorme e 20 voos não paga o conserto.

## Todos os voos

| folha | voos | Σ erro² (% do total) | RMSE da folha | y médio | pred média | regra |
|---|---|---|---|---|---|---|
| 1 | 558 | 42.4 % | 4.879 | 2.788 | 2.840 | dist_lo ≤ 1e-35 · cia ∈ {AAL, EJU, FIN, ITY, PGT, RYR…} |
| 2 | 268.798 | 20.1 % | 153 | 873 | 876 | dist_lo > 1e-35 · pred_campea ≤ 2050 · pred_campea ≤ 1326 · gap_EOBT_1_flt_AOBT_3_flt > -1439 |
| 3 | 45.528 | 12.0 % | 288 | 1.536 | 1.545 | dist_lo > 1e-35 · pred_campea ≤ 2050 · pred_campea > 1326 · ext_taxa_cia_ms ≤ 118.1 |
| 4 | 414 | 7.7 % | 2.422 | 5.532 | 5.258 | dist_lo ≤ 1e-35 · cia ∉ {AAL, EJU, FIN, ITY, PGT, RYR…} · pred_campea > 1807 |
| 5 | 15.991 | 4.2 % | 286 | 982 | 983 | dist_lo > 1e-35 · pred_campea ≤ 2050 · pred_campea ≤ 1326 · gap_EOBT_1_flt_AOBT_3_flt ≤ -1439 |
| 6 | 2.419 | 2.7 % | 587 | 1.566 | 1.592 | dist_lo > 1e-35 · pred_campea ≤ 2050 · pred_campea > 1326 · ext_taxa_cia_ms > 118.1 |
| 7 | 1.203 | 2.4 % | 788 | 2.907 | 2.891 | dist_lo > 1e-35 · pred_campea > 2050 · gap_SCHED_TIME_UTC_mvt_AOBT_3_flt ≤ -3299 · ext_taxa_cia ≤ 0.01033 |
| 8 | 400 | 2.1 % | 1.284 | 3.377 | 3.397 | dist_lo > 1e-35 · pred_campea > 2050 · gap_SCHED_TIME_UTC_mvt_AOBT_3_flt ≤ -3299 · ext_taxa_cia > 0.01033 |
| 9 | 4.066 | 2.0 % | 397 | 900 | 888 | dist_lo ≤ 1e-35 · cia ∉ {AAL, EJU, FIN, ITY, PGT, RYR…} · pred_campea ≤ 1807 · pred_campea ≤ 1443 |
| 10 | 1.615 | 1.8 % | 586 | 2.688 | 2.679 | dist_lo > 1e-35 · pred_campea > 2050 · gap_SCHED_TIME_UTC_mvt_AOBT_3_flt > -3299 · to_takeoff_from_SCHED_TIME_UTC_mvt > 3572 |
| 11 | 3.017 | 1.5 % | 390 | 2.349 | 2.355 | dist_lo > 1e-35 · pred_campea > 2050 · gap_SCHED_TIME_UTC_mvt_AOBT_3_flt > -3299 · to_takeoff_from_SCHED_TIME_UTC_mvt ≤ 3572 |
| 12 | 410 | 1.2 % | 951 | 1.676 | 1.588 | dist_lo ≤ 1e-35 · cia ∉ {AAL, EJU, FIN, ITY, PGT, RYR…} · pred_campea ≤ 1807 · pred_campea > 1443 |

## Sem os voos-loteria (11 voos, 37.9 % do erro²)

Árvore reajustada só nas linhas que sobram — é a fatia que o laço tenta melhorar.

| folha | voos | Σ erro² (% do total) | RMSE da folha | y médio | pred média | regra |
|---|---|---|---|---|---|---|
| 1 | 312.571 | 48.4 % | 174 | 946 | 949 | adsb_menos_pred > -2218 · pred_campea ≤ 2976 · pred_campea ≤ 1805 · to_takeoff_from_EOBT_1_flt ≤ 2696 |
| 2 | 556 | 14.6 % | 2.260 | 4.621 | 4.780 | adsb_menos_pred ≤ -2218 |
| 3 | 19.678 | 14.1 % | 373 | 1.179 | 1.179 | adsb_menos_pred > -2218 · pred_campea ≤ 2976 · pred_campea ≤ 1805 · to_takeoff_from_EOBT_1_flt > 2696 |
| 4 | 10.167 | 11.5 % | 469 | 2.117 | 2.122 | adsb_menos_pred > -2218 · pred_campea ≤ 2976 · pred_campea > 1805 · gap_LOBT_flt_AOBT_3_flt > -3810 |
| 5 | 415 | 6.3 % | 1.720 | 4.984 | 5.159 | adsb_menos_pred > -2218 · pred_campea > 2976 · gap_IOBT_flt_AOBT_3_flt ≤ -3450 |
| 6 | 439 | 3.9 % | 1.308 | 2.292 | 2.213 | adsb_menos_pred > -2218 · pred_campea ≤ 2976 · pred_campea > 1805 · gap_LOBT_flt_AOBT_3_flt ≤ -3810 |
| 7 | 582 | 1.3 % | 648 | 3.527 | 3.537 | adsb_menos_pred > -2218 · pred_campea > 2976 · gap_IOBT_flt_AOBT_3_flt > -3450 |
