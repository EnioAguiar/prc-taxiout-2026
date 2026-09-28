# 4-detector-de-mudanca

Classificador jan/jul 2025 × jan/jul 2026 com as features do modelo. AUC 0.875 (0,5 = indistinguível). Colunas que mais separam os anos (candidatas a deriva):

| coluna | importância | média 2025 | média 2026 |
|---|---|---|---|
| adsb_lon0 | 22.4% | 6.56 | 4.38 |
| adsb_lat0 | 10.4% | 48 | 47.5 |
| adsb_n_chao | 7.7% | 81.4 | 91.8 |
| ctx_arr_tin_apt_60 | 5.8% | 558 | 545 |
| hour | 4.7% | 13.1 | 13.1 |
| ctx_viz_fut_apt_60 | 4.5% | 1.02e+03 | 1.04e+03 |
| ctx_viz_pas_apt_60 | 3.5% | 1.01e+03 | 1.04e+03 |
| apt_arr_next_60m | 2.9% | 29.6 | 29.7 |
| rwy_dep_prev_60m | 2.7% | 21.6 | 21.5 |
| rwy_dep_next_60m | 2.6% | 21.6 | 21.5 |
| apt_dep_prev_60m | 2.5% | 33.5 | 33.6 |
| ctx_arr_tin_rwy_60 | 2.1% | 450 | 435 |
| apt_dep_next_60m | 1.9% | 33.6 | 33.7 |
| apt_arr_prev_60m | 1.8% | 30 | 30 |
| ctx_viz_fut_rwy_60 | 1.7% | 1.02e+03 | 1.04e+03 |
| dow | 1.7% | 2.95 | 3.06 |
| ctx_viz_pas_rwy_60 | 1.5% | 1.01e+03 | 1.04e+03 |
| ctx_arr_n_rwy_60 | 1.3% | 1.66 | 1.37 |
| adsb_gs0 | 1.0% | 9.95 | 8.71 |
| adsb_gap_max | 1.0% | 130 | 144 |
