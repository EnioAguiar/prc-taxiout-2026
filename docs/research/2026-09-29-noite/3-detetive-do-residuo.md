# 3-detetive-do-residuo

Modelo que prevê o **erro** da v16 (y ≤ 3 h, 5 folds por dia) com todas as colunas brutas (tempos como MVT − X, minuto e segundo; categorias). RMSE do erro 238.81 → 235.69. Se cair bem, há sinal não usado.

| coluna | importância (ganho, %) |
|---|---|
| cat_ADES_mvt | 13.4% |
| cat_STAND_mvt | 13.3% |
| cat_AIRCRAFT_TYPE_mvt | 11.9% |
| pred | 10.0% |
| cat_AIRCRAFT_OPERATOR_flt | 8.1% |
| cat_RUNWAY_mvt | 7.0% |
| dt_SCHED_TIME_UTC_mvt | 6.9% |
| cat_ADES_flt | 5.8% |
| dt_EOBT_1_flt | 4.7% |
| cat_ADES_FILED_flt | 3.8% |
| dt_LOBT_flt | 2.4% |
| cat_AIRCRAFT_TYPE_flt | 2.1% |
| dt_IOBT_flt | 2.0% |
| cat_CALLSIGN_flt | 1.7% |
| num_FLIGHT_ID_mvt | 1.4% |
| dt_AOBT_3_flt | 1.1% |
| cat_FLIGHT_mvt | 0.5% |
| cat_ADEP_mvt | 0.5% |
| min_LOBT_flt | 0.5% |
| dt_ARVT_1_flt | 0.5% |
| seg_ARVT_3_flt | 0.4% |
| min_ARVT_3_flt | 0.3% |
| min_MVT_TIME_UTC_mvt | 0.2% |
| min_AOBT_3_flt | 0.2% |
| min_EOBT_1_flt | 0.2% |
| cat_MARKET_SEGMENT_flt | 0.2% |
| cat_ADEP_flt | 0.2% |
| dt_ARVT_3_flt | 0.1% |
| min_SCHED_TIME_UTC_mvt | 0.1% |
| min_ARVT_1_flt | 0.1% |
