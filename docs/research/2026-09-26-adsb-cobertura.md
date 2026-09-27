# adsb.lol — cobertura por aeroporto e teto barato (plano 4, tarefas 2–3)

Data: 26/09/2026. Eventos de `src/adsb_events.py` (jan/jul 2025 e 2026, 815.651
decolagens casadas). Referência: campeã `20260924-215852-nm_retas_6h_r` no holdout
jan+jul/2025 (344.419 voos, RMSE 332,86). Nenhum treino do modelo principal.

## Estimadores do off-block

- `adsb_taxi = MVT − first_ground` (1º ponto do segmento contínuo no chão).
- `adsb_taxi_move = MVT − first_move` (1º ponto com gs > 1 kt).
- `adsb_gs0` = velocidade no 1º ponto; `gs0 ≤ 1` = avião visto **parado** (no portão).

Dia 15/01/2025, erro mediano de `adsb_taxi_move` contra `y`:

| Aeroporto | Visto parado: n / mediana / ±60 s | Visto andando: mediana (sinal) |
|---|---|---|
| EDDM | 149 / 34 s / 81 % | −67 s |
| EHAM | 107 / 22 s / 75 % | −180 s |
| LSZH | 58 / 26 s / 88 % | −337 s |
| LEBL | 29 / 28 s / 66 % | −370 s |
| EDDF | 31 / 50 s / 55 % | −383 s |
| EGLL | 14 / 357 s / 7 % | −821 s |

Visto andando = rastro começa depois do pushback → estimativa curta (viés negativo que
o modelo pode aprender).

## Cobertura no holdout (jan+jul/2025)

| Aeroporto | Voos | Casados | Vistos parados | Mediana (parados) | ±60 s | RMSE campeã (parados) | RMSE adsb (parados) |
|---|---|---|---|---|---|---|---|
| EHAM | 40.949 | 97 % | 28 % | 20 s | 79 % | 193,7 | 263,1 |
| LEBL | 28.986 | 87 % | 13 % | 30 s | 72 % | 229,5 | 251,2 |
| EDDF | 36.830 | 73 % | 8 % | 51 s | 54 % | 175,2 | 348,5 |
| LSZH | 22.346 | 73 % | 19 % | 27 s | 87 % | 158,2 | 240,6 |
| EDDM | 27.091 | 62 % | 24 % | 35 s | 81 % | 184,4 | 182,1 |
| LIRF | 26.528 | 45 % | 2 % | 411 s | 15 % | 680,7 | 1.875,5 |
| EGLL | 40.210 | 14 % | 1 % | 414 s | 2 % | 219,1 | 901,9 |
| LFPG | 39.612 | 7 % | ~0 % | 1.009 s | 3 % | 417,1 | 1.222,5 |
| LEMD | 35.328 | 2 % | ~0 % | 121 s | 14 % | 199,3 | 471,2 |
| LTFM | 46.539 | 0 % | 0 % | — | — | — | — |

Mediana boa, RMSE ruim: poucos voos com erro de horas (casamento errado ou rastro
cortado) dominam o RMSE. O ADS-B serve como **feature**, não como substituto.

## Teto barato (sobre as previsões salvas)

| Teste | RMSE |
|---|---|
| Campeã | 332,86 |
| Trocar `pred` por `adsb_taxi_move` em todos os parados (30.237 voos) | 345,24 |
| Idem, só se \|adsb − pred\| < 900 s | 332,78 |
| Mistura `w` por aeroporto (ajuste dias pares, medida dias ímpares) | 375,00 vs 376,33 |
| Oráculo: erro zero nos normais vistos parados (8,8 % dos voos) | 327,8 |
| Oráculo: erro zero nos normais com qualquer evento (42,4 %) | 289,5 |
| **Empilhamento fora do fold** (LightGBM prevê `y − pred` com `pred`, `adsb_*`, aeroporto, `nm_missing`, hora, `to_takeoff_from_*`; 5 folds por dia; 300 rodadas, 63 folhas) | **317,44** |

Empilhamento, detalhe: normais (y ≤ 1 h) 248,15 → 223,92; voos com evento 282,27 →
244,52; sem evento 365,65 → 361,83; jan 349,78 → 337,38; jul 318,58 → 300,40.
O ganho está onde há ADS-B. **Controle sem `adsb_*` pendente** (corrida interrompida).

## Decisão

Portão de 10 s da tarefa 3: passou (−15,4 s). Seguir para a tarefa 4 com o ano inteiro
de eventos (4a: features no treino da campeã) e o empilhamento como referência (4b).
Aeroportos sem cobertura (LTFM, LEMD, LFPG, EGLL) ficam com a campeã (NaN nas features).
