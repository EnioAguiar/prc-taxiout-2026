# 2-minerador-de-erros

Voos com y ≤ 3 h (v16, holdout). RMSE geral 238.8. Grupos (1 ou 2 condições, ≥ 300 voos) com mais erro² acima da média.

| grupo | n | RMSE do grupo | parte do erro² a mais |
|---|---|---|---|
| apt=LIRF | 26474 | 505 | 26.65% |
| apt=LIRF & mes=7 | 15164 | 602 | 23.58% |
| pista=25 | 22714 | 501 | 22.47% |
| apt=LIRF & pista=25 | 22714 | 501 | 22.47% |
| nm=1 | 5310 | 927 | 21.67% |
| mes=7 & pista=25 | 13768 | 587 | 20.19% |
| apt=LIRF & adsb=True | 11840 | 608 | 18.83% |
| mes=7 & nm=1 | 3888 | 996 | 18.51% |
| adsb=True & pista=25 | 11578 | 601 | 17.91% |
| apt=LIRF & nm=1 | 345 | 3069 | 16.44% |
| mes=7 | 190663 | 266 | 13.54% |
| nm=1 & adsb=True | 2148 | 1130 | 13.33% |
| apt=LIRF & nm=0 | 26129 | 366 | 10.21% |
| atraso=(60.0, 1000000000.0] | 7696 | 562 | 10.15% |
| nm=0 & atraso=(60.0, 1000000000.0] | 7696 | 562 | 10.15% |
| nm=1 & adsb=False | 3162 | 758 | 8.34% |
| nm=0 & pista=25 | 22415 | 359 | 8.19% |
| apt=LIRF & adsb=False | 14634 | 402 | 7.82% |
| apt=LIRF & wtc=M | 22614 | 353 | 7.76% |
| mes=7 & atraso=(60.0, 1000000000.0] | 5492 | 564 | 7.30% |
| mes=7 & adsb=True | 82801 | 272 | 7.13% |
| seg=Mainline & atraso=(60.0, 1000000000.0] | 5051 | 571 | 6.93% |
| adsb=False & atraso=(60.0, 1000000000.0] | 4740 | 578 | 6.70% |
| pista=25 & wtc=M | 20430 | 345 | 6.47% |
| atraso=(45.0, 60.0] | 17638 | 359 | 6.42% |
| nm=0 & atraso=(45.0, 60.0] | 17638 | 359 | 6.42% |
| mes=7 & adsb=False | 107862 | 262 | 6.41% |
| wtc=M & atraso=(60.0, 1000000000.0] | 5001 | 553 | 6.34% |
| apt=LIRF & dow=6 | 3569 | 636 | 6.31% |
| adsb=False | 198618 | 251 | 6.12% |
| apt=LIRF & dow=0 | 3595 | 600 | 5.54% |
| dow=6 & mes=7 | 24611 | 316 | 5.35% |
| dow=0 & pista=25 | 2599 | 677 | 5.30% |
| apt=LIRF & seg=Lowcost | 10392 | 386 | 4.89% |
| mes=7 & atraso=(45.0, 60.0] | 12404 | 365 | 4.83% |
| adsb=False & atraso=(45.0, 60.0] | 10728 | 381 | 4.80% |
| apt=LIRF & cia=RYR | 3261 | 588 | 4.79% |
| apt=LIRF & seg=Mainline | 14414 | 346 | 4.62% |
| adsb=False & pista=25 | 11136 | 371 | 4.55% |
| dow=6 & nm=1 | 1049 | 947 | 4.49% |
