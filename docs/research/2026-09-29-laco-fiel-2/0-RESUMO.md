# 0 — RESUMO do laço fiel, 2ª volta (29/09)

Cada linha desta tabela é o corretor de verdade da campeã v26: conjunto (LightGBM global + por aeroporto + CatBoost) treinado nas 1,74 M linhas cegas com `--externos --plano13 --dist-plano`, sobre o oof `20260928-131704-v20_cf` reaproveitado (`stack.py --reusar-oof`), medido no holdout jan+jul 2025 e comparado à campeã com a matemática do `src/compare.py`. 12 avaliações em 0.7 h.

Campeã `20260928-195148-v26_cf`: 300,64 na simulação, 251,10 no placar oficial. O ganho de cada linha é contra ela; a coluna `completo` é a métrica do placar.

O bloco novo desta volta é `superficie` (`src/superficie.py`, `--superficie`): quantos aviões estão no solo no push estimado (`MVT − pred`) e quantos decolam ou pousam durante o táxi estimado, sem nunca ler o off-block de outra decolagem. `superficie_adsb` é o mesmo bloco com o push do rastro ADS-B onde ele existe — só roda no laço, para separar sinal de congestionamento de erro da base.

**Portão.** aprovado = ganho > 0 no `completo`, IC 95% baixo > −0,5 s, ganho sem os 10 maiores voos > 0 e ganho > 0 em jan e em jul. `sem loteria` e `normais NM` são informação, não entram no portão.

**Fidelidade.** `referencia_v26` (nenhuma mudança) deu completo 300.65 contra 300,64 da campeã: ganho -0.01 s — é o ruído do CatBoost na GPU, e é o piso de qualquer ganho desta tabela.

## 1ª rodada

| candidato | completo | ganho | IC 95% | sem top 10 | jan / jul | sem loteria | normais NM | veredito | min |
|---|---|---|---|---|---|---|---|---|---|
| stand_prefixo | 299.75 | +0.90 | +0.32 a +1.64 | +0.30 | +0.4 / +1.4 | 235.6 | 194.03 | aprovado | 3.5 |
| fila | 300.11 | +0.54 | +0.09 a +1.02 | +0.15 | +0.2 / +0.8 | 235.63 | 193.62 | aprovado | 4.5 |
| p_copia | 300.29 | +0.36 | -0.58 a +1.45 | -0.39 | +0.5 / +0.3 | 236.47 | 194.31 | reprovado | 6.1 |
| rounds_500 | 300.34 | +0.30 | +0.04 a +0.62 | +0.08 | +0.3 / +0.3 | 236.15 | 193.77 | aprovado | 4.1 |
| peso_y3h_07 | 300.41 | +0.24 | -0.29 a +0.80 | -0.23 | -0.1 / +0.6 | 236.07 | 194.26 | reprovado | 3.2 |
| superficie | 300.88 | -0.24 | -1.57 a +0.94 | -1.27 | -0.1 / -0.4 | 236.33 | 193.4 | reprovado | 3.5 |
| superficie_adsb | 301.0 | -0.36 | -1.49 a +0.63 | -0.79 | -0.3 / -0.4 | 236.1 | 193.51 | reprovado | 3.7 |
| corretor_sem_ctx | 301.55 | -0.91 | -2.71 a +0.47 | -1.52 | -1.0 / -0.8 | 237.36 | 194.58 | reprovado | 2.8 |
| so_lgb | 303.55 | -2.91 | -5.08 a -1.12 | -3.67 | -1.5 / -4.2 | 239.77 | 196.32 | reprovado | 0.8 |
| so_catboost | 303.99 | -3.35 | -5.34 a -1.57 | -5.06 | -2.8 / -3.9 | 239.88 | 197.61 | reprovado | 1.0 |

## Combinação gulosa

| candidato | completo | ganho | IC 95% | sem top 10 | jan / jul | sem loteria | normais NM | veredito | min |
|---|---|---|---|---|---|---|---|---|---|
| combo_stand_prefixo+rounds_500 | 299.32 | +1.33 | +0.62 a +2.35 | +0.63 | +0.7 / +1.9 | 235.16 | 193.44 | aprovado | 4.4 |

## Veredito

- **aprovados na 1ª rodada (3)**: stand_prefixo, fila, rounds_500

## Amanhã

Melhor candidato: **`combo_stand_prefixo+rounds_500`** — ganho +1.33 s (IC 95% +0.62 a +2.35), completo 299.32, sem os 10 maiores +0.63 s.

Corrida cheia para registrar a corrida (com o oof reaproveitado, ~7 min):

```bash
bin/run src/stack.py <nome> --base 20260928-131214-base_ctx --crossfit --conjunto --externos --plano13 --dist-plano --reusar-oof 20260928-131704-v20_cf   # ATENÇÃO: stand, rounds ainda não existe no stack.py
# … e o envio, com o id que ela gravar:
bin/run src/train.py submit N --corrida <id>
```

Antes de enviar, passe pelo portão oficial: `bin/run src/compare.py <id> 20260928-195148-v26_cf`.
