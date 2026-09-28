# 0 — RESUMO do laço fiel (29/09)

Cada linha desta tabela é o corretor de verdade da campeã: conjunto (LightGBM global + por aeroporto + CatBoost) treinado nas 1,74 M linhas cegas com `--externos --plano13`, sobre o oof da campeã `20260928-131704-v20_cf` reaproveitado (`stack.py --reusar-oof`), medido no holdout jan+jul 2025 e comparado à campeã com a matemática do `src/compare.py`. 11 avaliações em 0.6 h.

Campeã `20260928-131704-v20_cf`: 301,73 na simulação, 252,34 no placar oficial. O ganho de cada linha é contra ela; a coluna `completo` é a métrica do placar.

**Portão.** aprovado = ganho > 0 no `completo`, IC 95% baixo > −0,5 s, ganho sem os 10 maiores voos > 0 e ganho > 0 em jan e em jul. `sem loteria` e `normais NM` são informação, não entram no portão.

**Fidelidade.** `referencia_v20` (nenhuma mudança) deu completo 301.72 contra 301,73 da campeã: ganho +0.01 s — é o ruído do CatBoost na GPU, e é o piso de qualquer ganho desta tabela.

## 1ª rodada

| candidato | completo | ganho | IC 95% | sem top 10 | jan / jul | sem loteria | normais NM | veredito | min |
|---|---|---|---|---|---|---|---|---|---|
| dist_plano | 300.64 | +1.09 | +0.14 a +2.13 | +0.01 | +0.6 / +1.6 | 236.5 | 194.21 | aprovado | 3.0 |
| dist_plano_sem_ctx | 301.56 | +0.17 | -2.06 a +1.91 | -1.08 | -0.4 / +0.8 | 237.37 | 194.57 | reprovado | 2.5 |
| stand_prefixo | 301.8 | -0.07 | -0.55 a +0.37 | -0.35 | +0.4 / -0.5 | 237.82 | 194.51 | reprovado | 3.1 |
| fila | 301.85 | -0.12 | -1.07 a +0.72 | -0.50 | +0.3 / -0.5 | 237.51 | 194.02 | reprovado | 3.9 |
| rounds_500 | 301.87 | -0.14 | -0.51 a +0.18 | -0.46 | +0.2 / -0.5 | 237.84 | 194.33 | reprovado | 5.3 |
| peso_y3h_07 | 301.93 | -0.20 | -0.93 a +0.49 | -0.65 | +0.1 / -0.5 | 237.63 | 194.61 | reprovado | 3.0 |
| p_copia | 302.07 | -0.34 | -0.84 a +0.05 | -0.66 | +0.1 / -0.7 | 238.09 | 194.77 | reprovado | 5.5 |
| corretor_sem_ctx | 302.42 | -0.69 | -1.83 a +0.35 | -1.25 | -0.5 / -0.9 | 238.43 | 194.85 | reprovado | 2.4 |
| so_lgb | 304.56 | -2.83 | -4.59 a -1.51 | -3.42 | -1.9 / -3.8 | 240.9 | 196.95 | reprovado | 0.7 |
| so_catboost | 305.98 | -4.25 | -6.86 a -2.26 | -4.92 | -2.3 / -6.2 | 241.94 | 197.91 | reprovado | 1.0 |

## Combinação gulosa

Nenhuma combinação foi avaliada (ninguém passou o portão na 1ª rodada).

## Veredito

- **aprovados na 1ª rodada (1)**: dist_plano

## Amanhã

Melhor candidato: **`dist_plano`** — ganho +1.09 s (IC 95% +0.14 a +2.13), completo 300.64, sem os 10 maiores +0.01 s.

Corrida cheia para registrar a corrida (com o oof reaproveitado, ~7 min):

```bash
bin/run src/stack.py <nome> --base 20260928-131214-base_ctx --crossfit --conjunto --externos --plano13 --dist-plano
# … e o envio, com o id que ela gravar:
bin/run src/train.py submit N --corrida <id>
```

Antes de enviar, passe pelo portão oficial: `bin/run src/compare.py <id> 20260928-131704-v20_cf`.
