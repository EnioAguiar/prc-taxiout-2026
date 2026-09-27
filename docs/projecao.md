# Projeção — gerada em 2026-09-27 18:47 UTC

Dias até o prazo (11/10 23:59:59, horário da Europa): **14.1**. Placar público: 2026-09-27T18:41:44Z, 186 equipes.

Nossa melhor nota: **266.81** → posição **30** (o placar ainda mostra 275.90)

## Corte por posição

| Posição | Hoje | Velocidade (s/dia) | No prazo: parado / desacelerando / ritmo atual | Falta para nós hoje |
|---|---|---|---|---|
| 1º | 224.50 | -3.27 | 224.5 / 199.6 / 178.3 | 42.3 s |
| 3º | 228.59 | -2.62 | 228.6 / 208.7 / 191.6 | 38.2 s |
| 10º | 237.87 | -3.57 | 237.9 / 210.7 / 187.4 | 28.9 s |
| 50º | 278.13 | — | 278.1 / 278.1 / 278.1 | -11.3 s |

## Nossa projeção (fila de saltos em `saltos.json`)

Saltos que cabem em 11.3 dias úteis: `catboost_residuo` (5 s × 50%), `media_mensal_oficial` (2 s × 40%), `clima` (3 s × 50%), `deriva_adsb` (2 s × 30%), `ablacao` (2 s × 40%), `cobertura` (10 s × 15%).

- Esperado: **259.1**; faixa 10–90 %: 251.8 a 264.8; se tudo funcionar: 242.8.
- Nosso ritmo nos últimos 7 dias: -38.8 s/dia (não extrapolar: vem de saltos, não de tendência).

| Meta | Chance: parado / desacelerando / ritmo atual |
|---|---|
| top 1 | 0% / 0% / 0% |
| top 3 | 0% / 0% / 0% |
| top 10 | 0% / 0% / 0% |
| top 50 | 100% / 100% / 100% |

Para o top 3 (cenário desacelerando, 208.7) faltam **58 s**: 4.1 s por dia, ou 1.72 % por dia composto.

Leitura: a chance vem só da fila de saltos. Salto novo com evidência → entra em `saltos.json`; experimento feito → status e ganho real atualizados. Estimativas sem medida são inferência.
