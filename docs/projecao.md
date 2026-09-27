# Projeção — gerada em 2026-09-27 17:28 UTC

Dias até o prazo (11/10 23:59:59, horário da Europa): **14.2**. Placar público: 2026-09-27T17:27:48Z, 186 equipes.

Nossa melhor nota: **275.90** → posição **45**

## Corte por posição

| Posição | Hoje | Velocidade (s/dia) | No prazo: parado / desacelerando / ritmo atual | Falta para nós hoje |
|---|---|---|---|---|
| 1º | 224.50 | -3.33 | 224.5 / 199.1 / 177.3 | 51.4 s |
| 3º | 228.59 | -2.67 | 228.6 / 208.3 / 190.8 | 47.3 s |
| 10º | 237.87 | -3.64 | 237.9 / 210.2 / 186.3 | 38.0 s |
| 50º | 278.13 | — | 278.1 / 278.1 / 278.1 | -2.2 s |

## Nossa projeção (fila de saltos em `saltos.json`)

Saltos que cabem em 11.4 dias úteis: `catboost_residuo` (5 s × 50%), `arr_vizinhos` (4 s × 50%), `media_mensal_oficial` (2 s × 40%), `clima` (3 s × 50%), `deriva_adsb` (2 s × 30%), `ablacao` (2 s × 40%), `cobertura` (10 s × 15%).

- Esperado: **266.2**; faixa 10–90 %: 258.9 a 271.9; se tudo funcionar: 247.9.
- Nosso ritmo nos últimos 7 dias: -38.0 s/dia (não extrapolar: vem de saltos, não de tendência).

| Meta | Chance: parado / desacelerando / ritmo atual |
|---|---|
| top 1 | 0% / 0% / 0% |
| top 3 | 0% / 0% / 0% |
| top 10 | 0% / 0% / 0% |
| top 50 | 100% / 100% / 100% |

Para o top 3 (cenário desacelerando, 208.3) faltam **68 s**: 4.8 s por dia, ou 1.96 % por dia composto.

Leitura: a chance vem só da fila de saltos. Salto novo com evidência → entra em `saltos.json`; experimento feito → status e ganho real atualizados. Estimativas sem medida são inferência.
