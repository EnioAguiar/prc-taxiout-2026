# Projeção — gerada em 2026-09-30 12:00 UTC

Dias até o prazo (11/10 23:59:59, horário da Europa): **11.4**. Placar público: 2026-09-30T11:42:11Z, 155 equipes.

Nossa melhor nota: **247.11** → posição **24**

## Corte por posição

| Posição | Hoje | Velocidade (s/dia) | No prazo: parado / desacelerando / ritmo atual | Falta para nós hoje |
|---|---|---|---|---|
| 1º | 219.64 | -2.68 | 219.6 / 201.3 / 189.1 | 27.5 s |
| 3º | 224.50 | -2.27 | 224.5 / 209.0 / 198.6 | 22.6 s |
| 10º | 235.92 | -2.24 | 235.9 / 220.6 / 210.3 | 11.2 s |
| 50º | 275.64 | -1.03 | 275.6 / 268.6 / 263.9 | -28.5 s |

## Nossa projeção (fila de saltos em `saltos.json`)

Saltos que cabem em 9.1 dias úteis: `regressor_sem_lirf` (3 s × 30%), `teto_aeroporto` (1 s × 40%), `ablacao` (2 s × 40%), `especialista_lobt` (2 s × 40%), `cobertura` (10 s × 15%).

- Esperado: **242.7**; faixa 10–90 %: 235.1 a 247.1; se tudo funcionar: 229.1.
- Nosso ritmo nos últimos 7 dias: -29.7 s/dia (não extrapolar: vem de saltos, não de tendência).

| Meta | Chance: parado / desacelerando / ritmo atual |
|---|---|
| top 1 | 0% / 0% / 0% |
| top 3 | 0% / 0% / 0% |
| top 10 | 12% / 0% / 0% |
| top 50 | 100% / 100% / 100% |

Para o top 3 (cenário desacelerando, 209.0) faltam **38 s**: 3.3 s por dia, ou 1.46 % por dia composto.

Leitura: a chance vem só da fila de saltos. Salto novo com evidência → entra em `saltos.json`; experimento feito → status e ganho real atualizados. Estimativas sem medida são inferência.
