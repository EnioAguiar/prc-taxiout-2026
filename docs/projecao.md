# Projeção — gerada em 2026-10-01 03:01 UTC

Dias até o prazo (11/10 23:59:59, horário da Europa): **10.8**. Placar público: 2026-10-01T02:55:39Z, 153 equipes.

Nossa melhor nota: **244.89** → posição **22**

## Corte por posição

| Posição | Hoje | Velocidade (s/dia) | No prazo: parado / desacelerando / ritmo atual | Falta para nós hoje |
|---|---|---|---|---|
| 1º | 219.64 | -2.41 | 219.6 / 203.6 / 193.6 | 25.3 s |
| 3º | 222.91 | -2.14 | 222.9 / 208.7 / 199.8 | 22.0 s |
| 10º | 235.52 | -2.03 | 235.5 / 222.1 / 213.7 | 9.4 s |
| 50º | 274.64 | -1.03 | 274.6 / 267.8 / 263.5 | -29.7 s |

## Nossa projeção (fila de saltos em `saltos.json`)

Saltos que cabem em 8.6 dias úteis: `regressor_sem_lirf` (3 s × 30%), `teto_aeroporto` (1 s × 40%), `ablacao` (2 s × 40%), `especialista_lobt` (2 s × 40%), `cobertura` (10 s × 15%).

- Esperado: **240.5**; faixa 10–90 %: 232.9 a 244.9; se tudo funcionar: 226.9.
- Nosso ritmo nos últimos 7 dias: -27.6 s/dia (não extrapolar: vem de saltos, não de tendência).

| Meta | Chance: parado / desacelerando / ritmo atual |
|---|---|
| top 1 | 0% / 0% / 0% |
| top 3 | 0% / 0% / 0% |
| top 10 | 16% / 0% / 0% |
| top 50 | 100% / 100% / 100% |

Para o top 3 (cenário desacelerando, 208.7) faltam **36 s**: 3.4 s por dia, ou 1.47 % por dia composto.

Leitura: a chance vem só da fila de saltos. Salto novo com evidência → entra em `saltos.json`; experimento feito → status e ganho real atualizados. Estimativas sem medida são inferência.
