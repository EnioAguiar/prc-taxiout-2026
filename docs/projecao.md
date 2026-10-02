# Projeção — gerada em 2026-10-02 12:00 UTC

Dias até o prazo (11/10 23:59:59, horário da Europa): **9.4**. Placar público: 2026-10-02T11:58:36Z, 142 equipes.

Nossa melhor nota: **244.18** → posição **23**

## Corte por posição

| Posição | Hoje | Velocidade (s/dia) | No prazo: parado / desacelerando / ritmo atual | Falta para nós hoje |
|---|---|---|---|---|
| 1º | 219.64 | -1.89 | 219.6 / 208.0 / 201.8 | 24.5 s |
| 3º | 221.65 | -1.84 | 221.6 / 210.4 / 204.3 | 22.5 s |
| 10º | 230.37 | -2.13 | 230.4 / 217.3 / 210.3 | 13.8 s |
| 50º | 271.76 | -1.42 | 271.8 / 263.1 / 258.4 | -27.6 s |

## Nossa projeção (fila de saltos em `saltos.json`)

Saltos que cabem em 7.5 dias úteis: `regressor_sem_lirf` (3 s × 30%), `teto_aeroporto` (1 s × 40%), `ablacao` (2 s × 40%), `especialista_lobt` (2 s × 40%), `cobertura` (10 s × 15%).

- Esperado: **239.7**; faixa 10–90 %: 232.2 a 244.2; se tudo funcionar: 226.2.
- Nosso ritmo nos últimos 7 dias: -24.8 s/dia (não extrapolar: vem de saltos, não de tendência).

| Meta | Chance: parado / desacelerando / ritmo atual |
|---|---|
| top 1 | 0% / 0% / 0% |
| top 3 | 0% / 0% / 0% |
| top 10 | 5% / 0% / 0% |
| top 50 | 100% / 100% / 100% |

Para o top 3 (cenário desacelerando, 210.4) faltam **34 s**: 3.6 s por dia, ou 1.57 % por dia composto.

Leitura: a chance vem só da fila de saltos. Salto novo com evidência → entra em `saltos.json`; experimento feito → status e ganho real atualizados. Estimativas sem medida são inferência.
