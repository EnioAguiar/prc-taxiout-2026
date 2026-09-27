# Janela do LOBT — |BLOCK − LOBT| ≤ 3606 s (plano 7, tarefa 1)

Data: 27/09/2026. Dados de 2025 (`data/training_2025-*.parquet`) e do ranking 2026.
Fonte da ideia: repositório `elegant-alligator` (`scripts/queue_features.py:68,82`), que
usa a diferença para a fila e tem um `assert` da janela no treino. Nenhum treino do
modelo principal foi rodado por esta nota.

## A regra no dado

Nas **2.062.577** decolagens de 2025 com `LOBT_flt`, |BLOCK − LOBT| ≤ 3606 s em
**100 %** dos voos (o máximo observado é exatamente 3606 s). É um limite duro do
provedor, não uma cauda estatística — por isso vale como restrição, e não como feature.

Cobertura dos demais horários planejados na mesma amostra (fração dentro de ±3606 s):

| Horário | Dentro da janela |
|---|---|
| LOBT | 100 % (máx. 3606 s) |
| IOBT | 99,997 % |
| EOBT_1 | 99,970 % |
| AOBT_3 | 99,914 % |
| SCHED | 94,844 % |

Só o LOBT é exato; os outros erram o suficiente para não servirem de limite.

## Quem fica de fora

Sem `LOBT_flt`: **22.470** decolagens (22.424 delas também sem NM). Esses voos
concentram **78,6 %** do Σy² da cauda y > 1 h — ou seja, a janela não toca justamente
onde o erro mora, mas também não estraga nada lá (sem LOBT a previsão fica como está,
e as retas NM do `two_stage_nm` continuam mandando nesses voos).

## Ganho medido (aplicando a janela às previsões já gravadas)

| Corrida | RMSE holdout | Limitada | Linhas alteradas |
|---|---|---|---|
| v6 | 323,50 | 320,30 | 79 |
| v7_cf | 320,67 | 317,98 | 53 |

Poucas linhas, ganho de ~3 RMSE: são erros de horas em voos que a janela prova
impossíveis (previsão de cópia do SCHED quando o SCHED está longe do LOBT).

## Teto no ranking oficial

Na submissão v6, **117** linhas caem fora da janela, com Σ(p − q)²/N = **16.125**
(p = previsão enviada, q = previsão projetada). Se a regra valer também em 2026, o
placar oficial de 314,76 cairia para no máximo

√(314,76² − 16.125) ≈ **288,0**.

É um limite superior: supõe que o valor projetado nunca fica pior que o original, o que
a janela garante quando a regra vale (o verdadeiro y está dentro dela).

## Como entra no código

- `models.JANELA_LOBT_S = 3606`, `models.janela_lobt(df) -> (lo, hi)` com
  `lo = MVT − LOBT − 3606` e `hi = MVT − LOBT + 3606` (NaN sem LOBT).
- `models.limitar_janela(pred, df)`: projeta em `[lo, hi]` e **depois** aplica o piso 0.
- `cfg["janela_lobt"]`: `TwoStage.predict` zera `p` onde o SCHED cai fora da janela (a
  cópia daria um BLOCK impossível) e `build_model` embrulha o modelo em
  `JanelaLOBT`, que projeta a previsão final uma única vez (inclusive a média do
  `SeedAvg`). Sem a flag, as previsões são idênticas às de hoje.
