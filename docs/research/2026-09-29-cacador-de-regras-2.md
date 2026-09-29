# Caçador de regras 2 (29/09): stand, mesmo voo, cópia do plano

Holdout jan+jul/2025, previsões da v28 (`20260929-093849-v28_cf`, 299,41). Scripts descartáveis
(`/tmp/h2.py`, `/tmp/h3.py`, `/tmp/h4.py`), nada entrou em `src/`. Resultado: **nenhuma regra nova**.

## 1. Sequência de pousos no mesmo stand

O ranking tem BLOCK e TAXITIME das ARR (100 %), então os pousos no stand são conhecidos na previsão.
Um stand é ocupado por um avião de cada vez: o off-block da partida fica entre o pouso do próprio
avião e o pouso do próximo ocupante.

Oráculo (sabendo qual pouso é qual), cortando a previsão da v28 no intervalo:

| Corte | RMSE |
|---|---|
| v28 (já com a janela do LOBT) | 299,38 |
| pouso anterior e próximo ocupante | **289,27** |
| só o próximo ocupante (piso `taxi ≥ MVT − A`) | 290,86 |
| só o pouso anterior (teto) | 297,83 |

O ganho vem quase todo de LIRF e EGLL e dos voos sem LOBT ou com táxi > 1 h.

Na prática não dá para saber se o último pouso no stand antes do MVT é o próprio avião ou o
próximo ocupante (não há matrícula). P(próximo ocupante) em 2025:

- tipo de avião diferente: 43–82 % conforme o aeroporto (EGLL 79 %, LTFM 82 %, resto 43–51 %);
- mesmo tipo e mesma companhia: 1,9 %;
- sem LOBT, tipo diferente, pouso a < 15 min do MVT: 96 % (1.291 voos em 2025).

Qualquer piso aplicado piora: o melhor recorte (tipo diferente, sem LOBT, gap < 15 min) dá 299,42
contra 299,41; com gap < 1 h, 299,58; sem limite de gap, explode. O teto (mesmo tipo e companhia,
sem LOBT) dá 299,70. O corretor já recebe essa informação pelas colunas `rot_*` do plano 13
(idade do último pouso no stand, mesma companhia, mesmo tipo).

## 2. Mesmo número de voo em outro mês

Resíduo médio da v28 por voo (encolhido com k = 5, 20 ou 50), aprendido em jan e aplicado em jul
e vice-versa, por `FLIGHT`, `FLIGHT+STAND` e `FLIGHT+tipo`: **piora sempre**, de 0,03 a 1,4 s no
completo e de 0,05 a 2,4 s nos normais. O erro por voo não persiste entre meses.

## 3. Cópia do plano

Já coberta: o caçador de 28/09 (`2026-09-29-noite/1-cacador-de-regras.md`) só achou a janela do
LOBT (±3606 s); `p_copia` foi reprovado no laço fiel 1 e 2; a regra de Roma já está no arquivo.
