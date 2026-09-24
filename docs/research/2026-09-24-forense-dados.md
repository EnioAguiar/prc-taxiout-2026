# Forense dos dados — PRC 2026 (taxi-out)

Medido em `data/` local: DEP 2025 (2.085.047 linhas) e `ranking.parquet`
(344.841 DEP de 2026). Scripts em `/tmp/prc-DataForensics/`. Validação em todo
o relatório: **treino = 10 meses de 2025, teste = jan+jul/2025**.

---

## 0. Resultado principal (medido, não inferido)

| Modelo (mesmas features, LightGBM L2, 500 rodadas) | RMSE jan+jul/2025 |
|---|---|
| Único (equivalente ao v2 atual) | **461,1** |
| Dois estágios: classificador `P(BLOCK≈SCHED)` × regressão | **391,1** |

Decomposição:

| | único | dois estágios |
|---|---|---|
| voos com registro NM | 278,1 | 247,3 |
| voos sem registro NM | 2.959,6 | 2.440,4 |
| y ≤ 1 h | 268,5 | 259,2 |
| y > 1 h | 7.028,9 | 5.495,3 |

O experimento reproduz 461,1 ≈ 460,4 da simulação do projeto, logo a escala é
comparável. Aplicando a razão oficial/simulado da v2 (384,7/460,4 = 0,836),
**391,1 → ≈ 327 s oficial** (hoje 384,7). [INFERÊNCIA] a razão se mantém.

Robusto à tolerância do rótulo: `eq` a ±60 s → 391,1 (AUC 0,881); ±120 s →
394,6 (AUC 0,907); ±300 s → 393,0 (AUC 0,953). Misturar com o modelo único só
piora (w=0,5 → 416,0; w=1,0 → 391,1): **usar o de dois estágios puro.**
Nas linhas com `ms > 2 h` (n = 9.094): 2.160 → **1.561**.

---

## 1. A hipótese "BLOCK é cópia de AOBT_3" está errada

| horário | `P(BLOCK==cand)` | `P(\|Δ\|≤60s)` | `sec==0` |
|---|---|---|---|
| AOBT_3 | 0,0065 | 0,2079 | 0,9665 |
| EOBT_1 | 0,0047 | 0,1252 | 0,9862 |
| LOBT / IOBT | 0,0043 | 0,111 | 0,9862 |
| SCHED | 0,0039 | 0,0957 | 1,0000 |
| **BLOCK** | — | — | **0,0640** |

`BLOCK` tem resolução de segundo (6,4 % com `sec==0`); AOBT_3/EOBT têm
resolução de minuto (96–99 %). O "38 % a ±2 min" do README é proximidade, não
cópia — logo as features `round_*` pouco entregam. Prever `MVT − AOBT_3` direto
já dá RMSE 384,9 (nível da v2). A cópia só existe na cauda: em `y > 3 h`,
84,7 % têm `|BLOCK − SCHED| ≤ 60 s`, mediana **−1 s**.

---

## 2. O mecanismo real dos outliers

`y = MVT − BLOCK`, e `MVT` é observável no ranking. Portanto y explode quando o
off-block fica ancorado no horário **programado** e a decolagem acontece horas
depois. Seja `eq = (|BLOCK − SCHED| ≤ 60 s)` e `ms = MVT − SCHED` (observável):

| faixa de y | n | `P(eq)` | mediana `BLOCK−SCHED` | mediana `ms` |
|---|---|---|---|---|
| y ≤ 1 h | 2.080.921 | 0,095 | 415 s | 1.398 s |
| 1 h < y ≤ 3 h | 3.924 | 0,403 | 58 s | 5.278 s |
| y > 3 h | 202 | **0,847** | **−1 s** | 15.505 s |

E o discriminante decisivo — `P(eq | ms > 2 h)` por aeroporto (2025):

| aeroporto | n (ms>2h) | `P(eq)` | `E[y]` |
|---|---|---|---|
| **LIRF** | 3.377 | **0,1267** | **3.252 s** |
| EGLL | 4.054 | 0,0007 | 1.616 s |
| LFPG | 6.057 | 0,0002 | 1.245 s |
| demais 7 | 26.421 | **0,0000** | 871–1.235 s |

`P(y > 1 h | eq ∧ ms > 2 h) = 1,000` (n = 432): **fora de LIRF, atraso grande
nunca vira taxi-out grande**; em LIRF vira em ~13 % dos casos.

---

## 3. O gatilho observável: `FLIGHT_ID_mvt` nulo (falha do join NM)

`FLIGHT_ID_mvt` nulo ⇔ todas as `*_flt` nulas (CALLSIGN, operador, segmento,
LOBT, EOBT_1, AOBT_3, WTC): 1,08 % das DEP de 2025.

| grupo | n | média y | `P(y>1h)` | `P(y>3h)` | desvio interno |
|---|---|---|---|---|---|
| NM presente | 2.062.623 | 987 | 0,0015 | 0,000008 | 417 |
| **NM ausente** | 22.424 | 1.385 | **0,0485** | **0,0083** | **3.401** |

Carrega **42,2 % do SSE** contra a média global, com 1 % das linhas.

Cruzando os dois flags em jan+jul/2025 (teste):

| LIRF | NM nulo | ms>2h | n | `P(eq)` | `E[y]` | **SSE** |
|---|---|---|---|---|---|---|
| n | n | n | 305.994 | 0,082 | 985 | 0,331 |
| n | n | s | 6.928 | 0,000 | 1.173 | 0,027 |
| n | s | n | 3.728 | 0,033 | 992 | 0,071 |
| n | s | s | 1.241 | 0,000 | 1.314 | 0,008 |
| **s** | n | n | 25.387 | 0,194 | 1.184 | 0,062 |
| **s** | n | s | 744 | 0,016 | 1.667 | 0,054 |
| **s** | s | n | 216 | 0,231 | 2.426 | 0,009 |
| **s** | **s** | **s** | **181** | **0,464** | **10.863** | **0,437** |

**181 linhas = 43,7 % de todo o erro quadrático.** No ranking 2026 essa mesma
célula tem **exatamente 181 linhas** (0,052 %), com `E[ms] = 15.397 s`
(vs 14.974 s em 2025). A estrutura de 2026 é a mesma.

---

## 4. 2025 × 2026: a proporção de outliers deve ser igual

| | 2025 jan+jul | 2026 ranking |
|---|---|---|
| `P(FLIGHT_ID nulo)` | 0,0156 | 0,0153 |
| `P(ms > 2 h)` | 0,0268 | 0,0286 |
| `P(MVT−EOBT_1 > 1 h)` | 0,0160 | 0,0152 |
| LIRF `P(NM nulo)` | 0,0150 | 0,0142 |
| célula de risco (n) | 181 | 181 |
| medianas `ms/me/ma/ml` | idênticas ±5 % | — |

EHAM piorou (NM nulo 3,2 % vs 1,97 %; `P(ms>2h)` 4,8 % vs 3,2 %), LSZH também;
EDDF/EGLL melhoraram. Taxi-in dos ARR: distribuição idêntica até o P99.

---

## 5. LIRF: por que é o pior

- 46,9 % de todo o SSE de 2025 (2ª colocada EGLL com 12,2 %).
- `P(BLOCK ≈ SCHED)` = 20,7 % em LIRF contra 5,8–10,5 % nos outros nove.
  Com NM nulo sobe para 48,5 %.
- `P(eq)` cresce com o atraso: 0,12 (ms 2–3 h) → 0,28 (6–12 h) → **0,61 (>12 h)**.
- Pista/stand não concentram (25 = 84 % do SSE porque é 90 % dos voos).
  Companhia também não: em `ms>2h`, `P(eq)` = 0,01 em Lowcost e Mainline — a
  massa toda vem das linhas **sem registro NM** (segmento nulo).
- NM nulo em LIRF: 0,47 % (mar) a 2,22 % (jul). **Julho está no ranking.**
- Regressão `y ~ a + b·ms` dentro de (LIRF, NM nulo): **r = 0,896**,
  inclinação 1,062, intercepto −2.818 s. RMSE cai de 11.218 → 4.972.

---

## 6. Rotação e duplicados: sem sinal

- Ligação DEP ↔ ARR anterior por (aeroporto, STAND) com `merge_asof`:
  cobertura **98,05 %** em 2025 e **97,53 %** no ranking.
- `tempo_de_solo = MVT_dep − onblock_arr`: **corr(y) = −0,009**. Nenhum poder.
  `P(y>1h)` é 0,0013–0,0029 em todas as faixas de tempo de solo.
- `taxi-in do voo anterior`: corr 0,103; média de y sobe 908 → 1.114 do 1º ao
  10º decil. Sinal fraco e quase todo já capturado por `ref_p10` de stand/pista.
- Média móvel do taxi-in dos ARR do aeroporto (legal: o alvo dos ARR **não** foi
  apagado no ranking): corr 0,09–0,10, mas **0,014 após remover o efeito fixo do
  aeroporto**. Não paga.
- Duplicados: 0 `MVT_ID` repetido; 3 `FLIGHT_ID` repetidos (6 linhas, y normal);
  9.983 pares (ADEP, MVT_TIME) idênticos (pistas paralelas). Nada explorável.

---

## 7. Teto de ganho

Todos com treino em 10 meses e teste em jan+jul/2025.

| cenário | RMSE |
|---|---|
| modelo único atual (medido) | 461,1 |
| **dois estágios (medido)** | **391,1** |
| oráculo: erro zero só nas linhas com NM nulo (resto = baseline grosseiro) | 391,8 (de 628,4) |
| oráculo: erro zero em todo `y > 1 h` (resto = baseline grosseiro) | 324,5 (de 628,4) |
| piso teórico: `nm_present` a 278 s e `nm_missing` perfeito | **287,7** (≈ 240 oficial) |

Líder oficial: 234,1 s. O piso "NM ausente perfeito" projeta ≈ 240 oficial.
**A competição é decidida nessas ~1,5 % de linhas.**

---

## 8. Os cinco sinais para implementar (ordem de retorno)

1. **`nm_missing = FLIGHT_ID_mvt.isna()` como feature explícita + interação com
   `ADEP` e `ms`.** Evidência: §3. Impacto: é o eixo que o modelo de dois
   estágios explora (−70 s medido em conjunto com o item 2). Esforço: 1 linha.
   Confiança: **alta**.
2. **Dois estágios: classificador binário `eq = |BLOCK−SCHED| ≤ 60 s` + regressor
   treinado só em `~eq`, combinados pela esperança
   `ŷ = p·(MVT − SCHED) + (1−p)·ŷ_normal`, com clipe em 0.**
   Medido: 461,1 → **391,1**; melhora inclusive os voos normais (278 → 247).
   Esforço: ~40 linhas em `train.py`. Confiança: **alta** (medido).
3. **Modelo dedicado para `nm_missing`, linear em `ms` por aeroporto, não
   LightGBM.** Medido no subgrupo (teste jan+jul): constante 3.962 →
   LightGBM 2.626 → **linear por aeroporto 1.993**. Com 17 k linhas de treino
   o GBM perde para `y ≈ a + b·(MVT−SCHED)`. Esforço: baixo. Confiança: **alta**.
4. **Célula de risco explícita `LIRF ∧ nm_missing ∧ ms > 2 h`** (181 linhas no
   ranking, 43,7 % do SSE): calibrar `p̂ ≈ 0,46` e prever
   `p̂·ms + (1−p̂)·ŷ_normal`. O modelo de dois estágios já a leva de
   13.492 → 9.956, mas a regressão linear dedicada do item 3 chega a ~4.700 na
   mesma população — **é aqui que sobra a maior folga**. Esforço: baixo.
   Confiança: **alta** para 2025, **média** para 2026 (a célula existe com o
   mesmo tamanho, mas `P(eq)` nela é estimado em 397 linhas/ano).
5. **Podar o que não paga:** `round_*` (§1), features de rotação/tempo de solo
   (§6) e a média móvel de taxi-in (§6). Remover reduz ruído e tempo de treino.
   Confiança: **alta**.

### Riscos

- `P(eq)` em LIRF varia por mês (0,167 jul → 0,240 mar). Estimar por
  (aeroporto × mês × faixa de `ms`) com suavização, não global.
- O classificador de `eq` é o ponto frágil: `P(eq | ms>2h)` é 0 fora de LIRF, e
  o modelo precisa **não** inflar os 26 k voos atrasados dos outros aeroportos.
  A combinação por esperança já faz isso, mas é o que deve ser monitorado.
- Nada aqui foi enviado ao placar; tudo é validação local.
