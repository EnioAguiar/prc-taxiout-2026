# Práticas de ML de competição aplicáveis à campeã (26/09/2026)

Pesquisa sobre o que times vencedores de competições **tabulares** fazem, o que
já fazemos, e o que cabe nos 15 dias restantes (prazo 11/10). Referência:
oficial **314,76 s** (v6), topo do placar **234,1 s** — a distância é de 25,6 %
relativos. 1 % ≈ 3,1 s.

## 1. Inventário do ambiente (`.venv/bin/pip list`, 26/09)

Instalado e relevante: **lightgbm 4.7.0**, numpy 2.5.3, pandas 3.0.6,
pyarrow 25.0.1, scipy 1.18.1, psutil, minio, pytest 9.1.1.

**Não instalado** (e o que cada um destravaria):

| Falta | Para quê | Vale instalar? |
|---|---|---|
| `scikit-learn` | isotônica, KNN, RFE, `NearestNeighbors` (join METAR do vencedor 2024) | sim, barato; hoje tudo é feito à mão com numpy |
| `xgboost` (CUDA) | 2º modelo para stacking/blend | **não** — no Discord ganhou peso zero |
| `catboost` | 3º modelo; *ordered target statistics* sem vazamento | talvez, só se sobrar tempo |
| `optuna` | busca de hiperparâmetros | **não** — tuning rende < 1 s aqui |
| `shap` | atribuição por voo/fatia para escolher o próximo experimento | sim, é diagnóstico, não modelo |
| `cudf-cu12` / `cuml` (RAPIDS) | feature engineering em massa na RTX 4060 | avaliar: 8 GB é apertado para 2,1 M linhas × centenas de colunas |
| `polars` | leitura/agregação mais rápida que pandas em CPU | opcional |

Nada foi instalado nesta tarefa.

## 2. Práticas, evidência e aplicabilidade

Legenda de prioridade: **P1** (fazer nos próximos 3 dias), **P2** (se P1 render),
**P3** (não fazer agora).

### 2.1 Validação que imita o teste
Estrutura da CV = estrutura do conjunto de avaliação (TimeSeriesSplit/GroupKFold,
não KFold aleatório).
Fonte: <https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data>
— *"Match your CV strategy to how the test data is structured"*.
**Já fazemos**: `experiment.py` simula jan+jul/2025 com o alvo apagado como no
oficial; `compare.py` faz bootstrap pareado por dia e exige IC > 0 em jan e jul
separados. Custo: zero. Ganho: já capturado. **P1 (manter)**.

### 2.2 Adversarial validation
Classificador binário treino×teste; AUC ≈ 0,5 = sem shift, AUC alta = shift, e
as features mais importantes do classificador são as que derivam.
Fonte: <https://blog.zakjost.com/post/adversarial_validation/> e
<https://arxiv.org/pdf/2112.10078> (método formalizado, passo a passo).
**Não fazemos.** Custo: ~10 min (um LightGBM binário `train2025` vs `ranking2026`
sobre as 58 features do cache). Ganho: [INFERÊNCIA] 0 s direto, mas é o único
jeito barato de descobrir quais features **não** sobrevivem a 2026 — e, com a
possível etapa final oculta em outro período, é seguro de generalização.
**P1**.

### 2.3 Média de múltiplas seeds (multi-seed / bagging de modelos)
Treinar o mesmo modelo com N seeds e promediar as previsões.
Fonte **do próprio domínio**, vencedor do PRC 2024 (team_likable_jelly, ENAC):
<https://github.com/PRC-Data-Challenge-2024/team_likable_jelly> — 1 seed
**1.612,41** kg → 10 seeds **1.564,09** (−3,0 %) → 20 seeds **1.561,63**
(−0,16 % adicional). Confirmado pelo playbook NVIDIA (técnica 7: 100 seeds de
XGBoost > seed única).
**Não fazemos**: `champion.json` fixa uma seed (`--seed N`, `deterministic=True`).
Custo: v3 leva 5m23s no `train.py submit`; 10 seeds ≈ 55 min de CPU, mais nada
(sem RAM extra, roda em série). Ganho: [INFERÊNCIA] 1–3 % ≈ **3–9 s**, com a
curva do 2024 dizendo que 8–10 seeds pegam quase tudo.
**P1 — melhor razão ganho/risco do documento.**

### 2.4 Retreinar em 100 % dos dados
Depois de fixar hiperparâmetros/rodadas na CV, refazer o modelo final no dataset
inteiro. Fonte: playbook NVIDIA, técnica 7.
**Já fazemos**: `train.py submit` usa `full2025` (2,085 M linhas) com
`best_iter × 1,2`. Custo/ganho: já capturado. **—**

### 2.5 Feature engineering em massa (`groupby(A)[B].agg(STAT)`)
Gerar centenas/milhares de agregações e deixar o modelo escolher.
Fonte: <https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/>
(>10.000 features geradas, as 500 melhores deram 1º lugar) e o vencedor 2024, que
criou **centenas** de features por fatia de altitude e deixou o LightGBM podar.
**Parcial**: temos 58 features feitas à mão (P10 por aeroporto×stand×pista,
congestionamento em 4 janelas, diferenças entre horários, `adsb_*`).
Custo: em pandas/CPU, cada bloco de ~50 features novas custa ~10–20 min de cache
+ 4 min de treino; com `cudf` seria minutos, mas exige instalar RAPIDS e caber em
8 GB. Ganho: [INFERÊNCIA] 1–2 % se houver eixo novo (ex.: agregações por
`AIRCRAFT_OPERATOR_flt` × pista × hora, ou estatísticas do taxi-out dos vizinhos
imediatos). **P2**.

### 2.6 Target encoding sem vazamento
Codificar categóricas pela média do alvo, calculada **fora do fold** (nested CV)
ou por *ordered target statistics*.
Fontes: <https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/>
(*"When COL2 is the target... we use nested cross-validation to avoid leakage"*) e
CatBoost, <https://arxiv.org/abs/1706.09516>.
**Parcial**: a referência P10 por (aeroporto, stand, pista) já é um target
encoding calculado só no fold de treino — o padrão certo. Falta TE em eixos
maiores (operadora, tipo de aeronave × pista, hora×pista).
Custo: baixo, o esqueleto existe em `features.py`. Ganho: [INFERÊNCIA] 0,5–1,5 %.
**P2**.

### 2.7 Stacking com previsões fora-do-fold
Modelo de 2º nível sobre as OOF do 1º nível (ou sobre os resíduos).
Fontes: playbook NVIDIA (técnica 5) e
<https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/>
— 500 modelos experimentais, 75 no nível 1, 3 níveis, 1º lugar.
**Já fazemos**: `src/stack.py` (corretor sobre OOF por dia) leva 332,86 → 317,57
na simulação, veredito MELHOR. **Bloqueado**: o corretor só existe para jan/jul,
então a v7 exige base treinada nos 10 meses restantes.
Custo: 2 treinos (base 10 meses + corretor) ≈ 15 min. Ganho: já medido **−4,6 %**
na simulação; oficial [INFERÊNCIA] −3 a −5 % ≈ 10–15 s pela razão
oficial/simulação de 0,97 da v6. **P1 — é o maior número já medido na bancada.**

### 2.8 Hill climbing de pesos do ensemble
Partir do melhor modelo e ir somando modelos/pesos que melhoram a OOF.
Fonte: <https://www.kaggle.com/code/cdeotte/gpu-hill-climbing-cv-0-05930> (1º
lugar em *Predict Calorie Expenditure*).
**Não fazemos.** Contraevidência local: no Discord do PRC 2026, pesos de blend
ajustados no holdout **perderam 4 de 4** contra pesos iguais, e XGBoost recebeu
peso zero. Custo: baixo (numpy). Ganho: [INFERÊNCIA] ~0 com 2 modelos; só faz
sentido com ≥ 5 modelos realmente diversos, que não temos.
**P3** — se usar ensemble, usar **média simples de seeds** (2.3).

### 2.9 Baselines diversos / famílias de modelo diferentes
Fonte: playbook NVIDIA, técnica 2.
Contraevidência local: tuning e 3º modelo não moveram o placar de ninguém no
Discord; o vencedor 2024 usou **só LightGBM** em 20 seeds. **P3**.

### 2.10 Pseudo-labeling
Rotular o conjunto sem alvo com o melhor modelo e retreinar junto.
Fonte: playbook NVIDIA, técnica 6 (com o alerta: em k-fold, gerar k conjuntos de
pseudo-rótulos para não vazar).
**Não fazemos.** Aplicável: o `ranking.parquet` tem as features de jan/jul 2026
sem alvo. Custo: 2 treinos ≈ 12 min. Ganho: [INFERÊNCIA] 0–1 %; em regressão com
cauda pesada, pseudo-rótulos reforçam o viés do modelo justamente na cauda que
domina nosso erro² (32 % vem de 11 voos "loteria"). Risco real de piorar.
**P3 nesta competição**, mas o primo útil dele é *transdutivo*: features
calculadas **dentro** do ranking (congestionamento, vizinhos), que já usamos.

### 2.11 Tratamento de shift temporal
Checar distribuições treino×teste e tendência/sazonalidade do alvo antes de
modelar. Fonte: playbook NVIDIA, técnica 1.
**Parcial**: `docs/research/2026-09-24-diagnostico-v4.md` já fez leave-one-month-out
e comparou cobertura 2025×2026. Contraevidência local: pesar linhas por
(aeroporto, mês) não funcionou para outro time. **P2** (só como diagnóstico, via 2.2).

### 2.12 Reparametrizar o alvo
Prever uma razão/resíduo em vez do alvo cru (vencedor 2024:
`(TOW−EOW)/(MTOW−EOW)` com peso `(MTOW−EOW)²`; Deotte: alvo, razão, resíduo e
imputação em paralelo).
**Já testado e descartado**: item 4 do plano 3b (`residual_aobt`, `residual_feat`)
empatou com a campeã porque `ref` já é feature. **P3** — a variante que ainda não
testamos é **peso por linha**, não reparametrização.

### 2.13 Ablação de features / poda
Medir quanto o ensemble piora removendo cada bloco.
Fonte: team_tiny_rainbow (3º em 2024), paper com estudo de ablação:
<https://github.com/PRC-Data-Challenge-2024/team_tiny_rainbow>.
**Não fazemos** (item 8 do roadmap). Custo: 1 treino por bloco (~4 min × 8 blocos).
Ganho: [INFERÊNCIA] 0–0,5 % direto, mas corta features que não generalizam — vale
pela etapa final oculta. **P2**.

### 2.14 Gestão de envios: placar público × final
O PRC **já fez isso em 2025**: fase 2 com `fuel_final_submission.parquet` separado,
com a justificativa explícita *"to avoid the possibility of leaning from the
leaderboard results"* —
<https://ansperformance.eu/study/data-challenge/dc2025/ranking.html>.
Em 2026 o organizador falou em "possibly a 1 final submission" com formato ainda
não decidido.
**Já fazemos em parte**: decidimos por `compare.py`/`teto.py` antes de enviar.
Regras que faltam formalizar: (a) nunca usar o placar como validação — no máximo
como verificação da razão oficial/simulação; (b) **1 envio/dia**, não 5; (c) o
pipeline inteiro (adsb.lol → `adsb_events.py` → cache → modelo → parquet) tem que
rodar em outro período com um comando. **P1**.

## 3. Top 5 para fazer agora

1. **v7 = empilhamento enviável** (`stack.py` com base treinada nos 10 meses fora de jan/jul e corretor no holdout inteiro): é o único ganho de −4,6 % já medido na bancada.
2. **Média de 8–10 seeds no modelo final**: única evidência in-domain quantificada (PRC 2024: −3,0 %), custa ~1 h de CPU e não muda uma linha de modelagem.
3. **Adversarial validation `train2025` × `ranking2026`** sobre as 58 features: diagnóstico de 10 min que diz quais features quebram em 2026 e protege contra a etapa final oculta.
4. **Reprodutibilidade de ponta a ponta em um comando** (download ADS-B → eventos → cache → submit para um período arbitrário), porque a etapa final pode exigir rodar em outro período/aeroportos.
5. **Ablação dos 8 blocos de features + poda**: remove o que não generaliza e barateia todos os ciclos seguintes; só depois disso, feature engineering em massa (2.5/2.6).

Não entram: tuning (< 1 s para todos), XGBoost/CatBoost como 3º modelo (peso
zero no Discord), pesos de blend ajustados (perderam 4/4), pseudo-labeling
(risco na cauda).

## 4. Como melhorar a campeã continuamente (ganho de X % por ciclo)

### 4.1 Quanto cada tipo de mudança costuma render (números de writeups)

| Tipo de mudança | Ganho relativo observado | Fonte |
|---|---|---|
| **Fonte de dados nova** (ADS-B, METAR, física) | −4,9 % (nosso v5→v6, 331,0 → 314,76); vencedor 2024 = trajetória + OpenAP + METAR, 1.561 contra 2.695 de quem usou só o básico (−42 %) | `README.md`; <https://conormclaughlin.net/2024/12/reviewing-the-winning-ml-models-from-the-opensky-prc-data-challenge-what-did-they-do-differently/> |
| **Feature engineering em massa** sobre fonte existente | 1º lugar com 500 das 10.000 features geradas; ordem de 1–3 % típica | <https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/> |
| **Stacking / 2º nível sobre OOF** | −4,6 % na nossa bancada (332,86 → 317,57); 1º lugar com stack de 3 níveis | `src/stack.py`; <https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/> |
| **Multi-seed** | −3,0 % com 10 seeds; +0,16 % ao dobrar para 20 | <https://github.com/PRC-Data-Challenge-2024/team_likable_jelly> |
| **Pós-processamento / regra por fatia** | −2,3 % (nossa v4→v5, 338,7 → 331,0) | `README.md` |
| **Tuning de hiperparâmetros** | < 1 s (≈ 0,3 %) para vários times do PRC 2026 | Discord `#prc-data-competition`, resumo em `README.md` |

Leitura: **fonte nova > stacking ≈ multi-seed > FE > pós-processamento > tuning**.
O tuning é a única linha com evidência local de ganho ~zero.

### 4.2 Como escolher o próximo experimento (em vez de tentar às cegas)

1. **Decompor o erro² por fatia antes de propor qualquer coisa.** Já temos isso:
   voos normais com NM 43 % do erro², alarmes falsos 10 %, cauda-cópia 10 %,
   11 voos "loteria" 32 %. O teto de uma linha de ataque é a parte do erro² da
   fatia que ela pode zerar — foi assim que o plano 3b matou três ideias.
2. **Checagem treino×teste (adversarial validation)** antes de confiar numa
   feature nova: <https://blog.zakjost.com/post/adversarial_validation/>.
3. **Atribuição por voo** (`shap`, ou ganho/split do LightGBM + importância por
   permutação restrita à fatia) para saber *qual* feature produz os alarmes
   falsos — o playbook chama isso de "catch when a model is failing... early".
4. **Teto antes de gastar** (`teto.py`): se o ganho simulado > 2 × teto, a
   simulação está medindo folga que o envio não tem (lição da v4: −42 s
   simulados viraram −1,5 s oficiais).

### 4.3 A curva de retornos decrescentes

Nossa própria série oficial é o melhor estimador:

| Passo | RMSE | Ganho relativo |
|---|---|---|
| v1 → v2 | 514,5 → 384,7 | −25,2 % |
| v2 → v3 | 384,7 → 338,7 | −12,0 % |
| v3 → v4 | 338,7 → 337,2 | −0,4 % |
| v4 → v5 | 337,2 → 331,0 | −1,8 % |
| v5 → v6 | 331,0 → 314,76 | −4,9 % |

Depois do conserto dos bugs grossos (v1–v3), o regime estável é **−2 a −5 % por
versão, e só quando entra informação nova**; mudanças internas ao modelo rendem
< 1 %. O mesmo formato aparece no vencedor 2024 (10 seeds −3,0 %, mais 10 seeds
−0,16 %): dentro de uma linha de ataque o ganho cai em ordem de grandeza a cada
dobra de esforço. Estimativa do próximo passo = **min(teto da fatia, ganho médio
da categoria da tabela 4.1) × 0,85** (nossa razão oficial/simulação recente).

### 4.4 Cadência

O playbook NVIDIA é explícito: o maior alavanca é *o número de experimentos de
qualidade por unidade de tempo*
(<https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data>);
Deotte selecionou 75 modelos de nível 1 **a partir de 500 experimentos**. Nossa
bancada: experimento de holdout 3–6 min, `submit` 5–6 min → **8–12 experimentos/dia**
é factível com `bin/run` em 6 núcleos. Envio: 5/dia são permitidos, mas a
organização monitora engenharia reversa do placar e já criou fase final separada
em 2025 — **1 envio/dia, só com veredito MELHOR (ou FRÁGIL + `teto.py` + ok do
usuário)**.

### 4.5 Ciclo proposto para nós (48 h por ciclo, ~7 ciclos até 11/10)

1. **Diagnóstico (30 min):** rodar a decomposição por fatia da campeã e escolher a
   fatia com maior `parte_erro2` ainda não atacada. Escrever o **teto** no plano.
2. **Hipótese única (1 linha) + custo estimado.** Se o teto < 1 % do RMSE, não
   começar.
3. **3 a 6 experimentos** na simulação, todos gravados em `experiments.jsonl`
   com seed fixa.
4. **Decisão** por `compare.py` (bootstrap pareado por dia; MELHOR exige ganho
   ≥ 10 s, IC > 0, ganho fora dos 10 maiores voos, IC > 0 em jan e jul).
5. **`teto.py`** contra o envio atual; enviar no máximo 1 arquivo/dia.
6. **Registrar no README** o ganho relativo realizado e a razão realizado/teto.

- **Meta por ciclo:** **−1,5 % a −3 %** (≈ 5 a 10 s) enquanto houver fonte ou
  fatia nova; **−0,5 % a −1 %** quando o ciclo for só multi-seed/stacking.
  Alvo acumulado em 7 ciclos com −2 % médio: 314,76 → ≈ **273 s**.
- **Critério de parada de uma linha de ataque:** abandonar quando **dois
  experimentos consecutivos** dela derem IC 95% contendo zero, **ou** quando o
  ganho medido ficar abaixo de **30 % do teto** calculado no passo 1, **ou**
  quando o custo do próximo passo dobrar para um ganho previsto < 1 %. Foi o que
  já aconteceu (e foi registrado) em `cal_celula`, `nm_hibrido`, `copy_mix` e
  `residual_aobt`.
- **Como medir:** ganho relativo `(RMSE_base − RMSE_novo)/RMSE_base` com IC
  pareado por dia; ganho por fatia (`normais_nm`, `alarmes_falsos`,
  `cauda_copia`, `sem_loteria`); razão **realizado/teto**; e, a cada envio, a
  razão **oficial/simulação** (0,97 na v6) — se ela cair, a simulação voltou a
  medir folga inexistente e a regra de promoção precisa endurecer de novo.
