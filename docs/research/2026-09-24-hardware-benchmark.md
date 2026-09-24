# Configuração de treino e sistema de progresso — bancada real (HardwareBench)

Tudo medido em `/tmp/prc-HardwareBench` com o **código do projeto sem modificação**
(`sys.path` + `features.load/build`). Venv própria: pandas 3.0.6, LightGBM 4.7.0,
XGBoost 3.4.1 (CUDA 13.3), CatBoost 1.2.10. Scripts: `bench.py`, `prep_full.py`,
`earlystop.py`, `runlog.py`, `ocl_test.py`; resultados em `results.jsonl` e
`experiments.jsonl`.

Topologia: `cpuN` e `cpuN+12` são irmãos HT → **6 núcleos físicos = `taskset -c 0-5,12-17`**.
Todos os testes "budget" rodaram nessa máscara, com `nice -n 5`, deixando os núcleos
6–11/18–23 livres para o usuário e para o outro agente (~1 núcleo).

---

## 1. `num_threads` padrão (24) custa 2–4× de tempo — **o maior ganho isolado**

| conjunto | 24 threads | 12 threads | 8 | 6 | 4 |
|---|---|---|---|---|---|
| 297 k linhas, 300 rodadas (máquina toda) | **69,0 s** (230 ms/r) | **16,4 s** (55 ms/r) | 22,4 s | 23,9 s | 30,0 s |
| 1,74 M linhas, 300 rodadas (6 núcleos) | 137,0 s (457 ms/r) | **69,6 s** (232 ms/r) | — | — | — |

RMSE idêntico (571,70 / 451,38) em todas as contagens de thread — é só desperdício.
A doc do LightGBM manda usar o nº de **núcleos reais**, não de threads
(<https://lightgbm.readthedocs.io/en/latest/Parameters.html#num_threads>).
Curiosamente `num_threads=12` **preso a 6 núcleos físicos** (15,2 s) foi mais rápido que
12 threads soltos na máquina toda (16,4 s): localidade de cache vence.
→ **Impacto: −50% do tempo de treino. Esforço: 1 linha. Confiança: alta.**

## 2. 1500 rodadas é desperdício: o ótimo é ~325

`lgb.early_stopping(150)` com validação jan+jul/2025 (1,74 M treino / 344 k validação):

| config | melhor rodada | RMSE ótimo | RMSE @300 | RMSE na última | treino |
|---|---|---|---|---|---|
| lr 0,05 · 255 folhas · min_leaf 100 | **325** | **451,18** | 451,51 | 452,64 (@475) | 1m40 |
| lr 0,02 · 255 folhas | 888 | 450,90 | 462,53 | 451,32 | 2m57 |
| lr 0,05 · **63 folhas** | 496 | 455,12 | 455,92 | — | **1m13** |
| lr 0,05 · 255 · min_leaf **20** | 79 | 457,57 | 460,67 | — | 1m41 |

Ou seja: as rodadas 326–1500 de hoje **pioram** o RMSE (451,2 → 452,6) e custam ~4,6× de
tempo. `lr 0,02` ganha 0,3 s de RMSE por 1,8× de tempo — não compensa agora.
`min_data_in_leaf=100` está certo (20 sobreajusta em 79 rodadas).
→ **Impacto: −4,6× no treino, −1,4 s de RMSE. Esforço: baixo. Confiança: alta** (um split).

## 3. Cache de features em parquet

12 meses: `load` 7,6 s + `build` 17,5 s, **pico de 5,09 GB de RSS**, 2.085.047 DEP.
Cache zstd = **89 MB**, releitura **0,4 s**.
→ **−25 s e −2,5 GB de pico por experimento. Esforço: trivial. Confiança: alta.**
O pico de 5,09 GB do `build` é o que manda no orçamento de RAM: gerar o cache **uma vez**,
sozinho, e nunca mais tocar nele.

Atenção: depois do cache, a fase "dados" (P10 de referência + `as_categories` + casts)
ainda leva **46–53 s** e passou a ser o maior custo fixo. Vale cachear também a matriz
já com `ref_p10` e categorias fixas quando o split não muda.

## 4. GPU: XGBoost sim, LightGBM não

Escala de produção (1,74 M linhas, 300 rodadas, 6 núcleos):

| engine | tempo | ms/rodada | RSS | CPU | GPU | VRAM | RMSE |
|---|---|---|---|---|---|---|---|
| LightGBM CPU 12 threads | 69,6 s | 232 | 3,01 GB | 973% | — | — | **451,38** |
| **XGBoost CUDA** | **37,2 s** | **124** | 3,83 GB | 168% | 57% | 1,36 GB | 487,02 |
| CatBoost GPU (depth 8) | 58,1 s | 194 | 2,92 GB | 159% | 66% | **7,53 GB** | 501,16 |
| XGBoost CPU 6 threads | 104,6 s* | 349* | 1,40 GB | 575% | — | — | 615,44* |

\* medido em 297 k linhas — XGBoost `lossguide` na CPU é inviável aqui.

XGBoost na GPU escala muito melhor (102 → 124 ms/r de 297 k para 1,74 M linhas) e gasta
**1,7 núcleo**. CatBoost GPU **reserva 7,5 GB dos 8 GB de VRAM** → bloqueia a GPU para
qualquer outra coisa (limite `gpu_ram_part` se for usá-lo).
Com parâmetros equivalentes, LightGBM na CPU ainda ganha em RMSE (451 vs 487/501); o valor
de XGB/CatBoost é **ensemble e paralelismo**, não substituição.

**LightGBM em GPU: descartado.** O wheel do PyPI tem OpenCL (`device_type='gpu'`) mas
**não** CUDA (`device_type='cuda'` → "CUDA Tree Learner was not enabled in this build").
E o OpenCL quebra com as categóricas grandes (`STAND_mvt` 1728 níveis, `ADES_mvt` 1190):
`bin size 1257 cannot run on GPU`. Removendo as três maiores categóricas ele roda e é
**2,5× mais lento que a CPU** (79 vs 32 ms/rodada, 6 threads). Compilar a versão CUDA
exigiria instalar CUDA toolkit + cmake (não há `nvcc` na máquina) e, segundo a doc, o
OpenCL só move os histogramas para a GPU
(<https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#build-cuda-version>).
→ **Não vale o esforço; use XGBoost quando quiser GPU.**

## 5. GPU + CPU em paralelo dentro do orçamento: **sim, vale**

`lgb` (6 threads em `0-5`) e `xgb` CUDA (6 threads em `12-17`), ao mesmo tempo:

| job | sozinho | junto | custo |
|---|---|---|---|
| LightGBM CPU | 23,9 s | 26,8 s | **+12%** |
| XGBoost GPU | 30,0 s | 31,1 s | **+4%** |

Dois experimentos pelo preço de 1,12. RAM somada: 2,4 GB (3 meses) / ~6,8 GB (produção —
no limite do orçamento, **não rode dois jobs de produção juntos sem cache**). Só um job de
GPU por vez, e **não** use CatBoost GPU como job paralelo (VRAM).

## 6. Parâmetros que **não** compensam

- `max_bin=63`: −12% de tempo, RMSE 620,9 vs 571,7 → **não**.
- `force_col_wise` (15,4 s) e `force_row_wise` (17,3 s) vs automático (15,2 s): o teste
  automático do LightGBM já acerta; fixar só evita ~0,2 s de sondagem.

---

## Recomendação

**Config de trabalho (padrão, dentro dos 50%):**

```bash
taskset -c 0-5,12-17 nice -n 5 .venv/bin/python src/train.py validate
```
```python
PARAMS = dict(..., num_threads=12)     # nunca deixar em 0/24
rounds = 350                            # não 1500
```
- Engine: **LightGBM CPU**, `lr 0,05`, `num_leaves 255`, `min_data_in_leaf 100`, `max_bin 255`.
- **Config leve para varredura:** `num_leaves=63`, ~500 rodadas → treino de 1m13.
- **GPU:** XGBoost `device="cuda", tree_method="hist", enable_categorical=True` — use-o em
  paralelo ao LightGBM para ensemble/experimentos, não no lugar dele.
- Cache: `data/cache/features_2025.parquet` (89 MB) gerado uma vez; invalide pelo hash de
  `features.py`.

**Tempo por experimento (produção, 1,74 M linhas):**

| cenário | tempo |
|---|---|
| hoje (24 threads, 1500 rodadas, sem cache) | **~12,5 min** |
| recomendado (12 threads, 350 rodadas, cache) | **~2,3 min** |
| varredura leve (63 folhas, 500 rodadas) | ~2,1 min |
| + XGBoost GPU em paralelo | +0 min (cabe no mesmo relógio) |

→ **5× mais experimentos por hora**, dentro de 6 núcleos e ≤ 4 GB de RSS por processo.

## Rodadas do modelo final (sem validação)

1. Rodar `validate` com `lgb.early_stopping(150)` em jan+jul/2025 → `best_iteration` (325).
2. Escalar pelo tamanho: treino final usa 2,085 M contra 1,741 M linhas → **×1,20 ≈ 390 rodadas**.
3. Quem preferir CV: `lgb.cv(..., callbacks=[lgb.early_stopping(150)])`; usar o `best_iteration`
   médio (ou o mínimo entre folds, mais conservador — discussão em
   <https://github.com/microsoft/LightGBM/issues/5683>) e escalar por `k/(k−1)`.
   Aqui o split temporal jan+jul imita o ranking melhor que CV aleatória.

## Progresso e registro: `runlog.py` (~120 linhas, zero dependência nova)

Testado nas 4 corridas acima. Saída real:

```
== early_stopping (ccfc64e40beb)
[  0%] dados...
[rec] RSS 2.2 GB (pico 2.2) CPU 112%
[  5%] dados: 46s
  rodada 100/2500 (4%) 294 ms/r restam 11m46s  valid-rmse=464.2
Early stopping, best iteration is: [325] valid's rmse: 451.179
== fim early_stopping: 2m27s | {'best_iter': 325.0, 'best_rmse': 451.179}
```

- **Fases com % e tempo** (`run.phase("carregar"/"features"/"treino"/"predição")`).
- **ETA por rodada** via callback do LightGBM (`env.iteration`, `env.end_iteration`) — dá
  ms/rodada e tempo restante; `every=50/100` evita poluir o log.
- **Recursos**: thread amostrando RSS/CPU (psutil) e `nvidia-smi` a cada 30 s.
- **Registro**: uma linha JSONL por corrida com `params`, `phases`, `metrics`, `rss_peak_gb`,
  `git` e `src_hash` (SHA-256 dos `.py` — pega mudança não commitada, que é o caso comum).
- Use `lgb.record_evaluation(hist)` para guardar a curva e `lgb.log_evaluation(period)`
  só se quiser o log nativo (<https://lightgbm.readthedocs.io/en/latest/Python-API.html>).

**tqdm: não.** Barra com `\r` polui log redirecionado e não sabe o custo por rodada; o print
periódico com ETA dá mais informação em menos ruído. **MLflow: não** — exige servidor
(`mlflow server`) e diretório `mlruns` para algo que `jq`/pandas resolvem sobre um JSONL de
algumas centenas de linhas (<https://mlflow.org/docs/latest/ml/tracking/quickstart/>).
Se um dia precisar de UI, o JSONL importa para MLflow em minutos.

## Pista secundária (baixa confiança)

Com `max_bin=63` e 3 meses, **remover** `STAND_mvt`/`ADES_mvt`/`AIRCRAFT_OPERATOR_flt`
melhorou o RMSE (609,8 vs 620,9). Pode ser ruído do split pequeno, mas sugere testar
agrupamento/target-encoding dessas categóricas de alta cardinalidade em vez de passá-las
cruas. [INFERÊNCIA] — não validado em escala de produção.
