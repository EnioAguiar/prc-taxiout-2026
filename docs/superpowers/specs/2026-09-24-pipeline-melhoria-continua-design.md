# Pipeline de melhoria contínua — PRC 2026 taxi-out

Data: 24/09/2026. Status: aprovado em conversa (abordagem A, partes 1–3).
Prazo da competição: 11/10/2026 23:59:59 CET.

## Objetivo

Sair de 384,7 s oficial (~132º de 188) para o mais perto possível do top 3
(~235 s), com um processo que:

1. usa o PC do jeito certo (metade da máquina, sem desperdício);
2. mostra progresso (% por fase, ETA, CPU/RAM/GPU) e registra tudo;
3. decide com evidência (comparação pareada) quando uma versão é melhor;
4. não depende de memória de conversa: estado vive em arquivos e no git.

Fora de escopo: painel web, notificação de desktop, MLflow, envio automático.

## Fatos de base (pesquisa de 24/09)

Fontes e números completos em `docs/research/2026-09-24-*.md` (literatura,
competições/placar, forense dos dados, benchmark de hardware).

- Hardware: Xeon E5-2670 v3 (12 físicos / 24 threads; `cpuN` e `cpuN+12`
  são irmãos HT), 15 GB RAM, RTX 4060 8 GB.
- LightGBM com 24 threads é 2–4× mais lento que com 12; RMSE idêntico.
- Ótimo de rodadas ~325 (lr 0,05, 255 folhas); 1500 rodadas piora 1,4 s.
- Cache de features 12 meses: 89 MB, releitura 0,4 s; montar custa pico 5,1 GB.
- XGBoost CUDA: 37 s/300 rodadas, 1,4 GB VRAM, roda em paralelo ao LightGBM
  (+12% no LightGBM). LightGBM GPU inviável (wheel sem CUDA; OpenCL quebra
  com categóricas grandes). CatBoost GPU reserva 7,5 GB de VRAM.
- O BLOCK_TIME oficial **não** é cópia de AOBT_3 (resolução de segundo vs
  minuto); é cópia do SCHED só na cauda (84,7% dos y > 3 h).
- `FLIGHT_ID_mvt` nulo (sem registro NM): 1% das DEP, 42% do SSE.
- Célula LIRF ∧ sem NM ∧ (MVT−SCHED) > 2 h: 181 linhas, 43,7% do SSE;
  o ranking 2026 tem as mesmas 181 linhas.
- 2026 tem a mesma proporção de outliers de 2025: a simulação jan+jul/2025 é
  calibrada. Relação oficial/simulação na v2: 384,7 / 460,4 = 0,836.
- Ruído: nota isolada ±56 s; diferença pareada resolve ~5 s.

## Parte 1 — Infraestrutura

### Orçamento de hardware (fixo)

- Todo processo pesado roda com `taskset -c 0-5,12-17 nice -n 5`,
  `num_threads=12`, RSS ≤ 7 GB somando os processos.
- No máximo um job de GPU por vez (XGBoost); CatBoost GPU só sozinho.
- Processos longos rodam supervisionados (sem limite de tempo de shell).

### Módulos (`src/`)

| Módulo | Responsabilidade | Interface |
|---|---|---|
| `runlog.py` | Progresso e registro | `Run(name, config)` como context manager; `run.phase(nome, peso)`; `run.lgb_callback(rodadas)`; amostra RAM/CPU (psutil) e GPU (`nvidia-smi`) a cada 30 s; ao sair (inclusive com erro) grava uma linha em `experiments.jsonl` na raiz (versionado). Log espelhado em `logs/<id>.log`. |
| `cache.py` | Cache de features | `load_split(nome)` → DataFrame; chave = hash de `features.py` + nomes/tamanhos dos parquet; guarda em `data/cache/`. Splits: `train2025` (10 meses), `holdout2025` (jan+jul montados à parte, BLOCK e alvo apagados nas DEP, verdade na coluna `y_true`), `full2025`, `ranking2026`. |
| `models.py` | Modelos com interface única | `prepare(train, outros) -> colunas`; `Modelo(cfg).fit(train, cols, run, valid) -> self`; `.predict(df) -> ndarray`; `.best_iter`. Começa com `SingleLGBM`; recebe os modelos da parte 2. |
| `experiment.py` | Experimento na simulação calibrada | `bin/run src/experiment.py <nome> --model ...`: treina em `train2025`, prevê `holdout2025`, calcula RMSE completo, sem outliers (0<y<3 h), por aeroporto e por grupo (NM presente/ausente, y>1 h); salva previsões em `runs/<id>.parquet`; grava a curva de RMSE no holdout e registra `best_iter` (mínimo da curva) sem parar o treino. |
| `compare.py` | Decisão | `bin/run src/compare.py <id_novo> [<id_base>] [--promover]`: bootstrap pareado por dia (1000 reamostras) → ganho, IC 95%, veredito. Campeão atual em `champion.json` na raiz (versionado). |
| `train.py submit N` | Versão final | Treina a config campeã em `full2025` com `rodadas = best_iter × 1,20`; gera `submissions/<time>_vN.parquet`; valida IDs contra o template. **Não envia.** |
| `s3.py submit` | Envio | Inalterado; só roda após ok do usuário. |

`sim_ranking.py` e o `validate` antigo de `train.py` são removidos
(substituídos por `experiment.py`).

### Progresso (formato)

```
== exp dois_estagios (a1b2c3d)            fase 3/4 treino
[treino  45% | total 71%] rodada 150/330 · 232 ms/r · ETA 41s · RAM 3,1 GB · CPU 970% · GPU 57%
```

Linha a cada 50–100 rodadas e a cada troca de fase; nada de barra com `\r`.

### Registro (`experiments.jsonl`, uma linha por corrida)

`id, nome, data, git_commit, src_hash, config, fases{nome: s}, rss_pico_gb,
best_iter, rmse{completo, sem_outliers, por_aeroporto, nm_presente,
nm_ausente, y_gt_1h}, previsoes_path, nota`.

## Parte 2 — Sequência de melhorias do modelo

Cada item é um experimento contra o campeão. Ordem fixa; ideias novas entram
no backlog do README com ganho esperado.

| # | Item | Evidência | Esperado (simulação) |
|---|---|---|---|
| 0 | Linha de base: modelo v2 na infraestrutura nova, 12 threads, early stopping | reproduzir ~460 | — |
| 1 | Dois estágios: `p = P(|BLOCK−SCHED| ≤ 60 s)` (classificador) + regressor L2 em `~eq`; `ŷ = p·(MVT−SCHED) + (1−p)·ŷ_normal`, piso 0 | 461 → 391 medido | −70 |
| 2 | Features `nm_missing` e `ms = MVT−SCHED` explícitas | parte do −70 | junto do 1 |
| 3 | Modelo linear por aeroporto em `ms` para `nm_missing` | subgrupo 2.626 → 1.993 | −10 a −30 |
| 4 | Célula LIRF ∧ nm_missing ∧ ms > 2 h com `p` calibrado por aeroporto × mês × faixa de ms | 43,7% do SSE | maior folga |
| 5 | Alvo residual `y − (MVT − AOBT_3)` no regressor normal | equipes 268–278 | incerto |
| 6 | Features de vizinhos: referência NM das DEP na mesma pista/stand antes/depois; pousos recentes | −1 a −6 por família | −3 a −10 |
| 7 | Ensemble: XGBoost CUDA em paralelo + 3–5 seeds LightGBM; pesos por OOF | vencedores 2024/25 | −1 a −3% |
| 8 | METAR (IEM): degelo, trovoada, LVP, vento, com interação por aeroporto | 1,8% do ganho num concorrente | −0 a −8 |
| 9 | Poda: `round_*`, features sem ganho | forense §1 | tempo/ruído |

Regras de modelagem: L2 no alvo bruto; sem log, Huber ou corte de outliers
(provado que piora); combinar por esperança, nunca argmax.

## Parte 3 — Ciclo de melhoria contínua

1. Pegar o próximo item do roadmap.
2. `experiment.py` (~2 min) → linha em `experiments.jsonl`.
3. `compare.py` contra o campeão.
   - Ganho ≥ 10 s e IC 95% acima de 0 → novo campeão (`champion.json`).
   - Senão → roadmap marca "descartado: <motivo>".
4. Novo campeão → `train.py submit N` → mostrar ao usuário: simulação,
   ganho ± IC, nota oficial projetada (× relação atual) → **esperar ok**.
5. Com ok: `s3.py submit`, ler `_result.json`, registrar nota oficial e a
   relação oficial/simulação no README.
6. Commit + push a cada passo concluído.

Limites: ≤ 3 envios/dia; nunca enviar para sondar o placar. Se a relação
oficial/simulação sair de 0,84 ± 0,05, parar e investigar antes de seguir.

Antes de 11/10: README com reprodução, repositório público GPLv3.

## Parte 4 — Dados externos: adsb.lol (adendo de 26/09)

Decisão de 25/09, depois de os planos 3b (itens 2–4) não moverem a nota e de o
Discord do desafio mostrar que times do topo usam adsb.lol (organizador: dado
aberto declarado é permitido; o modelo é pós-operação).

- Fonte: `adsblol/globe_history_2025` e `_2026` no GitHub (ODbL 1.0), um arquivo
  por dia (2–4 GB), lista de réplicas em `PREFERRED_RELEASES.txt`.
- Armazenamento: só o recorte perto dos 10 aeroportos (`src/adsb.py`), parquet
  zstd no SSD `/mnt/c0399cd8-…/prc-adsb/cut/`, ~7–19 MB por dia; bruto apagado.
- Uso no modelo: features `adsb_*` por `MVT_ID_mvt` (NaN sem cobertura); o
  classificador de cópia continua decidindo a cauda (o ADS-B vê o off-block real,
  não a cópia do SCHED).
- Processo mais barato: **teste barato antes de treino** (script sobre as
  previsões salvas; treino só se o teto ≥ 10 s); download/processamento como
  serviço `systemd-run --user`, fora da sessão; sem revisor para experimento
  descartado.
- Documentar a fonte no README (condição do prêmio).

Plano: `docs/superpowers/plans/2026-09-26-plano4-adsb.md`.

Estado em 26/09 (noite): 427 dias baixados (2025 inteiro + jan/jul 2026). Teto barato
(empilhamento fora do fold) −15,4 s → segue para features no modelo. Possível etapa
final oculta (Discord): o pipeline adsb → eventos → features → modelo precisa rodar
para qualquer período com um comando.

## Parte 5 — v7: corretor empilhado com cross-fitting por mês (adendo de 27/09)

Decisão de 27/09 (rota B, escolhida pelo usuário): o corretor do `stack.py` (317,57 na
simulação contra 323,50 da campeã) só era treinável no holdout; para enviar, a base
precisa de previsões fora do bloco no ano inteiro.

- **Blocos:** meses consecutivos 2 a 2, derivados dos meses presentes nos dados (nenhum
  mês fixo no código). `train2025` → {2,3} {4,5} {6,8} {9,10} {11,12}; `full2025` →
  {1,2} … {11,12}. Um mês cai em exatamente um bloco.
- **Split `blind2025`:** os 12 meses montados como o ranking (`build_blind`: BLOCK e alvo
  apagados nas DEP antes das features, verdade em `y_true`). As previsões fora do bloco
  são feitas nessas linhas, nunca nas de treino.
- **Sem vazamento:** em cada bloco, `prepare(treino_do_bloco, [cegas_do_bloco])` refaz a
  referência P10 só com os meses de treino do bloco.
- **Simulação** (`stack.py <nome> --crossfit`): previsões fora do bloco nos 10 meses do
  `train2025` (config da base sem escala de rodadas) → corretor em `y − pred` → aplicado
  às previsões da corrida base no holdout (`runs/<base>.parquet`, padrão campeã). Grava
  `runs/<id>.parquet`, `runs/<id>_oof.parquet` e a linha no `experiments.jsonl` com
  `model: "stack_cf"` e `base_config`; decide pelo `compare.py` como qualquer corrida.
- **Envio** (`train.py submit N` com campeã `stack_cf`): previsões fora do bloco no
  `blind2025` (6 blocos, rodadas sem escala) → corretor nos 12 meses → base final no
  `full2025` (rodadas × 1,2) prevê o ranking → corretor → piso 0. Só gera o arquivo.
- `stack.py <nome>` sem `--crossfit` continua como teste barato (corretor fora do fold só
  no holdout).
- Limites: RSS ≤ 7 GB; um treino pesado por vez; `teto.py` contra a v6 e ok do usuário
  antes de enviar.

Plano: `docs/superpowers/plans/2026-09-27-plano5-v7-crossfit.md`.

## Parte 6 — seeds e janela do LOBT (adendo de 27/09)

- Seeds (`--seeds N`, `models.build_model`/`SeedAvg`): v7 + 5 seeds = 320,36 contra 320,67,
  5× o custo → descartado (código fica; `seeds` ausente = comportamento antigo).
  Plano: `docs/superpowers/plans/2026-09-27-plano6-seeds.md`.
- Janela do LOBT: |BLOCK − LOBT| ≤ 3606 s em 100 % das DEP com LOBT (regra do dado).
  `--janela-lobt` projeta toda previsão em `MVT − LOBT ± 3606`, zera `p` da cópia quando o
  SCHED está fora da janela e dá ao corretor `dist_lo`/`dist_hi`. Oficial: v9 = 275,90,
  v10 (só projeção da v6) = 284,17. Plano: `docs/superpowers/plans/2026-09-27-plano7-janela-lobt.md`.
- Lição de validação: o holdout jan/jul 2025 quase não tem previsões fora da janela e tem
  pouca cobertura ADS-B em LEMD/EGLL; ganhos que dependem da distribuição de 2026 só
  aparecem no placar. Regras exatas do dado (verificadas em 100 % do treino) podem ser
  enviadas com a garantia calculada no ranking (projeção em conjunto convexo).

## Parte 7 — contexto e conjunto de corretores (adendo de 27/09, noite)

- `src/contexto.py` (plano 8): colunas `ctx_*` só no corretor, a partir dos movimentos brutos
  do mesmo período (taxi-in das ARR, última chegada no stand, média de `MVT − AOBT_3` das DEP
  vizinhas antes e depois). Nunca lê alvo/BLOCK de DEP. Oficial: v11 = 266,81.
- `--conjunto` (plano 9): média de LightGBM global, LightGBM por aeroporto e CatBoost no
  corretor. Oficial: v12 = 264,74.
- Descartados por teste barato ou medição: seeds, detector de pushback, fila ADS-B, CatBoost
  sozinho, média mensal oficial (exclui voos de degelo).
- Limite do placar: 5 envios por dia **UTC**; o `_result.json` de envio recusado traz
  `DAILY_LIMIT_REACHED`.

## Verificação

- Infra: `experiment.py baseline` reproduz 460 ± 5 s; tempo total ≤ 3 min;
  RSS pico ≤ 4 GB; log mostra fases, %, ETA e recursos; linha no JSONL.
- `compare.py` de um experimento contra ele mesmo → ganho 0, IC contendo 0.
- `train.py submit` gera arquivo com os 344.841 IDs do template, sem nulos.
- Cada item da parte 2: registrado no JSONL e no roadmap, com veredito.
