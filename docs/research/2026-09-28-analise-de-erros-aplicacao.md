# Aplicação ao PRC 2026: o que adotar em 13 dias (28/09/2026)

Recorte prático da pesquisa geral
[`docs/pesquisa/2026-09-28-analise-de-erros-e-busca-automatica.md`](../../../docs/pesquisa/2026-09-28-analise-de-erros-e-busca-automatica.md)
(análise de erros, descoberta de fatias, caudas pesadas, AutoML, laços de busca automática —
com as fontes). Complementa `2026-09-26-praticas-ml.md` (multi-seed, empilhamento, CV que
imita o teste), que não repito aqui.

**Estado**: campeã `20260928-113219-v18_cf`, oficial **253,95 s** (26º/166; 3º lugar 224,50).
Erro² do holdout: loteria (11 voos) 37,5 %; normais (y ≤ 1 h) 49,2 %, dos quais **LIRF
normais = 14,7 % do erro² total** (RMSE 424 s), e dentro de LIRF-normais ~46 % vem de 114
voos com previsão > 1 h (hedging para "cópia do planejado").

---

## P1 — fazer agora

### P1.1 Fatia `normais × aeroporto` e parte do erro² em `metrics()`
**O que é.** `src/experiment.py:metrics()` já devolve `por_aeroporto` e `sem_loteria`, mas
não cruza os dois: não existe "normais de LIRF". Sem a fatia no dicionário, nenhum
experimento consegue ser julgado pela fatia que carrega 14,7 % do erro². Adicionar
`normais_por_aeroporto` (RMSE) e `parte_erro2_por_fatia` (fração do erro² total por fatia,
que é o "error coverage" da árvore de erro da Microsoft).
**Onde.** `src/experiment.py` (dict de `metrics`), propagado de graça para `runlog`,
`compare.py` e `.superpowers/noite/noite.py:avaliar()` (que só seleciona chaves).
**Custo.** 1 h. **Risco.** Zero (é só medição); muda o formato de `experiments.jsonl`, então
scripts que leem as chaves antigas continuam funcionando, mas os novos campos só existem
daqui pra frente.

### P1.2 Árvore de erro sobre o resíduo da campeã
**O que é.** Treinar uma árvore rasa (profundidade 3–4) para prever `|y − pred|` (ou
`(y−pred)²`) do `runs/20260928-113219-v18_cf.parquet` e ler as folhas: cada folha é uma
coorte com regra explícita. É a versão caseira do Error Analysis da Microsoft / SliceLine —
enumeração de conjunções ordenada por `Σ erro² da fatia`.
**Onde.** Script descartável no padrão de `.superpowers/noite/noite.py` (passos 2 e 3 já
fazem mineração manual de erros; isto substitui o chute por enumeração).
**Custo.** 2 h. **Risco.** Baixo; risco real é confundir *taxa* de erro com *cobertura* —
sempre ordenar por soma do erro², nunca por média.

### P1.3 Portão anti-sobreajuste jan↔jul no corretor barato
**O que é.** Hoje `noite.py:avaliar()` usa `stack.day_folds` sobre jan+jul juntos: um
candidato pode ganhar porque decorou o holdout inteiro. Adicionar um segundo veredito:
treinar o corretor **só em janeiro** e avaliar em julho, e o inverso. Aceitar candidato só
se ganha nos dois sentidos. É a defesa prática contra sobreajuste adaptativo (Ladder) e
contra deriva — a causa dominante de má generalização segundo a meta-análise de Kaggle.
**Onde.** `.superpowers/noite/noite.py:avaliar()` (parâmetro `modo="cv"|"jan2jul"|"jul2jan"`);
o veredito final continua em `src/compare.py` (ganho ≥ 10 s, IC > 0, sobrevive sem os 10
maiores voos, IC > 0 em cada mês).
**Custo.** 2 h. **Risco.** Baixo. Vai **reprovar** candidatos que hoje passariam — é o ponto.

### P1.4 Laço noturno de busca (candidatos declarados em dados)
**O que é.** Transformar o `noite.py` de "6 passos escritos à mão" em um laço
propor → rodar → avaliar → manter/descartar sobre uma **lista de candidatos**, com relatório
append-only. Detalhe do desenho na seção final.
**Onde.** Novo script em `.superpowers/noite/`, reusando `stack.corrector_frame`,
`stack.oof_correction`, `experiment.metrics`; serviço via
`systemd-run --user --unit=busca --same-dir --collect taskset -c 0-8,12-20 nice -n 5 …`.
**Custo.** 4 h para a infraestrutura + 1 h por lote de candidatos.
**Risco.** Médio: o laço maximiza exatamente a métrica que recebe. Mitigado por P1.3 e pelo
portão de fatia (aceitar só se `sem_loteria` melhora **e** `normais_nm` não piora).

### P1.5 Atacar o hedging de LIRF com feature separadora, não com regra
**O que é.** Os 114 voos LIRF com previsão > 1 h são o caso de manual: sob RMSE a previsão
ótima é a média condicional, então um alvo bimodal (sai em 15 min × "copiou o planejado")
força a previsão para o meio. As saídas legítimas são (a) dar ao modelo a feature que separa
os modos, (b) modelar explicitamente como mistura `p·m₁ + (1−p)·m₂`. Já temos o
classificador de "cópia do planejado" no `two_stage_nm`; o que falta é medir sua calibração
**dentro de LIRF** e alimentar o corretor com `p` (probabilidade) em vez de só com a
previsão composta.
**Onde.** `src/stack.py:corrector_frame` (adicionar a probabilidade do classificador como
coluna do corretor), avaliado pelo laço P1.4.
**Custo.** 3 h. **Risco.** Médio — é a maior aposta de ganho (teto ≈ 46 % de 14,7 % do
erro²), mas depende de o classificador ter sinal em LIRF; se a probabilidade for ~constante
lá, o teto cai para perto de zero e a linha morre em um experimento.

---

## P2 — se P1 render

- **Validação adversarial `holdout2025` × `ranking2026` sobre as colunas do corretor**
  (`stack.corrector_frame`). O passo 4 do `noite.py` já compara distribuições; falta o
  classificador binário com importâncias. 2 h; risco zero; serve para não levar para o envio
  uma feature que só existe em 2025.
- **Ablação dos blocos do corretor** (ADS-B, contexto, externos, plano 13): medir quanto o
  conjunto piora sem cada bloco, no avaliador barato. É o que o MLE-STAR faz para escolher
  onde refinar. 2 h; risco baixo; corta peso morto e acelera todo ciclo seguinte.
- **Geração automática de features no estilo OpenFE** sobre as colunas já existentes do
  corretor (pares com operadores aritméticos/agregações, avaliados pelo ganho no resíduo).
  O nosso avaliador barato **já é** o FeatureBoost do OpenFE: corretor pequeno sobre
  `y − base`. 4 h; risco médio (explosão de candidatos e sobreajuste — exige o portão P1.3).
- **Peso por linha em vez de troca de perda**: amortecer os voos-loteria no treino do
  corretor (não do base) para que 11 voos não moldem as árvores. 2 h; risco médio (mexe no
  que a métrica premia; testar nos dois sentidos jan↔jul).

## P3 — não fazer nestes 13 dias

- **AutoML pronto** (AutoGluon/FLAML/H2O): ganho vem de empilhar dezenas de modelos, que é
  justamente o que `stack.py` já faz no nosso formato; instalar, adaptar o holdout
  calibrado e caber no tempo de treino não cabe no prazo.
- **Optuna / busca fina de hiperparâmetros**: evidência local de ganho < 1 s, e risco
  documentado de *overtuning*. O passo 6 do `noite.py` (busca aleatória de parâmetros do
  corretor) já cobre o pouco que há.
- **Agentes de ML (AIDE / MLE-STAR / R&D-Agent) rodando o pipeline**: a ideia (busca em
  árvore + ablação) vale e está copiada no P1.4; rodar o agente de verdade custa dias de
  integração, tem variância enorme entre execuções e esbarra na regra de não usar previsão
  gerada por LLM.
- **Conformal / NGBoost**: entregam intervalo, não ponto. Só valeria como *feature de
  decisão* para o pós-processamento, e P1.5 ataca o mesmo problema mais barato.
- **log/Box-Cox no alvo**: a métrica oficial é RMSE na escala original; transformar muda a
  estatística estimada.

---

## Esboço do laço noturno

```python
# .superpowers/noite/busca.py (descartável)
CANDIDATOS = [                      # espaço declarado em dados, não em código
  {"nome": "met_vento_rajada", "tipo": "features", "gerar": lambda h: ...},
  {"nome": "peso_loteria_0.3", "tipo": "peso",     "gerar": lambda h: ...},
  {"nome": "clip_por_janela_lobt", "tipo": "pos",  "gerar": lambda p, h: ...},
]

for c in CANDIDATOS:                              # propor
    r_cv   = avaliar(c, modo="cv")                # rodar (~1 min: corretor sobre o resíduo)
    if not melhora(r_cv, REF):                    # avaliar
        registrar(c, r_cv, "descartado"); continue
    r_a = avaliar(c, modo="jan2jul")              # guarda de sobreajuste
    r_b = avaliar(c, modo="jul2jan")
    veredito = "manter" if melhora(r_a, REF_A) and melhora(r_b, REF_B) else "frágil"
    registrar(c, {"cv": r_cv, "jan2jul": r_a, "jul2jan": r_b}, veredito)

def melhora(r, ref):                              # portão por fatia, não por número global
    return (ref["sem_loteria"] - r["sem_loteria"] >= 1.0      # ganho real fora da loteria
            and r["normais_nm"] <= ref["normais_nm"] + 0.5    # não quebra os normais
            and r["LIRF_normais"] <= ref["LIRF_normais"])     # fatia alvo (P1.1)
```

**Avaliação** = `stack.oof_correction` sobre `stack.corrector_frame` (~1 min por candidato,
porque o modelo base fica congelado e só o corretor do resíduo é treinado — mesmo truque do
FeatureBoost/OpenFE). **Fatias** = dicionário de `experiment.metrics` com os campos novos do
P1.1. **Promoção de verdade** continua exigindo `src/compare.py` (bootstrap pareado por dia,
ganho ≥ 10 s, IC > 0, robusto a remover os 10 maiores voos, IC > 0 em jan e em jul) sobre um
experimento completo — o laço noturno **só produz candidatos**, nunca campeão.

**Execução**:
`systemd-run --user --unit=busca --same-dir --collect taskset -c 0-8,12-20 nice -n 5 .venv/bin/python .superpowers/noite/busca.py`
→ relatório em `docs/research/<data>-noite/`.

**Escala esperada.** 1 min por candidato × 8 h ≈ 400–500 avaliações por noite, na mesma
ordem de grandeza dos 500 experimentos que renderam um 1º lugar em Kaggle
(<https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/>).

---

## Expectativa honesta

Com 13 dias e a curva de retornos decrescentes registrada em `2026-09-26-praticas-ml.md`
(§4.3), o realista é **−2 % a −5 % por linha de ataque que traga informação nova**, e
**< 1 %** para qualquer coisa interna ao modelo. O único item do P1 com teto grande é o
**P1.5** (hedging de LIRF); os demais são infraestrutura para não desperdiçar os dias
restantes testando às cegas. Busca automática não inventa informação: ela só encontra mais
rápido o que já está nos dados — e, sem o portão jan↔jul, encontra principalmente ruído.
