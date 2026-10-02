# Pseudo-rotulagem transdutiva com as partidas de 2026 (`--pseudo`)

Item 4 de `docs/pesquisa/2026-10-02-tecnicas-descoberta.md`: pôr as linhas do
`ranking.parquet` (jan e jul de **2026**, sem rótulo) no **treino do corretor**, com alvo
vindo de um modelo e peso menor. O que se compra não é rótulo — é **cobertura**: o
corretor passa a ver a distribuição de covariáveis de 2026 (cobertura ADS-B diferente,
dias de ruptura de jan/26), que é o problema declarado da deriva, e sai do forno um membro
menos correlacionado para a média.

Nada de verdade de 2026 é usado em lugar nenhum: o alvo das linhas de 2026 é sempre uma
**previsão**.

## 1. O que entrou no código

`--pseudo propria|campea` no `src/stack.py` (só com `--crossfit`), mais
`--pseudo-peso` (padrão 0,3) e `--pseudo-corte` (padrão 3.600 s). A mecânica está em
`src/pseudo.py` e vale igual na simulação (`stack.py`) e no envio (`train.py`), para o
membro medido e o membro enviado serem o mesmo objeto.

1. **Previsão da base para 2026** (`pseudo.base_2026`): um ajuste da `base_config` no
   `train2025` — os **10 meses sem jan/jul** — prevendo as 344.841 partidas do ranking.
   Fica em `data/cache/pseudo/base-<chave>.parquet`; a chave junta a config da base e os
   arquivos que a determinam (`models.py`, `crossfit.py`, `features.py`, `cache.py`), no
   espírito do `train.chave_oof`. Custo de uma vez por base; depois é leitura.
2. **Alvo das linhas de 2026** (`fonte`):
   - `propria`: a previsão do **próprio corretor desta corrida** (treinado nas cegas dos
     10 meses) sobre aquela base. Nenhum dos dois viu rótulo de jan/jul de 2025.
   - `campea`: a previsão da campeã no último arquivo de `submissions/` (hoje
     `outgoing-boat_v37.parquet`). Aquele modelo treinou no `full2025`, **com** jan e jul.
3. **Filtro do corpo** (`pseudo.corpo`): entram só alvo ≤ 3.600 s, voo com plano NM
   (`nm_missing == 0`), previsão da base guardada para aquele voo (sem ela o corretor
   aprenderia `alvo − NaN` e levaria o ajuste inteiro junto) e — quando a receita tem
   `--corretor-sem-regra` — fora das linhas que a base entrega à reta dos sem-NM. A cauda
   é onde o pseudo-rótulo é pior (ali a previsão já é hedge entre táxi e cópia do
   planejado) e é dona de metade do erro²; pseudo-rotulá-la só reforçaria o hedge.
4. **Peso e junção** (`pseudo.juntar`): as linhas de 2026 vão para o fim do quadro com
   peso 0,3 (as de 2025 ficam com 1) e o `fit_corrector` passa esses pesos aos quatro
   motores (LightGBM global, LightGBM por aeroporto, CatBoost, XGBoost). As categóricas de
   2026 são reescritas no vocabulário de 2025 (`alinhar_categorias`): o LightGBM lê
   categórica pelo **código**, e concatenar dois vocabulários trocaria os códigos das
   linhas de 2025 — e com eles o que o corretor prevê no holdout.

O holdout continua sendo medido do mesmo jeito, nas mesmas linhas: **as linhas do holdout
não entram no treino**, só as de 2026. Não há transdução sobre o conjunto de avaliação.

Quanto isso pesa, medido nos dados de hoje com a previsão da campeã v37:

| | |
|---|---|
| partidas no ranking 2026 | 344.841 (jan/26: 152.719 · jul/26: 192.122) |
| sem plano NM | 1,53 % |
| previsão > 3.600 s | 0,36 % |
| **corpo (entram no treino)** | **338.675 (98,2 %)** |
| peso delas no treino do corretor | 0,3 × 338.675 / (1.740.628 + 0,3 × 338.675) = **5,5 %** |

Vale registrar que o corte de 3.600 s quase não corta (0,36 %): o filtro que realmente
morde é o `nm_missing`. O corte está lá como trava contra o dia em que o professor
alucinar cauda, não porque hoje ele remova muita coisa.

## 2. O vazamento: por que a fonte `campea` não pode decidir no holdout

O ranking de 2026 é de **janeiro e julho**. O holdout de 2025 é de **janeiro e julho**. A
campeã que gerou o `v37.parquet` treinou no `full2025` — **inclusive jan e jul de 2025**.
Então, com `--pseudo campea`, o caminho existe:

> rótulos de jan/jul de 2025 → campeã (`full2025`) → previsão em jan/jul de **2026** →
> alvo de 5,5 % do peso do treino do corretor → previsão do corretor em jan/jul de **2025**
> → RMSE do holdout.

Nenhum voo do holdout é copiado: o que atravessa é a **função** que a campeã ajustou com
ajuda dos rótulos do holdout. Dois canais, medidos:

**Canal 1 — células que só existem no holdout.** Se uma célula (aeroporto, stand, pista)
aparece em jan/jul de 2025 e **não** nos 10 meses de treino, tudo o que a campeã sabe
dela veio de rótulo do holdout; se a mesma célula aparece em 2026, o aluno recebe esse
conhecimento de bandeja. Medido nos parquets brutos:

| chave | células só do holdout | linhas de 2026 nelas |
|---|---|---|
| aeroporto × stand | 48 | 39 (0,01 %) |
| aeroporto × stand × pista | 302 | 79 (0,02 %) |

Canal desprezível: 10 aeroportos e um ano inteiro de treino cobrem praticamente todas as
células que 2026 usa.

**Canal 2 — a estrutura de nível de jan/jul.** Este é o que importa. O táxi médio das
partidas é **1.012,1 s nos meses do holdout contra 987,1 s nos 10 meses de treino
(+25,1 s)**. Um professor que viu jan/jul de 2025 carrega essa estrutura para jan/jul de
2026; o aluno a aprende nas covariáveis de 2026 e a aplica no holdout de 2025.

Quanto isso pode valer? O teto é o que se ganharia **removendo do holdout o viés do
próprio holdout** (o limite do que esse canal consegue transportar, já que a campeã não
pode saber mais do que os rótulos do holdout dizem). Com os resíduos da campeã v37 nos
344.419 voos do holdout:

| condicionamento | grupos | teto de ganho no RMSE completo |
|---|---|---|
| global (um viés só) | 1 | **0,02 s** |
| por aeroporto | 10 | **0,08 s** |
| por aeroporto, só o corpo (`y ≤ 3600`) | 10 | **0,29 s** |
| por aeroporto × hora | 228 | **0,41 s** |

Para calibrar: o ganho que estamos perseguindo por membro é de **0,5 a 1,5 s simulados**,
e a v37 — que simulava −0,77 s — valeu **−0,02 s oficiais**. Um canal que pode fabricar
0,1 a 0,4 s de ganho fantasma no holdout é, na nossa escala, metade da medida. E os tetos
acima são **in-sample** e param no aeroporto × hora; condicionando mais fino (stand,
pista, célula do METAR) o teto só sobe.

Há ainda a confirmação empírica no doc de 02/10 (seção 0): aplicar na outra metade o viés
por aeroporto medido numa metade vale +0,07 s, e por aeroporto × hora **piora**
(−0,37/−0,26). Ou seja, esse viés é em boa parte ruído do próprio holdout — exatamente o
tipo de coisa que um professor contaminado transporta e que não existe no oficial.

**Decisão.** `--pseudo campea` fica no código, mas grava `pseudo_vazado: true` na config
da corrida e **o número de holdout dela não decide nada** — nem régua, nem promoção. Quem
quiser usá-la tem que aceitar enviar às cegas. O candidato que foi para a fila usa
`--pseudo propria`.

**Por que `propria` é limpa.** A base de 2026 é ajustada no `train2025` (10 meses, sem
jan/jul) e o professor é o corretor desta mesma corrida, treinado nas cegas dos 10 meses
de treino (o `oof_base` só prevê os meses presentes no `train`, então jan/jul nunca entram
no treino do corretor) com as tabelas auxiliares — taxa de cópia, células do `refcel`,
modelos do `roma` — todas ajustadas nos mesmos 10 meses. Nenhum rótulo de jan/jul de 2025
toca em nada. O holdout segue valendo como medida.

## 3. O que o holdout pode dizer sobre `--pseudo propria` (e o que não pode)

Pode dizer **se faz mal**. Não pode dizer se faz bem: o benefício alegado é cobertura das
covariáveis de **2026**, e o holdout é 2025. Por construção, o melhor resultado possível
no holdout é "empate".

Por isso o critério de entrada deste candidato **não** é o ganho sozinho, e sim o do doc de
02/10 (seção 4, e a conclusão 4 da seção 0):

1. não piorar o holdout — ganho ≥ 0 ou IC pareado por dia contendo 0, nas duas metades da
   régua; e
2. **ρ do resíduo contra a campeã < 0,97** (a mediana da biblioteca hoje é 0,987, e sobre
   167 corridas recentes `corr(ρ, ganho ao entrar como 4º membro) = −0,53`).

Empatar sozinho com ρ baixo é o resultado que vale: é o que `e113_m1` fez e rendeu o 4º
membro.

### Protocolo de relato (lição da v37)

Qualquer ganho desta frente é relatado com as três quebras, sobre o holdout da campeã
(294,86 completo hoje):

| fatia | n | RMSE da campeã | viés | fatia do erro² |
|---|---|---|---|---|
| `y ≤ 3600` | 343.438 | 203,11 | +7,73 | 47,3 % |
| `y > 3600` | 981 | 4.010,16 | −1.505,80 | 52,7 % |
| 20 piores voos | 20 | — | — | **43,0 %** |

Ganho concentrado em `y > 3600` ou nos 20 piores voos é ruído de cauda, não melhoria —
foi exatamente assim que −0,77 s simulados viraram −0,02 s oficiais. E este candidato é
**um** só: não é o melhor de uma grade, então não paga o pedágio da seleção adaptativa.

## 4. Candidato enfileirado

Receita do membro `20261002-015703-e140` (`--conjunto --externos --plano13 --dist-plano
--corretor-sem-regra --pista --retencao --mapa --corretor-ref --corretor-rounds 500`,
base `20260930-130008-e76_base`, `--reusar-oof 20260930-130921-e76_m0`) **mais
`--pseudo propria`**. Prioridade 12: entra depois da base 198, que já está na fila com a
mesma prioridade, e antes das 199–201.

- **`esteira id 202`**, família `bloco:--pseudo`, origem `pesquisa-item4`, receita
  `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500,
  "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true,
  "--pista": true, "--plano13": true, "--pseudo": "propria", "--retencao": true}`.
  A esteira identifica `20261002-015703-e140` como membro de origem (uma chave de
  diferença), então a base e o oof da corrida saem dele.
- Custo: ~10 min do ajuste da base de 2026 (uma vez por base, depois fica em cache) +
  um ajuste extra do conjunto para gerar o alvo + o corretor final ≈ 25–30 min.
- A corrida grava `pseudo: "propria"`, `pseudo_peso`, `pseudo_corte` na config; o
  `train.py` lê essas chaves e refaz as mesmas linhas no envio.

## 5. Riscos conhecidos

- **Auto-destilação não cria sinal.** Com `propria`, o alvo das linhas de 2026 é a própria
  previsão do modelo: o que muda é só *onde* as árvores gastam capacidade (as partições
  passam a cobrir a densidade de 2026). O efeito esperado sozinho é ~0 [INFERÊNCIA]; a
  aposta declarada é ρ.
- **Viés de confirmação** (arXiv:1908.02983): peso 0,3 e o corte do corpo são as duas
  travas; se o corretor estiver sistematicamente errado numa região de 2026, o
  pseudo-rótulo congela o erro lá. Não dá para detectar isso no holdout de 2025 — é o
  limite honesto desta frente.
- **Cache do `data/cache/pseudo/`**: a chave cobre a config da base e quatro arquivos de
  `src/`. Mexer no corretor não invalida (nem precisa), mas mexer em qualquer outra coisa
  que mude a previsão da base sem tocar nesses quatro arquivos exige apagar a pasta à mão.
- **Diferença deliberada entre simulação e envio**: no envio o professor é o corretor
  treinado nas cegas dos **12** meses (lá não existe holdout para proteger), enquanto a
  base das linhas de 2026 continua sendo a do `train2025`, lida do mesmo cache. É a mesma
  assimetria que o pipeline já tem entre a previsão fora do bloco (treino do corretor) e a
  base final do ranking.
