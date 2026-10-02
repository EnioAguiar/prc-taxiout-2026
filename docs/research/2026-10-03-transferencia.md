# 2026-10-03 — Checagem de transferência: o teste que teria rejeitado a v37

A v37 foi escolhida por uma simulação de **−0,77 s** de `completo` e devolveu **−0,02 s**
oficiais. Não foi azar: o ganho simulado estava quase todo na cauda (**+0,586 s** de
`completo` vindos de `y > 1 h`, contra **+0,183 s** do corpo), **20 voos valiam 129 % do
ganho**, e o membro tinha sido escolhido entre 266 corridas no mesmo holdout.

Este documento mede o que a régua não mede e fixa o portão novo. Código:
`src/transferencia.py`; backtest reproduzível em
`.superpowers/transferencia/backtest.py` (fora do índice do git, como os scripts de
02/10). Item 5 de `docs/pesquisa/2026-10-02-tecnicas-descoberta.md`.

## 1. A decomposição exata do ganho

O ganho de RMSE não é só um número: ele **se soma por voo**, sem aproximação. Com
`dᵢ = (yᵢ − baseᵢ)² − (yᵢ − novoᵢ)²` e peso `wᵢ`,

```
RMSE(base) − RMSE(novo) = Σᵢ wᵢ·dᵢ / ( Σw · (RMSE_base + RMSE_novo) )
```

porque `a − b = (a² − b²)/(a + b)`. Logo a mesma soma restrita a um subconjunto é a
**parcela do ganho que vem dele**, na mesma unidade (segundos de RMSE) e aditiva. Daí
saem as três leituras de `src/transferencia.py`:

| nome | subconjunto | pergunta |
|---|---|---|
| `corpo` | `y ≤ 3600 s` | quanto do ganho vem dos voos que dá para prever |
| `cauda` | `y > 3600 s` | quanto vem da loteria (`corpo + cauda = completo`) |
| `topo` | os 20 voos de maior `d` | o ganho é de uma população ou de um punhado |

`medir()` devolve cada parcela com IC de 95 % por bootstrap pareado **de dias** — o mesmo
reamostrador do `compare.paired_bootstrap`, do qual é uma generalização: sem máscara e sem
peso os dois dão exatamente o mesmo número (há teste).

## 2. O peso adversarial (IWCV)

Classificador "esta linha é 2025 ou 2026?" nas colunas que a base e o corretor
compartilham (`ctx_*`, `adsb_*`, `to_takeoff_from_*`, `gap_*`, `apt_*`, `rwy_*`,
`round_*`, `hour`, `dow`, `nm_missing`, `AIRPORT`), holdout jan+jul/2025 contra ranking
jan+jul/2026, **duas dobras** (cada linha recebe `p` de um modelo que não a viu).

**AUC fora da amostra = 0,8656** — reproduz a varredura de 02/10 (0,880, que usava uma
dobra só). O peso é `w = p/(1−p)` truncado no P95 e normalizado para média 1
(Sugiyama 2007 para o IWCV; truncagem porque o peso cru tem variância explosiva).
Custo: **39 s** na bancada leve, com cache em `data/cache/transferencia-*.parquet`.

Distribuição: mediana 0,75, P95 = P99 = 2,90 (5 % dos voos ficam no teto). Por aeroporto,
o peso diz exatamente o que a cobertura ADS-B já dizia em 02/10:

| aeroporto | peso médio | leitura |
|---|---|---|
| EGLL | 0,50 | em 2025 só 7–21 % dos voos tinham rastro; em 2026, 60–98 % |
| LEMD | 0,55 | 0–3 % em 2025 contra 44–95 % em 2026 |
| EDDF | 0,96 | cobertura caiu de 71–75 % para 51–59 % |
| LEBL / EDDM | 1,37 / 1,31 | os mais "parecidos com 2026" do holdout |

Ou seja: o holdout de 2025 **sub-representa** o regime em que EGLL e LEMD estarão em 2026,
e o peso corrige isso na hora de ler o ganho.

## 3. Backtest contra os deltas oficiais

Seis envios consecutivos com oficial conhecido. Os membros de cada versão vieram do
histórico do `champion.json` (campo `enviada`); a previsão de cada um é medida **no mesmo
holdout**, alinhada por voo. Ganho positivo = melhora.

| de → para | oficial | `completo` | `sem_loteria` | **`corpo`** | IC baixo do corpo | `cauda` | `completo_p` | `corpo_p` | top-20 (completo) | top-20 (corpo) | portão |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v30 → v32 | **+0,650** | +0,377 | +0,954 | **+0,652** | +0,271 | −0,276 | −0,255 | +0,610 | 169 % | 63 % | **envia** |
| v32 → v33 | **+2,212** | +0,912 | +1,653 | **+1,728** | +0,177 | −0,816 | +0,160 | +0,810 | 271 % | 125 % | **envia** |
| v33 → v34 | **+0,710** | +0,900 | +1,123 | **+0,870** | +0,570 | +0,030 | +0,883 | +0,852 | 42 % | 29 % | **envia** |
| v34 → v35 | +0,054 | +0,044 | +0,610 | +0,083 | −0,259 | −0,040 | −0,492 | +0,054 | 1398 % | 378 % | segura |
| v35 → v36 | +0,155 | +0,186 | +0,277 | +0,190 | −0,037 | −0,004 | +0,077 | +0,077 | 201 % | 146 % | segura |
| **v36 → v37** | **+0,022** | **+0,769** | **+0,952** | **+0,183** | **−0,390** | **+0,586** | +0,822 | +0,539 | **129 %** | **249 %** | **segura** |

Qualidade de previsão do delta oficial (6 pontos):

| métrica | Pearson | Spearman | razão oficial/simulado | EAM reescalado |
|---|---|---|---|---|
| `completo` | +0,591 | +0,657 | 1,19 | 0,442 s |
| `sem_loteria` (métrica da régua) | +0,848 | +0,771 | 0,68 | 0,361 s |
| **`corpo` (y ≤ 1 h)** | **+0,985** | **+0,943** | **1,03** | **0,146 s** |
| `cauda` (y > 1 h) | −0,857 | −0,600 | −7,32 | 1,790 s |
| `completo_p` (com peso) | +0,026 | +0,257 | 3,18 | 1,594 s |
| `corpo_p` (corpo com peso) | +0,659 | +0,771 | 1,29 | 0,407 s |

A razão acima é ajustada nos mesmos seis pontos em que é medida. Repetindo com
**deixa-um-de-fora** (razão ajustada nas outras cinco transições, erro na que ficou de
fora), a ordem não muda:

| métrica | EAM fora da amostra | pior caso |
|---|---|---|
| `completo` | 0,594 s | 1,575 s |
| `sem_loteria` | 0,469 s | 1,540 s |
| **`corpo`** | **0,222 s** | **0,823 s** |
| `corpo_p` | 0,539 s | 1,608 s |

Três coisas saem daqui, e duas são desconfortáveis:

1. **A parcela do corpo é praticamente o delta oficial.** Razão 1,03 e erro absoluto médio
   de **0,15 s** nas seis transições, contra 0,44 s do `completo` e 0,36 s do
   `sem_loteria`. É o melhor preditor que já medimos — e não precisa de peso nenhum.
2. **A cauda é ruído com sinal invertido** (Pearson −0,86): quanto mais do ganho simulado
   vem de `y > 1 h`, **pior** tende a ser o oficial. É a assinatura da v37.
3. **Reponderar o `completo` destrói a previsão** (Pearson +0,03) e reponderar o corpo
   também piora (+0,66 contra +0,985). O peso adversarial **não** é o que rejeita a v37:
   com peso, o corpo da v37 sobe para +0,539 com IC +0,17 — ele *aprovaria*. Quem rejeita
   é a decomposição por regime. Isso confirma a previsão do levantamento de 02/10 ("como
   reponderação na decisão, 0 a −1 s, e o papel é de seguro").

Nota sobre o `sem_loteria`: ele tira 11 voos (`y > 3 h` sem horário planejado que os
explique) mas **mantém** o resto da cauda — e é aí que mora o problema da v37, cujo ganho
de `sem_loteria` era +0,952 s. Cortar a loteria não é o mesmo que separar o regime.

## 4. O portão adotado

`transferencia.veredito()` — quatro testes, todos sobre a decomposição acima:

```
corpo > 0                       parcela do ganho vinda de y ≤ 3600 s
IC baixo do corpo > 0           bootstrap pareado por dia, 95 %
top-20 do corpo < 200 %         o ganho do corpo não mora em 20 voos
corpo com peso 2026 > 0         seguro de mudança de covariáveis (IWCV)
```

**Acerta 6 de 6 no backtest**: aprova as três transições que valeram ≥ 0,65 s oficiais e
segura as três que valeram ≤ 0,16 s, inclusive a v37.

Calibração dos dois limites que não são óbvios:

* **200 %, e não 50 %, no top-20.** Com ganhos de ~1 s espalhados em 344 mil voos, a
  concentração é a regra, não a exceção: a v33, que valeu **+2,21 s oficiais**, tem 125 %
  do ganho do corpo em 20 voos. Um limite de 50 % reprovaria 5 das 6 transições, incluindo
  as três boas. O que separa as boas das ruins é 200 % (0,63 / 1,25 / 0,29 contra
  3,78 / 1,46 / 2,49).
* **O top-20 é medido no ganho do *corpo*, não no completo.** No completo a v37 dá 129 % e
  a v35 dá 201 % — o `completo` da v37 parece *menos* concentrado que o da v35, porque a
  cauda infla o denominador. No corpo a ordem fica certa (249 % contra 146 %).
* **O teste do peso nunca virou uma decisão** nestas seis transições (os seis `corpo_p` são
  positivos). Fica como seguro contra o modo de falha que o holdout de 2025 não enxerga por
  construção — ganho que vive só onde EGLL/LEMD não tinham ADS-B em 2025 — e é barato.

O limiar de magnitude ficou em `corpo > 0` (e não em 0,3 s como o `GANHO_A` da régua) de
propósito: com seis pontos, fixar um limiar que separa 0,19 de 0,65 seria ajustar o portão
ao próprio backtest. Quem separa é o IC.

## 5. Como se usa

```
bin/run src/transferencia.py <id_novo...> [--contra <id...>] [--json]
bin/run src/transferencia.py --pesos          # recalcula o peso adversarial (39 s)
```

Sem `--contra`, compara com os membros da campeã. Saída na v37, reproduzindo o diagnóstico:

```
| parcela | sem peso | com peso 2026 |
| completo | +0.769 (IC +0.34 a +1.38) | +0.822 (IC +0.05 a +1.86) |
| corpo (y ≤ 3600 s) | +0.183 (IC -0.39 a +0.60) | +0.539 (IC +0.17 a +1.12) |
| cauda (y > 3600 s) | +0.586 (IC +0.08 a +1.24) | +0.283 (IC -0.42 a +1.17) |

top-20 voos: 129.4% do ganho completo, 249.3% do ganho do corpo
delta oficial previsto: +0.19 s (±0,15 s de erro médio no backtest)
falha: IC do corpo > 0; top-20 do corpo < 200%
```

`esteira.talvez_enviar` chama `esteira.transferencia_ok` antes de montar uma submissão: a
campeã continua sendo promovida pela régua (é ela que decide o que entra no
`champion.json`), mas o envio — que custa 2,6 h de `submit` e um dos 5 slots diários — só
acontece se a transferência passar. Quando não passa, a esteira grava o motivo em
`erro_envio` (visível em `docs/esteira.md`), marca `transferencia_reprovada` com a chave da
campeã para não remedir a mesma coisa a cada candidato, e **não** gasta a trava de 6 h.

A referência da comparação é a última versão **enviada**, não a campeã de agora. Para isso
`enviada` passou a guardar `membros` (`esteira.py enviado <N>` grava os ids do arquivo que
foi gerado, que podem já não ser os da campeã); o `champion.json` da v37 foi preenchido à
mão com os quatro membros dela. Sem esse campo o portão reprova e pede para rodar
`esteira.py enviado <versão>` de novo — é o único jeito honesto, porque sem os membros
antigos não há o que medir.

Verificação de ponta a ponta (02/10, com a campeã real no disco): marcando a v36 como
última enviada, `esteira.transferencia_ok` devolve
`(False, 'falha: IC do corpo > 0; top-20 do corpo < 200%')` — ou seja, a esteira **não
teria montado a v37**.

## 6. Limites honestos

* **Seis pontos.** O backtest tem seis transições, todas da mesma pilha e do mesmo mês.
  Pearson 0,985 em seis pontos é uma indicação forte, não uma lei; a razão 1,03 pode ser
  coincidência de escala entre o holdout (344.419 voos) e o ranking (344.841).
* **O portão não protege contra seleção adaptativa.** Ele mede *um* candidato; se a
  esteira testar 266 e enviar o que passar, o viés de seleção continua lá. Para isso
  continua valendo a metade B cega da régua.
* **O peso vem de um modelo.** AUC 0,866 significa que o adversário acha a diferença, não
  que `p/(1−p)` é a razão de densidades verdadeira; com truncagem no P95, 5 % dos voos
  ficam com peso achatado de propósito.
* **A cauda não some.** O portão decide *não enviar* quando o ganho é de cauda; ele não
  conserta a cauda. O diagnóstico de 01/10 (20 voos = 41 % do erro²) continua de pé.
