# Origem dos dados do desafio (APDF / NM) e o que isso diz sobre as "loterias"

Pesquisa de 2026-10-01. Objetivo: descobrir, na documentação oficial, **como os campos do
dataset são produzidos** e se existe **regra documentada** que torne as loterias
(taxi-outs de horas, off-block aparentemente copiado de horário/schedule) previsíveis a
partir de `AOBT_3`, `LOBT`, `SCHED` e dos registros NM.

Resultado principal: sim, há três mecanismos documentados que geram cauda pesada
(**off-block derivado pelo NM**, **Push & Hold / remote holding**, **de-icing remoto**) e
há limiares de qualidade oficiais que dizem **quanto lixo é tolerado** em cada aeroporto.

---

## Resumo (o que vale testar)

### 1. Detectar quando `AOBT_3_flt` é *derivado* (take-off − taxi padrão), não medido

**O quê.** Para cada aeroporto/pista, histogramar `MVT_TIME_UTC_mvt − AOBT_3_flt` e
procurar **modos discretos** (valores que se repetem exatamente, tipicamente múltiplos de
1 min). Criar: (a) flag `aobt3_derivado`, (b) o valor do modo como categórica
(= "taxi padrão CACD daquele aeroporto/pista"), (c) resíduo em relação ao modo.

**Por quê reduz RMSE.** A EUROCONTROL documenta explicitamente que o off-block "actual" do
NM **pode ser calculado como take-off menos um taxi padrão do aeroporto** quando o NM não
recebeu atualização de dados do voo (citação literal abaixo). Nesse caso
`MVT − AOBT_3` é **circular** (não traz informação) e o modelo hoje provavelmente está
tratando essa quantidade como âncora confiável. Os 10 aeroportos do desafio incluem
**exatamente um que não é A-CDM: LTFM (Istanbul)** — os outros 9 estão na lista oficial de
34 aeroportos A-CDM (logo mandam DPI ao NM). LTFM é justamente o pior aeroporto do nosso
erro. A separação "âncora real × âncora derivada" é uma informação nova obtida **só com
colunas que já temos**, e está disponível no ranking set (`AOBT_3` e `MVT` não são
apagados).

**Fonte/licença.** EUROCONTROL *Aviation Data Repository for Research – Metadata*
(descreve o mesmo campo NM) e *DPI Implementation Guide* ed. 2.700 — PDFs públicos em
eurocontrol.int, com nota de copyright "may be copied in whole or in part, providing that
the copyright notice and disclaimer are included". Lista A-CDM: página pública
eurocontrol.int/concept/airport-collaborative-decision-making.

**Esforço.** Baixo (só agregação sobre o dataset atual; nenhum download novo).

---

### 2. Tratar "Push & Hold" (remote holding) como a classe-geradora das loterias

**O quê.** Modelar explicitamente a discrepância `d = AOBT_3_flt − BLOCK_TIME_UTC_mvt` no
treino (no ranking set o alvo é `(MVT − AOBT_3) + d`). Testar se a população de `d`
grande-e-positivo (NM acha que o off-block foi *muito depois* do que o aeroporto reportou)
se concentra em: voos regulados/atrasados, aeroportos com escassez de posição, e **stands
cuja próxima chegada está próxima**.

**Por quê reduz RMSE.** O *DPI Implementation Guide* descreve o procedimento Push & Hold:
a aeronave é liberada do portão cedo **só para liberar a posição** e espera num ponto
remoto; e manda **adiar o A-DPI** justamente nesses casos. Ou seja: o AOBT do aeroporto é
cedo e real (taxi-out de horas, legítimo), enquanto o off-block do NM é tardio. Isso
explica uma fatia das loterias **sem** supor erro de dado, e dá features causais
(pressão de stand = tempo até a próxima chegada alocada àquele stand; atraso ATFM do dia).

**Fonte/licença.** *DPI Implementation Guide* ed. 2.700, seção 11.3 (público,
eurocontrol.int).

**Esforço.** Médio (nova feature de pressão de stand + reparametrização do alvo em `d`).

---

### 3. Usar as regras oficiais de filtro do indicador ATXOT para definir a classe "loteria"

**O quê.** Adotar como definição de regime os mesmos cortes que a EUROCONTROL usa para
calcular *Additional Taxi-Out Time*: **excluir taxi-out > 120 min**, **excluir voos com
de-icing depois do AOBT**, **excluir helicópteros e registros sem pista/stand/off-block**.
No nosso caso não dá para excluir (RMSE no conjunto completo), mas dá para usar isso como
**fronteira do classificador**: o que a dona do dado chama de "não-dado" é exatamente a
nossa cauda.

**Por quê reduz RMSE.** Alinha a definição de classe com a física/processo em vez de um
corte empírico; e explica por que a cauda é maior em janeiro (de-icing remoto entra
*dentro* do taxi-out por construção; de-icing no stand, não — ver citações).
Janeiro é um dos dois meses do ranking.

**Fonte/licença.** *Additional taxi-out time performance indicator document*, ed. 01.00,
16-03-2023 (ansperformance.eu/library, público).

**Esforço.** Baixo.

---

### 4. Calibrar a taxa de loteria por aeroporto-mês com os limiares de qualidade do APDF

**O quê.** A spec APDF define critérios de aceitação por item de dado. Para AOBT/ATOT:
`100% flights with AOBT ≤ ATOT`, `≥ 95% flights with AXOT > 0 min`,
`≥ 95% flights with AXOT < 60 min`. Calcular, por aeroporto-mês do treino, a fração de
`TAXITIME ≥ 3600 s`; aeroportos encostados em 5% estão **operando no limite tolerado** —
e aeroportos acima disso estão fora de conformidade (regime diferente).

**Por quê reduz RMSE.** Dá um *prior* quantitativo para a probabilidade de loteria por
aeroporto-mês (hoje aprendida só dos dados) e um teste de sanidade para 2026: se em
jan/jul 2026 a mistura mudou, o classificador precisa ser recalibrado por aeroporto.

**Fonte/licença.** EUROCONTROL-SPEC-175 ed. 1.0 (15/01/2019) e APDF Data Specification
v00-11 — PDFs públicos.

**Esforço.** Baixo.

---

### 5. Tabelas sazonais de taxi-time do CODA (incluem **janeiro de 2026**)

**O quê.** `Taxi-out times - winter 2025/2026` (publicado 20/05/2026) e
`Taxi-out times by wake turbulence category - winter 2025/2026`, mais
`Taxi times - Summer 2025` (publicado 29/01/2026). XLSX pequenos (25–50 KB) com, por
aeroporto (e por categoria de esteira): média, desvio-padrão, P10, mediana, P90 do
taxi-out da temporada.

**Por quê reduz RMSE.** A temporada IATA de inverno 2025/26 **contém janeiro de 2026**, um
dos dois meses do ranking: é uma calibração de nível e dispersão do período alvo, por
aeroporto × categoria de esteira (granularidade que a série mensal do ansperformance que
já usamos não tem). Atenção: é fonte **AODF (reportada pelas companhias)**, não APDF, e o
filtro é `taxi-out > 180 min` (diferente do ATXOT, 120 min) — serve como prior, não como
verdade.

**Fonte/licença.** eurocontrol.int/publication/taxi-times-winter-2025-2026 e
.../taxi-times-summer-2025 (download público). Metodologia:
ansperformance.eu/reference/dataset/planning-taxi-times.
[INFERÊNCIA] A edição *Summer 2026* não deve sair antes de 11/10/2026 (padrão observado:
S25 saiu em jan/2026, W25-26 em mai/2026).

**Esforço.** Baixo (3 planilhas pequenas).

---

## Como o dado é produzido — definições oficiais

### O alvo

Página de dados do desafio: `TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt`
(para a partida). Colunas `_mvt` vêm do **aeroporto que reporta**; colunas `_flt` vêm da
**lista de voos do Network Manager**. O organizador avisa:

> "Real world data is messy. You will find some inconsistencies between movement and
> flight information. We haven't tried to reconcile them because they are probably due to
> some limitations of the underlying processes trying to match different sources of data."
> — <https://prc-data-challenge-2026.netlify.app/data.html>

Ou seja: a inconsistência `_mvt` × `_flt` é **assumida** pelo organizador, não corrigida.
É nela que mora a loteria.

### De onde vêm as colunas `_mvt`: o Airport Operator Data Flow (APDF)

O APDF é o fluxo mensal pelo qual os operadores de aeroporto enviam dados à EUROCONTROL
(EUROCONTROL-SPEC-175). O arquivo de partidas tem exatamente os campos que aparecem no
desafio: `STD_UTC` (→ `SCHED_TIME_UTC_mvt`), `AOBT_UTC` (→ `BLOCK_TIME_UTC_mvt`),
`ATOT_UTC` (→ `MVT_TIME_UTC_mvt`), `DRWY` (→ `RUNWAY_mvt`), `DSTND` (→ `STAND_mvt`),
além de `DLY1..DLY5`/`TIME1..TIME5` (códigos IATA de atraso) e `DE_ANTI_ICING` — **esses
dois últimos existem no APDF e NÃO foram liberados no desafio**.

Definições literais (APDF Data Specification v00-11, cap. 5; idênticas em SPEC-175 ed.1.0):

- **AOBT (ODI 11)** — "*'actual off-block time' means the actual date and time the aircraft
  has vacated the parking position (pushed back or on its own power)*". Definição
  alternativa: "*Time the aircraft pushes back / vacates the parking position. (Equivalent
  to Airline / Handlers ATD – Actual Time of Departure, ACARS=OUT)*".
- **ATOT (ODI 12)** — "*'actual take off time' means the date and time that an aircraft has
  taken off from the runway (wheels-up)*" / "*(Equivalent to ATC ATD, ACARS = OFF)*".
- **STD (ODI 09)** — "*'scheduled time of departure (off-block)' means date and time when a
  flight is scheduled to depart from the departure stand*".
- **SOBT (ODI 21)** — slot aeroportuário (Reg. CEE 95/93). **Obrigatório só para aeroportos
  nível 3**; é um campo *diferente* de STD. O desafio só entrega um `SCHED_TIME_UTC_mvt`.
- **DE_ANTI_ICING (ODI 17)** — "*indication if de-icing or anti-icing operations occurred
  and if yes, where (before leaving the departure stand or in a remote position after
  departing the stand, i.e. after off-block)*"; domínio `A` (depois do AOBT), `B` (antes),
  `N` (não houve), `Z` (sem informação).

Critérios de qualidade que a spec impõe a AOBT e ATOT (literal):

> Accuracy: "100% flights with AOBT ≤ ATOT; ≥ 95% flights with AXOT > 0 min;
> ≥ 95% flights with AXOT < 60min"
> Consistency: "≥ 98% of completeness for IFR flights (flight rules I, Y or Z);
> ≥ 95% of associated flights with (Airport vs Airline) values in tolerance interval
> [-3,+3] min; ≥ 99% … [-120,+120] min"
> (AXOT = Actual Taxi-Out Time = ATOT − AOBT)

**Leitura direta:** a especificação **tolera até 5% de voos com taxi-out ≥ 60 min** e até
5% com AXOT ≤ 0. Para ATOT há ainda cheque "Airport vs NM" em ±3 min — para **AOBT não
existe cheque contra o NM**, só contra a companhia. Isto é consistente com o que vemos: o
take-off é confiável, o off-block é o campo frágil.

Outras regras do processo que acoplam AOBT e STD:

- `APDF-RFS-340-M`: "*operated departure flights shall be considered delayed when the
  delay threshold is breached, i.e. AOBT ≥ STD + 00:04:00 (240 seconds)*" — e nesse caso o
  registro **obriga** pelo menos um código de atraso e sua duração.
- `TIME1` quando não há causa conhecida: "*The associated field TIME1 shall be populated
  with the result of the actual delay, in minutes (AOBT minus STD)*".
- Regra de qualidade sobre as durações: "*≥ 90% of the flight records in the departure
  data files need to comply with: DLY_TIME ≥ (AOBT-STD) - 00:20:00*".
- Único valor-sentinela documentado em todo o APDF: `ATOC` (hora de cancelamento) ausente
  → "*the value 01-01-1990 00:00:00 shall be used*". **Não existe valor-sentinela
  documentado para AOBT**: se o aeroporto não tem a medida, ele preenche *alguma coisa*
  sem regra publicada.
- Detecção de duplicatas: "*flight records for which association with other data sources
  fails (e.g. NM database) and which show identical ADEP/ADES, aircraft registration, and
  time stamps (STA/STD, ALDT/ATOT, AIBT/AOBT)*".

### Como a EUROCONTROL usa esse dado (indicador Additional Taxi-Out Time)

Do documento do indicador (ed. 01.00, 2023), passo 1 do algoritmo mensal — literal:

> "*filter out the flights with some missing data (e.g. no runway information, no stand
> information or no off-block time), flights with taxi-out time longer than 120 min,
> helicopters and flights with de-icing after the off-block time (i.e. during the taxi-out
> phase).*"

E a referência por combo stand×pista é o **percentil 10** do último ano móvel, exigindo
"*at least 10 flights in the sample with a taxi-out time equal or shorter than the
calculated reference time*". Sobre precisão:

> "*The precision in the collection of the AOBT (Actual Off-Block Time) is key in the
> calculation of the taxi-out times. This event should reflect as much as possible when
> the aircraft starts moving on the apron.*" … "*in some cases the data provided might be
> limited to a HH:MM format, so in those cases the precision is lower.*"

**Consequência para nós:** a própria dona do dado considera tudo acima de 120 min como não
mensurável, e remove de-icing pós-AOBT. O desafio **não** removeu. Nossa cauda é
literalmente o conjunto que a metodologia oficial descarta.

### De onde vêm as colunas `_flt`: Network Manager

- `LOBT_flt` = "last known off-block time" (o EOBT corrente). Regra que o governa:
  "*A DLA message shall be sent for any change of EOBT greater than 15 minutes*"
  (ATFCM Users Manual, Wave 2.1) e, em aeroportos A-CDM, "*a DLA message shall be sent if
  your TOBT is more than 15 minutes after the last EOBT*" — com a NM podendo filar os DLA
  automaticamente a partir dos valores de TOBT nas mensagens DPI. Ou seja, em aeroporto
  A-CDM o `LOBT` tende a seguir o **TOBT**, não o schedule.
- `AOBT_3_flt` = off-block do modelo M3 (CTFM, "trajetória voada"). A definição oficial do
  mesmo campo no repositório de dados de pesquisa da EUROCONTROL é a chave de tudo:

> "**ACTUAL OFF BLOCK TIME** — Off-Block Time (UTC) based on the ATFM-updated flight plan.
> The time that an aircraft departs from its parking position. **This time may be known
> from flight data updates received by NM, or in the absence of such updates may be
> calculated from the known take-off time minus a standard taxi time value for the
> airport.**"
> — EUROCONTROL *Aviation Data Repository for Research – Metadata*, abr/2025

Como o NM "sabe" o off-block quando sabe: via mensagens DPI dos aeroportos A-CDM.
Detalhes relevantes do *DPI Implementation Guide* ed. 2.700 (30/06/2025):

- "*A-DPI (ATC DPI): It is sent at off-block and it provides the ETFMS with an accurate
  estimation of the actual off-block time*" (ATFCM Users Manual).
- Mas o campo AOBT propriamente dito **não é usado**: "*The aobt- and aobd-fields are
  currently not used. They will be used for information sharing purposes.*" O que trafega
  é o **TTOT** e o **taxitime**: "*If provided in an A-DPI, the ttot-field shall supply
  ETFMS with an estimate of the actual take-off-time, i.e., AOBT+EXOT*", e o off-block
  interno é reconstruído: "*(Derived OBT in DPI msg is TOT – last-received-taxi-time)*".
- `taxitime` (EXOT): "*The taxitime-field shall contain the time between take-off and
  off-block*"; "*shall at least depend on gate/stand/parking position and departure
  runway*"; "*shall also include any time it takes to de-ice. This applies to remote
  de-icing*"; e um limite duro: "**The taxitime-field value shall be between 1min and
  90min**".
- Quando não há DPI, o parâmetro vem do banco de ambiente: "*The default taxi time
  parameter is specified for each runway at an aerodrome in the CACD*" (ATFCM Users
  Manual, 8.1.10.1), sobreposto pelo taxi time que o operador põe no plano de voo.

**Consequência para nós:** `MVT − AOBT_3` é (a) o AOBT real, quando houve A-DPI; (b) um
taxi padrão por aeroporto/pista, quando não houve; e (c) **nunca** maior que ~90 min pela
rota DPI. Qualquer loteria de 3 h é, por construção, invisível em `AOBT_3`.

### A-CDM: quem manda DPI entre os 10 aeroportos

Lista oficial (34 aeroportos "fully implemented"), inclui: Amsterdam, Barcelona,
Frankfurt, London Heathrow, Madrid, Munich, Paris CDG, Rome Fiumicino, Zurich.
**Istanbul (LTFM) não está na lista.** Fonte:
<https://www.eurocontrol.int/concept/airport-collaborative-decision-making>.

Definições A-CDM citadas (Airport CDM Implementation Manual v5.0, 2017, glossário):

- **TOBT** — "*The time that an Aircraft Operator or Ground Handler estimates that an
  aircraft will be ready, all doors closed, boarding bridge removed, push back vehicle
  available and ready to start up / push back immediately upon reception of clearance from
  the TWR*".
- **TSAT** — "*The time provided by ATC taking into account TOBT, CTOT and/or the traffic
  situation that an aircraft can expect start up / push back approval*".
- **ARDT** — "*When the aircraft is ready for start up/push back or taxi immediately after
  clearance delivery*".
- **EXOT** — "*The estimated taxi time between off-block and take off. This estimate
  includes any delay buffer time at the holding point or remote de-icing prior to take
  off*".
- **TTOT** — "*The Target Take Off Time taking into account the TOBT/TSAT plus the EXOT*".

E da *EUROCONTROL Specification for A-CDM* ed. 1.0 (30/01/2025):

- `A-CDM-[DAT]-[300]`: "*The A-CDM System shall maintain the AOBT from agreed local
  sources, such as ATC systems, Aircraft Operator / Ground Handling Agent input, docking
  system, movement messages (MVT) and/or ACARS "out" and/or A-SMGCS, with a priority
  defined when several sources are received.*" — **a fonte do AOBT varia por aeroporto e
  pode variar por voo dentro do mesmo aeroporto.**
- Milestone 15 (Off-Block): "*When push-back or taxi starts, the AOBT is recorded in the
  A-CDM System. A reliable local process for the recording and update of the AOBT is
  essential. The start-up request from the Electronic Flight Strip may be used as the
  off-block event.*" — note: **o pedido de start-up pode ser usado como evento de
  off-block**, o que antecipa o AOBT em relação ao movimento real.
- Retorno ao stand: "*In the case where an aircraft is off-block and has returned to stand
  or holding remotely to resolve a problem, local procedures is highly recommended to be
  defined to establish who is responsible to generate C-DPI or update of the TTOT.*"
  (isto é: **não há regra única**; depende do aeroporto).

---

## As "loterias": hipóteses testáveis

Numeradas para virarem experimentos. Todas usam só colunas que já temos (ou ADS-B/METAR já
integrados), salvo indicação.

**H1 — AOBT_3 derivado (ruído circular).**
Em LTFM (não A-CDM) e em janelas sem DPI nos demais, `MVT − AOBT_3` assume poucos valores
discretos (taxi padrão CACD por pista). *Teste:* contar a moda de `MVT − AOBT_3` por
aeroporto × pista × mês; medir a massa exatamente na moda. Se >20% da massa estiver em 1–3
valores, a âncora é falsa ali. *Uso:* flag + dummy do modo; e remover a âncora do modelo
nesse subconjunto.

**H2 — Teto de 90 min no EXOT.**
Pela regra `taxitime ∈ [1, 90] min`, não deve existir `MVT − AOBT_3 > 90 min` quando o
valor veio de DPI. *Teste:* fração de voos com `MVT − AOBT_3 > 5400 s` por aeroporto. Onde
for ~0, confirma origem DPI; onde houver massa, o off-block do NM veio de outra rota (DLA
sucessivos / FAM). *Uso:* censura conhecida → o modelo deve saber que a âncora satura.

**H3 — Push & Hold: off-block cedo legítimo.**
Voos com atraso ATFM > 30–45 min em aeroportos com escassez de posição são empurrados cedo
para liberar o stand (procedimento documentado). *Assinatura esperada:*
`d = AOBT_3 − BLOCK_TIME` grande e **positivo** (NM "vê" o off-block tarde, porque o A-DPI
é adiado até a liberação do remote hold). *Teste:* no treino, cruzar `d > 20 min` com
(i) dia/hora de alto atraso ATFM, (ii) tempo até a próxima chegada alocada ao mesmo stand,
(iii) aeroporto. *Uso:* feature "pressão de stand" + classe dedicada.

**H4 — De-icing remoto entra no taxi-out; de-icing no stand, não.**
Documentado nos dois lados (ODI 17 distingue A/B; EXOT "inclui remote de-icing"; o
indicador ATXOT exclui voos com de-icing após AOBT). *Teste:* em jan/fev/dez 2025, a cauda
condicionada a METAR congelante deve ser muito maior nos aeroportos com pátio de de-icing
remoto do que nos que de-icem no stand. *Uso:* regime "de-icing" separado em janeiro (mês
do ranking); procedimento por aeroporto vem do AIP AD 2 (dado aberto nacional).

**H5 — Retorno ao stand / re-pushback.**
Sem regra única (a spec A-CDM só "recomenda" definir procedimento local). *Teste:* nos
aeroportos com cobertura ADS-B de solo, detectar trajetos que **voltam** a menos de ~50 m
do stand de partida depois de já terem se movido, e medir a fração da cauda explicada.
*Uso:* se a fração for alta, é um fenômeno real (não erro) e depende de variáveis que não
observamos no futuro → deve ir para a classe "ruído" com média condicional, não para o
regressor.

**H6 — AOBT = pedido de start-up, não movimento.**
Milestone 15 permite usar o start-up request como evento de off-block. *Teste:* comparar
`BLOCK_TIME_UTC_mvt` com o off-block estimado pelo ADS-B próprio (feature já existente):
se num aeroporto o APDF for **sistematicamente mais cedo** por um offset estável, a
diferença é procedimento, não ruído → corrigível por offset por aeroporto (e por stand
remoto × contato).

**H7 — "Cópia do schedule" tem formato detectável.**
Não existe sentinela documentado para AOBT, mas as regras de atraso acoplam AOBT a STD
(`TIME1 = AOBT − STD`). *Teste:* medir a massa exatamente em `BLOCK_TIME == SCHED`,
`BLOCK_TIME == LOBT`, `BLOCK_TIME == IOBT`, `BLOCK_TIME == EOBT_1` e em
`BLOCK_TIME` múltiplo exato de 5/10/60 min (arredondamento de entrada manual). A spec
avisa que alguns provedores só têm precisão HH:MM — então `segundos == 0` é um **indicador
de origem manual/arredondada**, por voo. *Uso:* features de "forma" do carimbo
(segundos==0, minutos múltiplos de 5, igualdade exata a cada candidato) alimentando o
classificador de loteria. Barato e disponível no treino; no ranking o alvo é `d`, então
usar como rótulo de treino do classificador, não como feature de inferência.

**H8 — Taxa de loteria segue o limiar de conformidade (5%).**
*Teste:* fração de `TAXITIME ≥ 3600 s` por aeroporto-mês contra 5%. Aeroportos que ficam
colados no limite estão "gerenciando" o indicador; os que estouram mudaram de regime.
*Uso:* prior/recalibração por aeroporto-mês do classificador, e alerta de drift para
jan/jul 2026.

**H9 — Mismatch movimento × NM é diagnóstico.**
O organizador diz que o join é imperfeito. *Teste:* comparar a taxa de loteria entre voos
com `FLIGHT_ID_mvt` nulo (sem match NM) e com match. Se divergirem muito, "sem match NM" é
feature de primeira ordem — e está disponível no ranking set.

---

## Descartado / irrelevante

- **Campos APDF que resolveriam quase tudo mas não foram liberados:** `DE_ANTI_ICING`
  (ODI 17), `DLY1..DLY5` + `TIME1..TIME5` (códigos IATA de atraso), `REG` (matrícula),
  `IFPLID`, `SOBT` separado de `STD`. Não há fonte aberta equivalente por voo. Só
  proxies (METAR para de-icing; séries diárias EUROCONTROL para atraso).
- **"Variable Taxi Time" (Annex C da spec A-CDM):** descreve 4 métodos de cálculo
  (default / média histórica / condições operacionais / cálculo complexo) mas **não
  publica os valores** por aeroporto. Sem dado aberto, serve apenas como justificativa
  conceitual para features stand×pista que já temos.
- **Taxi times CACD por pista:** parâmetro interno do NM (CACD), não publicado. Só
  recuperável por engenharia reversa via H1.
- **Repositório de dados de pesquisa da EUROCONTROL (R&D data archive):** só meses
  março/junho/setembro/dezembro, com atraso de 2 anos, e os termos proíbem
  redistribuição ("*I shall not share or distribute*") — **incompatível com a regra de
  dado aberto/documentado do prêmio** e sem cobertura de jan/jul 2026. Útil só como
  documentação (foi dele que veio a citação-chave sobre o off-block derivado).
- **Buscar um "paper do organizador" para 2026:** não existe ainda; os data papers
  publicados são das edições 2024 (takeoff weight) e 2025 (fuel burn), em JOAS. A página
  do desafio promete um data paper, mas não há nada publicado sobre o dataset de taxi-out.
- **A-CDM Implementation Manual v5.0 (2017):** útil só para o glossário (TOBT/TSAT/ARDT/
  EXOT/TTOT). A spec de 2025 (SPEC A-CDM ed. 1.0) é a fonte normativa atual.

## Fontes

1. PRC Data Challenge 2026 — Data. <https://prc-data-challenge-2026.netlify.app/data.html>
2. EUROCONTROL, *Airport Operator Data Flow – Data Specification*, ed. 00-11, 04/09/2014.
   <https://www.eurocontrol.int/sites/default/files/publication/files/apdf-data-specification-v00-11.pdf>
3. EUROCONTROL-SPEC-175, *Operational ANS Performance Monitoring – Airport Operator Data
   Flow*, ed. 1.0, 15/01/2019.
   <https://www.eurocontrol.int/sites/default/files/content/documents/single-sky/specifications/EUROCONTROL%20Specification%20APDF%20Edition%201.0%20final%20web.pdf>
4. EUROCONTROL, *Additional taxi-out time performance indicator document*, ed. 01.00,
   16/03/2023. <https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf>
5. EUROCONTROL, *Aviation Data Repository for Research – Metadata*, abr/2025.
   <https://www.eurocontrol.int/sites/default/files/2025-04/eurocontrol-aviation-data-repository-research-metadata.pdf>
6. EUROCONTROL NM, *DPI Implementation Guide*, ed. 2.700, 30/06/2025.
   <https://www.eurocontrol.int/publication/departure-planning-information-dpi-implementation-guide>
7. EUROCONTROL NM, *ATFCM Users Manual*, Wave 2.1, validade 04/11/2025.
   <https://www.eurocontrol.int/sites/default/files/2026-02/eurocontrol-atfcm-users-manual-wave-2-1.pdf>
8. EUROCONTROL, *Specification for Airport Collaborative Decision Making (A-CDM)*, ed. 1.0,
   30/01/2025.
   <https://www.eurocontrol.int/sites/default/files/2025-01/eurocontrol-specification-for-acdm.pdf>
9. EUROCONTROL/ACI/IATA, *Airport CDM Implementation Manual*, v5.0, 31/03/2017.
   <https://www.eurocontrol.int/sites/default/files/publication/files/airport-cdm-manual-2017.PDF>
10. EUROCONTROL, *Airport collaborative decision-making (A-CDM)* — lista de aeroportos.
    <https://www.eurocontrol.int/concept/airport-collaborative-decision-making>
11. EUROCONTROL, *Taxi times – Winter 2025-2026* (20/05/2026) e *Summer 2025* (29/01/2026).
    <https://www.eurocontrol.int/publication/taxi-times-winter-2025-2026> /
    <https://www.eurocontrol.int/publication/taxi-times-summer-2025>
12. EUROCONTROL AIU, *Taxi-time planning values* (metodologia).
    <https://ansperformance.eu/reference/dataset/planning-taxi-times>
