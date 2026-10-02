# Regras do organizador e precedentes (2024, 2025, 2026)

Pesquisa de 01/10/2026. Pergunta central: **o organizador vai aceitar usar o rastro
adsb.lol do próprio voo, depois do push-back, como estimativa de off-block?**
Resposta curta: **sim, e com folga** — não só nenhuma regra proíbe, como o próprio
EUROCONTROL/PRC publica dados abertos que fazem exatamente isso (OPDI) e publicou em
setembro de 2026 um site medindo quanta cobertura ADS-B de solo existe por aeroporto.

Segunda parte: a lista completa de requisitos de elegibilidade para o prêmio, com o
texto literal das regras.

---

## Resumo (o que vale testar)

Ordenado por (ganho esperado) × (probabilidade), do mais alto para o mais baixo.

### 1. OPDI `flight_events` v0.0.2 — eventos de solo do próprio organizador (jan e jul 2026 disponíveis)

**O quê.** O EUROCONTROL/PRC publica, aberto, um dataset de *eventos de voo* derivado
de ADS-B da OpenSky que inclui os tipos `entry-parking_position`, `exit-parking_position`,
`entry-apron`, `entry-runway`, `exit-runway` e — não documentado na página de
metodologia, mas presente nos arquivos — **`entry-deicing_pad`**. Cada evento traz
`event_time`, lat/lon e um campo `info` em JSON com `osm_airport`, `osm_ref` (número do
stand / designador da pista), `osn_icao24` e `osn_flight_id` (callsign). Verifiquei lendo
só o rodapé Parquet do arquivo de 09–19/07/2026 (65 KB por *range request*, nada de
download pesado): os tipos acima aparecem nas estatísticas de coluna, e o `info` traz
exemplos reais como `{"osm_id":995933596,"osm_aeroway":"parking_position","osm_ref":"320","osm_airport":"LSZH",...}`.

**Por que pode baixar o RMSE para nós.**
- `exit-parking_position` é um off-block observado — a mesma feature que já nos deu
  −16 s via adsb.lol — **mas vindo de outra rede de receptores (OSN)**. A união das duas
  redes cobre mais voos do que qualquer uma sozinha; o rastro adsb.lol e o evento OPDI
  raramente faltam no mesmo voo.
- `entry-deicing_pad` é informação que hoje não temos de jeito nenhum: ela separa o voo
  que ficou 40 min na fila do voo que foi para o pad de degelo. Metade do ranking é
  **janeiro**, e a cauda de janeiro é exatamente isso. Nosso proxy de degelo hoje é só
  METAR (temperatura/ponto de orvalho), que é condição do aeroporto, não do voo.
- `entry-runway` com `osm_ref` dá o instante de entrada na pista do *próprio* voo e de
  todos os outros: fila de pista medida, não contada por janela temporal.

**Fonte e licença.** <https://www.opdi.aero/flight-event-data.html>, arquivos em
`https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_{YYYYMMDD}_{YYYYMMDD+10}.parquet`.
É dado aberto do próprio organizador: "The Open Performance Data Initiative and the
relevant data sets are made openly available in order to promote transparent and
reproducible performance analysis" (<https://www.opdi.aero/concepts.html>); "Open data
can be freely used, reused, and distributed provided that the data source is attributed"
(<https://www.opdi.aero/about.html>). Risco de elegibilidade: **zero** — já usamos o
`flight_list` do OPDI em `src/externos.py`, é a mesma fonte.

**Esforço.** Cada arquivo de 10 dias tem ~550 MB (confirmado: `content-length: 547733088`
para 09–19/07/2026; os arquivos de 2026 existem até 29/07–08/08). Jan + jul de 2025 e
2026 = ~13 arquivos ≈ 7 GB de download, filtrando por `osm_airport` em fluxo e guardando
só os 10 aeroportos (deve sobrar bem menos de 100 MB). Ano inteiro de 2025 + jan/jul 2026
= ~44 arquivos ≈ 24 GB — só vale se o teste nos meses de ranking pagar. Meio dia de
trabalho para o adaptador, reaproveitando o padrão de `src/adsb.py`.

**Cuidado.** O OPDI sai da OSN, e a OSN **não ouve o solo** em LFPG, EGLL, LIRF, EDDM e
LTFM (ver item 3, com os números do próprio organizador). Então isto *não* resolve os
aeroportos cegos; melhora LSZH (0,996), LEBL (0,797), EDDF (0,622), EHAM (0,286) e LEMD
(0,175) — e aí o ganho é redundância com o adsb.lol, fora o degelo e a pista, que são
novos em todo lugar.

### 2. Documentar os dados externos num arquivo próprio — e tirar o X-Plane

**O quê.** Criar `docs/dados-externos.md` (ou uma seção fechada no README) com uma tabela
"fonte → para quê → URL → licença → data" cobrindo adsb.lol (ODbL 1.0), IEM/METAR
(domínio público), OurAirports (domínio público), séries diárias EUROCONTROL e OPDI, com o
texto literal de cada licença. E **remover a feature de distância de táxi do `apt.dat` do
X-Plane**: a licença dele não é documentável em fonte primária (achado 11) e a feature já
foi medida em ~0 de ganho.

**Por que.** É condição de prêmio, não enfeite: *"All used external datasets are openly
accessible/usable and documented"* e *"All **additional datasets** used are openly
available under an open source license"*
(<https://prc-data-challenge-2026.netlify.app/eligibility.html>). Estamos em 20º com
~20 s de distância do 3º; se chegarmos ao pódio e a documentação estiver espalhada pelo
README, a comissão avalia "open availability of the code and the relevant documentation"
como critério (precedente 2024/2025, ver abaixo). O concorrente `ahmetabdullahgultekin`
já mantém exatamente esse arquivo (`docs/external_data.md`), com o texto das licenças
transcrito.

**Esforço.** Uma a duas horas. Ganho de RMSE: zero (a remoção do X-Plane é neutra por
medição nossa). Ganho esperado de prêmio: alto.

### 3. Usar a tabela de cobertura OSN do próprio organizador como *prior* por aeroporto

**O quê.** Em setembro de 2026 a PRU publicou `euctrl-pru/opensky-airport-coverage`
(<https://euctrl-pru.github.io/opensky-airport-coverage/>), com CSV/Excel para download,
medindo por aeroporto a fração mediana do táxi-out que chega aos receptores da OSN.
Para os nossos 10 (amostra 5–7/06/2026):

| ICAO | voos vistos | táxi-out recebido (mediana) | início do rastro vs decolagem (min) |
| --- | --- | --- | --- |
| LSZH | 99,8 % | **0,996** | −13,2 |
| LEBL | 99,8 % | **0,797** | −12,4 |
| EDDF | 99,2 % | **0,622** | −8,6 |
| EHAM | 99,7 % | 0,286 | −2,5 |
| LEMD | 99,6 % | 0,175 | −3,1 |
| EDDM | 99,6 % | **0,000** | +0,2 |
| LFPG | 99,9 % | **0,000** | +0,6 |
| EGLL | 99,7 % | **0,000** | +0,8 |
| LIRF | 99,8 % | **0,000** | +0,3 |
| LTFM | 97,8 % | **0,000** | +2,0 |

(<https://euctrl-pru.github.io/opensky-airport-coverage/downloads/opensky-airport-coverage-all-aerodromes-2026.csv>)

**Por que importa para nós, em duas direções.**
- É a confirmação independente do nosso diagnóstico: o erro se concentra onde não há
  ADS-B de solo. E mostra que **adsb.lol ≠ OSN**: nosso recorte do adsb.lol tem cobertura
  boa em EDDM, onde a OSN mede 0,000. O adsb.lol agrega FlyItalyADSB e TheAirTraffic
  (<https://github.com/adsblol/globe_history_2026>), redes que a OSN não tem. Ou seja,
  somar as duas é ganho real, não duplicação (reforça o item 1).
- A coluna "Times measured?" vale **yes** para os 10 aeroportos: o `BLOCK_TIME_UTC_mvt` do
  desafio vem do APDF ("The airport's own records give the real times: off-block,
  take-off, landing, in-block"). Isso é argumento forte para a nossa hipótese de
  "loteria": quando o off-block reportado é cópia do SCHED/LOBT, é falha de preenchimento
  do APDF, não um táxi real de horas.
- O CSV é pequeno (22 KB) e o site tem página por aeroporto com histórico 2024/2025/2026:
  dá uma feature de "quanto esperar de ADS-B aqui" e, principalmente, um *peso* honesto
  para quando confiar no `adsb_*` e quando cair no fallback.

**Licença.** Site GitHub Pages do `euctrl-pru` (EUROCONTROL PRU), construído sobre OPDI.
Mesmo guarda-chuva do OPDI.

**Esforço.** Uma hora (é um CSV de 22 KB).

### 4. Não sondar o leaderboard — e saber que isso é a única prática que o organizador chamou de desleal

**O quê.** A página de ranking de 2026 abre com:
> "We will monitor submissions for attempts to learn from or exploit the ranking process.
> We consider such practices unfair and inconsistent with the purposes and goals of the
> Data Challenge."
(<https://prc-data-challenge-2026.netlify.app/ranking.html>, texto introduzido em
11/08/2026, commit `9ac59cdb`)

**Por que.** É a *única* coisa que as regras de 2026 declaram desleal, e ela é sobre o
processo de ranking (submissões de sondagem, aprender o alvo pelo RMSE devolvido), não
sobre dados externos. Em 2025 a mesma preocupação virou um desenho: a fase 2 usava um
conjunto final escondido *"to avoid the possibility of leaning from the leaderboard
results"* (<https://ansperformance.eu/study/data-challenge/dc2025/ranking.html>, nota de
rodapé 1). Consequência prática para nós: com limite de 5 submissões/dia, **não** fazer
submissões desenhadas para medir o alvo (ex.: constante por aeroporto para inferir
médias). Submeter só candidatas de modelo.

**Esforço.** Zero, é disciplina.

### 5. Preparar a entrega do repositório antes do dia 11, incluindo o canal de entrega

**O quê.** O repositório precisa estar **público, no GitHub, com LICENSE GPLv3**, e o
organizador precisa saber a URL. A página de 2026 não diz como avisar; o precedente de
2025 diz: *"Please let us know the URL of your **public** Github repository either via
email at `challenge AT opensky-network DOT org` or on Discord. **NOTE**: we will consider
till last commit before the deadline."*
(<https://ansperformance.eu/study/data-challenge/dc2025/submission.html>)

**Por que.** "till last commit before the deadline" é literal: o que estiver commitado às
23:59:59 CET de 11/10/2026 é o que conta. Em 2025, 11 dos 20 repositórios da organização
`prc-data-challenge-2025` aparecem com licença GPL-3.0 detectada pelo GitHub e 9 sem
nenhuma — ou seja, metade dos times não cumpriu o item mais fácil da lista.

**Esforço.** Uma hora, mas tem que ser **antes** do dia 11.

---

## Achados, com as fontes

### 1. O que as regras de 2026 dizem, literalmente

Elegibilidade (<https://prc-data-challenge-2026.netlify.app/eligibility.html>):

> The winning submissions/solutions will only be eligible for the awards if:
> - The team is NOT related in any way to sanctioned non-EUROCONTROL states.
> - All used external datasets are openly accessible/usable and documented.
> - All produced **source code** are made openly available on GitHub under the **GNU
>   GPLv3 license**. Note: It will then be forked by the Challenge GitHub account for
>   organisational purposes.
> - All **additional datasets** used are openly available under an open source license.
> - Sufficient **documentation** is provided **to understand and reproduce** the results.
> - The solution is *original*: Teams must use their own original solutions. Re-using any
>   existing implementation is only allowed if the original authors grant you the rights
>   to use their solution and if you made significant modifications to the algorithm or
>   model. In particular, simply re-using existing code and rewriting the data input and
>   output mechanism is not sufficient. Adding parameters to the model and modifying
>   filters to match the specific peculiarities of the data, however, can be considered
>   sufficient.

Ranking (<https://prc-data-challenge-2026.netlify.app/ranking.html>):

> Teams will be ranked based on the **best Root Mean Square Error (RMSE)** achieved across
> their submissions.

Dados (<https://prc-data-challenge-2026.netlify.app/data.html>): a página **não tem**
seção sobre dados externos — ao contrário de 2025, que tinha uma explícita: *"The use of
additional and/or external dataset is permitted if open data and documented."*
(<https://ansperformance.eu/study/data-challenge/dc2025/data.html>). A regra continua, só
migrou inteira para a página de elegibilidade.

**Nada, em nenhuma das três páginas, restringe o *instante* da informação externa.** Não
existe "causal", "antes do off-block", "em tempo real", "no-future-data". A única
restrição é sobre a *origem* (aberta, documentada, licenciada) e sobre o *processo de
ranking*.

### 2. Três mudanças de regra em 2026 que vale conhecer

Li o histórico do repositório do site (<https://github.com/euctrl-pru/prc_data_challenge_website_2026>):

- **13/08/2026, commit `65758293` "remove req"**: foi *removida* a linha
  `To receive the prize money, a **paper** on the methodology must be published in the
  open access paper JOAS.` Ou seja: em 2026, publicar no JOAS **não é mais condição de
  prêmio**; a página inicial só diz que "strongly encourage" (😉 no original). Isso muda
  o cálculo de esforço no fim.
- **04/08/2026, commit `8b47b927`**: a exigência de abrir "source code **and additional
  data sets**" sob GPLv3 foi partida em duas. Hoje: o *código* vai sob GPLv3; os *datasets
  adicionais* só precisam já ser abertos sob licença open source. **Não temos que
  republicar o recorte do adsb.lol sob GPLv3** — ele é ODbL e basta documentar e apontar
  para a origem. (Republicar dados ODbL sob GPLv3 seria, aliás, incompatível.)
- **18/09/2026, commit `a666240f` "fix 10 airports not 11"**: o site dizia 11 aeroportos
  e passou a dizer 10. Repositórios de concorrentes congelados antes dessa data (p.ex.
  `ahmetabdullahgultekin`, que documenta LTAI/Antalya) modelaram um aeroporto a mais.

### 3. Fase final escondida: em 2026 ela **não existe** nas regras publicadas

Precedentes:

- **2024**: duas listas. `submission_set.csv` (105.959 voos) durante o desafio e
  `final_submission_set.csv` (+52.190 voos, total 158.149) publicado *uma semana antes do
  prazo*; o ranking final usou só o segundo
  (<https://ansperformance.eu/study/data-challenge/dc2024/rankings.html>, confirmado no
  data paper <https://doi.org/10.59490/joas.2025.8252>).
- **2025**: duas fases com datas (fase 1 até 09/11, fase 2 de 10/11 a 30/11), arquivo
  separado `fuel_final_submission.parquet` e um arquivo de entrega com nome fixo
  `<team-name>_final.parquet`; e a justificativa em nota de rodapé: *"This is to avoid the
  possibility of leaning from the leaderboard results."*
  (<https://ansperformance.eu/study/data-challenge/dc2025/ranking.html>)
- **2026**: o site tem exatamente cinco páginas (`index`, `data`, `eligibility`,
  `ranking`, `rationale` — verificado no `_quarto.yml` do repositório fonte). A página de
  dados lista **um** `ranking.parquet` (jan+jul 2026) e **um** `submitting.parquet`. Não
  há `final_`, não há fase 2, não há data intermediária. O prazo é único: 11/10/2026
  23:59:59 CET.

[INFERÊNCIA] O risco de uma fase final surpresa não é zero — o `index.html` avisa *"We
reserve the right to change the rules or even stop the challenge due to any
circumstances."* e o canal oficial para "urgent and important news" é o Discord
`#prc-data-competition`, não o site. Mas, pelo padrão das duas edições anteriores, uma
fase final seria **anunciada com ~1 semana de antecedência e consistiria em mais linhas
do mesmo período**, não em outro período. Como o nosso pipeline já roda o ano inteiro de
2025 e jan/jul 2026 ponta a ponta, uma lista maior dos mesmos meses não exigiria nada
novo; um *período* novo (ex.: fev 2026) exigiria baixar adsb.lol e METAR desses dias — o
que `src/adsb.py` já faz por dia. Vale manter essa capacidade viva até o dia 11.

### 4. Como o pódio foi decidido nas edições anteriores (e o que 2026 omite)

2024 e 2025 tinham, idêntico, um parágrafo "Final Prize":

> The Selection Committee will review and discuss the top ranking teams and select final
> Awardees considering 1. their final ranking score, 2. the originality of the proposed
> model 3. the open availability of the code and the relevant documentation provided.
> By participating in the Challenge, you agree to be bound by the final decision of the
> PRC Data Challenge selection committee.
(<https://ansperformance.eu/study/data-challenge/dc2024/rankings.html> e
<https://ansperformance.eu/study/data-challenge/dc2025/ranking.html>)

**Em 2026 esse parágrafo não foi publicado.** A página de elegibilidade de 2026 termina no
item de originalidade; a de ranking só diz "best RMSE". [INFERÊNCIA] Não acho que a
comissão tenha deixado de existir — a frase sobre "monitorar submissões" implica
julgamento humano, e os 5000 EUR para os 3 primeiros são os mesmos de 2025. O prudente é
assumir que os três critérios de 2024/2025 continuam valendo de fato: **posição + modelo
original + código e documentação abertos**. Isso dá peso aos itens 2 e 5 do resumo.

Sobre empates: nenhuma edição publicou regra de desempate. Em 2024 o ranking foi por RMSE
com 2 casas decimais e a tabela final mostra a *melhor* submissão de cada time com a data
e o número da versão (<https://ansperformance.eu/study/data-challenge/dc2024/rankings.html>).
[INFERÊNCIA] Com RMSE em segundos e ~4,1 M de movimentos, empate exato é improvável; se
houvesse, a data da submissão é o desempate natural e é o único campo extra na tabela.

### 5. O argumento decisivo: o próprio PRC deriva off-block de ADS-B e chama isso de taxi-out

A página de conceitos do OPDI, do EUROCONTROL/PRC, define explicitamente:

> "The taxi-out sub-phase can be framed by the ground portion from `off-block` to
> `rwy-entry` event"
(<https://www.opdi.aero/concepts.html>)

e lista `off-block` (T04), `end of push back` (T05) e `enter runway for take-off` (T06)
como eventos a extrair **de trajetórias ADS-B** ("These tables are currently derived from
ADS-B trajectory data kindly made available by OpenSky Network"). A metodologia v0.0.2
descreve como: posições casadas contra grades H3 resolução 12 construídas sobre
`parking_position`/`runway`/`taxiway` do OpenStreetMap
(<https://www.opdi.aero/methodology.html>), com a ferramenta aberta `HexAeroPy`
(<https://github.com/euctrl-pru/HexAeroPy>).

Ou seja: **a nossa feature de off-block por ADS-B é a reimplementação do método do próprio
organizador.** Não é só permitida; é a técnica que ele publica, documenta e mantém em
código aberto.

### 6. Por que a objeção "usa informação posterior ao evento" não se aplica

A página de justificativa de 2026 diz para que serve o modelo:

> "A model to estimate taxi-out time can be used in **post-operations analysis** to
> identify constrained operations intervals and measure excess fuel burnt/CO2 production
> compared to unconstrained status."
(<https://prc-data-challenge-2026.netlify.app/rationale.html>)

Análise *pós-operacional*. O caso de uso declarado é retrospectivo: reconstruir o táxi-out
onde o aeroporto não o mede. Um modelo que só pode usar informação anterior ao off-block
seria um modelo de *previsão em tempo real* — que não é o que foi pedido. O desenho da
tarefa confirma: o dataset de ranking entrega `MVT_TIME_UTC_mvt` (a **decolagem**, que é
posterior ao off-block) e esconde só `BLOCK_TIME_UTC_mvt` e `TAXITIME_SEC_mvt`
(<https://prc-data-challenge-2026.netlify.app/data.html>). O alvo é literalmente
`MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt`, com o minuendo dado. **O organizador nos entrega
de mão beijada um dado posterior ao evento que estamos estimando.** Não há leitura
coerente das regras em que isso seja legal e observar o mesmo avião no rastro ADS-B não
seja.

Precedente de 2024 na mesma direção: a lista de voos de treino incluía `taxiout_time`
como *feature* e o organizador instruiu a usar o intervalo
`[actual_offblock_time + taxiout_time, arrival_time]` para recortar a parte aérea do
rastro (<https://ansperformance.eu/study/data-challenge/dc2024/data.html>) — misturando
livremente tempos de solo observados com trajetória.

### 7. O que os concorrentes estão fazendo (sinal de que a leitura é consensual)

- O README do nosso repositório já registra, do Discord do desafio, que o 3º colocado
  (SoK) usa adsb.lol + clima + stands do X-Plane e que "o organizador confirmou que dado
  aberto declarado vale" (`README.md`, roadmap item A). Não consegui verificar o Discord
  de forma independente (exige conta na OSN e o canal não é indexado), então trato como
  **[INFERÊNCIA] confirmada só pela nossa própria anotação**.
- Repositórios públicos de 2026 que declaram abertamente dados externos: `victoralcadi`
  ("every external dataset used is openly accessible, openly licensed, and documented"),
  `ahmetabdullahgultekin` (tabela de licenças com METAR/IEM, OurAirports, séries
  EUROCONTROL), `javidmardanov` (GPLv3 + pesos TimesFM com licença própria). Nenhum
  repositório público menciona qualquer restrição temporal imposta pelo organizador.
- Em 2025, times publicaram soluções usando ERA5, METAR e até tráfego da OpenSky como
  dados externos (`nurturing-telephone`, `bright-lobster`, `unique-honey`) e ficaram no
  pódio/organização sem objeção.

### 8. OpenSky: por que o Trino continua fora e o adsb.lol continua dentro

Os Termos de Uso da OpenSky concedem licença *"solely for the purpose of non-profit
research and non-profit education"*, exigem licença escrita para entidades comerciais e
para uso operacional da API REST
(<https://opensky-network.org/about/terms-of-use>). Isso **não é uma licença open source**
no sentido do critério "All additional datasets used are openly available under an open
source license" — ela não é redistribuível nem livre de finalidade. Mantém-se a decisão
de não usar o banco histórico Trino.

O adsb.lol é o oposto: *"This database is made available under the Open Database License:
http://opendatacommons.org/licenses/odbl/1.0/"*
(<https://github.com/adsblol/globe_history_2026>). ODbL é licença de dados abertos
reconhecida (Open Knowledge Foundation), satisfaz o critério ao pé da letra, e exige
apenas atribuição e *share-alike* para bancos derivados distribuídos — como **não
redistribuímos o banco**, só publicamos código que o baixa e features derivadas por voo,
a obrigação prática é atribuição (já no README e a repetir em `docs/dados-externos.md`).

O OPDI vem com atribuição também: *"Open data can be freely used, reused, and distributed
provided that the data source is attributed"* (<https://www.opdi.aero/about.html>).

### 9. Detalhes operacionais da submissão que não podem falhar

Da página de ranking de 2026:

- Nome do arquivo: `<team-name>_v<incremental integer>.parquet`, no bucket do time, via
  **MinIO CLI** — *"You will **NOT** be able to submit via the webinterface"* (aviso
  acrescentado em 30/09/2026, commit `a9486526`; ou seja, é novo).
- A submissão tem que bater linha a linha com `submitting.parquet`: o script dá erro se
  um `MVT_ID_mvt` não casa, se faltam linhas ou se sobram linhas.
- **5 submissões por dia**, máximo **1 GB por bucket** (limite introduzido em 06/09/2026,
  aumentado em 07/09). Com 1 GB de teto e ~1,1 MB por submissão, o teto não aperta, mas
  vale limpar submissões velhas se houver reenvios em massa.
- O ranking oficial sai da API:
  `https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard`
  (paginada por `cursor=`).

### 10. Brasil não é estado EUROCONTROL — e isso **não** nos desqualifica

Esta é a única cláusula de elegibilidade que poderia nos derrubar, então vale o detalhe.
A EUROCONTROL tem **42 Estados-Membros e 2 Estados de Acordo Abrangente** (Israel e
Marrocos); o Brasil não é nenhum dos dois
(<https://www.eurocontrol.int/our-member-and-comprehensive-agreement-states>).

O texto de 2026 é ambíguo se lido isolado:
> "Restrictions apply to teams originating from, or affiliated with, institutions or
> companies in countries that are either not EUROCONTROL Member States or Comprehensive
> Agreement States, or are subject to sanctions."

Três evidências de que a leitura correta é **conjuntiva** (não-EUROCONTROL **e**
sancionado), e não disjuntiva:

1. A mesma página começa com *"Participation is open to teams worldwide, subject to
   applicable sanctions and restrictions"*, e a página inicial fecha a questão:
   *"Note: The participation is open to all, but teams from sanctioned countries are
   excluded."* (<https://prc-data-challenge-2026.netlify.app/>)
2. A redação **anterior** da mesma frase, antes do commit `8b47b927` de 04/08/2026, era
   explicitamente conjuntiva: *"The restrictions are mainly about teams originating or
   affilaited to institutions/companies from (non ECTL) countries subject to sanctions."*
   O motivo declarado é prático e continua no texto atual: a impossibilidade de
   transferir o prêmio para países sancionados.
3. A prova empírica: contei os países declarados nas 480 páginas de times do site oficial.
   **72 times da Índia, 59 dos Estados Unidos, 14 do Canadá, 10 de Singapura, 8 do Japão,
   5 do Brasil** — nenhum deles é Estado-Membro ou de Acordo Abrangente, todos foram
   aprovados pelo organizador (a inscrição é revisada: *"Your request will be assessed by
   the PRC Data Challenge team for approval"*) e publicados. Nosso próprio time está lá,
   com `subtitle: Brazil`
   (<https://prc-data-challenge-2026.netlify.app/teams/outgoing-boat.html>).

O Brasil não consta no mapa de sanções da UE (<https://www.sanctionsmap.eu/>). Conclusão:
**elegíveis**. O item 1 da checklist pode ser considerado cumprido, com estas citações
como lastro se alguém perguntar.

### 11. A única pendência de licença: `apt.dat` do X-Plane

`src/mapa.py` afirma, no docstring, que o `apt.dat` do X-Plane Scenery Gateway é
"GPL v2+". **Não consegui confirmar isso em fonte primária.** A página oficial do Gateway
(<https://gateway.x-plane.com/>) e o artigo de referência da Laminar Research
(<https://developer.x-plane.com/article/airport-scenery-gateway/>) descrevem o processo de
submissão e os status das cenas, mas **não declaram licença nenhuma** para os dados. Os
arquivos que já baixamos (`data/mapa/EDDF.dat` etc.) começam em `1200 Generated by
WorldEditor 2.6.0r1` — **sem cabeçalho de licença**. A afirmação "GPL v2+" circula em
fóruns de desenvolvedores (ex.: <https://www.fsdeveloper.com/forum/threads/apt-dat-usage-and-license.449292>),
não em documento da Laminar.

Isso colide de frente com o critério *"All additional datasets used are openly available
under an open source license"*.

**Recomendação: remover a feature.** O custo é zero — o README já registra que a distância
de táxi stand→cabeceira do X-Plane rendeu "~0 de ganho" quando testada. Manter um dado sem
licença documentável para ganhar nada é trocar o prêmio por ruído. Se por algum motivo a
feature voltar a valer, a alternativa com licença limpa é o **OpenStreetMap** (ODbL,
`aeroway=taxiway|parking_position|runway`) — que é, aliás, exatamente o que o OPDI usa
para construir seus eventos de solo (<https://www.opdi.aero/methodology.html>).

---

## Checklist de elegibilidade para o prêmio

Marque tudo antes de 11/10/2026 23:59:59 CET.

| # | Requisito | Texto-fonte | Como provamos | Estado |
| --- | --- | --- | --- | --- |
| 1 | Time não ligado a estado sancionado | eligibility.html | **Verificado** (achado 10): o Brasil não é Estado-Membro nem de Acordo Abrangente, mas também não é sancionado, e o site oficial publica times aprovados de Índia (72), EUA (59), Japão (8) e Brasil (5), inclusive o nosso | **ok** |
| 2 | Código-fonte **público no GitHub** | eligibility.html | `https://github.com/EnioAguiar/prc-taxiout-2026` responde **404 hoje (01/10)**, ou seja, está privado | **a fazer, bloqueante** |
| 3 | Licença **GNU GPLv3** no repositório | eligibility.html | **Verificado**: `LICENSE` existe e começa com "GNU GENERAL PUBLIC LICENSE / Version 3, 29 June 2007" | **ok** |
| 4 | Aceitar que o organizador **forke** o repositório | eligibility.html ("It will then be forked by the Challenge GitHub account") | nada a fazer, mas o repositório não pode ser apagado/privado depois | — |
| 5 | **Todo** dataset externo é aberto, usável e **documentado** | eligibility.html (dois itens separados) | `docs/dados-externos.md` com fonte, URL, licença, texto literal da licença e data | **a fazer** (item 2 do resumo) |
| 6 | Datasets adicionais sob **licença open source** | eligibility.html | adsb.lol = ODbL 1.0 (texto no README do `globe_history_2026`); IEM/METAR = domínio público; OurAirports = domínio público; OPDI/EUROCONTROL = open data com atribuição; **X-Plane `apt.dat` = problema, ver achado 11** | 1 pendência |
| 7 | **Nenhum** dado sob licença restritiva | eligibility.html | OpenSky Trino **fora** (ToU não é open source); nada de ADS-B Exchange pago, Flightradar24, Cirium | ok, por decisão |
| 8 | Documentação suficiente **para entender e reproduzir** | eligibility.html | README com pipeline ponta a ponta, comandos, variáveis de ambiente, `docs/mapa.md`; idealmente um "como reproduzir do zero" com tempos e espaço em disco | revisar |
| 9 | Solução **original** (não reaproveitamento de implementação alheia) | eligibility.html | nosso código é próprio; citar inspirações (método OPDI/HexAero, ATXOT) como *referência*, não como código copiado. Se alguma função vier de terceiros, declarar licença e a modificação significativa | revisar |
| 10 | Avisar a URL do repositório ao organizador | precedente 2025 (submission.html) | e-mail `challenge AT opensky-network DOT org` **e** Discord `#prc-data-competition` | a fazer |
| 11 | Último commit **antes** do prazo | precedente 2025 ("till last commit before the deadline") | congelar o repositório em 11/10 | a fazer |
| 12 | Não sondar o leaderboard | ranking.html 2026 | submeter só candidatas de modelo, nunca sondas | disciplina |
| 13 | Paper no JOAS | **não é mais condição** em 2026 (removido no commit `65758293`) | opcional, "strongly encouraged" | opcional |

---

## Descartado / irrelevante

- **"Regra de causalidade" / proibição de dados futuros.** Procurei nas cinco páginas de
  2026, nas páginas de 2024 e 2025, nos dois data papers do JOAS
  (<https://doi.org/10.59490/joas.2025.8252>, <https://doi.org/10.59490/joas.2026.8750>) e
  no histórico de commits do site. **Não existe.** Parar de investigar isso.
- **Issues/discussões públicas no GitHub do organizador.** As organizações
  `prc-data-challenge-2024` e `prc-data-challenge-2025` são só forks de soluções, com
  issues desabilitadas ou vazias (busca na API retornou 0 resultados para qualquer termo
  de regra). `prc-data-challenge-2026` **ainda não existe** (404) — será criada depois do
  prazo para receber os forks. Não há fórum público de regras; o canal é o Discord.
- **Banco histórico Trino da OpenSky.** Confirmado fora por licença (item 8 acima), não
  por regra do desafio. A decisão anterior do time está certa e documentada.
- **Publicar no JOAS como condição.** Era condição no rascunho, foi removida em 13/08/2026.
  Não gastar esforço nisso antes do prazo.
- **Republicar o recorte do adsb.lol sob GPLv3.** A regra de 2026 separa código (GPLv3) de
  datasets (só precisam ser abertos). Republicar dados ODbL sob GPLv3 seria errado e
  desnecessário.
- **ADS-B Exchange / Flightradar24 / adsbhub como fontes extras para LFPG, EGLL, LIRF,
  LTFM.** Nenhuma tem licença open source para histórico; cairiam no item 6 da checklist.
  Os aeroportos cegos continuam um problema de modelagem, não de nova antena — a não ser
  que apareça fonte aberta nova.
- **Regra de desempate.** Nenhuma edição publicou uma. Não há nada a explorar aí.
