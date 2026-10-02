# Dados abertos de aeroporto (pista em uso, degelo, obras, greves) — 2026-10-01

Escopo: fontes **abertas e baixáveis** que cubram jan/jul 2025 **e** jan/jul 2026 para os 10
aeroportos (EDDM, EDDF, EGLL, EHAM, LFPG, LEMD, LEBL, LIRF, LSZH, LTFM).
Tudo que não tem licença clara está marcado. Nada foi baixado além de cabeçalhos HTTP;
ganhos são hipótese, não medição.

## Resumo (o que vale testar)

1. **OPDI `flight_events` v0.0.2 (EUROCONTROL/PRC + OpenSky) — jan/2022 a 31/07/2026.**
   *O quê:* parquet por janelas de 10 dias com milestones 4D por voo: `off-block` (T04),
   `end of pushback` (T05), `enter runway for take-off` (T06), `lift-off` (T08), com campo `info`
   trazendo **posição de estacionamento** no off-block e **identificador de pista** no take-off
   ([conceitos](https://www.opdi.aero/concepts), [download](https://www.opdi.aero/flight-event-data.html)).
   *Por que pode derrubar RMSE:* é exatamente o par (início de táxi, entrada na pista) que falta nos
   aeroportos sem cobertura de solo do adsb.lol (LTFM, LFPG, EGLL, LIRF). Mesmo onde o off-block não
   for detectado, T06/T08 dão **pista de partida por voo** (feature nova, direta) e permitem reconstruir a
   **configuração de pista por faixa de 30 min** nos 10 aeroportos com uma única fonte. Também é um
   segundo medidor independente para marcar "cópia/loteria" (se o OPDI vê off-block ~X e o AOBT_3
   reportado é 3 h antes, é cópia).
   *Fonte/licença:* `https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_{YYYYMMDD}_{YYYYMMDD+10}.parquet`;
   o PRC declara OPDI como open data, "free to use, reuse and distribute provided the source is
   attributed" ([about](https://www.opdi.aero/about)) — não achei string CC-BY formal.
   *Atenção à regra:* a proibição do organizador é à **Trino/histórico do OpenSky**; o OPDI é um dataset
   publicado pelo EUROCONTROL (derivado de OSN). É uso de dataset aberto publicado, não de query ao
   banco do OpenSky — mas convém confirmar com o organizador antes de depender dele. [INFERÊNCIA: a
   distinção é nossa leitura da regra.]
   *Esforço:* médio. Já usamos o `flight_list` do OPDI, então o pipeline de download/join existe.
   Detalhe prático verificado por `curl -I` em 01/10/2026: a grade de 10 dias começa em **2022-01-01**,
   então os arquivos **não** caem no dia 1 do mês. Os que cobrem nossos meses são
   `20250105_20250115` (245 MB), `20250115_20250125` (225 MB), … `20250704_20250714` (415 MB),
   `20250714_20250724` (389 MB), `20251231_20260110` (41 MB — bem menor que os vizinhos, conferir),
   `20260110_20260120` (272 MB), `20260120_20260130` (239 MB), `20260629_20260709` (506 MB),
   `20260719_20260729` (554 MB). São ~13 arquivos, 3–4 GB no total: ler com `pyarrow`/`duckdb`
   por row group e filtrar pelos `flight_id` dos ADEP alvo antes de materializar, nunca `read_parquet`
   inteiro em pandas.

2. **Códigos de causa diários do EUROCONTROL (APT-DLY e ERT-DLY) — jan/2019 a ago/2026, CSV/XLSX.**
   *O quê:* o dataset *Airport Arrival ATFM Delay* traz, **por aeroporto e por dia**, minutos de atraso
   ATFM por código: **D = De-icing**, **G = Aerodrome Capacity** (obras/capacidade de pista),
   **I = Industrial Action (ATC)**, **N = Industrial Action (não-ATC)**, **W = Weather**,
   **S = ATC Staffing**, **E/T = equipamento**, **P = evento especial**
   ([dicionário de colunas](https://ansperformance.eu/reference/dataset/airport-arrival-atfm-delay/)).
   O equivalente en-route por FIR (`En-Route_ATFM_Delay_FIR`, diário até ago/2026) dá os mesmos códigos
   para o espaço aéreo ([lista de datasets](https://ansperformance.eu/data/)).
   *Por que:* resolve, com um CSV, três itens do pedido de uma vez — **dia de degelo** (código D, sinal
   operacional, não meteorológico: só aparece quando houve regulação por degelo), **dia de obra/capacidade
   de aeródromo** (G) e **dia de greve** (I/N/S) — sem precisar de arquivo de NOTAM. Nossa cauda pesada
   está concentrada em dias assim; hoje o modelo só vê METAR e as séries diárias de slot adherence /
   pre-departure delay.
   *Licença:* EUROCONTROL Aviation Intelligence Portal, mesma origem das séries diárias que já usamos.
   *Esforço:* baixo (um download, join por `APT_ICAO` × data; en-route por FIR local).
   *Ressalva:* é atraso de **chegada** no aeroporto; serve como indicador do dia, não do voo.

3. **EGLL: programa oficial de alternância de pista 2025 e 2026 (PDF da Heathrow).**
   *O quê:* tabela semana-a-semana com a pista de **pouso** em operação a oeste, separada em
   `06:00–15:00` e `15:00 até a última partida`, mais o ciclo noturno de 4 semanas
   ([2026](https://www.heathrow.com/content/dam/heathrow/web/common/documents/company/local-community/noise/operations/runway-alternation/Heathrow_Runway_Alternation_Programme_2026.pdf),
   [2025](https://www.heathrow.com/content/dam/heathrow/web/common/documents/company/local-community/noise/operations/runway-alternation/Heathrow_Runway_Alternation_Programme_2025.pdf)).
   O 2026 cobre as semanas de 5 jan a 28 dez, incluindo jul/2026.
   *Por que:* em modo segregado a **partida usa a outra pista** (pousa 27L → decola 27R e vice-versa), e
   a troca às 15:00 muda a distância stand→pista para metade do aeroporto. EGLL é um dos aeroportos sem
   cobertura de solo; isso dá pista de partida determinística com ~2 bits por voo, usando o vento do
   METAR (que já temos) só para decidir oeste (≈70% do ano) × leste.
   *Licença:* PDF marcado "Classification: Public", sem licença formal declarada.
   *Esforço:* baixo (duas tabelas de ~52 linhas digitadas/parseadas; regra `westerly → dep = a outra pista`).
   *Ressalva do próprio documento:* alternância é suspensa em mau tempo/obra, e há desvio no período noturno
   por causa do recapeamento ([runway resurfacing](https://www.heathrow.com/company/local-community/noise/news/runway-resurfacing)).

4. **Planos de degelo publicados (FRA, MUC, LHR, ZRH) → feature estrutural de degelo, não só meteorológica.**
   *O quê:* documentos operacionais públicos que dizem **onde** se faz degelo e **quanto custa em tempo**:
   Frankfurt *Aircraft Deicing Plan* (pads DP1/DP2 só servem partidas da RWY 18, com restrição para 07;
   DP3 N7 e DP4 V159/V161 preferencialmente para 07/25; estados A-CDM `ICE=P` posição × `ICE=R` remoto;
   ADIT/EDIT) ([DIP 2024/25](https://cdm.frankfurt-airport.com/content/dam/fraport-company-cdm/documents/binary/documents/deicing-procedure/EN-DIP%202024-2025.pdf/_jcr_content/renditions/original./EN-DIP%202024-2025.pdf));
   Munique *Deicing Plan 2025/2026* ([PDF](https://www.munich-airport.de/_b/0000000000000035898332bb68e77735/deicing-plan-2025-2026.pdf));
   Heathrow *HADIP* temporada 2025-26, com pads JEDI/remoto
   ([PDF](https://www.heathrow.com/content/dam/heathrow/web/common/documents/company/team-heathrow/airside/winter-operations/Heathrow_Aircraft_De-Icing_Plan_(HADIP)_Winter_Season_2025-26.pdf));
   Zurique *Winter Operations*
   ([PDF](https://media.flughafen-zuerich.ch/-/jssmedia/airport/portal/dokumente/business/airlines-and-handling/flight-operations/winter-operations/winter_operations_2025.pdf)).
   *Por que:* metade do ranking é **janeiro**. Hoje o `met_degelo` é só "frio + úmido". O que muda o táxi-out
   é a *política*: degelo remoto insere uma parada de vários minutos entre o off-block e a pista (e em FRA
   o pad depende da pista de partida), degelo na posição é absorvido antes do off-block. Dá para montar
   `degelo_esperado = f(aeroporto, política, pista de partida, porte da aeronave, condição METAR)` como
   offset explícito, em vez de deixar o modelo descobrir sozinho com poucos exemplos.
   *Licença:* PDFs públicos dos sites dos aeroportos, **sem licença explícita** (uso como documentação de
   regra, não como dataset; o dado por voo — ACZT/AEZT/ADIT — não é público).
   *Esforço:* médio-baixo (ler 4 PDFs, codificar ~10 regras por aeroporto). Combinar com o código D do item 2.

5. **Obras e fechamentos: arquivo aberto de NOTAM do Reino Unido + calendários públicos de obra.**
   *O quê:* `Jonty/uk-notam-archive` — arquivo horário de **todos os NOTAM do UK** (PIB completo em XML,
   via feed sem autenticação do NATS AIS), código MIT, histórico no git desde 2023
   ([repo](https://github.com/Jonty/uk-notam-archive)) → fechamentos de pista/taxiway em EGLL em jan/jul de
   2025 e 2026 reconstruídos por commit. Complementos por aeroporto: página de recapeamento da Heathrow
   (obras noturnas 2026 + 7 semanas de taxiway até 19/12)
   ([link](https://www.heathrow.com/company/local-community/noise/news/runway-resurfacing)),
   previsão semanal de tráfego/manutenção da Schiphol
   ([link](https://www.schiphol.nl/nl/schiphol-als-buur/vooruitblik-vliegverkeer)) e as obras do Groupe ADP
   ([link](https://entrevoisins.groupeadp.fr/projets)).
   *Por que:* fechamento de doublet/pista muda a pista de partida de blocos inteiros do aeroporto por semanas.
   *Licença:* repo MIT (código); o conteúdo NOTAM vem do NATS AIS e **não tem licença aberta declarada** —
   marcar como risco se formos usar no envio final.
   *Esforço:* médio (andar no histórico do git + parser de NOTAM Q/A/B/C/E). Cobre só EGLL; para os outros 9
   não achei arquivo aberto equivalente — usar o código **G** do item 2 como proxy diário.

---

## Tabela: fonte × aeroporto × licença × cobertura

| Fonte | Aeroportos | O que dá | Resolução | Cobre jan/jul 2025 | Cobre jan/jul 2026 | Licença | Dificuldade |
|---|---|---|---|---|---|---|---|
| [OPDI `flight_events` v0.0.2](https://www.opdi.aero/flight-event-data.html) | todos os 10 | off-block, fim de pushback, entrada na pista, lift-off; `info` = stand / **pista** | por voo | sim | sim (até 31/07/2026) | "open data" do PRC/EUROCONTROL com atribuição; sem nome formal de licença ([about](https://www.opdi.aero/about)) | média (parquet 10 dias; filtrar antes de materializar) |
| [OPDI `flight_list` / `measurements`](https://www.opdi.aero/data) | todos os 10 | lista de voos, métricas por evento | por voo | sim | sim | idem | já usamos o flight_list |
| [Airport Arrival ATFM Delay (APT-DLY)](https://ansperformance.eu/reference/dataset/airport-arrival-atfm-delay/) | todos os 10 | causa do atraso: **D degelo**, **G capacidade de aeródromo/obra**, **I/N greve**, S staffing, W, P, E/T | dia × aeroporto | sim | sim (até ago/2026) | EUROCONTROL Aviation Intelligence Portal (aberto, com atribuição) | baixa |
| [En-Route ATFM Delay FIR (ERT-DLY)](https://ansperformance.eu/data/) | FIRs de todos | mesmas causas no espaço aéreo (greve francesa etc.) | dia × FIR | sim | sim (até ago/2026) | idem | baixa |
| [ATC Pre-Departure Delay / ATFM Slot Adherence](https://ansperformance.eu/data/) | todos os 10 | já em uso | dia × aeroporto | sim | sim | idem | — (já usado) |
| [Heathrow Runway Alternation Programme 2025/2026](https://www.heathrow.com/content/dam/heathrow/web/common/documents/company/local-community/noise/operations/runway-alternation/Heathrow_Runway_Alternation_Programme_2026.pdf) | EGLL | pista de pouso por semana × meio-dia (⇒ pista de partida) + ciclo noturno | semana × 2 blocos | sim | sim (5 jan–28 dez 2026) | "Classification: Public", sem licença formal | baixa |
| [Heathrow runway resurfacing](https://www.heathrow.com/company/local-community/noise/news/runway-resurfacing) | EGLL | desvios da alternância por obra (noite), taxiways até 19/12 | evento/texto | parcial | sim | público, sem licença | baixa (manual) |
| [Jonty/uk-notam-archive](https://github.com/Jonty/uk-notam-archive) | EGLL | NOTAM completos do UK (PIB XML), histórico horário | hora | sim (git desde 2023) | sim (se o job segue ativo) | repo MIT; **conteúdo NOTAM do NATS sem licença aberta** | média |
| [Frankfurt Aircraft Deicing Plan](https://cdm.frankfurt-airport.com/content/dam/fraport-company-cdm/documents/binary/documents/deicing-procedure/EN-DIP%202024-2025.pdf/_jcr_content/renditions/original./EN-DIP%202024-2025.pdf) | EDDF | pads DP1–DP4 ligados à pista de partida, posição × remoto, EDIT/ADIT, janela 15/10–30/04 | regra estática | regra da temporada | regra da temporada | PDF público, sem licença | baixa |
| [Munich Deicing Plan 2025/26](https://www.munich-airport.de/_b/0000000000000035898332bb68e77735/deicing-plan-2025-2026.pdf) | EDDM | procedimento de degelo (motores ligados, posições) | regra estática | sim | sim | PDF público, sem licença | baixa |
| [Heathrow HADIP 2025-26](https://www.heathrow.com/content/dam/heathrow/web/common/documents/company/team-heathrow/airside/winter-operations/Heathrow_Aircraft_De-Icing_Plan_(HADIP)_Winter_Season_2025-26.pdf) | EGLL | pads JEDI/remoto, plano de inverno | regra estática | sim | sim | PDF público, sem licença | baixa |
| [Zurich Winter Operations](https://media.flughafen-zuerich.ch/-/jssmedia/airport/portal/dokumente/business/airlines-and-handling/flight-operations/winter-operations/winter_operations_2025.pdf) | LSZH | organização do inverno/degelo | regra estática | sim | sim | PDF público, sem licença | baixa |
| [Schiphol — vooruitblik/baanonderhoud](https://www.schiphol.nl/nl/schiphol-als-buur/vooruitblik-vliegverkeer) | EHAM | uso de pista esperado e manutenção de pista por semana | semana | não (só semana corrente) | só prospectivo | site público, sem licença | baixa, mas sem histórico |
| [LVNL baangebruik](https://www.lvnl.nl/omgeving/actueel-baangebruik-schiphol) | EHAM | pista em uso | tempo real | **não** | não | sem licença | — |
| [Flughafen Zürich — Bewegungsstatistik](https://www.flughafen-zuerich.ch/de/unternehmen/verantwortung/laerm-und-schallschutz/bewegungsstatistik) | LSZH | partidas/chegadas com **pista escolhida**, mas só últimos 10 dias; PDFs anuais de rotas | 10 dias / ano | **não** | não | sem licença | — |
| [Groupe ADP — Entre voisins / travaux](https://entrevoisins.groupeadp.fr/projets) | LFPG | obras anunciadas | evento | parcial | parcial | site público, sem licença | média (texto livre) |
| [Fraport — Messberichte Fluglärm](https://www.fraport.com/) | EDDF | distribuição por Betriebsrichtung (07/25/18) | mês | sim | provável | PDF público, sem licença | baixa, mas resolução mensal |
| [EUROCONTROL European Aviation Overview (semanal)](https://www.eurocontrol.int/publication/eurocontrol-european-aviation-overview-archive-2025) | rede | atrasos por causa, dias de greve comentados | semana (PDF) | sim | sim | publicação EUROCONTROL | média (PDF) |
| [NOP public — headline news / industrial action](https://www.public.nm.eurocontrol.int/PUBPORTAL) | rede | avisos de greve com NOTAM (ex.: `LF_STRIKE_10_09_25.pdf`) | evento | parcial (sem arquivo) | parcial | portal público, sem licença | alta (não arquiva) |
| LTFM, LIRF, LEMD, LEBL: fonte própria de pista em uso | LTFM/LIRF/LEMD/LEBL | — | — | **não encontrado** | não | — | usar OPDI (item 1) |

Notas de cobertura: a página de datasets do ansperformance ainda anuncia OPDI "Jan 2022 - May 2026",
mas o portal do próprio OPDI e a lista de arquivos de `flight_events` vão até **31/07/2026** — ou seja,
cobre os dois meses do ranking.

## Achados por tema

### Pista em uso / configuração
- **A via uniforme é o OPDI**: o evento de take-off carrega o identificador de pista no campo `info`, e o
  próprio documento de conceitos cita "usando o touch-down com a informação contextual (RWY) podemos
  calcular a utilização de pista" ([concepts](https://www.opdi.aero/concepts)). Isso resolve os 10
  aeroportos com uma fonte só, inclusive LTFM/LIRF/LEMD/LEBL, onde **não achei** publicação de pista em uso.
- **EGLL** é o único com calendário determinístico publicado (item 3 do resumo). Em operação a leste não há
  alternância diurna; o documento de 2026 explica os códigos e que oeste é ≈70% do ano.
- **EHAM**: LVNL e Schiphol publicam uso de pista **atual** e previsão semanal, não histórico baixável.
  O histórico oficial existe em relatórios de fiscalização da ILT (`Handhavingsrapportage Schiphol`), mas em
  agregado anual/por gebruiksjaar — resolução inútil para nós.
- **LSZH**: a página de Bewegungsstatistik tem movimentos com pista escolhida, mas só dos **últimos 10 dias**
  (sem arquivo), e PDFs anuais de ocupação de rotas.
- **EDDF**: relatórios de ruído mensais com repartição por Betriebsrichtung (07/25/18) — mensal, pouco útil
  dado que já temos vento do METAR.

### Degelo
- Não existe (achei) nenhum dado **por voo** de degelo aberto: os tempos ACZT/AEZT/ADIT de Frankfurt circulam
  só nos sistemas A-CDM dos parceiros (CSA-Tool, INFO*plus*).
- O que existe aberto e com granularidade diária é o **código D (De-icing)** do APT-DLY — e ele é um sinal
  *operacional*, não meteorológico.
- O que existe aberto e estrutural são os planos de degelo (FRA/MUC/LHR/ZRH). O detalhe mais acionável:
  em FRA, os pads de degelo estão amarrados à **pista de partida** (DP1/DP2 só para partidas da RWY 18;
  DP3/DP4 preferencialmente para 07/25), e o plano avisa que "restrições de capacidade por inverno,
  distâncias entre posição e pista designada podem resultar em **tempos de táxi estendidos**", recomendando
  tabelas de HOT mais longas. Ou seja: em dia de degelo remoto o táxi-out tem um patamar próprio por pista.

### Obras e fechamentos
- Arquivo aberto de NOTAM **só para o Reino Unido** (repo MIT com histórico horário). Para FR/DE/NL/ES/IT/CH/TR
  não achei arquivo aberto; os serviços comerciais cobrem (ver "Descartado").
- Fallback aberto e uniforme: código **G (Aerodrome Capacity)** do APT-DLY por dia e aeroporto, mais, quando
  precisar de causa, as páginas de obra dos próprios aeroportos (Heathrow resurfacing, ADP travaux,
  Schiphol baanonderhoud).

### Greves e disrupção de rede
- Estruturado e aberto: códigos **I** (greve ATC), **N** (greve não-ATC) e **S** (staffing) no APT-DLY por
  aeroporto/dia e no ERT-DLY por FIR/dia (ex.: greve francesa de 3–4 jul/2025, que o EUROCONTROL documentou
  em publicação própria — [Aviation Trends 9](https://www.eurocontrol.int/publication/impact-french-atc-strike-3-4-july-2025-european-aviation)).
  **Jul/2025 está no nosso treino e jul/2026 está no ranking**, então um flag de greve por FIR tem chance real.
- Não estruturado: avisos do NOP público (PDFs por evento, sem arquivo histórico) e os PDFs semanais
  *European Aviation Overview*.

### Solo em LTFM / LFPG / EGLL / LIRF
- Única rota aberta que achei: **OPDI** (eventos derivados de ADS-B do OpenSky, publicados pelo EUROCONTROL).
  Vale medir a taxa de cobertura de `off-block`/`enter runway` por aeroporto antes de construir feature —
  se o OPDI também não enxergar o solo em LTFM, sobra o `lift-off` (T08) e a pista, que não dependem de
  receptor de solo.
- EGLL ganha ainda a pista de partida determinística pelo calendário de alternância (item 3).
- Para LFPG/LIRF/LEMD/LEBL não achei publicação de pista em uso nem feed de solo aberto.

## Descartado / irrelevante

- **ADS-B Exchange histórico** — arquivo desde 2020, mas é produto comercial por assinatura; sem licença
  aberta ([historical data](https://www.adsbexchange.com/products/historical-data)). Rejeitado pela regra do prêmio.
- **notamify.com / notamhistory.com** — arquivos históricos de NOTAM, pagos e proprietários. Rejeitados.
- **Laminar Data Hub (Cirium) / DTN NOTAM API** — comerciais. Rejeitados.
- **EUROCONTROL ADRR (Aviation Data Repository for Research)** — trajetórias 4D planejadas e reais,
  download gratuito, mas exige conta OneSky Online, pedido de acesso e **assinatura dos Terms of Use**
  por usuário; e a cobertura anunciada é **2015–2024, ~4 meses de dados por ano**
  ([docs](https://www.eurocontrol.int/dashboard/aviation-data-research)). Não é dado aberto redistribuível
  e não cobre jan/jul 2026. Rejeitado.
- **EAD/PAMS, eAIP com login (DFS, skyguide, DHMİ)** — exigem conta; licença de redistribuição não aberta.
  AIP SUP por ciclo AIRAC seria o caminho "limpo" para obras, mas é PDF por ciclo, por país, sem licença aberta
  e com esforço alto para 9 países — não compensa frente ao código G do APT-DLY.
- **Casper NoiseLab / WebTrak (Aena)** — visualizadores proprietários de ruído/trajetória; sem exportação em
  massa nem licença ([Aena mapas interactivos de ruido](https://www.aena.es/es/corporativa/sostenibilidad-ambiental/ruido/mapas-interactivos-de-ruido.html)).
- **Médias mensais de uso de pista em relatórios de ruído (FRA/MUC/ZRH/ILT)** — resolução mensal; já temos
  vento horário do METAR, que domina esse sinal. Mesmo motivo pelo qual a média mensal do ansperformance foi
  descartada antes.
- **Declarações de capacidade dos coordenadores de slot (ACL, COHOR, Fluko, AECFA, Assoclearance)** — trazem
  redução de capacidade por obra, mas são PDFs sazonais sem licença aberta e com granularidade de temporada.
- **LVNL "actueel baangebruik" e Zurich "aktuelle Starts"** — só tempo real / 10 dias, sem arquivo: inúteis
  para jan/jul 2025.
- **Dados de degelo por voo** — não são públicos em nenhum dos 10 aeroportos (ficam no A-CDM).
