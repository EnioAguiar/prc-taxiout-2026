# Pesquisa de domínio: previsão de taxi-out, horários A-CDM, METAR e cauda pesada

(Não salvei em local://research-*.md porque a ferramenta write só aceita xd:// fora do plan mode. O texto abaixo é o relatório completo.)

## 0. Alertas antes de planejar

- **O conjunto de ranking parece ter mudado durante a competição.** A equipe *vibrant-lollipop* (ahmetabdullahgultekin) mediu em 01/09: 215.876 DEP; julho só com EDDF/EGLL/EHAM; janeiro = 71% das linhas. Ela e o placar citam um "mid-competition evaluation-set change". Nosso submitting.parquet tem 344.841 IDs.
  - Fontes: https://github.com/ahmetabdullahgultekin/prc-taxiout-2026/blob/master/docs/facts.md (R03, R04); https://github.com/LeeMarshall1113/prc-taxiout-2026 (resultado de busca; a página agora dá 404).
  - **Ação:** contar DEP por aeroporto × mês no ranking.parquet e repesar o sim_ranking.py para a mesma mistura. Esforço: baixo. Confiança: alta.
  - Notas do placar antigo e do novo não são comparáveis [INFERÊNCIA].
- **Resolução da validação.** Com o erro concentrado em poucas centenas de linhas, o bootstrap de uma nota isolada tem desvio de ~56 s. A diferença pareada entre dois modelos resolve ~5 s. Um ganho local significativo (−5 s) virou +0,6 s no placar.
  - Fonte: https://github.com/ahmetabdullahgultekin/prc-taxiout-2026/blob/master/docs/experiments.md
  - **Ação:** usar bootstrap pareado e só enviar ganhos maiores que ~10 s.

## 1. Features de domínio priorizadas

| # | Achado / feature | Evidência | Impacto esperado no RMSE | Esforço | Confiança |
|---|---|---|---|---|---|
| 1 | **Probabilidade de o BLOCK ser cópia de SCHED, EOBT/LOBT/IOBT ou AOBT_3** (seção 4) e os candidatos MVT−T_k | Nosso achado: 38/20/17%. Outliers acima de 2 h são erro de rótulo: 94% são plausíveis (<2 h) segundo o NM; mediana do NM 18 min contra 2,3 h no APDF (facts.md e experiments.md da equipe acima). y>3 h responde por ~37% do erro quadrático. | Maior alavanca. Recuperar 30% do erro quadrático da cauda dá ~−6% de RMSE (≈−20 a −25 s) [INFERÊNCIA] | médio | média-alta |
| 2 | **Referência P10 por (aeroporto, stand, pista)**, janela de 12 meses e ≥10 voos ≤P10; é o método oficial ATXOT | https://www.sesperformance.eu/dataportal/metadata/additional-taxi-out-time/ ; notas ATXOT: https://github.com/ahmetabdullahgultekin/prc-taxiout-2026/blob/master/docs/reference/atxot-notes.md. Foi a feature com maior ganho (18,8%). | Já existe no projeto. Usar como alvo residual não trouxe ganho mensurável (±2–5 s, dentro do ruído). | — | alta |
| 3 | **Offsets MVT−SCHED, MVT−EOBT e MVT−AOBT_3.** Carregam o estado de atraso e parte do congestionamento. | Ganho: eobt 16%, sched 15,4%, naive AOBT_3 10,4% (facts.md M22). RMSE só com MVT−AOBT_3: 385 s; com o modelo: 378 contra 531 da previsão ingênua no holdout. | Já existe. | — | alta |
| 4 | **Fila à la Idris (N-control)**: contar as DEP j com off-block estimado (AOBT_3_j) antes do off-block estimado de i e decolagem entre off-block e decolagem de i; o mesmo para ARR (categorias d1–d4 e a1–a4). Também a média de (ATOT−AOBT_3) dos vizinhos da mesma pista nos 30 min anteriores. | Idris et al.: fila é o principal preditor (β≈0,68). Stacking+SHAP: fila de DEP e fila de ARR no topo; efeito não linear acima de 8 aeronaves. https://pmc.ncbi.nlm.nih.gov/articles/PMC13022428/ . No ranking, ATOT de todos e AOBT_3 são observáveis. **Porém** contagens em janela fixa deram só 3% do ganho no concorrente. | Pequeno a moderado (−3 a −10 s) [INFERÊNCIA]. Contagens ancoradas no intervalo off-block→decolagem de cada voo são mais informativas que janelas fixas em torno da decolagem. | médio | média |
| 5 | **Configuração de pista em uso**: mix de pistas das últimas N DEP e ARR, pista do voo × configuração, alternância de EGLL | Família runway_configuration: 5,9% do ganho (experiments.md). O documento ATXOT admite que não modela a rota de táxi. | Pequeno a moderado | baixo | média |
| 6 | **Tipo de aeronave, esteira, operador (proxy de handler e terminal)**. O PRC exclui o tipo de propósito do agrupamento. | ATXOT p.11 §3.2 (atxot-notes.md). Clustering de companhias por taxi médio ajuda (artigo PMC acima). | Pequeno; o GBM já usa a categórica | baixo | média |
| 7 | **Degelo e meteorologia (METAR)**: seção 3 | Proxy de degelo via METAR contra a fração de voos sem referência no indicador oficial: r=0,76 no geral e 0,87–0,98 em LTFM, LSZH, EDDF, EDDM e LFPG. Degelo em jan/2026: LSZH 18%, EHAM 13%, EDDM 11%, LTFM 10%, EDDF 8%. Os regimes diferem: em EHAM o degelo entra no taxi-out; em EDDM e LSZH esses voos saem do indicador oficial mas ficam no nosso alvo. https://github.com/ahmetabdullahgultekin/prc-taxiout-2026/blob/master/docs/deicing_analysis.md . O ganho total do grupo meteorologia foi só 1,8%. | Pequeno no total (−2 a −8 s), concentrado em janeiro nos 5 aeroportos frios. Precisa de interação aeroporto × clima. | baixo-médio | média |
| 8 | **ATFM diário da EUROCONTROL** (% de DEP reguladas, atraso ATFM por causa): cobre jan e jul/2026 | https://www.eurocontrol.int/performance/data/download/xls/ATFM_Slot_Adherence.xlsx e .../Airport_Arrival_ATFM_Delay.xlsx (citados em https://github.com/ahmetabdullahgultekin/prc-taxiout-2026/blob/master/docs/external_data.md). Ganho de 3,7%. A causa "D" (degelo) está vazia. | Pequeno | baixo | média |

Descartados pelo concorrente: eventos de estacionamento do OPDI, que só existem em LSZH e EDDF. A cobertura ADS-B em solo em LTFM é quase nula.

## 2. Como surgem AOBT, EOBT, LOBT, IOBT e SCHED, e por que o BLOCK é "cópia"

- **SCHED_TIME_mvt**: horário de partida do schedule/slot do aeroporto, com granularidade de minutos.
- **IOBT/EOBT_1**: EOBT do plano de voo (M1), atualizado por DLA/CHG. Todo desvio acima de 15 min tem de ir ao IFPS.
- **TOBT**: nos aeroportos A-CDM, fixado pelo operador ou handler. O NM recebe o TOBT por DPI e pode gerar DLA automaticamente, alinhando o EOBT ao TOBT (padrão: diferença >15 min; TOBT confirmado ~30–40 min antes, no TSAT). Fonte: https://www.eurocontrol.int/service/estimated-block-time-update-service . Por isso o EOBT/LOBT pode ser na prática um TOBT arredondado.
- **Mensagens DPI**: E-DPI, T-DPI (TOBT), T-DPI-s (TSAT) e **A-DPI**, enviada no AOBT, "the time at which the Airport Operator records the departure from the stand". Fontes: https://www.eurocontrol.int/sites/default/files/2024-01/eurocontrol-flight-dispatcher-days-2024-2-dorat-airline-demonstration-understanding-acdm-with-fedex.pdf ; manual A-CDM de Schiphol: https://assets.ctfassets.net/biom0eqyyi6b/7ERl8iHeLELDtgFsnK0mGi/474b9801a07e239cb41adc5f8b2fe8d2/A-CDM_Manual_Schiphol_Airport_v1.0.pdf
- **LOBT_flt** tem a mesma qualidade do IOBT: RMSE ingênuo de 740 para ambos. É horário planejado, não real (facts.md M23/M24).
- **AOBT_3_flt (M3)**: nos aeroportos A-CDM vem do A-DPI, portanto do mesmo registro do aeroporto. Isso explica a coincidência de 38% com o BLOCK. Quando não há A-DPI, o NM provavelmente estima o off-block a partir da decolagem ou do EOBT [INFERÊNCIA; não achei documentação pública do cálculo M3].
- **BLOCK_TIME_UTC_mvt** = AOBT do fluxo APDF. É a única fonte aceita: ATXOT Tabela 3 não tem fonte alternativa para AOBT (atxot-notes.md). A qualidade depende do "type of AOBT recording at the airports (manual vs. automated)" (metadados oficiais acima). Stand de contato com A-VDGS integrado ao A-CDM/AODB tende a registrar o AOBT automaticamente. Stand remoto, handler manual ou falha de registro podem cair no valor planejado (SCHED/TOBT) [INFERÊNCIA]. Isso produz BLOCK==SCHED e taxi-out = atraso inteiro, com horas de duração. Isso bate com LIRF: 20% de BLOCK==SCHED e 1,2% acima de 1 h.
- **Status A-CDM**: EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF e LSZH estão na lista de 34 aeroportos com A-CDM completo. **LTFM não está.** Fonte: https://www.eurocontrol.int/concept/airport-collaborative-decision-making
  - Esperado [INFERÊNCIA]: em LTFM o AOBT_3 é menos confiável. O concorrente mediu RMSE ingênuo de AOBT_3 de 531 em LTFM e 557 em LIRF, contra 255 em EDDF.
  - Em A-CDM com CTOT o avião espera no stand (TSAT), então o taxi-out não absorve o atraso ATFM. Sem A-CDM, a espera tende a ir para a fila na pista.
- **Testes baratos que sugiro no planejamento:**
  1. Por aeroporto, o histograma de MVT−AOBT_3 tem picos em constantes? Isso indicaria AOBT_3 recalculado (decolagem − taxi padrão). Se tiver, criar a flag "AOBT_3 derivado".
  2. Taxa de BLOCK==SCHED por aeroporto × stand × operador × hora × mês, para medir estabilidade e drift de 2025 até jan/jul.
  3. O AOBT_3 coincide com EOBT ou SCHED? Nesse caso o NM também não recebeu A-DPI e a chance de o APDF ser cópia sobe [INFERÊNCIA].

## 3. METAR: fonte testada

**Iowa Environmental Mesonet (IEM)**, domínio público (https://mesonet.agron.iastate.edu/disclaimer.php, via external_data.md).

- URL testada: `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=LIRF&data=tmpc&data=dwpc&data=sknt&data=gust&data=vsby&data=wxcodes&data=metar&year1=2025&month1=1&day1=15&year2=2025&month2=1&day2=16&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=3&report_type=4`
  - Retornou CSV `station,valid,tmpc,dwpc,sknt,gust,vsby,wxcodes,metar`, 48 linhas no dia (METAR a cada 30 min), por exemplo: `LIRF,2025-01-15 00:20,2.00,-4.00,5.00,M,6.21,M,LIRF 150020Z 03005KT CAVOK 02/M04 Q1023 NOSIG`.
  - vsby vem em milhas; sknt em nós.
- Teste para jan/jul 2026: `station=EDDF&data=metar&year1=2026&month1=7&day1=1...` devolveu 48 METAR, incluindo `+TSRA` e `CB`.
- Limites (https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help): rate limit por IP, 503 sob carga e HTTP 422 acima de 1.000 estações-ano por requisição. **Plano: 1 requisição por aeroporto de 2025-01-01 a 2026-08-01**, 10 no total (~306 mil linhas no concorrente, <0,03% faltante).
  - O concorrente recomenda **não** filtrar report_type para não perder os SPECI. No meu teste, report_type=3/4 retornou os 48 por dia.
  - Fazer em série, com pausa e retentativa.
- Alternativas (não testadas): NOAA ISD global-hourly (https://www.ncei.noaa.gov/data/global-hourly/access/) e Ogimet.
- Variáveis a derivar, cada uma no instante da decolagem −20 min e como máximo/qualquer ocorrência em [decolagem−90 min, decolagem]:
  - flag de degelo (tmpc ≤ 3 °C e wxcodes com SN/FZ/RA/DZ/PL/GS/IC/BR/FG, ou tmpc−dwpc ≤ 2 com T ≤ 0);
  - trovoada (TS/CB), que fecha o pátio por raios; relevante em julho em EDDF e EHAM;
  - LVP (vsby < ~0,5 mi ou teto BKN/OVC ≤ 200 ft, extraído do texto METAR), que reduz a vazão da pista;
  - vento e rajada, e o componente de través por pista;
  - neve acumulada nas últimas 6–12 h.
  - Usar interação com aeroporto.
- **Por que o ganho do tempo é pequeno:** a vazão realizada (as decolagens e pousos observados) já reflete o clima. O METAR serve sobretudo para o degelo pós-off-block, que não aparece na contagem de tráfego [INFERÊNCIA].

## 4. Modelagem para cauda pesada otimizando RMSE

1. **Manter L2 no alvo bruto.** O RMSE é minimizado pela média condicional.
   - Evidência medida: tirar do treino y > 120 min piorou +24 s; y > 60 min piorou +36 s. Em LIRF o RMSE foi de 966 para 1.205 (experiments.md, E03). É o mesmo erro da nossa v1.
   - Huber, corte do alvo e limite na previsão entram na mesma armadilha.
2. **Log-transform prejudica.** exp(E[log y]) estima a média geométrica, abaixo da média. A cauda fica sistematicamente subestimada, e o custo é quadrático.
   - A correção por smearing de Duan (1983) supõe resíduos i.i.d. e falha com heterocedasticidade, que aqui é extrema. Fonte: https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478017 ; sobre o viés com heterocedasticidade: https://www.sciencedirect.com/science/article/abs/pii/S0167629600000461
   - Tweedie/Gamma (link log, mas prevê a média) só como membro do ensemble. Não aceita alvo negativo: LSZH tem 0,22% de negativos.
3. **Mistura em dois estágios (recomendação principal).**
   - Rótulo de classe k = qual horário T_k ∈ {SCHED, EOBT_1, LOBT/IOBT, AOBT_3} fica a ≤60–120 s do BLOCK; senão, "real/outro".
   - LightGBM multiclasse (logloss) com previsões out-of-fold. Features: aeroporto, stand, operador, hora, mês, flags de arredondamento, diferenças entre horários, atraso MVT−SCHED.
   - Regressor L2 só na classe "real", com as features de fila.
   - Previsão ŷ = Σ_k p_k·(MVT−T_k) + p_real·ŷ_real. **Nunca usar argmax**: errar a classe custa horas ao quadrado.
   - Calibrar p (isotônica ou temperatura, por aeroporto) e conferir pelo diagrama de confiabilidade na validação.
   - Variante mais robusta: empilhamento. Um LightGBM final recebe p_k, os candidatos MVT−T_k e os produtos p_k·(MVT−T_k), treinado em OOF.
   - Impacto: é a principal esperança de ir de ~385 para ~300 ou menos [INFERÊNCIA]. Esforço: médio (1–2 dias). Confiança: média.
4. **Quantis só como features** (P10/P50/P90 por stand-pista), não como saída.
5. **Ruído de rótulo.**
   - A parte estruturada (cópia de horário) é aprendível; tratar como no item 3.
   - A parte não estruturada (ex.: 36 h) é imprevisível, mas está na nota. Não remover do treino.
   - No máximo, reduzir o peso de uma linha só se a validação completa mostrar ganho pareado maior que 10 s.
6. **Pós-calibração por aeroporto** (y ≈ a + b·ŷ, ajustada em OOF) para encolher previsões extremas pouco confiáveis. Esforço: baixo.
7. **Validação.**
   - Espelhar a mistura aeroporto × mês do ranking atual (seção 0).
   - Bootstrap pareado.
   - Média de 3–5 seeds (método do vencedor de 2024, segundo o concorrente).
   - Parada antecipada: o concorrente achou o ótimo por volta da rodada 330 com lr 0,05 e sobreajuste com 800 rodadas. Nossos 1.500 provavelmente sobreajustam [INFERÊNCIA].

## 5. Ordem sugerida (ganho × esforço)

1. Repesar a validação para a mistura atual do ranking.
2. Parada antecipada e seeds.
3. Mistura/empilhamento por origem do BLOCK.
4. Pós-calibração por aeroporto.
5. Fila N-control e configuração de pista.
6. METAR/degelo com interação por aeroporto.
7. ATFM diário.
8. Tweedie e CatBoost no ensemble.