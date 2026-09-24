# O que as equipes fortes fazem no PRC Data Challenge

## (a) Edições anteriores: o que venceu

Os problemas eram outros (2024: peso de decolagem; 2025: combustível), mas as lições se aplicam.

- **2024, 1º lugar (likable_jelly, ENAC):**
  - Média de LightGBMs de 50 mil árvores com os mesmos parâmetros e seeds diferentes: RMSE 1612 com 1 modelo, 1564 com 10 e 1561 com 20 (−3%).
  - Alvo reescalonado por grupo, de (TOW−EOW)/(MTOW−EOW), com peso = escala² para continuar otimizando o RMSE absoluto.
  - Busca aleatória de parâmetros num conjunto de validação.
  - METAR do IEM e correção de horários sujos.
  - Fonte: https://raw.githubusercontent.com/richardalligier/prc/main/README.md
- **2024, 2º lugar (tiny_rainbow, ITA):** CatBoost, XGBoost, LightGBM e redes neurais; combinação treinada sobre as previsões de validação; Optuna; bases externas públicas. Fontes: https://raw.githubusercontent.com/marcosmaximo/prc-challenge-ita/main/README.md e https://doi.org/10.59490/joas.2025.7963
- **2025, 1º (resourceful-quiver, 200,83):** features de domínio, reaproveitamento do estimador de massa de 2024, limpeza própria, LightGBM e artigo no formato JOAS.
- **2025, 2º (jubilant-vase):** ensemble de três bibliotecas GBDT.
- **2025, júri:** pesa nota, originalidade, tratamento de outliers e qualidade da documentação. Na média, 2.127 envios para 53 times.
  - Fonte: https://prc-data-challenge-2025.netlify.app/outcome.html
- **GPU:** citada só de passagem. Os vencedores treinaram em CPU (ThreadRipper); em 2026, as equipes que publicaram código usam CatBoost em GPU de 8 GB.

## (b) O que as equipes de 2026 publicaram

Notas tiradas do espelho do placar e dos READMEs.

| Time / repositório | Nota oficial | Abordagem | Outliers e horário copiado |
|---|---|---|---|
| elegant-alligator (javidmardanov) | **268,56** (v6) | Referência MVT−AOBT_3 com fallback de 900 s + CatBoost em GPU sobre a diferença, previsões TimesFM‑3 por aeroporto, combinação SLSQP, especialista para AOBT ausente, contexto de pousos | Classificador que estima se o horário de Roma é cópia de SCHED; expert parcial de LIRF; limites finais (MVT−LOBT)±3606 s; piso em 0 |
| zestful-fountain (Phoenix-Ops) | **273,56** (v12) | CatBoost + LightGBM sobre a diferença para a referência NM; 183 features (pousos, referência NM de decolagens vizinhas antes e depois, até 2 h) | Especialista para LIRF sem IOBT (referência = MVT−SCHED, média de 5 seeds); regra: se AOBT_3−LOBT > 7200 s, prever max(0, MVT−LOBT) (mudou 31 linhas, −0,2 s oficial); erro limitado a ±7200 s no treino; alvos negativos descartados |
| knowledgeable-helicopter (satam2) | **278,01** | CatBoost em três rotas: diferença para a referência NM, direto quando a referência é negativa, especialista quando falta AOBT_3 | Treino da diferença só em referências de 0–7200 s; limitação declarada: rótulos extremos |
| vibrant-lollipop (ahmetabdullahgultekin) | **310,55** | Referência P10 oficial da EUROCONTROL + diferença; METAR/IEM; OurAirports | Análise de degelo; resultado negativo com OPDI |
| victoralcadi | ? | LightGBM, contagens ancoradas na decolagem | Deixa AOBT_3 de fora (erro, na nossa leitura) |

Fontes:
- https://github.com/javidmardanov/PRC-Data-Challenge-2026
- https://raw.githubusercontent.com/Phoenix-Ops-LTD/prc2026-taxiout/main/METHOD.md
- https://github.com/satam2/knowledgeable-helicopter
- https://github.com/ahmetabdullahgultekin/prc-taxiout-2026
- https://github.com/victoralcadi/prc-data-challenge-2026

Nenhum dos seis primeiros publicou código; a busca por nome de time não achou nada.

## (c) Placar

Dados de 24/09 às ~20h UTC: 190 times com nota, 2.689 envios pontuados. O competição começou em 04/09.

**Distribuição:** 1º 234,1 · 10º 248,4 · 20º 265,3 · 50º ~280,3 · mediana (95º) ~305 · 132º 384,7. Dois times estão empatados em 728,49, provavelmente um valor constante.

[INFERÊNCIA] Somos o outgoing-boat: 2 envios e melhor nota 384,7377, igual à nossa v2.

**Top 10** (n = envios pontuados; o número do arquivo vai mais alto porque inclui arquivos rejeitados):

| # | Time | Melhor | n | Saltos grandes |
|---|---|---|---|---|
| 1 | vigorous-whistle | 234,14 | 25 (v26) | 13/09: 421→332; 14/09: 323→277; **15/09: 267→248**; 23/09: 238,8→235,3 |
| 2 | jolly-lobster | 234,51 | 54 (v104) | 05–06/09: 408→356; **16–17/09: 272→257**; 24/09: 238,1→235,8 |
| 3 | gentle-tractor | 236,31 | 29 (v50) | **19–20/09: 258,6→247,1**; 21/09: 243→239,5 |
| 4 | zesty-puzzle | 237,91 | 58 (v62) | 05/09: 361→291; estacionado em ~264 de 12 a 19/09; **19–21/09: 263→244,6** |
| 5 | jovial-uniform | 239,77 | 18 | 264 (10/09) → 253 → 247,5 (15/09) → 240,7 (19/09) |
| 6 | gentle-igloo | 240,14 | 47 (v226) | melhora gradual |
| 7 | enthusiastic-daisy | 243,29 | 85 (v744) | gradual |
| 8 | upstanding-firefly | 243,93 | 12 | **266,4 (09/09) → 244,9 (23/09)** |
| 9 | youthful-giraffe | 245,02 | 21 | **08/09: 262,7→248,5**; parado desde 10/09 |
| 10 | bubbly-telephone | 248,39 | 63 | 5 envios em poucos minutos |

**Melhor nota ao fim de cada dia (UTC):**

| Dias | Melhor nota | Quem |
|---|---|---|
| 04 | 312,8 | generous-jungle |
| 05 | 288,3 | zesty-puzzle |
| 06 | 265,8 | youthful-giraffe |
| 07 | 263,5 | youthful-giraffe |
| 08 | **248,5** | youthful-giraffe |
| 09 | 246,4 | youthful-giraffe |
| 10–15 | 245,0 | youthful-giraffe |
| 16 | 241,75 | vigorous-whistle |
| 18 | 241,2 | vigorous-whistle |
| 19 | 239,6 | vigorous-whistle |
| 20 | 239,1 | vigorous-whistle |
| 22 | 238,65 | gentle-tractor |
| 23 | 235,3 | vigorous-whistle |
| 24 | 234,1 | vigorous-whistle |

Limitação: montei a curva só com o histórico dos 10 primeiros, das duas pontas do ranking geral e de alguns times anteriores. Não li as posições 51–144 do ranking geral, então um recorde entre 11 e 15/09 pode ter ficado de fora.

**Como ler:** seis times saíram independentemente de um patamar de ~263–267 s e caíram 15–20 s, para ~244–248 s, em datas diferentes (08, 15, 16–17, 19–21 e 23/09).

[INFERÊNCIA] Parece uma mesma descoberta, e ela ainda não está nos repositórios públicos. Os repositórios públicos, que já usam referência NM + especialistas, param em 268–275. Os melhores estão ganhando ~1 s por dia com ajustes finos e 3–5 envios diários.

## (d) Regras sobre dados externos

- **Permitidos**, desde que abertos, com licença aberta e documentados. O prêmio exige código GPLv3 no GitHub, documentação que permita reproduzir e solução original (eligibility.html).
- METAR do IEM e OurAirports já foram aceitos em 2024 e são usados em 2026.
- ranking.html avisa que a organização vai monitorar tentativas de aprender com o placar ou explorá-lo.
- O data.html avisa que as divergências entre os dados de movimento e os de voo foram mantidas de propósito, sem conciliação.
- Pelo swagger só há dois GETs (leaderboard e teams).

## (e) Ideias em ordem de prioridade

1. **Treinar sobre a diferença para a referência NM.**
   - Referência = MVT − AOBT_3 quando o aeroporto de origem NM bate e o valor fica entre 0 e 7200 s; senão LOBT → EOBT → mediana. O modelo prevê y − referência, com o erro limitado a ±7200 s no treino.
   - Evidência: javid v1 fez 285 s com isso; o satam2, 278.
   - Impacto esperado: −60 a −100 s sobre os nossos 384,7 [INFERÊNCIA]. Esforço baixo. Confiança alta.
2. **Modelo de mistura pelo horário copiado.**
   - Classificador multiclasse: o horário registrado é cópia de AOBT_3, LOBT, EOBT, IOBT, SCHED ou de nenhum (a nossa conta: 38%, ~20%, 17%…).
   - Previsão = Σ p_k·(MVT − horário_k) + p_nenhum·regressão. É o ótimo para erro quadrático e ataca os ~37% do erro quadrático que vêm de y > 3 h.
   - Evidência: o classificador de Roma do javid e a regra AOBT−LOBT>2 h da zestful são casos particulares disso.
   - Impacto: −10 a −30 s [INFERÊNCIA]; é a candidata mais provável para o salto 265→248. Esforço médio. Confiança média.
3. **Especialista para LIRF sem IOBT / AOBT ausente** (referência = MVT − SCHED), média de 5 seeds. Impacto: vários segundos. Esforço baixo. Confiança alta.
4. **Features de contexto visíveis no ranking:**
   - pousos: taxi-in médio por pista/stand em 15/30/60 min, tempo e taxi-in do último pouso no mesmo stand;
   - referência NM das decolagens vizinhas (mesma pista/stand, janelas antes e depois de 15/60 min, próxima decolagem).
   - Evidência: zestful ganhou −0,8 a −6 s locais por família.
   - Esforço médio. Confiança média-alta.
5. **Ensemble CatBoost (GPU na RTX 4060) + LightGBM, com várias seeds.** Impacto: −1 a −3% (2024). Cuidado com RAM: float32 e subamostra. Esforço baixo.
6. **Pós-processamento:** piso em 0 e limites (MVT−LOBT)±3600 s quando LOBT existe (javid). Impacto pequeno, risco baixo.
7. **Validação que imite o ranking:** jan+jul 2025 fora do treino, peso por mês, bootstrap por dia. Sempre medir o RMSE completo, com outliers — não aprender com o placar.
8. **METAR** (degelo em janeiro, vento/pista). Impacto incerto, de −0 a −3 s. Esforço médio.
9. **Para o prêmio:** repositório GPLv3 limpo, documentação de reprodução e artigo no formato JOAS. Pesa na decisão do júri.