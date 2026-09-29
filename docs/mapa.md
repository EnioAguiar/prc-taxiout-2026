# Mapa do projeto (atualizado em 29/09, 19h UTC)

Resumo para se localizar. Detalhes: `README.md` (modelo, roadmap, envios) e `CONTEXTO.md` (linha do tempo).

## Objetivo: top 3 (decisão do usuário, 28/09)

Entramos para ganhar. O top 3 não vem de melhorar os voos normais: pela conta da v11,
a cauda (voos de horas, off-block gravado como horário programado ou padrão, aeroportos sem
ADS-B) é ~60 % do erro², e mesmo zerando o erro dos normais não chegaríamos a 228. O topo
erra menos na cauda.

Princípios:

1. **Se um competidor achou, a gente acha.** Buscar ativamente dados abertos que expliquem a
   cauda e cubram os aeroportos sem ADS-B (LTFM, LFPG, EGLL), e regras exatas no próprio dado
   (como a janela do LOBT). Ideias de repositórios públicos valem; código deles, não (regra de
   originalidade).
2. **Busca automática em massa onde ela ajuda:** centenas de combinações do corretor barato
   (~1 min cada) rodando à noite, com a simulação decidindo. Tuning sozinho não ganhou para
   ninguém (Discord); a busca serve para combinar features e fontes novas.
3. **Decidir pela simulação**, enviar só o que melhora, e guardar os 5 envios diários para
   confirmar.

Estudo da cauda (28/09, `docs/research/2026-09-28-estudo-cauda.md`): loterias são um piso comum
a todos os times; a distância para o topo está na parte previsível (normais e cópias).

Frentes abertas: fontes de dados (OPDI, OpenSky, regulações ATFM, meteorologia, layout de
aeroporto); estudo da cauda; plano 11 (regressor limpo).

## Onde estamos

| Item | Situação |
|---|---|
| Nota oficial | **247,83 s** (v29) |
| Posição | 24º de 162 |
| 1º colocado | 220,40 (faltam 27,4 s) |
| 3º colocado | 224,50 (faltam 23,3 s) |
| 10º colocado | 236,98 (faltam 10,9 s) |
| Prazo | 11/10, 23:59 (horário da Europa) |
| Envios | 5 por dia UTC (zera às 21h de Brasília) |
| Repositório | privado; abrir entre 08 e 10/10 (condição do prêmio) |

## O caminho até aqui

```mermaid
flowchart LR
    A["v1 514<br/>modelo básico"] --> B["v2 385<br/>outliers no treino"]
    B --> C["v3 339<br/>2 estágios"]
    C --> D["v5 331<br/>retas voos sem NM"]
    D --> E["v6 315<br/>ADS-B adsb.lol"]
    E --> F["v9 276<br/>janela do LOBT<br/>+ corretor"]
    F --> G["v11 267<br/>taxi-in chegadas<br/>+ vizinhos"]
    G --> H["v12 265<br/>3 corretores"]
    H --> I["v13 264,35<br/>regra de Roma"]
    I --> J2["v17 256,86<br/>regressor limpo +<br/>companhia, séries diárias, OPDI"]
    J2 --> K["v19 253,95<br/>METAR, rotação no stand,<br/>consistência NM, companhia"]
    K --> L["v21 252,34<br/>base com contexto (ctx_*)"]
    L --> M["v27 251,10<br/>distância aos horários planejados"]
    M --> N["v28 249,16<br/>pátio do stand + 500 rodadas"]
    N --> O["v29 247,83<br/>regressor por aeroporto<br/>+ célula stand × pista"]
```

## Como o modelo funciona

```mermaid
flowchart TD
    D1["Dados do organizador<br/>2025 + ranking 2026"] --> F["Features<br/>horários NM, congestionamento,<br/>stand/pista"]
    D2["ADS-B adsb.lol<br/>5,7 GB no SSD"] --> F
    F --> B["BASE: 2 estágios<br/>classificador 'cópia do SCHED'<br/>+ regressor normal<br/>+ retas voos sem NM"]
    B --> C["CORRETOR (3 modelos em média)<br/>+ taxi-in chegadas + vizinhos"]
    C --> J["Janela do LOBT<br/>previsão limitada a ±3606 s"]
    J --> R["Regra de Roma (4 voos)"]
    R --> S["Arquivo de envio"]
```

## Onde está o erro (simulação da v11)

| Parte | % do erro² | Dá para melhorar? |
|---|---|---|
| "Loterias": 11 voos de horas sem explicação | ~34 % | quase impossível |
| Voos normais sem ADS-B (LTFM, LFPG, LEMD…) | ~31 % | sim, pela base |
| Cauda que é cópia de horário | ~13 % | pouco |
| Voos normais com ADS-B | ~11 % | pouco |
| Alarmes falsos (normal previsto > 1 h) | ~11 % | um pouco |

## Descartados

Seeds, detector de pushback, fila vista pelo ADS-B, CatBoost sozinho, média mensal oficial,
tirar lat/lon do ADS-B, alvo residual, calibração de p, mistura multiclasse.

## Para onde vamos (a partir de 29/09)

Feito em 28–29/09: v19 253,95 → v21 252,34 → v27 251,10 → v28 249,16. Descartados contra a campeã: base
com plano 13, `--fila` completo, base com mais rodadas, hedge de Roma, cobertura ADS-B 2026, tirar ctx do
corretor, pesos, p(cópia), só LGB/só CatBoost, XGBoost na base e como 4º corretor, LightGBM na GPU,
superfície (ATD2), **mapa do aeroporto** (X-Plane apt.dat; na base e no corretor +0,3–0,4 s, reprovado
sem top 10; código pronto e desligado), **caçador de regras 2** (sequência de pousos no stand, mesmo
voo em outro mês: nada; `docs/research/2026-09-29-cacador-de-regras-2.md`), **voos normais sem ADS-B**
(item 4e, `docs/research/2026-09-29-base-sem-adsb.md`: as duas hipóteses ganham na fatia-alvo mas
param em +0,2 s no completo; código pronto e desligado).

### Item 4e: voos normais sem ADS-B (29/09)

Diagnóstico sobre o oof da v28: nenhum agrupamento novo sobrou (stand, companhia, tipo, destino:
R² leave-one-out ≈ 0), fila de pushback pelo AOBT_3 e configuração de pista não correlacionam com o
resíduo (|r| ≤ 0,03), o OPDI não vê o solo nesses aeroportos (casa 0–3 % dos voos) e a previsão já está
calibrada (inclinação 0,97–1,03). Sobra um estado de aeroporto por bloco de 30 min (desvio 58–92 s,
vale −3 a −14 s por aeroporto) que **não é observável** no ranking.

| | completo | normais NM | os 4 sem ADS-B |
|---|---|---|---|
| v28 | 299,41 | 193,45 | 212,92 |
| `--corretor-ref` (célula aeroporto × stand × pista) | 299,23 | 192,85 | 212,08 |
| `--base-por-apt` (regressor por aeroporto na base) | 299,23 | 191,90 | 211,78 |

A base com regressor por aeroporto sozinha é o maior ganho de base desde `--base-ctx`
(311,18 → 308,75, +2,4 s, IC +1,8 a +3,3, passa o portão); o corretor da v28 absorve quase tudo.
Nenhuma passa o portão no `completo` (IC baixo < −0,5, sem top 10 < 0, julho < 0).

```mermaid
flowchart TD
    B["Voos normais sem ADS-B (item 4e)<br/>medido 29/09: ~1-2 s na fatia, 0,2 s no completo"] --> C["H1 + H2 juntas<br/>(--base-por-apt + --corretor-ref)"]
    C --> FIM["08–10/10: README em inglês,<br/>repositório público GPLv3"]
```

| Etapa | Ganho esperado (oficial) | Quando |
|---|---|---|
| Base mais forte sem ADS-B | medido: 0,2 s no completo, 1,5 s nos normais | feito 29/09 |
| H1 + H2 juntas (v29) | simulação 299,10 (+0,3 s, IC −0,7 a 1,3; sem loteria +1,2, IC 0,7 a 1,8); **oficial 247,83 (−1,33 s)** | enviada 29/09 |
| Meta | top 10 (~237); top 3 pede ~220 | até 11/10 |

Lições de 28/09: a simulação acerta o oficial (−2,8→−2,91; −2,1→−1,61; −1,1→−1,24); o corretor
barato de 1 LightGBM não prevê o corretor real (usar `--reusar-oof` / laço fiel); GPU não é a
alavanca (limite é informação, não cálculo).

Lição de 29/09 (v29): com o `completo` preso no ruído das loterias, o ganho `sem_loteria` com IC
todo positivo previu o oficial melhor que o portão (+1,2 s simulado sem loteria → −1,33 s oficial).

## Regras

- Medir tudo na simulação (jan+jul/2025) e enviar só o que melhora; nada de envio para sondar.
- Todo envio com ok do usuário.
- Treinos longos como serviço do sistema; estado no git, nos `.md` e na memória.
