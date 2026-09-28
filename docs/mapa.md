# Mapa do projeto (atualizado em 28/09, 02h30 UTC)

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
| Nota oficial | **253,95 s** (v19) |
| Posição | ~26º de 176 |
| 3º colocado | 226,90 (faltam 30 s) |
| 10º colocado | 237,06 (faltam 20 s) |
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

```mermaid
flowchart TD
    A["Plano 13: mais sinais no corretor<br/>METAR (tempo), prefixo da companhia<br/>nos voos com NM, rotação no stand<br/>(tipo/companhia da última chegada),<br/>consistência NM (ARVT_1/ARVT_3)"] --> T{"ganhou na simulação?<br/>(sem_loteria e normais)"}
    T -->|sim| E["v18: submit + regra de Roma,<br/>conferir, mostrar, enviar"]
    T -->|não| B
    E --> B["Validação jan→jul e jul→jan<br/>(dica do GREKI)"]
    B --> C["Plano 14: especialista no resíduo<br/>do LOBT, teto por aeroporto,<br/>calibração fora da amostra"]
    C --> FIM["08–10/10: README em inglês,<br/>repositório público GPLv3"]
```

| Etapa | Ganho esperado (oficial) | Quando |
|---|---|---|
| Plano 13 (sinais no corretor) | −3 a −6 s | 29–30/09 |
| Validação jan↔jul | melhora as decisões | 30/09 |
| Plano 14 | −2 a −4 s | até 03/10 |
| Meta | ~245 (top 10–15); top 3 pede ~227 | até 11/10 |

Lição de 28/09: juntar vários sinais pequenos no corretor rende mais no oficial que na
simulação (v17: −3,1 s simulado, −7,5 s oficial).

## Regras

- Medir tudo na simulação (jan+jul/2025) e enviar só o que melhora; nada de envio para sondar.
- Todo envio com ok do usuário.
- Treinos longos como serviço do sistema; estado no git, nos `.md` e na memória.
