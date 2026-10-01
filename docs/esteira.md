# Esteira de experimentos

Atualizado em 2026-10-01 19:08. Spec: `docs/superpowers/specs/2026-09-30-esteira-design.md`.

## Campeã

- Membros: 20261001-185827-e122, 20260930-140011-e76_m1
- Simulação: completo 295.81 · sem loteria 228.97
- Última enviada: {"versao": 34, "sem_loteria": 229.58}
- Arquivo pronto esperando ok: nenhum

## Vazão

- Fila: 3 · rodando: 0 · feitos nas últimas 24 h: 62 · consultas à metade B: 5

## Últimos candidatos

| id | origem | receita | decisão | A sem lot. | B sem lot. | min |
|---|---|---|---|---|---|---|
| 122 | agente | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--retencao": true}` | aprovado | +0.63 | +0.59 | 10.4 |
| 121 | agente | `{"base": ["--model", "two_stage_nm", "--nm-min-ms", "21600", "--janela-lobt", "--reg-corte", "7200", "--base-ctx", "--base-por-apt", "--base-p13", "--base-ext"]}` | A: não seleciona | -0.56 | — | 87.3 |
| 120 | agente | `{"base": ["--model", "two_stage_nm", "--nm-min-ms", "21600", "--janela-lobt", "--reg-corte", "7200", "--base-ctx", "--base-por-apt", "--base-p13", "--base-ret"]}` | A: não seleciona | -0.07 | — | 74.3 |
| 113 | agente | `{"base": ["--model", "two_stage_nm", "--nm-min-ms", "21600", "--janela-lobt", "--reg-corte", "7200", "--base-ctx", "--base-por-apt", "--base-p13", "--cls-peso", "quad"]}` | A: não seleciona | +0.58 | — | 72.3 |
| 118 | agente | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--retencao": true, "--superficie": true}` | A: não seleciona | +0.13 | — | 12.6 |
| 117 | agente | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--retencao": true}` | aprovado | +0.64 | +0.94 | 9.8 |
| 116 | agente | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--pista": true, "--plano13": true, "--superficie": true}` | A: não seleciona | +0.07 | — | 13.6 |
| 115 | agente | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true}` | A: não seleciona | -0.15 | — | 9.6 |
| 112 | agente | `{"base": ["--model", "two_stage_nm", "--nm-min-ms", "21600", "--janela-lobt", "--reg-corte", "7200", "--base-ctx", "--base-por-apt", "--base-p13", "--cls-peso", "abs"]}` | A: não seleciona | -2.35 | — | 70.6 |
| 108 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | -0.04 | — | 11.3 |
| 101 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--externos": true, "--mapa": true, "--plano13": true}` | A: não seleciona | -0.27 | — | 8.3 |
| 107 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | -0.23 | — | 10.9 |
| 106 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--plano13": true, "--superficie": true}` | A: não seleciona | -0.14 | — | 11.6 |
| 100 | gerador | `{"--conjunto": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true}` | A: não seleciona | -0.35 | — | 7.6 |
| 99 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--plano13": true}` | A: não seleciona | -0.28 | — | 8.1 |
| 111 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | -0.09 | — | 10.4 |
| 105 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true}` | A: não seleciona | -0.07 | — | 11.4 |
| 109 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-ctx": true, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | +0.26 | — | 10.7 |
| 102 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-ctx": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true}` | A: não seleciona | +0.25 | — | 7.8 |
| 98 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | -0.23 | — | 9.4 |
| 110 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true, "--superficie": true}` | A: não seleciona | +0.25 | — | 11.1 |
| 104 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true}` | A: não seleciona | -0.23 | — | 9.8 |
| 103 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.30 | — | 9.1 |
| 92 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | -0.04 | — | 11.4 |
| 83 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.33 | — | 8.6 |
| 97 | gerador | `{"--conjunto": true, "--corretor-params": {"learning_rate": 0.05, "num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | -0.00 | — | 11.8 |
| 96 | gerador | `{"--conjunto": true, "--corretor-params": {"learning_rate": 0.03, "num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | -0.19 | — | 12.0 |
| 91 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | -0.28 | — | 10.9 |
| 90 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--plano13": true, "--superficie": true}` | A: não seleciona | -0.15 | — | 11.5 |
| 88 | gerador | `{"--conjunto": true, "--corretor-params": {"learning_rate": 0.05}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.22 | — | 9.0 |

## Famílias

| família | n | média A | melhor | promovidos | situação |
|---|---|---|---|---|---|
| bloco:--corretor-sem-regra | 1 | +0.627 | +0.63 | 1 | explorando |
| bloco:--retencao | 2 | +0.386 | +0.64 | 1 | explorando |
| bloco:--corretor-sem-ctx | 8 | +0.096 | +0.39 | 0 | ativa |
| outro | 11 | +0.026 | +0.34 | 1 | ativa |
| fila:nenhum | 7 | +0.016 | +0.30 | 1 | ativa |
| param:min_data_in_leaf | 6 | -0.000 | +0.11 | 0 | ativa |
| bloco:--superficie | 9 | -0.011 | +0.36 | 0 | ativa |
| param:num_leaves | 10 | -0.018 | +0.26 | 0 | cortada |
| fila:--stand-prefixo | 2 | -0.027 | +0.25 | 0 | explorando |
| bloco:--pista | 2 | -0.043 | +0.07 | 0 | explorando |
| rodadas | 8 | -0.070 | +0.19 | 0 | ativa |
| param:lambda_l2 | 9 | -0.115 | +0.22 | 0 | ativa |
| param:learning_rate | 10 | -0.164 | +0.07 | 0 | cortada |
| bloco:--mapa | 8 | -0.189 | +0.06 | 0 | ativa |
| fila:--fila | 2 | -0.198 | -0.16 | 0 | explorando |
| base | 8 | -0.236 | +2.37 | 0 | ativa |
| bloco:--corretor-ref | 8 | -0.243 | -0.08 | 0 | ativa |
| bloco:--dist-plano | 8 | -0.266 | +0.06 | 0 | ativa |

## Revisões

2026-09-30: Estreia: limite de RAM 10 → 8 GB (desktop aberto deixa ~9,3 GB; pico do corretor 7,8 GB). Sonda de ruído (repetição do membro 1) na frente da fila.
2026-09-30: Estreia (30/09 00h57–02h40 -03): 10 candidatos (sonda + 9 vizinhos), 0 falhas, 8,8–11,3 min cada, pico RAM ~7,3 GB. Todos reprovados em A (melhor: superficie somado +0,19; sonda de ruído −0,13, conferida contra compare.py). Régua mantida: nenhum candidato chegou a 0,3 em A, então o IC não foi o gargalo. Serviço habilitado (enable). Próxima revisão: 24 h ou primeiro envio.
2026-09-30: 30/09 10h40: revisão pelo rendimento — 53 testes do corretor com 1 promoção (+0,20 s). Mudanças: candidato de base vira média completa (mesmos corretores refeitos sobre a base nova, alinhados pelo voo); grade do gerador só num_leaves/learning_rate e sem variar rodadas; na fila, bases da retrospectiva: --base-mapa (75), --base-p13 (76), --seeds 3 (77).
2026-09-30: Critério do usuário (30/09 11h): a esteira precisa render pelo menos o ritmo manual. Referência: manual 29/09 v28→v32 em ~13 h = 235,21→232,35 sem loteria (−2,86 s simulação; −2,05 s oficial) ≈ 0,2 s/h. Checkpoint 01/10 11h (24 h): esteira precisa somar ≥ 2,0 s sem loteria sobre 232,35 (campeã ≤ 230,35) ou um envio com ganho oficial ≥ 1 s. Se não: pausar o serviço e voltar ao manual (fila vira ferramenta sob demanda). Enquanto a esteira roda, experimentos manuais pesados não cabem na RAM.

