# Esteira de experimentos

Atualizado em 2026-09-30 21:20. Spec: `docs/superpowers/specs/2026-09-30-esteira-design.md`.

## Campeã

- Membros: 20260930-130921-e76_m0, 20260930-140011-e76_m1
- Simulação: completo 296.76 · sem loteria 230.7
- Última enviada: {"versao": 33, "sem_loteria": 230.7}
- Arquivo pronto esperando ok: nenhum

## Vazão

- Fila: 20 · rodando: 0 · feitos nas últimas 24 h: 59 · consultas à metade B: 2

## Últimos candidatos

| id | origem | receita | decisão | A sem lot. | B sem lot. | min |
|---|---|---|---|---|---|---|
| 78 | agente | `{"base": ["--model", "two_stage_nm", "--nm-min-ms", "21600", "--janela-lobt", "--reg-corte", "7200", "--base-ctx", "--base-por-apt", "--base-p13", "--seeds", "3"]}` | A: não seleciona | -0.47 | — | 192.3 |
| 77 | agente | `{"base": ["--model", "two_stage_nm", "--nm-min-ms", "21600", "--janela-lobt", "--reg-corte", "7200", "--base-ctx", "--base-por-apt", "--seeds", "3"]}` | cancelado: receita sem --base-p13 (campeã mudou para a v33) | — | — | 8.4 |
| 76 | agente | `{"base": ["--model", "two_stage_nm", "--nm-min-ms", "21600", "--janela-lobt", "--reg-corte", "7200", "--base-ctx", "--base-por-apt", "--base-p13"]}` | B: não confirma | +2.37 | +0.50 | 72.8 |
| 75 | agente | `{"base": ["--model", "two_stage_nm", "--nm-min-ms", "21600", "--janela-lobt", "--reg-corte", "7200", "--base-ctx", "--base-por-apt", "--base-mapa"]}` | A: não seleciona | -0.12 | — | 73.2 |
| 55 | gerador | `{"--conjunto": true, "--corretor-params": {"min_data_in_leaf": 100}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | +0.01 | — | 9.7 |
| 54 | gerador | `{"--conjunto": true, "--corretor-params": {"lambda_l2": 50}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.25 | — | 10.1 |
| 53 | gerador | `{"--conjunto": true, "--corretor-params": {"lambda_l2": 10}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.29 | — | 10.2 |
| 52 | gerador | `{"--conjunto": true, "--corretor-params": {"lambda_l2": 0}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | +0.00 | — | 10.1 |
| 51 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 255}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.06 | — | 13.0 |
| 50 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.21 | — | 11.1 |
| 49 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 63}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.11 | — | 9.9 |
| 48 | gerador | `{"--conjunto": true, "--corretor-params": {"learning_rate": 0.05}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.11 | — | 9.7 |
| 47 | gerador | `{"--conjunto": true, "--corretor-params": {"learning_rate": 0.03}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.36 | — | 10.1 |
| 46 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 700, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.05 | — | 10.7 |
| 45 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 300, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.27 | — | 8.8 |
| 44 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true}` | A: não seleciona | +0.01 | — | 9.5 |
| 43 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true}` | A: não seleciona | -0.31 | — | 11.0 |
| 42 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-ctx": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.34 | — | 9.0 |
| 41 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.78 | — | 9.7 |
| 40 | gerador | `{"--conjunto": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.26 | — | 8.5 |
| 39 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.19 | — | 9.5 |
| 38 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true, "--superficie": true}` | A: não seleciona | +0.08 | — | 10.5 |
| 36 | gerador | `{"--conjunto": true, "--corretor-params": {"min_data_in_leaf": 200}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | -0.24 | — | 11.8 |
| 35 | gerador | `{"--conjunto": true, "--corretor-params": {"min_data_in_leaf": 100}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | +0.09 | — | 11.6 |
| 34 | gerador | `{"--conjunto": true, "--corretor-params": {"lambda_l2": 50}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | -0.12 | — | 11.6 |
| 33 | gerador | `{"--conjunto": true, "--corretor-params": {"lambda_l2": 10}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | -0.03 | — | 11.5 |
| 32 | gerador | `{"--conjunto": true, "--corretor-params": {"lambda_l2": 0}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | -0.10 | — | 11.9 |
| 31 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 255}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | +0.26 | — | 15.6 |
| 30 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | aprovado | +0.30 | +0.11 | 12.9 |
| 29 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 63}, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | +0.03 | — | 11.5 |

## Famílias

| família | n | média A | melhor | promovidos | situação |
|---|---|---|---|---|---|
| base | 3 | +0.591 | +2.37 | 0 | ativa |
| fila:nenhum | 3 | +0.010 | +0.07 | 0 | ativa |
| outro | 9 | -0.034 | +0.30 | 1 | ativa |
| param:num_leaves | 8 | -0.037 | +0.26 | 0 | ativa |
| bloco:--superficie | 4 | -0.045 | +0.19 | 0 | ativa |
| param:min_data_in_leaf | 3 | -0.055 | +0.11 | 0 | ativa |
| bloco:--mapa | 3 | -0.108 | +0.06 | 0 | ativa |
| rodadas | 6 | -0.110 | +0.13 | 0 | ativa |
| bloco:--corretor-sem-ctx | 3 | -0.140 | +0.18 | 0 | ativa |
| bloco:--corretor-ref | 3 | -0.157 | -0.08 | 0 | ativa |
| param:learning_rate | 4 | -0.214 | +0.00 | 0 | ativa |
| param:lambda_l2 | 6 | -0.226 | +0.00 | 0 | ativa |
| bloco:--dist-plano | 3 | -0.504 | +0.00 | 0 | ativa |

## Revisões

2026-09-30: Estreia: limite de RAM 10 → 8 GB (desktop aberto deixa ~9,3 GB; pico do corretor 7,8 GB). Sonda de ruído (repetição do membro 1) na frente da fila.
2026-09-30: Estreia (30/09 00h57–02h40 -03): 10 candidatos (sonda + 9 vizinhos), 0 falhas, 8,8–11,3 min cada, pico RAM ~7,3 GB. Todos reprovados em A (melhor: superficie somado +0,19; sonda de ruído −0,13, conferida contra compare.py). Régua mantida: nenhum candidato chegou a 0,3 em A, então o IC não foi o gargalo. Serviço habilitado (enable). Próxima revisão: 24 h ou primeiro envio.
2026-09-30: 30/09 10h40: revisão pelo rendimento — 53 testes do corretor com 1 promoção (+0,20 s). Mudanças: candidato de base vira média completa (mesmos corretores refeitos sobre a base nova, alinhados pelo voo); grade do gerador só num_leaves/learning_rate e sem variar rodadas; na fila, bases da retrospectiva: --base-mapa (75), --base-p13 (76), --seeds 3 (77).
2026-09-30: Critério do usuário (30/09 11h): a esteira precisa render pelo menos o ritmo manual. Referência: manual 29/09 v28→v32 em ~13 h = 235,21→232,35 sem loteria (−2,86 s simulação; −2,05 s oficial) ≈ 0,2 s/h. Checkpoint 01/10 11h (24 h): esteira precisa somar ≥ 2,0 s sem loteria sobre 232,35 (campeã ≤ 230,35) ou um envio com ganho oficial ≥ 1 s. Se não: pausar o serviço e voltar ao manual (fila vira ferramenta sob demanda). Enquanto a esteira roda, experimentos manuais pesados não cabem na RAM.

