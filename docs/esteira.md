# Esteira de experimentos

Atualizado em 2026-09-30 02:36. Spec: `docs/superpowers/specs/2026-09-30-esteira-design.md`.

## Campeã

- Membros: 20260929-154118-v29_mapa_cf, 20260929-220513-e2_fila_sup
- Simulação: completo 297.67 · sem loteria 232.35
- Última enviada: {"versao": 32, "sem_loteria": 232.35}
- Arquivo pronto esperando ok: nenhum

## Vazão

- Fila: 27 · rodando: 0 · feitos nas últimas 24 h: 10 · consultas à metade B: 0

## Últimos candidatos

| id | origem | receita | decisão | A sem lot. | B sem lot. | min |
|---|---|---|---|---|---|---|
| 9 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 700, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | +0.08 | — | 10.8 |
| 8 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 300, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.29 | — | 8.8 |
| 7 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true}` | A: não seleciona | -0.06 | — | 9.6 |
| 6 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true}` | A: não seleciona | -0.14 | — | 11.3 |
| 5 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-ctx": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.26 | — | 9.1 |
| 4 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.74 | — | 10.1 |
| 3 | gerador | `{"--conjunto": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.13 | — | 9.1 |
| 2 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.19 | — | 9.6 |
| 1 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true, "--superficie": true}` | A: não seleciona | +0.19 | — | 11.0 |
| 37 | agente | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--stand-prefixo": true}` | A: não seleciona | -0.13 | — | 9.8 |

## Revisões

2026-09-30: Estreia: limite de RAM 10 → 8 GB (desktop aberto deixa ~9,3 GB; pico do corretor 7,8 GB). Sonda de ruído (repetição do membro 1) na frente da fila.

