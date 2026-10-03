# Esteira de experimentos

Atualizado em 2026-10-03 09:55. Spec: `docs/superpowers/specs/2026-09-30-esteira-design.md`.

## Campeã

- Membros: 20261001-185827-e122, 20260930-140011-e76_m1, 20261002-015703-e140, 20261001-160427-e113_m1
- Bases: 20260930-130008-e76_base, 20261001-150429-e113_base
- Simulação: completo 294.86 · sem loteria 227.74
- Última enviada: {"versao": 37, "sem_loteria": 227.74, "membros": ["20261001-185827-e122", "20260930-140011-e76_m1", "20261002-015703-e140", "20261001-160427-e113_m1"]}
- Arquivo pronto esperando ok: nenhum

## Vazão

- Fila: 17 · rodando: 0 · feitos nas últimas 24 h: 46 · consultas à metade B: 6

## Últimos candidatos

| id | origem | receita | decisão | A sem lot. | B sem lot. | min |
|---|---|---|---|---|---|---|
| 205 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true}` | A: não seleciona | -0.04 | — | 9.8 |
| 220 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-ctx": true, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | +0.33 | — | 11.8 |
| 215 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-ctx": true, "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.22 | — | 10.7 |
| 209 | gerador | `{"--conjunto": true, "--corretor-params": {"num_leaves": 127}, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-ctx": true, "--dist-plano": true, "--externos": true, "--fila": true, "--mapa": true, "--plano13": true, "--superficie": true}` | A: não seleciona | +0.10 | — | 12.0 |
| 204 | gerador | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-ctx": true, "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.44 | — | 9.5 |
| 188 | div:seed3 | `{"--conjunto": true, "--corretor-params": {"bagging_seed": 3, "feature_fraction_seed": 3, "seed": 3}, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.16 | — | 12.0 |
| 187 | div:bag08 | `{"--conjunto": true, "--corretor-params": {"bagging_fraction": 0.8, "bagging_freq": 1, "seed": 11}, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.03 | — | 11.7 |
| 186 | div:ff07 | `{"--conjunto": true, "--corretor-params": {"feature_fraction": 0.7, "seed": 7}, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.11 | — | 12.1 |
| 185 | div:xgb | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-regra": true, "--corretor-xgb": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | código 1 | — | — | 12.3 |
| 196 | abl:to_takeoff_iobt | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["to_takeoff_from_IOBT_flt"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.27 | — | 12.1 |
| 195 | abl:adsb_move | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["adsb_menos_pred", "adsb_taxi_move"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.30 | — | 11.8 |
| 194 | abl:adsb_taxi | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["adsb_taxi"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | código 1 | — | — | 5.9 |
| 184 | abl:cia | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["cia"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.12 | — | 11.5 |
| 183 | abl:janela | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["dist_hi", "dist_lo"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.17 | — | 11.9 |
| 182 | abl:adsb_geo | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["adsb_lat0", "adsb_lon0"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.11 | — | 12.0 |
| 181 | abl:ctx_stand | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["ctx_stand_ult_idade", "ctx_stand_ult_tin"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.17 | — | 11.8 |
| 180 | abl:ext_opdi | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["ext_mesmo_apt", "ext_solo_s"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.12 | — | 11.8 |
| 179 | abl:ext_cia | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["ext_taxa_cia", "ext_taxa_cia_ms"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.21 | — | 12.0 |
| 178 | abl:ret_over | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["ret_overdue_apt", "ret_overdue_rwy"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.09 | — | 11.9 |
| 177 | abl:cel_p50 | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["cel_p50", "cel_pred_menos_p50"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.09 | — | 11.9 |
| 176 | abl:dist_agg | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["dist_min", "n_planos", "n_planos_longos"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.13 | — | 12.1 |
| 175 | abl:rot | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["rot_idade", "rot_mesma_cia", "rot_mesmo_tipo"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.04 | — | 12.0 |
| 174 | abl:mapa | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["map_dist", "map_dmin", "map_extra"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.06 | — | 11.8 |
| 173 | abl:ret_fluxo | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["ret_ativos_rwy", "ret_atraso_15m", "ret_ewma_dep"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.15 | — | 11.6 |
| 193 | deriva:adsb_top4 | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["adsb_gs0", "adsb_lat0", "adsb_lon0", "adsb_n_chao"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.05 | — | 12.0 |
| 192 | deriva:adsb_n_chao | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["adsb_n_chao"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.15 | — | 11.8 |
| 172 | abl:pista_fila | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["pista_dep_h30", "pista_fila_pesadas", "pista_fila_peso", "pista_span20"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.02 | — | 11.6 |
| 171 | abl:cel_disp | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["cel_dp", "cel_n", "cel_nivel", "cel_p90"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.02 | — | 11.7 |
| 170 | abl:ext_diarias | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["ext_atc_min", "ext_late_frac", "ext_pre_min", "ext_reg_frac", "ext_reg_n"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.11 | — | 11.8 |
| 169 | abl:adsb_qual | `{"--conjunto": true, "--corretor-ref": true, "--corretor-rounds": 500, "--corretor-sem-feature": ["adsb_frac_mlat", "adsb_gap_max", "adsb_gs0", "adsb_n_chao", "adsb_takeoff_err"], "--corretor-sem-regra": true, "--dist-plano": true, "--externos": true, "--mapa": true, "--pista": true, "--plano13": true, "--retencao": true}` | A: não seleciona | +0.11 | — | 12.0 |

## Famílias

| família | n | média A | melhor | promovidos | situação |
|---|---|---|---|---|---|
| bloco:--fe-auto | 1 | +0.292 | +0.29 | 0 | explorando |
| bloco:--pseudo | 1 | +0.215 | +0.21 | 0 | explorando |
| bloco:--corretor-sem-ctx | 18 | +0.174 | +0.54 | 0 | ativa |
| bloco:--corretor-sem-feature | 29 | +0.117 | +0.36 | 0 | ativa |
| bloco:--corretor-sem-regra | 6 | +0.088 | +0.63 | 1 | ativa |
| bloco:--retencao | 8 | +0.044 | +0.64 | 1 | ativa |
| outro | 15 | +0.029 | +0.34 | 1 | ativa |
| fila:--stand-prefixo | 8 | +0.008 | +0.25 | 0 | ativa |
| param:min_data_in_leaf | 6 | -0.000 | +0.11 | 0 | ativa |
| param:num_leaves | 10 | -0.018 | +0.26 | 0 | cortada |
| fila:nenhum | 9 | -0.031 | +0.30 | 1 | ativa |
| bloco:--pista | 5 | -0.037 | +0.32 | 1 | ativa |
| bloco:--superficie | 10 | -0.064 | +0.36 | 0 | cortada |
| rodadas | 8 | -0.070 | +0.19 | 0 | ativa |
| param:lambda_l2 | 9 | -0.115 | +0.22 | 0 | ativa |
| fila:--fila | 6 | -0.131 | +0.09 | 0 | ativa |
| param:learning_rate | 10 | -0.164 | +0.07 | 0 | cortada |
| bloco:--mapa | 10 | -0.244 | +0.06 | 0 | cortada |
| bloco:--dist-plano | 10 | -0.303 | +0.06 | 0 | cortada |
| bloco:--corretor-ref | 10 | -0.311 | -0.08 | 0 | cortada |
| base | 15 | -0.399 | +2.37 | 0 | cortada |

## Revisões

2026-09-30: Estreia: limite de RAM 10 → 8 GB (desktop aberto deixa ~9,3 GB; pico do corretor 7,8 GB). Sonda de ruído (repetição do membro 1) na frente da fila.
2026-09-30: Estreia (30/09 00h57–02h40 -03): 10 candidatos (sonda + 9 vizinhos), 0 falhas, 8,8–11,3 min cada, pico RAM ~7,3 GB. Todos reprovados em A (melhor: superficie somado +0,19; sonda de ruído −0,13, conferida contra compare.py). Régua mantida: nenhum candidato chegou a 0,3 em A, então o IC não foi o gargalo. Serviço habilitado (enable). Próxima revisão: 24 h ou primeiro envio.
2026-09-30: 30/09 10h40: revisão pelo rendimento — 53 testes do corretor com 1 promoção (+0,20 s). Mudanças: candidato de base vira média completa (mesmos corretores refeitos sobre a base nova, alinhados pelo voo); grade do gerador só num_leaves/learning_rate e sem variar rodadas; na fila, bases da retrospectiva: --base-mapa (75), --base-p13 (76), --seeds 3 (77).
2026-09-30: Critério do usuário (30/09 11h): a esteira precisa render pelo menos o ritmo manual. Referência: manual 29/09 v28→v32 em ~13 h = 235,21→232,35 sem loteria (−2,86 s simulação; −2,05 s oficial) ≈ 0,2 s/h. Checkpoint 01/10 11h (24 h): esteira precisa somar ≥ 2,0 s sem loteria sobre 232,35 (campeã ≤ 230,35) ou um envio com ganho oficial ≥ 1 s. Se não: pausar o serviço e voltar ao manual (fila vira ferramenta sob demanda). Enquanto a esteira roda, experimentos manuais pesados não cabem na RAM.
2026-10-02: Gerador de vizinhos esgotado (162 cand., 5 aprovados); fila realimentada com 34 candidatos de descoberta bruta: 25 ablacoes de sub-bloco do corretor via --corretor-sem-feature (flag nova, ~11 min porque mantem --reusar-oof), 5 dirigidas pela deriva 2025->2026 (AUC adversaria 0,880; adsb_lon0/lat0/n_chao e ctx_*_60 lideram) e 4 membros de diversidade (xgb, feature_fraction, bagging, seed). Fatias do residuo descartadas como pos-regra: oraculo <= 0,60 s e toda estimativa fora do mes e negativa (docs/research/2026-10-02-descoberta-bruta.md).

