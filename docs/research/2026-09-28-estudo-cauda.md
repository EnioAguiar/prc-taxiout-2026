# Estudo da cauda — 28/09/2026

Scripts descartáveis sobre 2025 inteiro (DEP) e sobre as previsões da v12 no holdout jan+jul 2025.

## Onde está o erro da v12 (holdout, RMSE 309,78)

- **Concentração extrema:** os 25 maiores voos = 40,8 % do erro²; 100 = 48,1 %; 1.000 = 61,8 %.
  Só 2 voos do LFPG sem NM (y = 84.240 e 58.206 s, previstos ~800 s, `MVT − SCHED` ~0,5 h) = **31 %**.
- **Voos sem LOBT (sem NM) = 53 % do erro²** (32,2 % com `MVT − SCHED` ≤ 1 h, onde estão as loterias).
- Com LOBT: `MVT − LOBT` ≤ 1 h = 39 % do erro² (RMSE 160 e 287); 1–1,5 h = 6,4 %; acima = 1,5 %.

## Loterias são um piso comum, não o que separa os times

Prever uma loteria exige saber qual voo; prevendo a média condicional q·V num grupo de N
voos, o ganho é N·(q·V)² ≈ 3·10⁷ s² para o LFPG sem NM (q ≈ 1/600, V ≈ 60.000) → ~0,16 s.
Ninguém ganha isso. **Consequência [inferência]:** todos os times pagam as loterias de 2026;
a diferença para o topo está na parte previsível. Se as loterias de 2026 custarem o mesmo que
no holdout (~1/3 do erro²), o 3º (228,6) erra ~metade do que nós na parte previsível.

## Achados

- **A janela do LOBT captura a forma da cauda com NM:** com `MVT − LOBT` > 2 h, y > 1 h em 100 %;
  BLOCK − LOBT mediano 16 min. Nos 32 desses voos com ADS-B, 28 mostram táxi real curto
  (< 30 min) mas y > 1 h: o registro é defeituoso (BLOCK perto do LOBT/SCHED); o sensor vê a
  realidade, não o registro (o que o GREKI disse no Discord).
- **A cópia do SCHED depende da companhia** (prefixo de `FLIGHT_mvt`), estável entre meses
  (correlação 1,00 entre 10 meses e jan/jul): LIRF sem NM com atraso > 1 h, RYR 33 %, WMT 49 %,
  quase todas as outras ~0 %. A base não tem a companhia nos voos sem NM
  (`AIRCRAFT_OPERATOR_flt` vem do NM).
  - Categórica aeroporto:companhia no corretor: 309,27 → 306,31, **mas sem loteria só −0,1** e
    normais +1,3 → o ganho veio de voos loteria (sorte). Descartado.
  - Taxa de cópia suavizada por (aeroporto, companhia, NM) aprendida só nos 10 meses: −1,0 s
    (sem loteria −1,3). Pequeno; guardado para somar.

## Lição de validação

O RMSE completo do holdout é dominado por 2 voos; decisões devem olhar `sem_loteria` e
`normais_nm`, não só o completo (o `compare.py` já imprime as duas como informação).
