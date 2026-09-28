# 5-plano13-testes-baratos

Corretor barato (5 folds por dia) sobre a base `20260927-211418-reg_corte` com `ctx_*` e janela. Decidir por **sem loteria** e **normais NM**; portão ~1 s.

| teste | completo | sem loteria | normais NM | LTFM | LFPG | ganho sem loteria | ganho normais |
|---|---|---|---|---|---|---|---|
| referência | 309.96 | 248.63 | 201.24 | 259.1 | 565.3 |
| METAR | 309.57 | 247.8 | 200.47 | 258.0 | 566.3 | +0.83 | +0.77 |
| rotação no stand | 309.52 | 248.22 | 201.08 | 258.0 | 564.8 | +0.41 | +0.16 |
| consistência NM | 309.58 | 248.29 | 200.88 | 258.4 | 564.5 | +0.34 | +0.36 |
| companhia (todas) | 307.8 | 247.9 | 201.37 | 258.3 | 555.2 | +0.73 | -0.13 |
| todos juntos | 307.48 | 246.7 | 200.35 | 256.9 | 558.2 | +1.93 | +0.89 |

Cobertura (linhas com algum valor): {'METAR': '100%', 'rotação no stand': '100%', 'consistência NM': '100%', 'companhia (todas)': '100%'}
