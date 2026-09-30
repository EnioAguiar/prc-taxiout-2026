#!/usr/bin/env bash
# Variantes do corretor para a média (base e oof da v30).
set -u
cd "$(dirname "$0")/../.."
V30="--base 20260929-125421-base_ctx_por_apt --crossfit --conjunto --externos --plano13 --dist-plano --corretor-ref --mapa --corretor-rounds 500 --reusar-oof 20260929-154118-v29_mapa_cf"
bin/run src/stack.py e2_fila_sup $V30 --fila --superficie
bin/run src/stack.py e2_xgb $V30 --stand-prefixo --corretor-xgb
bin/run src/stack.py e2_fila_sup_xgb $V30 --fila --superficie --corretor-xgb
echo FIM
