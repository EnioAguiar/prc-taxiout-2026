#!/usr/bin/env bash
# Base com regressor global sem LIRF (--reg-sem-lirf) e o corretor da v30 por cima.
set -eu
cd "$(dirname "$0")/../.."
bin/run src/experiment.py base_ctx_por_apt_semlirf --model two_stage_nm --nm-min-ms 21600 \
  --janela-lobt --reg-corte 7200 --base-ctx --base-por-apt --reg-sem-lirf
BASE=$(tail -n 1 experiments.jsonl | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
bin/run src/compare.py "$BASE" 20260929-125421-base_ctx_por_apt
bin/run src/stack.py v31_sem_lirf_cf --base "$BASE" --crossfit --conjunto --externos --plano13 \
  --dist-plano --stand-prefixo --corretor-rounds 500 --corretor-ref --mapa
ID=$(tail -n 1 experiments.jsonl | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
bin/run src/compare.py "$ID"
