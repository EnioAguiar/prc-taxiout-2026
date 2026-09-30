#!/usr/bin/env bash
# Risco da regra do ADS-B (Discord 30/09, arnavhm13): a campeã v32 refeita sem o rastro do
# próprio voo (colunas adsb_*) na base e no corretor. Roda com a esteira pausada.
set -eu
cd "$(dirname "$0")/../.."
ultimo() { tail -n 1 experiments.jsonl | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])"; }
while [ "$(.venv/bin/python -c "import sys;sys.path.insert(0,'src');import esteira;print(esteira.Fila(esteira.DB).contar('rodando'))")" != 0 ]; do sleep 30; done
SEM=""; for c in adsb_taxi adsb_taxi_move adsb_gs0 adsb_takeoff_err adsb_n_chao adsb_gap_max adsb_frac_mlat adsb_lat0 adsb_lon0; do SEM="$SEM --sem-feature $c"; done
bin/run src/experiment.py base_sem_adsb_proprio --model two_stage_nm --nm-min-ms 21600 --janela-lobt \
  --reg-corte 7200 --base-ctx --base-por-apt $SEM
BASE=$(ultimo)
bin/run src/stack.py sem_adsb_m0 --base "$BASE" --crossfit --sem-adsb --conjunto --corretor-ref --corretor-rounds 500 \
  --dist-plano --externos --mapa --plano13 --stand-prefixo
M0=$(ultimo)
bin/run src/stack.py sem_adsb_m1 --base "$BASE" --crossfit --sem-adsb --reusar-oof "$M0" --conjunto \
  --corretor-params '{"num_leaves": 127}' --corretor-ref --corretor-rounds 500 --dist-plano --externos --fila \
  --mapa --plano13 --superficie
M1=$(ultimo)
.venv/bin/python -c "
import sys; sys.path.insert(0,'src'); import regua, campeao
c = campeao.carregar(); r = regua.avaliar_conjunto([m['id'] for m in c['membros']], ['$M0', '$M1'])
print('SEM ADSB PROPRIO', 'completo', round(r['completo'], 2), 'A', r['a'], 'B', r['b'], r['motivo'])
m = campeao.previsao(['$M0', '$M1']); import numpy as np
print('rmse completo', round(float(np.sqrt(((m.y_true - m.pred) ** 2).mean())), 2))"
bin/run src/esteira.py retomar
