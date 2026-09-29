#!/usr/bin/env bash
# Laço fiel sobre a v30 (29/09): cada candidato é o corretor de verdade, base e oof da v30.
set -u
cd "$(dirname "$0")/../.."
V30="--base 20260929-125421-base_ctx_por_apt --crossfit --conjunto --externos --plano13 --dist-plano --corretor-ref --mapa --reusar-oof 20260929-154118-v29_mapa_cf"
OUT=.superpowers/noite2/resultados.txt
: > "$OUT"
roda() {  # nome, flags extras
  local nome=$1; shift
  bin/run src/stack.py "n2_$nome" $V30 "$@" > ".superpowers/noite2/$nome.log" 2>&1
  local id
  id=$(grep -o "== fim [0-9-]*[^:]*" ".superpowers/noite2/$nome.log" | awk '{print $3}')
  { echo "### $nome ($*) → $id"; bin/run src/compare.py "$id" 2>&1; echo; } >> "$OUT"
}
roda ref_v30        --stand-prefixo --corretor-rounds 500
roda superficie     --stand-prefixo --corretor-rounds 500 --superficie
roda fila           --fila --corretor-rounds 500
roda peso_lot_03    --stand-prefixo --corretor-rounds 500 --peso-loteria 0.3
roda rounds_700     --stand-prefixo --corretor-rounds 700
echo FIM >> "$OUT"
