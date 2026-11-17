#!/bin/bash
# run_e1.sh -- E1 serial scale curve (Experiment_Plan E1): classic CG, tol=0, fixed iters = clamp(100*4096^2/N,100,1000), 3 repeats, median.
#   bash scripts/run_e1.sh            Env: N2="..." N3="..." REPEAT=3 BIN= RES= QUICK=1 (tiny sizes)
# Output: $RES/e1/e1.csv (one row per n), $RES/e1/e1.log, $RES/e1/hosts.csv (node + CPU model, for noise/hardware attribution)
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
NAME=${NAME:-e1}; OUT=$RES/$NAME; mkdir -p "$OUT"   # NAME=e1_run2 for a repeat run (csv inside is always e1.csv)
if [ "${QUICK:-0}" = 1 ]; then N2=${N2:-"32 64 128"}; N3=${N3:-"8 16 32"}; else
  N2=${N2:-"64 128 256 512 1024 2048 4096 8192"}; N3=${N3:-"16 32 64 100 128 200 256 384"}; fi
REPEAT=${REPEAT:-3}
host=$(hostname -s | tr A-Z a-z)
cpu=$( (lscpu 2>/dev/null | sed -n 's/^Model name: *//p') || true ); [ -n "$cpu" ] || cpu=$(uname -m)
echo "$(date +%FT%T),serial,$host,\"$cpu\",${PBS_JOBID:-local}" >> "$OUT/hosts.csv"
rm -f "$OUT/e1.csv"; : > "$OUT/e1.log"
FAILS=0
for d in 2 3; do
  ns=$N2; [ $d = 3 ] && ns=$N3
  list=$(echo $ns | tr ' ' ',')
  echo "##### E1 dim=$d n=$list host=$host" | tee -a "$OUT/e1.log"
  "$BIN/serial" --dim $d --n "$list" --iters auto --repeat "$REPEAT" --csv "$OUT/e1.csv" --tag "E1_$host" 2>&1 | tee -a "$OUT/e1.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { FAILS=$((FAILS + 1)); echo "!! FAILED dim=$d"; }
done
echo "E1 DONE: $(($(wc -l < "$OUT/e1.csv") - 1)) rows, $FAILS failed  ($OUT)"
[ "$FAILS" -eq 0 ]
