#!/bin/bash
# run_e2s.sh -- E2 "same node" version: ALL rank counts inside ONE whole-node job, so every rank count sees the same
# hardware / neighbours, and the 1-rank baseline comes from the same job.
#   2D: n in {1024,4096} x decomp {1d,2d} x halo {block,overlap};  3D: n in {100,256} x 3d x halo {block,overlap}
#   classic CG, threads=1, tol=0, iters=auto, repeat=3 (median).  Config-outer / rank-inner loop (all R of a config are close in time);
#   PASSES=2 repeats everything, the second pass in reversed rank order (exposes drift / order effects).
#   Env: RANKS="1 2 4 8 16 32 64 72" PASSES=2 SIZES2D SIZES3D DEC2D HALOS REPEAT NAME=e2s QUICK=1 BIN RES
# Output: $RES/$NAME/$NAME.csv (tag E2S_p<pass>), $NAME.log, hosts.csv.   Ranks above the allowed core count are skipped (never oversubscribed).
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
NAME=${NAME:-e2s}; OUT=$RES/$NAME; mkdir -p "$OUT"
if [ "${QUICK:-0}" = 1 ]; then SIZES2D=${SIZES2D:-"64 128"}; SIZES3D=${SIZES3D:-"16 24"}; else
  SIZES2D=${SIZES2D:-"1024 4096"}; SIZES3D=${SIZES3D:-"100 256"}; fi
RANKS=${RANKS:-"1 2 4 8 16 32 64 72"}; PASSES=${PASSES:-2}
DEC2D=${DEC2D:-"1d 2d"}; HALOS=${HALOS:-"block overlap"}; REPEAT=${REPEAT:-3}
# qsub -v cannot carry spaces/commas inside a value: accept ':' or ',' separated lists too
for _v in RANKS NODES_LIST SIZES2D SIZES3D DEC2D HALOS; do eval "[ -n \"\${$_v:-}\" ] && $_v=\$(echo \"\$$_v\" | tr ':,' '  ')"; done
export OMP_NUM_THREADS=1 OMP_PLACES=${OMP_PLACES:-cores} OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export MPI_OVERSUB=""; export MPI_EXTRA=${MPI_EXTRA-"--bind-to none"}
BINDER="$(dirname "${BASH_SOURCE[0]}")/bind_rank.sh"; [ "${NOBIND:-0}" = 1 ] && BINDER=""
ncpu=$(awk '/^Cpus_allowed_list/ {print $2}' /proc/self/status 2>/dev/null | tr ',' '\n' | awk -F- 'NF==2 {s+=$2-$1+1; next} NF==1 && $1!="" {s+=1} END{print s+0}')
[ "${ncpu:-0}" -gt 0 ] || ncpu=$(sysctl -n hw.ncpu 2>/dev/null || echo 1)
host=$(hostname -s | tr A-Z a-z)
cpu=$( (lscpu 2>/dev/null | sed -n 's/^Model name: *//p') || true ); [ -n "$cpu" ] || cpu=$(uname -m)
echo "$(date +%FT%T),$host,$ncpu,\"$cpu\",${PBS_JOBID:-local}" >> "$OUT/hosts.csv"
CSV=$OUT/$NAME.csv; LOG=$OUT/$NAME.log; rm -f "$CSV"; : > "$LOG"
cfgs=()
for n in $SIZES2D; do for dc in $DEC2D; do for h in $HALOS; do cfgs+=("2 $n $dc $h"); done; done; done
for n in $SIZES3D; do for h in $HALOS; do cfgs+=("3 $n 3d $h"); done; done
FAILS=0; NRUN=0; SKIP=0
echo "run_e2s: host=$host allowed cores=$ncpu ranks=[$RANKS] passes=$PASSES configs=${#cfgs[@]}" | tee -a "$LOG"
for pass in $(seq 1 "$PASSES"); do
  rl=($RANKS); if [ $((pass % 2)) = 0 ]; then rl=($(for r in $RANKS; do echo $r; done | sort -rn)); fi   # reversed order on even passes
  for c in "${cfgs[@]}"; do
    set -- $c
    for R in "${rl[@]}"; do
      if [ "$R" -gt "$ncpu" ]; then SKIP=$((SKIP + 1)); echo "skip R=$R > $ncpu cores" >> "$LOG"; continue; fi
      NRUN=$((NRUN + 1))
      echo "### pass=$pass R=$R dim=$1 n=$2 decomp=$3 halo=$4 host=$host" >> "$LOG"
      mpi_run "$R" $BINDER "$BIN/poisson" --dim "$1" --n "$2" --decomp "$3" --halo "$4" --cg classic --iters auto \
          --repeat "$REPEAT" --csv "$CSV" --tag "E2S_p$pass" >> "$LOG" 2>&1 \
        || { FAILS=$((FAILS + 1)); echo "!! FAILED pass=$pass R=$R dim=$1 n=$2 $3 $4"; }
    done
  done
  echo "pass $pass done ($(date +%T))"
done
echo "$NAME DONE: $NRUN runs, $FAILS failed, $SKIP skipped, $(($(wc -l < "$CSV") - 1)) CSV rows  ($CSV)"
[ "$FAILS" -eq 0 ] && [ "$SKIP" -eq 0 ]
