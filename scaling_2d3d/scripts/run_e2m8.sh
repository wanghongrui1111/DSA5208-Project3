#!/bin/bash
# run_e2m8.sh (copy of run_e2m.sh with the `bash -c ... shift` bug fixed and early stop after MAX_FAILS failures)
# run_e2m.sh -- E2 "same node group" version for multi-node points: inside ONE multi-node allocation run
#   R = PPN * k for k in NODES_LIST (default 1 2 4 nodes => 64 128 256 ranks at PPN=64) on the FIRST k nodes of $PBS_NODEFILE,
#   so the 1-node point (baseline) and the 2-/4-node points use the same nodes, same job.  Plan: multi-node only at 2D n=4096, 3D n=256.
#   Env: NODES_LIST="1 2 4" PASSES=2 SIZES2D=4096 SIZES3D=256 DEC2D HALOS REPEAT NAME=e2m DRY=1 (print commands only) PPN BIN RES
# Env also: BIG_ITERS=1000 BIG_REPEAT=5 MULTI_MIN_NODES=2 (see below).  Output: $RES/$NAME/$NAME.csv (tag E2M_p<pass>), $NAME.log, nodes.csv (hostname + CPU model of every node used).
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
NAME=${NAME:-e2m}; OUT=$RES/$NAME; mkdir -p "$OUT"
NF=${PBS_NODEFILE:?PBS_NODEFILE not set (run inside a PBS job)}
NODES=(); while IFS= read -r l; do NODES+=("$l"); done < <(awk '!s[$0]++' "$NF")   # unique nodes in allocation order (no mapfile: bash 3 portable)
PPN=${PPN:-$(sort "$NF" | uniq -c | head -1 | awk '{print $1}')}
if [ "${QUICK:-0}" = 1 ]; then SIZES2D=${SIZES2D:-"128"}; SIZES3D=${SIZES3D:-"24"}; else SIZES2D=${SIZES2D:-"4096"}; SIZES3D=${SIZES3D:-"256"}; fi
NODES_LIST=${NODES_LIST:-"1 2 4"}; PASSES=${PASSES:-2}
# multi-node points (>= MULTI_MIN_NODES nodes) are only ~0.05-0.1 s long at the plan's iteration count => OS/network jitter dominates:
# use BIG_ITERS iterations and BIG_REPEAT repeats there (per-iteration time is what is reported; with MULTI_MIN_NODES=1 every point of this job uses the same protocol; set 2 to keep the 1-node point at iters=auto)
BIG_ITERS=${BIG_ITERS:-1000}; BIG_REPEAT=${BIG_REPEAT:-5}; MULTI_MIN_NODES=${MULTI_MIN_NODES:-1}   # 1 => the 1-node baseline point uses the same protocol (speedup base must not be noisier than the points)
DEC2D=${DEC2D:-"1d 2d"}; HALOS=${HALOS:-"block overlap"}; REPEAT=${REPEAT:-3}
# qsub -v cannot carry spaces/commas inside a value: accept ':' or ',' separated lists too
for _v in RANKS NODES_LIST SIZES2D SIZES3D DEC2D HALOS; do eval "[ -n \"\${$_v:-}\" ] && $_v=\$(echo \"\$$_v\" | tr ':,' '  ')"; done
export OMP_NUM_THREADS=1 OMP_PLACES=${OMP_PLACES:-cores} OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export MPI_OVERSUB=""
BINDER="$(dirname "${BASH_SOURCE[0]}")/bind_rank.sh"; [ "${NOBIND:-0}" = 1 ] && BINDER=""
CSV=$OUT/$NAME.csv; LOG=$OUT/$NAME.log; rm -f "$CSV" "$OUT/nodes.csv"; : > "$LOG"
echo "run_e2m: ${#NODES[@]} node(s) in allocation, PPN=$PPN, node counts [$NODES_LIST], passes=$PASSES" | tee -a "$LOG"
# hostfile per node count: first k unique nodes, PPN slots each
HFD=$(mktemp -d); trap 'rm -rf "$HFD"' EXIT
for k in $NODES_LIST; do
  [ "$k" -le "${#NODES[@]}" ] || { echo "!! need $k nodes, allocation has ${#NODES[@]}"; exit 2; }
  : > "$HFD/hf_$k"; for ((i=0; i<k; i++)); do for ((j=0; j<PPN; j++)); do echo "${NODES[$i]}" >> "$HFD/hf_$k"; done; done
done
# record the hardware of every node (extra evidence for hetero CPU/GPU node mixes)
NMAX=${#NODES[@]}
if [ -z "${DRY:-}" ]; then
  MPI_EXTRA="--hostfile $NF --map-by ppr:1:node --bind-to none" mpi_run "$NMAX" bash -c \
    'echo "$(hostname -s | tr A-Z a-z),\"$(lscpu | sed -n "s/^Model name: *//p")\",$(awk "/^Cpus_allowed_list/ {print \$2}" /proc/self/status)"' \
    2>/dev/null | grep -v '^Authorized' > "$OUT/nodes.csv"; cat "$OUT/nodes.csv"
fi
cfgs=()
for n in $SIZES2D; do for dc in $DEC2D; do for h in $HALOS; do cfgs+=("2 $n $dc $h"); done; done; done
for n in $SIZES3D; do for h in $HALOS; do cfgs+=("3 $n 3d $h"); done; done
FAILS=0; NRUN=0
for pass in $(seq 1 "$PASSES"); do
  kl=($NODES_LIST); if [ $((pass % 2)) = 0 ]; then kl=($(for k in $NODES_LIST; do echo $k; done | sort -rn)); fi   # reversed order on even passes
  for c in "${cfgs[@]}"; do
    set -- $c
    for k in "${kl[@]}"; do
      R=$((k * PPN)); NRUN=$((NRUN + 1))
      if [ "$k" -ge "$MULTI_MIN_NODES" ]; then ITR=$BIG_ITERS; RPT=$BIG_REPEAT; else ITR=auto; RPT=$REPEAT; fi
      export MPI_EXTRA="--hostfile $HFD/hf_$k --map-by ppr:$PPN:node --bind-to none"
      echo "### pass=$pass nodes=$k R=$R dim=$1 n=$2 decomp=$3 halo=$4  nodes: ${NODES[*]:0:$k}" >> "$LOG"
      if [ -n "${DRY:-}" ]; then echo "DRY: MPI_EXTRA='$MPI_EXTRA' mpi_run $R $BINDER $BIN/poisson --dim $1 --n $2 --decomp $3 --halo $4 ..."; continue; fi
      # call mpi_run directly (shell function; `timeout` cannot run functions, so the time limit is the PBS walltime + MPI_TIMEOUT below)
      mpi_run "$R" $BINDER "$BIN/poisson" --dim "$1" --n "$2" --decomp "$3" --halo "$4" --cg classic --iters "$ITR" \
          --repeat "$RPT" --csv "$CSV" --tag "E2M_p$pass" >> "$LOG" 2>&1 \
        || { FAILS=$((FAILS + 1)); echo "!! FAILED pass=$pass nodes=$k R=$R dim=$1 n=$2 $3 $4"; tail -5 "$LOG" | grep -vE "UCX +WARN|parser.c"; [ "$FAILS" -ge "${MAX_FAILS:-3}" ] && { echo "!! $FAILS failures: stopping early (MAX_FAILS)"; break 3; }; }
    done
  done
  echo "pass $pass done ($(date +%T))"
done
echo "$NAME DONE: $NRUN runs, $FAILS failed, $([ -f "$CSV" ] && echo $(($(wc -l < "$CSV") - 1)) || echo 0) CSV rows  ($CSV)"
[ "$FAILS" -eq 0 ]
