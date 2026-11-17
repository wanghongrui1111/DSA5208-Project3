#!/bin/bash
# run_e8.sh -- E8 weak scaling: every rank owns (about) the same number of points, the global grid grows with the rank count.
#   2D: local 1024^2 points per rank, global n = round(1024*sqrt(P))        3D: local 100^3 points per rank, global n = round(100*P^(1/3))
#   configs: 2D 2d-blocks x {block,overlap} x {classic,pipelined};  3D 3d-blocks x {block,overlap} x classic   (plan section 6.1: no 3D CG variants)
#   Performance protocol of E1/E2 (publication rule P2): tol=0, per-iteration time reported, median of the repeats.
#     single: ITERS=auto (plan formula clamp(100*4096^2/N,100,1000) at the GLOBAL N), REPEAT=3
#     multi : ITERS=1000, REPEAT=5 at every node count (as E2b: short runs at 128-256 ranks are jitter dominated)
#
# Two layouts, chosen by the allocation (override with MODE=single|multi):
#   single : ONE whole node (72 cores), ranks RANKS="1 2 4 8 16 32 64 72"                      (all ranks of the job on one node, as in E2a)
#   multi  : 8 chunks x 32 cores, node counts NODES_LIST="1 2 4 8" -> 32,64,128,256 ranks      (first k nodes of the allocation, as in E2b)
# The two layouts are NOT the same hardware layout (multi uses ONE socket per node): they are reported as separate series.
#
# Env: NAME=e8s_r1 PASSES=2 ITERS=200 REPEAT=3 LOCAL2=1024 LOCAL3=100 HALOS="block overlap" CGS2D="classic pipelined" CGS3D=classic
#      RANKS NODES_LIST PPN BIN RES QUICK=1 (tiny sizes) DRY=1 (print only) MAX_FAILS=3 NOBIND=1.  Lists may use ':' or ',' (qsub -v).
# Output: $RES/$NAME/$NAME.csv (tag E8_p<pass>), $NAME.log, hosts.csv (single) or nodes.csv (multi).
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
NAME=${NAME:-e8s_r1}; OUT=$RES/$NAME; mkdir -p "$OUT"
PASSES=${PASSES:-2}
if [ "${QUICK:-0}" = 1 ]; then LOCAL2=${LOCAL2:-32}; LOCAL3=${LOCAL3:-12}; else LOCAL2=${LOCAL2:-1024}; LOCAL3=${LOCAL3:-100}; fi
HALOS=${HALOS:-"block overlap"}; CGS2D=${CGS2D:-"classic pipelined"}; CGS3D=${CGS3D:-classic}
RANKS=${RANKS:-"1 2 4 8 16 32 64 72"}; NODES_LIST=${NODES_LIST:-"1 2 4 8"}
for _v in RANKS NODES_LIST HALOS CGS2D CGS3D; do eval "[ -n \"\${$_v:-}\" ] && $_v=\$(echo \"\$$_v\" | tr ':,' '  ')"; done

if [ -n "${PBS_NODEFILE:-}" ]; then NNODE=$(sort -u "$PBS_NODEFILE" | wc -l | tr -d ' '); else NNODE=1; fi
MODE=${MODE:-$([ "$NNODE" -gt 1 ] && echo multi || echo single)}
if [ "$MODE" = multi ]; then ITERS=${ITERS:-1000}; REPEAT=${REPEAT:-5}; else ITERS=${ITERS:-auto}; REPEAT=${REPEAT:-3}; fi
if [ "${QUICK:-0}" = 1 ]; then ITERS=${QITERS:-20}; REPEAT=1; fi
export OMP_NUM_THREADS=1 OMP_PLACES=${OMP_PLACES:-cores} OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export MPI_OVERSUB=""
BINDER="$(dirname "${BASH_SOURCE[0]}")/bind_rank.sh"; [ "${NOBIND:-0}" = 1 ] && BINDER=""
CSV=$OUT/$NAME.csv; LOG=$OUT/$NAME.log; rm -f "$CSV" "$OUT/nodes.csv"; : > "$LOG"

# global grid edge for P ranks with LOCAL points per rank along each axis:  n = round(LOCAL * P^(1/dim))
wn() { awk -v p="$1" -v d="$2" -v l="$3" 'BEGIN { if (d == 2) n = l * sqrt(p); else n = l * exp(log(p) / 3); printf "%d\n", n + 0.5 }'; }

if [ "$MODE" = single ]; then
  ncpu=$(awk '/^Cpus_allowed_list/ {print $2}' /proc/self/status 2>/dev/null | tr ',' '\n' | awk -F- 'NF==2 {s+=$2-$1+1; next} NF==1 && $1!="" {s+=1} END{print s+0}')
  [ "${ncpu:-0}" -gt 0 ] || ncpu=$(sysctl -n hw.ncpu 2>/dev/null || echo 1)
  host=$(hostname -s | tr A-Z a-z)
  cpu=$( (lscpu 2>/dev/null | sed -n 's/^Model name: *//p') || true ); [ -n "$cpu" ] || cpu=$(uname -m)
  echo "$(date +%FT%T),$host,$ncpu,\"$cpu\",${PBS_JOBID:-local}" >> "$OUT/hosts.csv"
  export MPI_EXTRA=${MPI_EXTRA-"--bind-to none"}
  UNITS=($RANKS); WHERE="host=$host"
  echo "run_e8: mode=single host=$host allowed cores=$ncpu ranks=[$RANKS] passes=$PASSES iters=$ITERS repeat=$REPEAT local2=$LOCAL2 local3=$LOCAL3" | tee -a "$LOG"
else
  NF=${PBS_NODEFILE:?PBS_NODEFILE not set (multi mode runs inside a PBS job)}
  NODES=(); while IFS= read -r l; do NODES+=("$l"); done < <(awk '!s[$0]++' "$NF")
  PPN=${PPN:-$(sort "$NF" | uniq -c | head -1 | awk '{print $1}')}
  UNITS=($NODES_LIST); WHERE="multi"
  echo "run_e8: mode=multi ${#NODES[@]} node(s), PPN=$PPN, node counts [$NODES_LIST] passes=$PASSES iters=$ITERS repeat=$REPEAT local2=$LOCAL2 local3=$LOCAL3" | tee -a "$LOG"
  HFD=$(mktemp -d); trap 'rm -rf "$HFD"' EXIT
  for k in $NODES_LIST; do
    [ "$k" -le "${#NODES[@]}" ] || { echo "!! need $k nodes, allocation has ${#NODES[@]}"; exit 2; }
    : > "$HFD/hf_$k"; for ((i=0; i<k; i++)); do for ((j=0; j<PPN; j++)); do echo "${NODES[$i]}" >> "$HFD/hf_$k"; done; done
  done
  NMAX=${#NODES[@]}
  if [ -z "${DRY:-}" ]; then
    MPI_EXTRA="--hostfile $NF --map-by ppr:1:node --bind-to none" mpi_run "$NMAX" bash -c \
      'echo "$(hostname -s | tr A-Z a-z),\"$(lscpu | sed -n "s/^Model name: *//p")\",$(awk "/^Cpus_allowed_list/ {print \$2}" /proc/self/status)"' \
      2>/dev/null | grep -v '^Authorized' > "$OUT/nodes.csv"; cat "$OUT/nodes.csv"
  fi
fi

cfgs=()
for h in $HALOS; do for cg in $CGS2D; do cfgs+=("2 2d $h $cg"); done; done
for h in $HALOS; do for cg in $CGS3D; do cfgs+=("3 3d $h $cg"); done; done

FAILS=0; NRUN=0; SKIP=0
for pass in $(seq 1 "$PASSES"); do
  ul=("${UNITS[@]}"); if [ $((pass % 2)) = 0 ]; then ul=($(printf '%s\n' "${UNITS[@]}" | sort -rn)); fi   # reversed order on even passes
  for c in "${cfgs[@]}"; do
    set -- $c; dim=$1; dc=$2; h=$3; cg=$4
    for u in "${ul[@]}"; do
      if [ "$MODE" = single ]; then
        R=$u; if [ "$R" -gt "$ncpu" ]; then SKIP=$((SKIP + 1)); echo "skip R=$R > $ncpu cores" >> "$LOG"; continue; fi
      else
        R=$((u * PPN)); export MPI_EXTRA="--hostfile $HFD/hf_$u --map-by ppr:$PPN:node --bind-to none"
      fi
      if [ "$dim" = 2 ]; then n=$(wn "$R" 2 "$LOCAL2"); else n=$(wn "$R" 3 "$LOCAL3"); fi
      NRUN=$((NRUN + 1))
      echo "### pass=$pass R=$R dim=$dim n=$n decomp=$dc halo=$h cg=$cg $WHERE" >> "$LOG"
      if [ -n "${DRY:-}" ]; then echo "DRY: MPI_EXTRA='${MPI_EXTRA:-}' mpi_run $R $BINDER $BIN/poisson --dim $dim --n $n --decomp $dc --halo $h --cg $cg --iters $ITERS --repeat $REPEAT"; continue; fi
      mpi_run "$R" $BINDER "$BIN/poisson" --dim "$dim" --n "$n" --decomp "$dc" --halo "$h" --cg "$cg" --iters "$ITERS" \
          --repeat "$REPEAT" --csv "$CSV" --tag "E8_p$pass" >> "$LOG" 2>&1 \
        || { FAILS=$((FAILS + 1)); echo "!! FAILED pass=$pass R=$R dim=$dim n=$n $dc $h $cg"; tail -5 "$LOG" | grep -vE "UCX +WARN|parser.c"
             [ "$FAILS" -ge "${MAX_FAILS:-3}" ] && { echo "!! $FAILS failures: stopping early (MAX_FAILS)"; break 3; }; }
    done
  done
  echo "pass $pass done ($(date +%T))"
done
echo "$NAME DONE: $NRUN runs, $FAILS failed, $SKIP skipped, $([ -f "$CSV" ] && echo $(($(wc -l < "$CSV") - 1)) || echo 0) CSV rows  ($CSV)"
[ "$FAILS" -eq 0 ] && [ "$SKIP" -eq 0 ]
