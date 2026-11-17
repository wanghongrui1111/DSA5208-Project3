#!/bin/bash
# Performance experiments; each appends rows to $CG_OUT/<experiment>.csv.
#   bash scripts/bench.sh <experiment>...
#
# Experiments
#   serial      serial baselines T_s for the strong-scaling sizes
#   strong      strong scaling, pure MPI. Classic CG with every halo strategy
#               (1d/2d x blocking/overlap/rma, 1d allgather); the other two CG
#               variants with 1d/2d x blocking/overlap
#   placement   one node, ranks packed onto the first socket vs alternated between sockets
#   hybrid      all cores, every ranks x threads split, halo overlap vs thread
#   repro       cost of reproducible reductions: fast vs repro over the rank counts
#   weak        weak scaling, pure MPI, CG_WEAK_LOCAL^2 points per rank
#   gpu         one GPU, several problem sizes, the three CG variants
#   gpu_strong  strong scaling over GPUs: {blocking,overlap} x {staged,aware} x {classic,pipe}
#   gpu_weak    weak scaling over GPUs, CG_GPU_WEAK_LOCAL^2 points per GPU
#
# Settings (environment)
#   CG_OUT             output directory (default results/<host>/bench)
#   CG_CORES           cores per node to use (default: physical cores of this node)
#   CG_NODES           nodes in the allocation (default 1)
#   CG_MIN_RANKS       skip process counts below this in strong/weak/repro (default 1)
#   CG_GPUS            GPUs per node (default 1)
#   CG_N, CG_N_SMALL   strong-scaling sizes (default 4096 and 1024)
#   CG_N_GPU           GPU strong-scaling size (default 8192)
#   CG_ITERS           iterations for n = 4096; smaller grids run proportionally
#                      more, up to 1000 (default 100)
#   CG_REPEAT          timed solves per run, the median is reported (default 3)
#   CG_WEAK_LOCAL      local block edge for CPU weak scaling (default 1024)
#   CG_GPU_WEAK_LOCAL  local block edge for GPU weak scaling (default 4096)
set -euo pipefail
source "$(dirname "$0")/lib.sh"

OUT="${CG_OUT:-$CG_ROOT/results/$(hostname -s)/bench}"
mkdir -p "$OUT"
FAIL="$OUT/failures.txt"
physical_cores() { lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l; }
CORES="${CG_CORES:-$(physical_cores)}"
NODES="${CG_NODES:-1}"
GPUS="${CG_GPUS:-1}"
TOTAL=$((CORES * NODES))
TOTAL_GPUS=$((GPUS * NODES))
N_BIG="${CG_N:-4096}"
N_SMALL="${CG_N_SMALL:-1024}"
N_GPU="${CG_N_GPU:-8192}"
ITERS="${CG_ITERS:-100}"
REPEAT="${CG_REPEAT:-3}"
export OMPI_MCA_opal_cuda_support=true

# Powers of two up to $1, plus $1 itself, skipping counts below CG_MIN_RANKS
# (multi-node jobs set it so they do not repeat the single-node runs).
counts_up_to() {
  local p=1 out="" min="${CG_MIN_RANKS:-1}"
  while [ $p -lt "$1" ]; do
    [ $p -ge "$min" ] && out="$out $p"
    p=$((p * 2))
  done
  echo "$out $1"
}

# Iterations for grid size n: about the same work per run as n = 4096.
iters_for() {
  local it=$((ITERS * 4096 * 4096 / ($1 * $1)))
  [ $it -lt "$ITERS" ] && it=$ITERS
  [ $it -gt 1000 ] && it=1000
  echo $it
}

# Global grid edge with about $1^2 points per process for $2 processes.
weak_n() { awk -v l="$1" -v p="$2" 'BEGIN { printf "%d\n", l * sqrt(p) + 0.5 }'; }

run_cpu() {  # run_cpu <csv> <ranks> <threads> <n> [solver args...]
  local csv=$1 np=$2 nt=$3 n=$4
  shift 4
  try "$FAIL" run_mpi "$np" "$nt" "$CG_BIN/cg_mpi_omp" --n "$n" --tol 0 --maxit "$(iters_for "$n")" \
    --repeat "$REPEAT" --csv "$OUT/$csv.csv" "$@"
}

run_gpu() {  # run_gpu <csv> <ranks> <n> [solver args...]
  local csv=$1 np=$2 n=$3
  shift 3
  CG_PPN=$GPUS try "$FAIL" run_mpi "$np" 1 "$CG_BIN/cg_mpi_cuda" --n "$n" --tol 0 --maxit "$(iters_for "$n")" \
    --repeat "$REPEAT" --csv "$OUT/$csv.csv" "$@"
}

exp_serial() {
  local n cg
  for n in $N_SMALL $N_BIG; do
    for cg in classic cgcg pipe; do
      try "$FAIL" "$CG_BIN/cg_serial" --n "$n" --tol 0 --maxit "$(iters_for "$n")" --repeat "$REPEAT" --cg $cg \
        --csv "$OUT/serial.csv"
    done
  done
}

exp_strong() {
  local n np dc cg halos
  for n in $N_SMALL $N_BIG; do
    for np in $(counts_up_to $TOTAL); do
      for cg in classic cgcg pipe; do
        if [ "$np" = 1 ]; then  # no neighbours: decomposition and halo mode do not matter
          run_cpu strong 1 1 "$n" --cg $cg
          continue
        fi
        halos="1d:blocking 1d:overlap 2d:blocking 2d:overlap"
        [ $cg = classic ] && halos="$halos 1d:rma 2d:rma 1d:allgather"
        for dc in $halos; do
          # allgather moves the whole vector every iteration: one timed solve is enough
          run_cpu strong "$np" 1 "$n" --cg $cg --decomp "${dc%:*}" --comm "${dc#*:}" \
            $([ "${dc#*:}" = allgather ] && echo --repeat 1)
        done
      done
    done
  done
}

exp_placement() {
  if [ "$CG_MPI_FLAVOR" != openmpi ]; then
    echo "placement needs Open MPI's --map-by; skipped"
    return
  fi
  local n np map
  for n in $N_SMALL $N_BIG; do
    for np in $(CG_MIN_RANKS=2 counts_up_to "$CORES"); do
      for map in slot socket; do
        CG_MAP=$map run_cpu placement "$np" 1 "$n" --cg classic --tag "map=$map"
      done
    done
  done
}

exp_hybrid() {
  local n nt cg comm
  for n in $N_SMALL $N_BIG; do
    for nt in 1 2 3 4 6 8 9 12 16 18 24 32 36 48 64 72 96 128; do
      [ $((CORES % nt)) = 0 ] || continue
      for cg in classic pipe; do
        for comm in overlap thread; do
          # CORES / nt ranks on every node; without it a multi-node run
          # would put one rank per slot and oversubscribe the cores
          CG_PPN=$((CORES / nt)) run_cpu hybrid $((TOTAL / nt)) "$nt" "$n" --cg $cg --comm $comm
        done
      done
    done
  done
}

exp_repro() {
  local n np reduce
  for n in $N_SMALL $N_BIG; do
    for np in $(counts_up_to $TOTAL); do
      for reduce in fast repro; do
        run_cpu repro "$np" 1 "$n" --cg classic --comm overlap --reduce $reduce
      done
    done
  done
}

exp_weak() {
  local local_n="${CG_WEAK_LOCAL:-1024}" np n comm cg
  for np in $(counts_up_to $TOTAL); do
    n=$(weak_n "$local_n" "$np")
    for comm in blocking overlap; do
      for cg in classic pipe; do
        run_cpu weak "$np" 1 "$n" --cg $cg --comm $comm --tag "local=$local_n"
      done
    done
  done
}

exp_gpu() {
  local n cg
  for n in 1024 2048 4096 8192; do
    for cg in classic cgcg pipe; do
      run_gpu gpu 1 "$n" --cg $cg
    done
  done
}

exp_gpu_strong() {
  local np comm gm cg modes
  modes=$(gpu_mpi_modes)
  echo "GPU halo modes: $modes"
  for np in $(CG_MIN_RANKS=1 counts_up_to $TOTAL_GPUS); do
    for cg in classic pipe; do
      if [ "$np" = 1 ]; then
        run_gpu gpu_strong 1 "$N_GPU" --cg $cg
        continue
      fi
      for comm in blocking overlap; do
        for gm in $modes; do
          run_gpu gpu_strong "$np" "$N_GPU" --cg $cg --comm $comm --gpu-mpi $gm
        done
      done
    done
  done
}

exp_gpu_weak() {
  local local_n="${CG_GPU_WEAK_LOCAL:-4096}" np n comm cg gm
  gm=$(gpu_mpi_modes)
  gm=${gm##* }  # aware if available
  for np in $(CG_MIN_RANKS=1 counts_up_to $TOTAL_GPUS); do
    n=$(weak_n "$local_n" "$np")
    for comm in blocking overlap; do
      for cg in classic pipe; do
        run_gpu gpu_weak "$np" "$n" --cg $cg --comm $comm --gpu-mpi "$gm" --tag "local=$local_n"
      done
    done
  done
}

[ $# -gt 0 ] || { sed -n '2,32p' "$0"; exit 1; }
{
  echo "date: $(date -Is)"
  echo "host: $(hostname)  cores/node used: $CORES  nodes: $NODES  gpus/node: $GPUS"
  lscpu | grep -E 'Model name|Socket|Core\(s\)|NUMA node\(s\)' || true
  command -v nvidia-smi > /dev/null && nvidia-smi -L || true
  mpirun --version 2>&1 | head -1
} >> "$OUT/machine.txt"
for e in "$@"; do
  echo "== $e ($(date +%T))"
  "exp_$e"
done
echo "== done ($(date +%T)), results in $OUT"
if [ -s "$FAIL" ]; then
  echo "== $(wc -l < "$FAIL") run(s) failed, see $FAIL"
fi
