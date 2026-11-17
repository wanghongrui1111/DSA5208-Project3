#!/bin/bash
# Correctness checks. Writes CSVs and residual histories to OUTDIR
# (default results/<host>/validation), then runs check_validation.py.
#   bash scripts/validate.sh [OUTDIR]
#   CG_VALIDATE_GPU=0 skips the MPI+CUDA solver, CG_VALIDATE_CPU=0 the
#   MPI+OpenMP runs (e.g. in a GPU job after a CPU job has covered them).
set -euo pipefail
source "$(dirname "$0")/lib.sh"
OUT="${1:-$CG_ROOT/results/$(hostname -s)/validation}"
mkdir -p "$OUT/history"
rm -f "$OUT"/*.csv "$OUT"/history/* "$OUT/failures.txt"
FAIL="$OUT/failures.txt"
CPU="${CG_VALIDATE_CPU:-1}"
GPU=0
if [ "${CG_VALIDATE_GPU:-1}" = 1 ] && [ -x "$CG_BIN/cg_mpi_cuda" ]; then GPU=1; fi
export OMPI_MCA_opal_cuda_support=true
if [ $GPU = 1 ]; then
  GPU_MODES=$(gpu_mpi_modes)
  GPU_BEST=${GPU_MODES##* }
  echo "GPU halo modes: $GPU_MODES"
fi

echo "== 1. discretisation error under grid refinement (serial)"
for n in 31 63 127 255 511 1023; do
  try "$FAIL" "$CG_BIN/cg_serial" --n $n --tol 1e-12 --csv "$OUT/convergence.csv"
done

echo "== 2. parallel runs against the serial reference"
# Pipelined CG cannot reach 1e-10 reliably: its recurrence residual stagnates
# (see the residual histories), so it is checked at 1e-8.
N=257
tol_for() { if [ "$1" = pipe ]; then echo 1e-8; else echo 1e-10; fi; }
for cg in classic cgcg pipe; do
  try "$FAIL" "$CG_BIN/cg_serial" --n $N --tol "$(tol_for $cg)" --cg $cg --tag reference --csv "$OUT/consistency.csv"
done
CPU_HALOS="1d:blocking 1d:overlap 1d:thread 1d:rma 1d:allgather 2d:blocking 2d:overlap 2d:thread 2d:rma"
for rt in "1 4" "2 2" "3 1" "4 2" "6 1" "8 1"; do
  [ "$CPU" = 1 ] || break
  read -r np nt <<< "$rt"
  for dc in $CPU_HALOS; do
    for cg in classic cgcg pipe; do
      try "$FAIL" run_mpi "$np" "$nt" "$CG_BIN/cg_mpi_omp" --n $N --tol "$(tol_for $cg)" --decomp "${dc%:*}" \
        --comm "${dc#*:}" --cg $cg --csv "$OUT/consistency.csv"
    done
  done
done
if [ $GPU = 1 ]; then
  for np in 1 2 4; do
    for decomp in 1d 2d; do
      for comm in blocking overlap; do
        for gm in $GPU_MODES; do
          for cg in classic cgcg pipe; do
            try "$FAIL" run_mpi "$np" 1 "$CG_BIN/cg_mpi_cuda" --n $N --tol "$(tol_for $cg)" --decomp $decomp \
              --comm $comm --gpu-mpi $gm --cg $cg --csv "$OUT/consistency.csv"
          done
        done
      done
    done
  done
fi

echo "== 3. residual histories (capped at 4000 iterations: pipelined CG stagnates above 1e-10)"
N=511
HIST="--tol 1e-10 --maxit 4000"
for cg in classic cgcg pipe; do
  try "$FAIL" "$CG_BIN/cg_serial" --n $N $HIST --cg $cg --csv "$OUT/history.csv" \
    --history "$OUT/history/serial_$cg.txt"
  [ "$CPU" = 1 ] && try "$FAIL" run_mpi 8 1 "$CG_BIN/cg_mpi_omp" --n $N $HIST --cg $cg --comm overlap \
    --csv "$OUT/history.csv" --history "$OUT/history/mpi_omp_$cg.txt"
  if [ $GPU = 1 ]; then
    try "$FAIL" run_mpi 2 1 "$CG_BIN/cg_mpi_cuda" --n $N $HIST --cg $cg --comm overlap --gpu-mpi "$GPU_BEST" \
      --csv "$OUT/history.csv" --history "$OUT/history/mpi_cuda_$cg.txt"
  fi
done

echo "== 4. reproducibility: the same run repeated with 1, 2 and 4 threads per rank"
N=257
for reduce in fast repro; do
  for comm in overlap thread; do
    [ "$CPU" = 1 ] || break
    for nt in 1 2 4; do
      for run in 1 2; do
        try "$FAIL" run_mpi 4 $nt "$CG_BIN/cg_mpi_omp" --n $N --tol 1e-10 --comm $comm --reduce $reduce \
          --tag "run$run" --csv "$OUT/reproducibility.csv"
      done
    done
  done
  if [ $GPU = 1 ]; then
    for run in 1 2 3; do
      try "$FAIL" run_mpi 2 1 "$CG_BIN/cg_mpi_cuda" --n $N --tol 1e-10 --comm overlap --gpu-mpi "$GPU_BEST" \
        --reduce $reduce --tag "run$run" --csv "$OUT/reproducibility.csv"
    done
  fi
done

if command -v python3 > /dev/null; then
  python3 "$CG_ROOT/scripts/check_validation.py" "$OUT"
else
  echo "python3 not found: run scripts/check_validation.py $OUT later"
fi
