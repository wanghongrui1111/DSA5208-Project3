# Helpers sourced by validate.sh and bench.sh.
#
# Launcher settings (environment):
#   CG_MPI_FLAVOR   openmpi | hydra (MPICH / Intel MPI); detected if unset
#   CG_PPN          ranks per node; unset = fill nodes in order
#   CG_MAP          Open MPI mapping unit, e.g. "socket" to alternate ranks
#                   between sockets; unset = fill cores in order ("slot")
#   CG_MPIRUN_EXTRA extra launcher arguments (e.g. --oversubscribe, a hostfile)

CG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CG_BIN="${CG_BIN:-$CG_ROOT/${CG_BUILD_DIR:-build}}"

if [ -z "${CG_MPI_FLAVOR:-}" ]; then
  if mpirun --version 2>&1 | grep -q "Open MPI"; then CG_MPI_FLAVOR=openmpi; else CG_MPI_FLAVOR=hydra; fi
fi

# run_mpi <ranks> <threads> <executable> [args...]
run_mpi() {
  local np=$1 nt=$2
  shift 2
  export OMP_NUM_THREADS=$nt OMP_PLACES=cores OMP_PROC_BIND=close
  if [ "$CG_MPI_FLAVOR" = openmpi ]; then
    local map="${CG_MAP:-slot}"
    [ -n "${CG_PPN:-}" ] && map="ppr:${CG_PPN}:node"
    local envs=(-x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND)
    local v
    for v in CUDA_VISIBLE_DEVICES OMPI_MCA_opal_cuda_support LD_LIBRARY_PATH; do
      [ -n "${!v:-}" ] && envs+=(-x "$v")
    done
    # shellcheck disable=SC2086
    mpirun -np "$np" --map-by "${map}:PE=${nt}" --bind-to core "${envs[@]}" ${CG_MPIRUN_EXTRA:-} "$@"
  else
    local ppn=()
    [ -n "${CG_PPN:-}" ] && ppn=(-ppn "$CG_PPN")
    # shellcheck disable=SC2086
    I_MPI_PIN_DOMAIN=omp mpiexec -n "$np" "${ppn[@]}" -genv OMP_NUM_THREADS "$nt" \
      -genv OMP_PLACES cores -genv OMP_PROC_BIND close ${CG_MPIRUN_EXTRA:-} "$@"
  fi
}

# Prints one progress line from the key/value output of a solver run.
summarize() {
  awk '$1 ~ /^(impl|n|ranks|threads|decomp|comm|gpu_mpi|reduce|cg|iters|time_s|err_inf)$/ {printf "%s=%s ", $1, $2} END {print ""}'
}

# try <log> <command...>: runs one solver launch; a failure is reported and
# appended to <log> instead of ending the whole script, so a single bad
# configuration does not waste a batch job.
try() {
  local log=$1
  shift
  if ! "$@" | summarize; then
    echo "FAILED: $*" | tee -a "$log" >&2
  fi
}

# GPU halo modes this MPI library supports: "staged aware" if it is CUDA-aware
# at run time, otherwise "staged". Tested with two ranks (on one GPU if need be).
gpu_mpi_modes() {
  if OMPI_MCA_opal_cuda_support=true run_mpi 2 1 "$CG_BIN/cg_mpi_cuda" --n 64 --tol 0 --maxit 2 \
    --gpu-mpi aware > /dev/null 2>&1; then
    echo "staged aware"
  else
    echo "staged"
  fi
}
