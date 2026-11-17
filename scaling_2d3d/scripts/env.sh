#!/bin/bash
# env.sh -- shared by all scripts.  Source it:  . scripts/env.sh
#   BIN      directory with serial / poisson          (default <root>/bin)
#   RES      results directory                        (default <root>/results_rerun; the supplied results/ is never written)
#   mpi_run NP [args...]   launcher wrapper: Open MPI (-x env, --oversubscribe) or MPICH/Hydra
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN=${BIN:-$SRC/bin}
RES=${RES:-$SRC/results_rerun}
mkdir -p "$RES"

MPIRUN=${MPIRUN:-mpirun}
if $MPIRUN --version 2>&1 | grep -qi "open mpi\|open-mpi"; then
  MPI_FLAVOR=ompi
else
  MPI_FLAVOR=hydra
fi

# mpi_run NP cmd args...   (extra launcher flags via MPI_EXTRA, e.g. "--map-by ppr:64:node --bind-to core")
mpi_run() {
  local np=$1; shift
  if [ "$MPI_FLAVOR" = ompi ]; then
    $MPIRUN -np "$np" ${MPI_OVERSUB---oversubscribe} ${MPI_EXTRA:-} --mca coll ^hcoll,ucc -x UCX_POSIX_USE_PROC_LINK=n \
      -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND -x LD_LIBRARY_PATH "$@"
  else
    $MPIRUN -n "$np" ${MPI_EXTRA:-} -genv OMP_NUM_THREADS "${OMP_NUM_THREADS:-1}" "$@"
  fi
}
