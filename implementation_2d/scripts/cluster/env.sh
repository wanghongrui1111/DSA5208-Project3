# Environment on the NUS clusters. Source it with the cluster's name:
#   source scripts/cluster/env.sh atlas
#   source scripts/cluster/env.sh vanda
#
# Modules: CG_MODULES if set, else the toolchain that worked on both Atlas
# and Vanda (Nov 2026), else the first names that load.

CG_CLUSTER="${1:-${CG_CLUSTER:-}}"
case "$CG_CLUSTER" in
  atlas | vanda) export CG_CLUSTER ;;
  *)
    echo "usage: source scripts/cluster/env.sh atlas|vanda" >&2
    return 1
    ;;
esac

# Atlas: the module tree only appears after loading its environment (Lecture 8).
if [ "$CG_CLUSTER" = atlas ] && [ -f /app1/ebenv ]; then
  # shellcheck disable=SC1091
  source /app1/ebenv
fi

cg_load_first() {  # loads the first module name that exists
  local m
  for m in "$@"; do
    if module load "$m" > /dev/null 2>&1; then
      echo "  loaded module $m"
      return 0
    fi
  done
  return 1
}
cg_cmake_ok() {  # CMake >= 3.18 on PATH
  command -v cmake > /dev/null &&
    [ "$(printf '%s\n3.18\n' "$(cmake --version | awk 'NR == 1 {print $3}')" | sort -V | head -1)" = 3.18 ]
}

CG_DEFAULT_MODULES="GCC/12.3.0 OpenMPI/4.1.5-GCC-12.3.0 CMake/3.26.3-GCCcore-12.3.0"
if [ -n "${CG_MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load $CG_MODULES
elif module load $CG_DEFAULT_MODULES > /dev/null 2>&1; then
  echo "  loaded modules $CG_DEFAULT_MODULES"
  cg_load_first Python/3.11.3-GCCcore-12.3.0 Python python3 > /dev/null || true  # for check_validation.py
  if [ "$CG_CLUSTER" = vanda ]; then
    command -v nvcc > /dev/null || cg_load_first CUDA cuda nvhpc ||
      echo "  WARNING: no CUDA module found; cg_mpi_cuda will not be built" >&2
  fi
else
  command -v mpicxx > /dev/null || cg_load_first foss OpenMPI openmpi gompi mpich intel-mpi impi ||
    echo "  WARNING: no MPI module found; set CG_MODULES (see: module avail)" >&2
  cg_cmake_ok || cg_load_first CMake cmake ||
    echo "  WARNING: no CMake >= 3.18; try: python3 -m pip install --user cmake" >&2
  if [ "$CG_CLUSTER" = vanda ]; then
    command -v nvcc > /dev/null || cg_load_first CUDA cuda nvhpc ||
      echo "  WARNING: no CUDA module found; cg_mpi_cuda will not be built" >&2
  fi
fi
export PATH="$HOME/.local/bin:$PATH"  # pip --user installs cmake here

echo "  mpicxx: $(command -v mpicxx || echo missing)  cmake: $(command -v cmake || echo missing)  nvcc: $(command -v nvcc || echo -)"
