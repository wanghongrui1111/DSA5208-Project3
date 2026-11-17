# Sourced by the PBS job scripts after setting CG_CLUSTER: environment, a fresh
# build on the compute node, allocation size and output directory.
trap 'echo "ERROR: job stopped at line $LINENO of ${BASH_SOURCE[0]}: $BASH_COMMAND" >&2' ERR
cd "$PBS_O_WORKDIR"
set +u  # module and environment scripts read unset variables
source scripts/cluster/env.sh "$CG_CLUSTER"
set -u

export CG_OUT="$PBS_O_WORKDIR/results/$CG_CLUSTER/$PBS_JOBNAME"
mkdir -p "$CG_OUT"

# Build here, not on the login node: Vanda's login nodes are VMs (Lecture 8),
# and -march=native must match the compute node's CPU. One build directory
# per job, so jobs running at the same time do not overwrite each other.
export CG_BUILD_DIR="build-$CG_CLUSTER-$PBS_JOBNAME"
if [[ "${CG_CMAKE_ARGS:-}" != *CG_ENABLE_CUDA=OFF* ]] && command -v nvidia-smi > /dev/null; then
  # Native code for the GPU on this node, e.g. "8.0" -> 80. Older drivers do
  # not know the field; then the portable PTX default is used.
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | head -1 | tr -d '. ') || cap=""
  if [[ "$cap" =~ ^[0-9]+$ ]]; then
    export CG_CMAKE_ARGS="${CG_CMAKE_ARGS:-} -DCMAKE_CUDA_ARCHITECTURES=$cap"
  fi
fi
echo "building into $CG_BUILD_DIR"
if ! bash scripts/build.sh > "$CG_OUT/build.log" 2>&1; then
  tail -30 "$CG_OUT/build.log"
  echo "build failed, see $CG_OUT/build.log"
  exit 1
fi

# Allocation as PBS sees it: cores per node from the select statement, nodes
# from the node file, GPUs from the list PBS hands to the job.
export CG_CORES="${CG_CORES:-${NCPUS:-$(nproc)}}"
export CG_NODES="${CG_NODES:-$(sort -u "$PBS_NODEFILE" | wc -l)}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CG_GPUS="${CG_GPUS:-$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)}"
elif command -v nvidia-smi > /dev/null; then
  export CG_GPUS="${CG_GPUS:-$(nvidia-smi -L | wc -l)}"
fi
export CG_MPIRUN_EXTRA="${CG_MPIRUN_EXTRA:---hostfile $PBS_NODEFILE}"
{
  echo "job $PBS_JOBID ($PBS_JOBNAME) on $(sort -u "$PBS_NODEFILE" | tr '\n' ' ')"
  echo "cores/node=$CG_CORES nodes=$CG_NODES gpus/node=${CG_GPUS:-0}"
  module list 2>&1
} | tee "$CG_OUT/job_info.txt"
