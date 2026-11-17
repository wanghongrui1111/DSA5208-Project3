# Parallel Conjugate Gradient for the 2D Poisson Equation

DSA5208 Project 3: MPI + OpenMP on CPU nodes and MPI + CUDA on GPUs, measured
on the Atlas and Vanda clusters. Results appear in the hybrid, communication-strategy,
CG-variant, placement and GPU sections of the project report.

We solve −Δu = f on the unit square with zero Dirichlet boundary conditions.
The equation is discretised with the 5-point stencil on an n × n grid and solved
with the conjugate gradient (CG) method. The method is matrix-free: each
matrix-vector product is a stencil sweep. A manufactured solution gives an exact
reference for every run.

## Implementation

The CG iteration is written once (`src/cg_driver.hpp`) and drives three
backends, so all versions perform the same arithmetic:

| Program | Parallelism |
|---|---|
| `cg_serial` | none (baseline) |
| `cg_mpi_omp` | MPI ranks on a 1D or 2D Cartesian process grid; OpenMP threads inside each rank |
| `cg_mpi_cuda` | one MPI rank per GPU; all vectors stay on the device |

Parallelisation choices compared in the report:

| Choice | Options |
|---|---|
| Domain decomposition | 1D strips, 2D blocks |
| Halo exchange | `MPI_Sendrecv`; `Isend`/`Irecv` overlapped with the interior; master thread communicates while the others compute; one-sided `MPI_Put` + `MPI_Win_fence`; full copy of the vector on every rank (`MPI_Allgatherv`) |
| CG formulation | classic (2 reductions per iteration), Chronopoulos–Gear (1), pipelined (1, non-blocking and overlapped) |
| Shared memory | every ranks × threads split of a node; packed vs spread rank placement |
| Reductions | library (`MPI_Allreduce`, OpenMP) vs fixed-order, bit-reproducible |
| GPU halos | staged through host memory vs CUDA-aware MPI |

## Validation

- **Discretisation error**: observed order 2.000 under grid refinement (n = 31 … 1023).
- **Agreement with the serial solver**: 234 parallel configurations — every halo strategy and CG variant, 1–8 ranks × 1–4 threads, and 1–4 ranks on Vanda's A40 GPUs. Classic CG reproduces the serial iteration count in all of them.
- **Reproducibility**: with fixed-order reductions the result is bit-identical across repeated runs and thread counts.

## Main results

The CPU results come from Atlas (up to 4 nodes × 24 cores) and a 72-core Vanda node; the GPU results from a Vanda node with 2 × NVIDIA A40.

- **Within a node the solver is limited by memory bandwidth.** It reaches 19× on 72 cores at n = 4096. Spreading ranks over both sockets is up to 1.8× faster than packing them onto one.
- **Global reductions limit multi-node scaling.** In classic CG with 2D decomposition and blocking halos, reductions account for 83% of solve time at n = 1024 and 96 ranks. With 2D decomposition and overlapped halos at 64 and 96 ranks, Chronopoulos–Gear and pipelined CG are approximately 1.5–1.7× faster per iteration than classic CG. Selected hybrid MPI + OpenMP configurations with 2–4 threads per rank also outperform pure MPI.
- **A full copy of the vector on every rank** is about 50× slower than neighbour halos. **One-sided `MPI_Put`** loses to two-sided messages because `MPI_Win_fence` synchronises all ranks.
- **Weak scaling** from 1 to 4 full nodes keeps 75–84% efficiency.
- **GPU (A40):**
  - One GPU reaches about 70% of peak memory bandwidth.
  - Two GPUs give 2.1×, and weak scaling keeps 97–100% efficiency.
  - CUDA-aware MPI gives no gain for these 64 kB halos.
  - One A40 is slower than the 72-core CPU node: 1.16× at n = 4096 and 4.4× at n = 1024.

## Contents

```
src/                 CG driver, CPU and CUDA kernels; serial, MPI+OpenMP and MPI+CUDA backends
CMakeLists.txt       build
scripts/             build, validation, benchmark and plotting scripts
scripts/cluster/     module environment and PBS job scripts for Atlas and Vanda
results/atlas/       Atlas data: 1-node and 4-node jobs, CPU validation
results/vanda/       Vanda data: 72-core CPU node; A40 GPU node with GPU validation
```

## Reproducing the results

### 1. Extract and upload the project

1. Extract the project ZIP on your local computer. From the directory containing
   the extracted `implementation_2d` folder, upload it to both clusters. Replace
   `YOUR_NUSNET_ID` with your username:
   ```bash
   scp -r implementation_2d YOUR_NUSNET_ID@atlas8.nus.edu.sg:~/
   scp -r implementation_2d YOUR_NUSNET_ID@vanda.nus.edu.sg:~/
   ```
2. Log in to the relevant cluster and enter the project directory:
   ```bash
   cd ~/implementation_2d
   ```
   Run all cluster commands below from this directory. The original experiments
   used `~/project3`; the directory in this submission is `implementation_2d`.
3. Before rerunning, rename the supplied `results/` directory to
   `results_supplied/` on each cluster. Do not overwrite an existing backup.
   The benchmark scripts append to CSV files, so each new run must start with
   a separate results directory. If a previous build directory exists, move it
   aside before changing the compiler or MPI environment.

The workflow submits four jobs. Each job compiles the programs automatically on
its allocated compute nodes; no separate compilation or interactive job is needed.

| Job script | Experiments |
|---|---|
| `scripts/cluster/atlas_cpu_1node.pbs` | CPU validation, serial baseline and single-node CPU experiments |
| `scripts/cluster/atlas_cpu_4node.pbs` | CPU scaling experiments using up to four nodes |
| `scripts/cluster/vanda_cpu_1node.pbs` | Serial baseline, MPI/OpenMP combinations and rank placement on 72 CPU cores |
| `scripts/cluster/vanda_gpu.pbs` | GPU validation, single-GPU performance, strong scaling and weak scaling |

**Dependencies.** Compilation requires a C++17 compiler, MPI, OpenMP and
CMake 3.18 or later; the GPU programs also require CUDA. Validation requires
Python 3.6 or later. The job environments are specified separately below.
Local plotting requires Python 3.10 or later, NumPy, pandas and Matplotlib:

```bash
python3 -m pip install numpy pandas matplotlib
```

On Windows, use `py -3 -m pip install numpy pandas matplotlib`, and replace
`python3` with `py -3` in the plotting commands.

### 2. Atlas: CPU experiments

The reported Atlas experiments used the following configuration:

| Item | Configuration |
|---|---|
| Processor | AMD EPYC 7R13; 24 CPU cores allocated per node |
| Single-node allocation | `select=1:ncpus=24:mpiprocs=24` |
| Four-node allocation | `select=4:ncpus=24:mpiprocs=24`, `place=scatter`; 96 allocated cores in total |
| Queue and walltime limit | `parallel`, `06:00:00` |
| Compiler and MPI | GCC 12.3.0, Open MPI 4.1.5 |
| Build and validation tools | CMake 3.26.3, Python 3.11.3 |
| Supplied successful results | `results/atlas/atlas_1node/`, `results/atlas/atlas_4node_r2/` |

#### A1. Configure the job scripts

Use `scripts/cluster/atlas_cpu_1node.pbs` and
`scripts/cluster/atlas_cpu_4node.pbs`. Their resource directives already specify
the allocations, queue and walltime listed above. Retain `place=scatter` in the
four-node script.

In **both PBS files**, insert the following configuration immediately before
`source scripts/cluster/job_common.sh`. These lines belong in the job scripts:

```bash
set +u
if [ -f /app1/ebenv ]; then source /app1/ebenv; fi
module purge
set -u
export CG_MODULES="GCC/12.3.0 OpenMPI/4.1.5-GCC-12.3.0 CMake/3.26.3-GCCcore-12.3.0 Python/3.11.3-GCCcore-12.3.0"
export CG_CORES=24
```

The 24-core setting refers to the allocation, not the physical core count of
the node. CUDA is already disabled in both CPU scripts.

The submitted code incorporates the fixes used for the successful experiments:
the jobs enter `"$PBS_O_WORKDIR"` before loading project files, the MPI launcher
forwards `LD_LIBRARY_PATH` to remote processes, and hybrid runs set
`CG_PPN=$((CORES / nt))` to distribute ranks correctly across nodes.

#### A2. Submit the jobs

```bash
qsub scripts/cluster/atlas_cpu_1node.pbs
qsub scripts/cluster/atlas_cpu_4node.pbs
qstat -u "$USER"
```

Submit both jobs from the project root. Retain the job IDs printed by `qsub`.
In `qstat`, `Q` means queued and `R` means running.

#### A3. Review the outputs

The default job names produce `results/atlas/atlas_1node/` and
`results/atlas/atlas_4node/`. Logs are written in the project root, for example
`atlas_1node.o<job_number>`.

- The single-node `validation/validation_summary.md` must report `Overall: PASS`.
- Neither the results directories nor their validation subdirectories should
  contain a non-empty `failures.txt`.
- `qstat -fx JOB_ID` should show `job_state = F` and `Exit_status = 0`.
  Here, `F` means finished. The exit status alone does not establish that every
  experiment passed; also review the validation summary and failure records.
- The four-node performance CSVs should contain 90 `strong`, 64 `hybrid`,
  12 `repro` and 12 `weak` data rows, excluding headers. All four-node hybrid
  records should have `nodes = 4`.

After downloading the new results in Section 4, generate plots locally:

```bash
python3 scripts/plot.py --label Atlas --out figures/atlas results/atlas/atlas_1node results/atlas/atlas_4node
```

`atlas_4node_r2` contains the supplied historical results; it is **not** the
default output directory for a new job. To plot the supplied data, use
`atlas_1node` and `atlas_4node_r2` instead, under `results_supplied/atlas/` if
they have been moved there. If you change `#PBS -N`, update the plotting paths
to match the new job names.

### 3. Vanda: CPU and GPU experiments

The reported Vanda experiments used the following configuration:

| Item | Configuration |
|---|---|
| CPU node | 2 × Intel Xeon Platinum 8452Y; 36 cores per socket, 72 cores in total |
| CPU allocation | `select=1:ncpus=72:mpiprocs=72` |
| CPU environment | GCC 12.3.0, Open MPI 4.1.5, CMake 3.26.3; CUDA disabled |
| GPU node | 2 × NVIDIA A40 |
| GPU allocation | `select=1:ncpus=72:mpiprocs=72:ngpus=2` |
| GPU environment | `foss/2024a` (GCC 13.3.0, Open MPI 5.0.3), CMake 3.29.3, CUDA 12.9.1 |
| Walltime limit | `01:30:00` for each job |
| Supplied successful results | `results/vanda/vanda_cpu_r2/`, `results/vanda/vanda_gpu/` |

#### V1. Configure the resources and accounting project

Obtain the site-provided CPU and GPU job templates and your accounting project:

```bash
hpc cpu > vanda_cpu_template.txt
hpc gpu > vanda_gpu_template.txt
hpc project
```

In `scripts/cluster/vanda_cpu_1node.pbs` and
`scripts/cluster/vanda_gpu.pbs`, retain the resource allocations and walltime
listed above. Set any required queue and project directives using the
corresponding site template:

- Replace the placeholders in `##PBS -q <queue>` and `##PBS -P <project>`.
  Remove one leading `#` from each directive that must be enabled.
- `-q` selects a queue; `-P` specifies the accounting project. They are separate
  settings. Use your own project from `hpc project`; the original experiments
  used `personal-e1561379`.

#### V2. Set the CPU and GPU job environments

In **`vanda_cpu_1node.pbs`**, insert the following immediately before
`source scripts/cluster/job_common.sh`:

```bash
set +u
module purge
set -u
export CG_MODULES="GCC/12.3.0 OpenMPI/4.1.5-GCC-12.3.0 CMake/3.26.3-GCCcore-12.3.0"
export CG_CORES=72
export CG_CMAKE_ARGS="-DCG_ENABLE_CUDA=OFF"
```

The successful CPU run used Open MPI 4.1.5. The earlier Open MPI 5.0.3 run
rejected CPU binding for the 9 × 8, 3 × 24 and 1 × 72 rank/thread combinations.
The successful rerun retained the numerical parameters and experiment groups.

In **`vanda_gpu.pbs`**, use the following block at the same location, replacing
the existing `CG_MODULES` assignment rather than duplicating it:

```bash
set +u
module purge
set -u
export CG_MODULES="foss/2024a CMake/3.29.3-GCCcore-13.3.0 CUDA/12.9.1"
export CG_CORES=72 CG_GPUS=2
```

The CPU and GPU jobs use different toolchains. Keep the GPU configuration
above to reproduce the A40 experiments. Both scripts already enter
`"$PBS_O_WORKDIR"` before loading `job_common.sh`.

#### V3. Submit the jobs

```bash
qsub scripts/cluster/vanda_cpu_1node.pbs
qsub scripts/cluster/vanda_gpu.pbs
qstat -u "$USER"
```

Submit both jobs from the project root and retain their job IDs. The GPU job
automatically detects CUDA-aware MPI. If it is unavailable, only staged
communication is tested, so the full staged/aware comparison is not reproduced.

#### V4. Review the outputs

New jobs produce `results/vanda/vanda_cpu/` and `results/vanda/vanda_gpu/`.
Logs are written in the project root as `vanda_cpu.o<job_number>` and
`vanda_gpu.o<job_number>`.

- `qstat -fx JOB_ID` should show `F` and `Exit_status = 0`.
- Neither the results directories nor their validation subdirectories should
  contain a non-empty `failures.txt`.
- The GPU `validation/validation_summary.md` must report `Overall: PASS`.
- CPU performance data rows, excluding headers: `serial = 6`, `hybrid = 96`,
  `placement = 28`.
- GPU performance data rows, excluding headers: `gpu = 12`, `gpu_strong = 10`,
  `gpu_weak = 8`.

After downloading the new results in Section 4, generate plots locally:

```bash
python3 scripts/plot.py --label Vanda --out figures/vanda results/vanda/vanda_cpu results/vanda/vanda_gpu
```

`vanda_cpu_r2` contains the supplied historical CPU results. To plot the
supplied data, use `vanda_cpu_r2` and `vanda_gpu` instead, under
`results_supplied/vanda/` if they have been moved there. If you change
`#PBS -N`, update the plotting paths accordingly.

Performance experiments on both clusters use `tol=0` and a fixed iteration
count. Correctness is assessed by the separate validation experiments, rather
than by requiring convergence in every performance record.

### 4. Package and download the results

After the jobs finish, run the following from the project root on **each cluster**:

```bash
shopt -s nullglob
tar czf "results_$(hostname -s).tgz" results/ *.o* *.e* vanda_*_template.txt
```

On your local computer, download both archives into the local project root:

```bash
cd implementation_2d
scp "YOUR_NUSNET_ID@atlas8.nus.edu.sg:~/implementation_2d/results_*.tgz" .
scp "YOUR_NUSNET_ID@vanda.nus.edu.sg:~/implementation_2d/results_*.tgz" .
```

Preserve the supplied local `results/` as `results_supplied/` before extracting
the new archives; do not overwrite an existing backup. Extract each downloaded
archive from the local project root, replacing the filename below:

```bash
tar xzf RESULTS_ARCHIVE.tgz
```

The Atlas and Vanda archives populate separate subdirectories under `results/`.
Use the plotting commands in A3 and V4 with these new results. Retain both
archives, the job logs and the PBS scripts used for the runs.
