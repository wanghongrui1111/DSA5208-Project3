# Scaling Study: MPI + OpenMP CG for the 2D and 3D Poisson Equation

This directory has the code, job scripts, raw measurements and analysis for the
**strong-scaling (E2)** and **weak-scaling (E8)** results in the report. It also
includes the **E1** serial runs that serve as the speedup baseline. All
measurements were made on Vanda CPU nodes.

We solve −Δu = f on the unit square (2D) or unit cube (3D) with zero Dirichlet
boundary conditions. The equation is discretised with the 5-point or 7-point
stencil and solved by matrix-free conjugate gradient (CG): each matrix–vector
product is a stencil sweep over the local subdomain plus a halo exchange with
neighbouring ranks.

## Implementation

| Program | Source | Parallelism |
|---|---|---|
| `serial` | `serial_main.cpp`, `serial.cpp` | none (E1 baseline) |
| `poisson` | `mpi_main.cpp`, `solver.cpp` | MPI ranks on a 1D, 2D or 3D Cartesian process grid; one OpenMP thread per rank in all reported runs |

Parallelisation choices measured in E2 and E8:

| Choice | Options | Used in |
|---|---|---|
| Domain decomposition | 1D strips (`1d`), 2D blocks (`2d`), 3D blocks (`3d`); grids from `MPI_Dims_create` | E2: all three; E8: `2d`, `3d` |
| Halo exchange | blocking paired `MPI_Sendrecv` (`block`); `Isend`/`Irecv` overlapped with the interior stencil (`overlap`) | E2, E8 |
| CG formulation | classic (2 blocking reductions per iteration); pipelined (1 non-blocking reduction overlapped with the stencil) | E2: classic; E8 2D: both; E8 3D: classic |
| Rank placement | each rank pinned to one core of the PBS cpuset (`scripts/bind_rank.sh`) | all |

### Command-line options of `poisson`

| Option | Meaning |
|---|---|
| `--dim 2\|3 --n N` | dimension and grid points per axis |
| `--decomp 1d\|2d\|3d` | decomposition |
| `--halo block\|overlap` | halo strategy |
| `--cg classic\|pipelined` | CG formulation |
| `--iters K\|auto` | fixed iteration count with `tol=0`; `auto` = clamp(100·4096²/N, 100, 1000) |
| `--tol T` | run to a relative residual instead (validation) |
| `--repeat R` | timed solves per run; the CSV row stores the median-time solve |
| `--check-serial 1` | also solve serially on every rank and compare the solutions |
| `--csv file --tag name` | append one result row |

Each CSV row contains these fields:

- time per iteration, as the maximum over ranks;
- compute, halo and reduction counters, each maximised over ranks separately (they do not add up to the total);
- halo bytes and messages;
- error norms, process grid and node count.

## Contents

```
src/                     solver source + Makefile  (make -C src  ->  bin/serial, bin/poisson)
scripts/env.sh           paths and the mpirun wrapper (Open MPI: --mca coll ^hcoll,ucc, UCX_POSIX_USE_PROC_LINK=n)
scripts/bind_rank.sh     pins each rank to one core of the PBS cpuset
scripts/run_e1.sh        E1  serial size sweep
scripts/run_e2s.sh       E2a strong scaling on one 72-core node
scripts/run_e2m8.sh      E2b strong scaling on 8 nodes x 32 ranks
scripts/run_e8.sh        E8  weak scaling (single-node or 8 x 32 mode, chosen from the allocation)
scripts/submit_retry.sh  qsub wrapper that resubmits a job PBS failed to start
pbs/                     e1.pbs, e2s.pbs, e2m8.pbs, e8.pbs
results/                 supplied raw data: one directory per PBS job (CSV + hosts.csv/nodes.csv)
analysis/aggregate.py    raw CSVs -> published points (e2_points.csv, e8_points.csv), standard library only
analysis/generate_figures.py   published points -> report figures 02-06
```

## Reproducing the results

### 1. Extract and upload the project

1. Extract the ZIP on your local computer and upload this directory to Vanda.
   Replace `YOUR_NUSNET_ID` with your username:
   ```bash
   scp -r scaling_2d3d YOUR_NUSNET_ID@vanda.nus.edu.sg:~/
   ```
2. Log in and enter the directory. **Run all cluster commands below from this
   directory.** The PBS scripts use `$PBS_O_WORKDIR` as the project root, so
   `qsub` must be called from here:
   ```bash
   cd ~/scaling_2d3d
   ```
3. New jobs write to `results_rerun/<NAME>/`. The supplied `results/` is never
   modified. Use a new `NAME` for every job, because rerunning a name replaces
   that directory's CSV.

Every job compiles the solver on its own compute node with `-march=native`. No
separate compilation step or interactive job is needed. Single-node jobs build
into `/tmp`; multi-node jobs build into the project directory so that every node
can see the binary, and they delete it afterwards.

**Dependencies.** On Vanda the jobs load `nvhpc-hpcx-cuda12/26.1` (NVHPC 26.1,
HPC-X Open MPI 4.1.9). Override this with `-v MODULES=...` if the module
changes. Elsewhere, any C++17 compiler with MPI and OpenMP works:
`make -C src MPICXX=mpicxx OMPFLAG=-fopenmp`. Local analysis needs Python 3.8+;
the figures also need matplotlib (`python3 -m pip install matplotlib`).

**Quick local check** (no cluster, tiny sizes):

```bash
make -C src
QUICK=1 bash scripts/run_e1.sh
mpirun -np 4 bin/poisson --dim 2 --n 257 --decomp 2d --tol 1e-10 --check-serial 1
QUICK=1 NOBIND=1 RANKS="1 2 4" PASSES=1 bash scripts/run_e8.sh
```

### 2. Vanda configuration used in the report

| Item | Configuration |
|---|---|
| CPU node | 2 × Intel Xeon Platinum 8452Y, 36 cores per socket, 72 cores per node |
| Single-node allocation | `select=1:ncpus=72` (whole node; ranks 1–72) |
| Multi-node allocation | `select=8:ncpus=32:mpiprocs=32`, `place=scatter` (8 nodes × 32 cores; one 32-core vnode per node) |
| Network | 100 GbE |
| Compiler and MPI | NVHPC 26.1 (`nvhpc-hpcx-cuda12/26.1`), HPC-X Open MPI 4.1.9, `-O3 -march=native` |
| MPI settings | `--mca coll ^hcoll,ucc`, `UCX_POSIX_USE_PROC_LINK=n`, `--bind-to none` + `bind_rank.sh` |
| Threads | `OMP_NUM_THREADS=1` |
| Node class | not restricted (`CPU_ONLY=0`); `-v CPU_ONLY=1` accepts only `cn-*` nodes |

The scripts request CPUs only, never `ngpus`, so the jobs go to the CPU queue
even if PBS places them on a GPU-partition (`gn-*`) node. If your account must
name a project or queue, add `-P <project>` and/or `-q <queue>` to the `qsub`
lines below. Running `hpc project` lists your projects.

Common run protocol, built into the scripts:

- **Timing.** `tol=0` and a fixed iteration count, reporting time per iteration. Each run makes several timed solves (repeats) and keeps the median.
- **Single-node points.** `--iters auto` = clamp(100·4096²/N, 100, 1000), evaluated at the global N, with 3 repeats.
- **Multi-node points.** 1000 iterations and 5 repeats at every node count, including the one-node baseline. Short runs at 128–256 ranks would otherwise be dominated by jitter.
- **Passes.** Each job repeats its whole sweep `PASSES` times; even passes use the reverse rank order, to expose drift and order effects.
- **Same allocation for every node count.** A multi-node job runs node counts 1, 2, 4 and 8 on the **first k nodes of its own allocation**, so the baseline and the scaled points share hardware.

### 3. E1: serial baseline

The serial times are the denominator of every E2 speedup.

| Item | Setting |
|---|---|
| Sizes | 2D n = 64, 128, 256, 512, 1024, 2048, 4096, 8192; 3D n = 16, 32, 64, 100, 128, 200, 256, 384 |
| Protocol | classic CG, `--iters auto`, 3 repeats |
| Jobs in the report | `e1`, `e1_r2` (two independent jobs) |

```bash
qsub -v NAME=e1    pbs/e1.pbs
qsub -v NAME=e1_r2 pbs/e1.pbs
```

**Check:** `results_rerun/e1*/e1.csv` should have 16 data rows, and the job log
`p3_e1.o<id>` should end with `E1_DONE rc=0`. A size is accepted only if the two
jobs agree within 10%.

### 4. E2a: strong scaling on one node (1–72 ranks)

| Item | Setting |
|---|---|
| Problems | 2D n = 1024 and 4096; 3D n = 100 and 256 (fixed global size) |
| Configurations | 2D: {`1d`, `2d`} × {`block`, `overlap`}; 3D: `3d` × {`block`, `overlap`}; classic CG (12 configurations) |
| Ranks | 1, 2, 4, 8, 16, 32, 64, 72 on one 72-core node |
| Protocol | `--iters auto`, 3 repeats, 2 passes |
| Walltime | 1 h 30 min (default in `e2s.pbs`); the 2-rank/32–72-rank supplementary jobs took about 10 min |

Base jobs, at least two independent ones:

```bash
qsub -v NAME=e2s_r1 pbs/e2s.pbs
qsub -v NAME=e2s_r2 pbs/e2s.pbs
```

**Check:** `results_rerun/e2s_r*/e2s_r*.csv` should have 12 × 8 × 2 = **192**
rows, and the log `p3_e2s.o<id>` should end with `E2S_DONE rc=0`.
`hosts.csv` records the node.

Supplementary jobs in the supplied data repeated only the points whose spread
was wide (2D n = 1024 and 3D n = 100 at 2, 32, 64 and 72 ranks; three passes).
Each such job gives (4 + 2) × 4 × 3 = **72** rows:

```bash
for j in 1 2 3 4 5; do
  qsub -v NAME=e2sw_r$j,PASSES=3,RANKS=2:32:64:72,SIZES2D=1024,SIZES3D=100 -l walltime=00:20:00 pbs/e2s.pbs
done
```

The supplied jobs `e2sw_r1..r5` ran on `cn-*` nodes and `e2swg_r3..r5` on
`gn-*` nodes; the commands are the same. Lists passed through `qsub -v` must be
separated by `:` because PBS does not allow commas or spaces inside a value.

### 5. E2b: strong scaling on 8 nodes × 32 ranks (32–256 ranks)

| Item | Setting |
|---|---|
| Problems | 2D n = 4096; 3D n = 256 |
| Configurations | 2D: {`1d`, `2d`} × {`block`, `overlap`}; 3D: `3d` × {`block`, `overlap`}; classic CG (6 configurations) |
| Ranks | 32 per node on the first 1, 2, 4, 8 nodes = 32, 64, 128, 256 ranks |
| Protocol | 1000 iterations, 5 repeats, 2 passes (base jobs) or 3 (supplementary) |
| Walltime | 45 min for three passes over all node counts (31.9 min measured) |

Each job first checks that its nodes can reach one another and runs a 64-rank
smoke test. It stops with exit 3 or 4 if either fails; resubmit the job.
Base jobs:

```bash
for j in 1 2; do
  qsub -v NAME=e2m8_r$j -l select=8:ncpus=32:mpiprocs=32 -l place=scatter -l walltime=00:45:00 pbs/e2m8.pbs
done
```

**Check:** 6 × 4 × 2 = **48** rows per job, `E2M8_DONE rc=0` in `p3_e2m8.o<id>`,
and `nodes.csv` lists the 8 nodes.

Supplementary jobs in the supplied data added samples at selected node counts.
`NODES_LIST` chooses the node counts; the allocation must contain at least the
largest one:

| Supplied jobs | Variables | Rows |
|---|---|---|
| `e2m8c_r4`, `e2m8cg_r3` | `PASSES=3,NODES_LIST=1:2:4:8` | 72 |
| `e2m8c_r1`, `e2m8c_r2`, `e2m8w_r1`, `e2m8w_r2` | `PASSES=3,NODES_LIST=4:8` | 36 |
| `e2m8cg_r4`, `e2m8cg_r5` | `PASSES=3,NODES_LIST=1:2` with `select=2:ncpus=32:mpiprocs=32` | 36 |

`e2m8cg_r3` was cut by its 30-minute walltime and kept 59 of its 72 rows. The
completed rows are valid and were used.

```bash
qsub -v NAME=e2m8c_r5,PASSES=3,NODES_LIST=1:2:4:8 -l select=8:ncpus=32:mpiprocs=32 -l place=scatter -l walltime=00:45:00 pbs/e2m8.pbs
```

### 6. E8s: weak scaling on one node (1–72 ranks)

| Item | Setting |
|---|---|
| Local problem | 1024² points per rank (2D), 100³ (3D); global n = round(1024·√P) or round(100·P^(1/3)) |
| Configurations | 2D `2d` × {`block`, `overlap`} × {`classic`, `pipelined`}; 3D `3d` × {`block`, `overlap`} × `classic` (6 configurations) |
| Ranks | 1, 2, 4, 8, 16, 32, 64, 72 on one 72-core node |
| Protocol | `--iters auto` at the global N, 3 repeats, 2 passes |
| Walltime | about 10 min estimated; request 30 min |
| Jobs in the report | `e8s_r1`, `e8s_r2`, `e8s_r3` (three independent jobs) |

The mode is chosen from the allocation: one node gives the single-node mode.

```bash
for j in 1 2 3; do
  qsub -v NAME=e8s_r$j,PASSES=2 -l select=1:ncpus=72 -l walltime=00:30:00 pbs/e8.pbs
done
```

**Check:** 6 × 8 × 2 = **96** rows per job, `E8_DONE rc=0` in `p3_e8.o<id>`.
The log `results_rerun/e8s_r*/e8s_r*.log` prints the global `n` of each run.

### 7. E8m: weak scaling on 8 nodes × 32 ranks (32–256 ranks)

| Item | Setting |
|---|---|
| Local problem and configurations | as E8s |
| Ranks | 32 per node on the first 1, 2, 4, 8 nodes = 32, 64, 128, 256 ranks |
| Protocol | 1000 iterations, 5 repeats, 1 pass |
| Walltime | about 36 min estimated; request 60 min |
| Jobs in the report | `e8m_r2` only (see below) |

```bash
for j in 1 2; do
  qsub -v NAME=e8m_r$j,PASSES=1,NODES_LIST=1:2:4:8,EXPECT_NN=8 \
       -l select=8:ncpus=32:mpiprocs=32 -l place=scatter -l walltime=01:00:00 pbs/e8.pbs
done
```

**Check:** 6 × 4 = **24** rows per job and `E8_DONE rc=0`. `nodes.csv` lists
the nodes. Of the two supplied jobs, `e8m_r1` was about 1.7 times slower at
every point, including the one-node 32-rank point, and is excluded in
`aggregate.py`. The multi-node weak-scaling values therefore come from one job
and are descriptive. Two clean jobs are needed to meet the two-job rule.

### 8. Expected run-to-run variation on Vanda

Rerun timings will not match the supplied numbers exactly, and should not be
expected to. Vanda is a shared cluster, and the following affect each run:

- **Node assignment.** PBS places jobs on whichever nodes are free. CPU-partition (`cn-*`) and GPU-partition (`gn-*`) nodes both appeared in our data. GPU nodes were about 25% slower and noisier at 128–256 ranks.
- **Shared nodes and network.** Other users' jobs on the same node, or on the 100 GbE network, change the available memory bandwidth and message latency. Multi-node points are set by the slowest node.
- **Allocation layout.** A request for 32 cores per node lands in one vnode (inferred to be one socket). A whole 72-core node spans both sockets. The same rank count can therefore differ by about 1.7× in per-point cost, so the single-node and multi-node curves must not be joined.
- **Frequency, cache and ordering effects.** On one node, the second pass at 4–16 ranks was systematically up to 20% faster than the first. We did not find the cause.
- **Individual slow allocations.** One multi-node weak-scaling job (`e8m_r1`) was about 1.7× slower at every point and was excluded.

In the supplied data, 11 of 24 multi-node strong-scaling points have fewer than
two thirds of their runs within ±10% of the median. Expect differences of
roughly 10–20% per point, and larger at 128–256 ranks. To reduce the effect:

- run at least two independent jobs per point;
- check `hosts.csv` / `nodes.csv` for the nodes each job used;
- compare medians and interquartile ranges, not single runs.

The trends should still reproduce: memory-bandwidth saturation on one node,
near-flat multi-node weak scaling, and reductions that grow with the rank count.

### 9. Package, download and analyse the results

On Vanda, from the project directory:

```bash
shopt -s nullglob
tar czf "results_rerun_$(hostname -s).tgz" results_rerun/ p3_*.o*
```

Locally:

```bash
cd scaling_2d3d
scp "YOUR_NUSNET_ID@vanda.nus.edu.sg:~/scaling_2d3d/results_rerun_*.tgz" .
tar xzf results_rerun_*.tgz
```

Aggregate and plot either the supplied data or your new runs:

```bash
python3 analysis/aggregate.py                  # supplied data in results/
python3 analysis/aggregate.py results_rerun    # your new jobs
python3 analysis/generate_figures.py           # -> figures/02_single_node ... 06_weak_components (.pdf/.png)
```

`aggregate.py` writes `analysis/e2_points.csv` (120 points from the supplied
data) and `analysis/e8_points.csv` (72 points). It applies the report's rules:

- Each published value is the median over every run of every job and pass, with 25th–75th percentiles.
- A point needs at least two PBS jobs; the multi-node weak series is the exception and is flagged `single_job`.
- The iteration protocol above is checked.
- `wide` marks points with fewer than two thirds of runs within ±10% of the median. It describes spread and does not reject data.
- Speedup uses the E1 serial median of the same problem.
- `cn_gn` counts the runs from `cn-*` and `gn-*` nodes.

The job directory names decide which series a job joins: `e1*`, `e2s*_r*`,
`e2m8*_r*`, `e8s_r*` and `e8m_r*`. Keep these prefixes when naming new jobs.
Run both commands from this directory; `generate_figures.py` reads the CSVs
written by the most recent `aggregate.py`.
