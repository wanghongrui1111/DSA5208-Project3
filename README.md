# DSA5208 Project 3 — source code

**Parallel Conjugate Gradient for 2D and 3D Poisson Problems** (MPI + OpenMP, MPI + CUDA)  
Wang Hongrui (A0355459N), Zhang Yikai (A0355997B), Yao Jiajie (A0330617L)

There are two independent implementations of matrix-free CG. Each directory has
its own README with build, run and reproduction instructions.

| Directory | What it contains | Report sections |
|---|---|---|
| [`scaling_2d3d/`](scaling_2d3d/README.md) | Serial and MPI + OpenMP CG, 2D and 3D. Vanda strong scaling (E2: one 72-core node, and 8 nodes × 32 ranks up to 256 cores) and weak scaling (E8), with the E1 serial baseline | Serial cost, Strong scaling, Weak scaling |
| [`implementation_2d/`](implementation_2d/README.md) | 2D CG with serial, MPI + OpenMP and MPI + CUDA backends. Validation, halo strategies, CG variants (classic, Chronopoulos–Gear, pipelined), hybrid ranks × threads, rank placement, reproducible reductions, A40 GPU runs on Atlas and Vanda | Validation, Communication choices and hybrid execution, GPU |

Each directory also contains the raw measurement CSVs used in the report. Both
sets of figures can be regenerated locally without cluster access:

```bash
(cd scaling_2d3d && python3 analysis/aggregate.py && python3 analysis/generate_figures.py)
(cd implementation_2d && python3 scripts/plot.py --label Vanda --out figures/vanda results/vanda/vanda_cpu_r2 results/vanda/vanda_gpu)
```

**Note on reproducibility.** Atlas and Vanda are shared clusters. Measured
times vary with which nodes PBS assigns, background load from other jobs, the
network and the allocation layout. A rerun can therefore differ from the
supplied data by roughly 10–20% per point, and by more on multi-node runs. The
scaling trends should still reproduce. See
`scaling_2d3d/README.md` → *Expected run-to-run variation on Vanda*.

Requirements: a C++17 compiler, MPI and OpenMP. CUDA is needed for the GPU
backend and CMake 3.18+ for `implementation_2d`. Plotting needs Python 3 with
matplotlib; `implementation_2d/scripts/plot.py` also needs numpy and pandas.
