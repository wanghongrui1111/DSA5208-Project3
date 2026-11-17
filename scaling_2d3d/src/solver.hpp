// solver.hpp -- distributed (MPI + OpenMP) matrix-free CG for the 2D / 3D Poisson problem.
//
// Parallel design
//   * Cartesian block decomposition on a px x py x pz process grid (MPI_Cart_create, x varies fastest).
//     decomp = 1d (slowest axis only: y in 2D, z in 3D) | 2d (2D: px x py;  3D: y,z) | 3d (all axes).
//   * Matrix-free 5/7-point stencil; each rank owns an lx x ly x lz block plus a 1-cell halo (padded layout, problem.hpp).
//   * Halo faces are packed into contiguous buffers (no derived datatypes): the message for face f = 2*axis+side.
//   * halo modes:
//       block     paired two-phase MPI_Sendrecv per axis (even/odd coordinate pairing, no waiting chain), then the whole stencil
//       blockseq  the original order lo,hi,.. (rank k waits for rank k-1: chain of length P in 1D) -- kept for comparison
//       nonblock  Isend/Irecv for all faces + Waitall, then the whole stencil
//       overlap   Isend/Irecv, interior stencil while messages are in flight, Waitall, boundary shell
//       thread    like overlap, but only the OpenMP master thread communicates; the other threads compute the interior
//       rma       MPI_Put into the neighbours' receive buffers between two MPI_Win_fence (no overlap)
//       allgather 1D only: every rank MPI_Allgather's its two boundary planes (deliberately wasteful baseline)
//   * CG variants:
//       classic    2 MPI_Allreduce per iteration  (p.Ap, r.r)
//       cgcg       Chronopoulos-Gear: 1 Allreduce (2 values) per iteration
//       pipelined  Ghysels-Vanroose: 1 MPI_Iallreduce per iteration, hidden behind the matvec
//   * All local kernels are OpenMP (schedule(static), simd), first-touch initialised.
#pragma once
#include <mpi.h>

#include <string>
#include <vector>

#include "problem.hpp"

struct Config {
  ProblemKind kind = PROB_BUMP;
  int dim = 2, n = 256;
  std::string decomp = "";  // "" => 2d for dim 2, 3d for dim 3
  int px = 0, py = 0, pz = 0;  // 0 => automatic (explicit values override decomp)
  std::string halo = "block", cg = "classic";
  double tol = 1e-8;
  int maxit = 100000;
  bool barrier_reduce = false;  // MPI_Barrier before each blocking Allreduce: split wait vs pure reduction time
  bool true_res = false;        // compute ||b-Ax||/||b|| after the solve (not timed)
  int true_every = 0;           // >0: also every K iterations (history only)
  bool history = false;         // record relres every iteration (rank 0 gets the table)
};

struct Result {
  int dim = 2, n = 0, px = 1, py = 1, pz = 1;
  int l[3] = {0, 0, 0}, o[3] = {0, 0, 0};  // local interior size / global 0-based offset of this rank
  int iters = 0;
  bool converged = false;
  double relres = 0, relres_true = -1;
  double err_inf = 0, err_l2 = 0;
  // max over ranks [s]
  double t_total = 0, t_compute = 0, t_halo = 0, t_reduce = 0, t_reduce_wait = 0, t_reduce_pure = 0;
  long long n_matvec = 0, n_reduce = 0;
  double halo_bytes_per_matvec = 0, halo_msgs_per_matvec = 0;  // max over ranks
  double halo_bytes_total = 0;                                 // sum over ranks, whole run
  double halo_bytes_theory = 0;                                // closed form, whole run
  std::vector<double> hist_it, hist_rec, hist_true;            // rank 0 only, if cfg.history
  std::vector<double> u_local;                                 // padded local solution, if keep_solution
};

// Collective over `comm`.  Returns the same global quantities on every rank.
Result solve(MPI_Comm comm, const Config& cfg, bool keep_solution);

// Process-grid selection shared with the driver (collective-free).
void choose_grid(int P, int dim, const std::string& decomp, int& px, int& py, int& pz);
