#pragma once
// MPI process grid shared by the MPI+OpenMP and MPI+CUDA solvers.

#include <mpi.h>

#include "cg_driver.hpp"

namespace cg {

struct ProcGrid {
  MPI_Comm comm = MPI_COMM_NULL;  // Cartesian communicator, no periodicity
  int rank = 0, size = 1;
  int dims[2] = {1, 1};  // processes along x (rows) and y (columns)
  int coords[2] = {0, 0};
  int north = MPI_PROC_NULL, south = MPI_PROC_NULL;  // neighbours at i - 1 and i + 1
  int west = MPI_PROC_NULL, east = MPI_PROC_NULL;    // neighbours at j - 1 and j + 1
  int local_rank = 0, local_size = 1;                // position within the node
  int nodes = 1;
};

// Message tags name the direction a halo strip travels in.
enum HaloTag { kToNorth = 0, kToSouth = 1, kToWest = 2, kToEast = 3 };

// 1d: a column of P processes, each owning a strip of full rows.
// 2d: the near-square grid from MPI_Dims_create, each owning a block.
inline ProcGrid make_proc_grid(Decomp d) {
  ProcGrid g;
  MPI_Comm_size(MPI_COMM_WORLD, &g.size);
  if (d == Decomp::TwoD) {
    g.dims[0] = g.dims[1] = 0;
    MPI_Dims_create(g.size, 2, g.dims);
  } else {
    g.dims[0] = g.size;
    g.dims[1] = 1;
  }
  const int periods[2] = {0, 0};
  MPI_Cart_create(MPI_COMM_WORLD, 2, g.dims, periods, 0, &g.comm);
  MPI_Comm_rank(g.comm, &g.rank);
  MPI_Cart_coords(g.comm, g.rank, 2, g.coords);
  MPI_Cart_shift(g.comm, 0, 1, &g.north, &g.south);
  MPI_Cart_shift(g.comm, 1, 1, &g.west, &g.east);

  MPI_Comm node;
  MPI_Comm_split_type(g.comm, MPI_COMM_TYPE_SHARED, g.rank, MPI_INFO_NULL, &node);
  MPI_Comm_rank(node, &g.local_rank);
  MPI_Comm_size(node, &g.local_size);
  MPI_Comm_free(&node);
  const int leader = g.local_rank == 0 ? 1 : 0;
  MPI_Allreduce(&leader, &g.nodes, 1, MPI_INT, MPI_SUM, g.comm);
  return g;
}

inline Block make_block(const ProcGrid& g, long n) {
  Block b;
  b.n = n;
  partition(n, g.dims[0], g.coords[0], b.i0, b.nx);
  partition(n, g.dims[1], g.coords[1], b.j0, b.ny);
  return b;
}

inline void fill_run_info(RunInfo& info, const ProcGrid& g, const Options& o) {
  info.ranks = g.size;
  info.nodes = g.nodes;
  info.px = g.dims[0];
  info.py = g.dims[1];
  info.decomp = to_string(o.decomp);
  info.comm = to_string(o.comm);
  info.reduce = to_string(o.reduce);
}

// Global sums of the CG scalars, timed into Timers::reduce.
//   fast:  MPI_Allreduce / MPI_Iallreduce; the library chooses how partial
//          sums are combined, and the standard does not promise the same
//          order in every run.
//   repro: every rank gathers all partial sums and adds them in rank order,
//          so all ranks and all runs get bit-identical results.
class Reducer {
 public:
  Reducer(MPI_Comm comm, int size, Reduce mode, Timers* tm)
      : comm_(comm), size_(size), repro_(mode == Reduce::Repro), tm_(tm) {}

  void sum(double* v, int n) {
    const double t = now();
    if (repro_) {
      stage(v, n);
      MPI_Allgather(send_.data(), n, MPI_DOUBLE, recv_.data(), n, MPI_DOUBLE, comm_);
      add_in_rank_order(v, n);
    } else {
      MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, comm_);
    }
    tm_->reduce += now() - t;
  }

  void max(double* v, int n) { MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_MAX, comm_); }

  // Non-blocking sum; v must stay valid until sum_end().
  void sum_begin(double* v, int n) {
    const double t = now();
    pending_ = v;
    pending_n_ = n;
    if (repro_) {
      stage(v, n);
      MPI_Iallgather(send_.data(), n, MPI_DOUBLE, recv_.data(), n, MPI_DOUBLE, comm_, &req_);
    } else {
      MPI_Iallreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, comm_, &req_);
    }
    tm_->reduce += now() - t;
  }

  void sum_end() {
    const double t = now();
    MPI_Wait(&req_, MPI_STATUS_IGNORE);
    if (repro_) add_in_rank_order(pending_, pending_n_);
    tm_->reduce += now() - t;
  }

 private:
  void stage(const double* v, int n) {
    send_.assign(v, v + n);
    recv_.resize(static_cast<size_t>(n) * size_);
  }
  void add_in_rank_order(double* v, int n) const {
    for (int c = 0; c < n; ++c) {
      double s = 0.0;
      for (int r = 0; r < size_; ++r) s += recv_[static_cast<size_t>(r) * n + c];
      v[c] = s;
    }
  }

  MPI_Comm comm_;
  int size_;
  bool repro_;
  Timers* tm_;
  std::vector<double> send_, recv_;
  double* pending_ = nullptr;
  int pending_n_ = 0;
  MPI_Request req_ = MPI_REQUEST_NULL;
};

// Prints on rank 0 and aborts all ranks.
[[noreturn]] inline void fail(const std::string& msg) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) std::fprintf(stderr, "error: %s\n", msg.c_str());
  MPI_Abort(MPI_COMM_WORLD, 1);
  std::abort();
}

// Parses the options on every rank; help or a bad option ends the program.
inline Options parse_or_exit(int argc, char** argv) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  try {
    return parse_options(argc, argv);
  } catch (const HelpRequested&) {
    if (rank == 0) std::printf("usage: %s [options]\n%s", argv[0], usage_text());
    MPI_Finalize();
    std::exit(0);
  } catch (const std::exception& e) {
    if (rank == 0) std::fprintf(stderr, "error: %s\n%s", e.what(), usage_text());
    MPI_Finalize();
    std::exit(1);
  }
}

inline void check_grid(const ProcGrid& g, long n) {
  if (n < g.dims[0] || n < g.dims[1])
    fail("grid too small: n = " + std::to_string(n) + " for a " + std::to_string(g.dims[0]) + " x " +
         std::to_string(g.dims[1]) + " process grid");
}

}  // namespace cg
