#pragma once
// Framework-free pieces shared by all CG Poisson solvers: the model problem,
// command-line options, local block geometry, timing and result output.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cg {

constexpr double kPi = 3.14159265358979323846;

// Model problem: -Laplace(u) = f on (0,1)^2 with u = 0 on the boundary, with
// the manufactured solution u = g(x) h(y), g = e^x sin(pi x), h = e^2y sin(2 pi y).
// The exponential factors matter: a pure sine product is an eigenvector of the
// discrete Laplacian, and CG would converge in a single iteration. u is also
// not symmetric in x and y, so an x/y mix-up in the decomposition shows up.
inline double exact_solution(double x, double y) {
  return std::exp(x + 2.0 * y) * std::sin(kPi * x) * std::sin(2.0 * kPi * y);
}
inline double source_term(double x, double y) {
  const double sx = std::sin(kPi * x), cx = std::cos(kPi * x);
  const double sy = std::sin(2.0 * kPi * y), cy = std::cos(2.0 * kPi * y);
  const double g = std::exp(x) * sx, gxx = std::exp(x) * ((1.0 - kPi * kPi) * sx + 2.0 * kPi * cx);
  const double h = std::exp(2.0 * y) * sy, hyy = std::exp(2.0 * y) * ((4.0 - 4.0 * kPi * kPi) * sy + 8.0 * kPi * cy);
  return -(gxx * h + g * hyy);
}

enum class Variant { Classic, ChronopoulosGear, Pipelined };
enum class Decomp { OneD, TwoD };
// How the halo of the vector in a matrix-vector product reaches a rank:
//   blocking   MPI_Sendrecv with the four neighbours, then the stencil
//   overlap    Isend/Irecv, interior computed while the messages travel
//   thread     like overlap, but the master thread communicates while the
//              other OpenMP threads compute the interior (one parallel region)
//   rma        MPI_Put into the neighbours' ghost layers, MPI_Win_fence epochs
//   allgather  every rank gathers the whole vector (MPI_Allgatherv), the
//              "full copy on every rank" design; needs the 1d decomposition
enum class Comm { Blocking, Overlap, Thread, Rma, Allgather };
enum class GpuMpi { Staged, Aware };
// fast: MPI_Allreduce and OpenMP reductions, whose summation order may change
// with the thread count and from run to run. repro: fixed summation order, so
// results are bit-identical between runs and thread counts.
enum class Reduce { Fast, Repro };

inline const char* to_string(Variant v) {
  switch (v) {
    case Variant::Classic: return "classic";
    case Variant::ChronopoulosGear: return "cgcg";
    default: return "pipe";
  }
}
inline const char* to_string(Decomp d) { return d == Decomp::OneD ? "1d" : "2d"; }
inline const char* to_string(Comm c) {
  switch (c) {
    case Comm::Blocking: return "blocking";
    case Comm::Overlap: return "overlap";
    case Comm::Thread: return "thread";
    case Comm::Rma: return "rma";
    default: return "allgather";
  }
}
inline const char* to_string(GpuMpi g) { return g == GpuMpi::Staged ? "staged" : "aware"; }
inline const char* to_string(Reduce r) { return r == Reduce::Fast ? "fast" : "repro"; }

struct Options {
  long n = 1024;      // interior grid points per dimension (global grid is n x n)
  int maxit = 20000;  // iteration cap
  double tol = 1e-8;  // stop when ||r|| / ||b|| < tol; 0 runs exactly maxit iterations
  Variant cg = Variant::Classic;
  Decomp decomp = Decomp::TwoD;
  Comm comm = Comm::Blocking;
  GpuMpi gpu_mpi = GpuMpi::Staged;
  Reduce reduce = Reduce::Fast;
  int repeat = 1;       // timed solves; the median time is reported
  int chunks = 4;       // interior chunks between MPI progress polls (CPU overlap mode)
  std::string csv;      // append a result row to this CSV file
  std::string history;  // write the residual history of the last solve here
  std::string tag;      // free-form label copied into the CSV row
};

struct HelpRequested {};

inline const char* usage_text() {
  return "Options:\n"
         "  --n N           interior grid points per dimension (default 1024)\n"
         "  --maxit K       maximum CG iterations (default 20000)\n"
         "  --tol T         relative residual tolerance; 0 runs exactly maxit iterations (default 1e-8)\n"
         "  --cg V          classic | cgcg | pipe (default classic)\n"
         "  --decomp D      1d | 2d process grid (MPI builds, default 2d)\n"
         "  --comm C        blocking | overlap | thread | rma | allgather (MPI builds, default blocking;\n"
         "                  the GPU build supports blocking and overlap)\n"
         "  --gpu-mpi G     staged | aware: host-staged or CUDA-aware MPI (CUDA build, default staged)\n"
         "  --reduce R      fast | repro: library reductions or fixed-order, reproducible sums (default fast)\n"
         "  --chunks K      interior chunks between MPI_Testall polls, CPU overlap mode (default 4)\n"
         "  --repeat R      timed solves, the median is reported (default 1)\n"
         "  --csv FILE      append a result row to FILE\n"
         "  --history FILE  write the residual history (iteration, ||r||/||b||) to FILE\n"
         "  --tag STR       label stored in the CSV row\n";
}

inline Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      throw HelpRequested{};
    } else if (a == "--n") {
      o.n = std::stol(value());
    } else if (a == "--maxit") {
      o.maxit = std::stoi(value());
    } else if (a == "--tol") {
      o.tol = std::stod(value());
    } else if (a == "--cg") {
      const std::string v = value();
      if (v == "classic") o.cg = Variant::Classic;
      else if (v == "cgcg") o.cg = Variant::ChronopoulosGear;
      else if (v == "pipe") o.cg = Variant::Pipelined;
      else throw std::runtime_error("unknown --cg " + v);
    } else if (a == "--decomp") {
      const std::string v = value();
      if (v == "1d") o.decomp = Decomp::OneD;
      else if (v == "2d") o.decomp = Decomp::TwoD;
      else throw std::runtime_error("unknown --decomp " + v);
    } else if (a == "--comm") {
      const std::string v = value();
      if (v == "blocking") o.comm = Comm::Blocking;
      else if (v == "overlap") o.comm = Comm::Overlap;
      else if (v == "thread") o.comm = Comm::Thread;
      else if (v == "rma") o.comm = Comm::Rma;
      else if (v == "allgather") o.comm = Comm::Allgather;
      else throw std::runtime_error("unknown --comm " + v);
    } else if (a == "--gpu-mpi") {
      const std::string v = value();
      if (v == "staged") o.gpu_mpi = GpuMpi::Staged;
      else if (v == "aware") o.gpu_mpi = GpuMpi::Aware;
      else throw std::runtime_error("unknown --gpu-mpi " + v);
    } else if (a == "--reduce") {
      const std::string v = value();
      if (v == "fast") o.reduce = Reduce::Fast;
      else if (v == "repro") o.reduce = Reduce::Repro;
      else throw std::runtime_error("unknown --reduce " + v);
    } else if (a == "--chunks") {
      o.chunks = std::stoi(value());
    } else if (a == "--repeat") {
      o.repeat = std::stoi(value());
    } else if (a == "--csv") {
      o.csv = value();
    } else if (a == "--history") {
      o.history = value();
    } else if (a == "--tag") {
      o.tag = value();
    } else {
      throw std::runtime_error("unknown option " + a);
    }
  }
  if (o.n < 1 || o.maxit < 1 || o.repeat < 1 || o.chunks < 1 || o.tol < 0)
    throw std::runtime_error("option values out of range");
  return o;
}

// Contiguous partition of n items over p parts; part r gets [start, start + count).
inline void partition(long n, int p, int r, long& start, long& count) {
  const long base = n / p, extra = n % p;
  count = base + (r < extra ? 1 : 0);
  start = r * base + std::min<long>(r, extra);
}

// The part of the global n x n interior grid owned by one process, stored
// row-major with one ghost layer: local point (i, j), 0 <= i < nx, 0 <= j < ny,
// lives at index (i + 1) * ld() + (j + 1). The row index i runs along x.
struct Block {
  long n = 0;         // global interior points per dimension
  long nx = 0, ny = 0;
  long i0 = 0, j0 = 0;  // global index of local point (0, 0)
  long ld() const { return ny + 2; }
  long size() const { return (nx + 2) * (ny + 2); }
  double h() const { return 1.0 / static_cast<double>(n + 1); }
};

// Half-open rectangle [i0, i1) x [j0, j1) of local interior points.
struct Rect {
  long i0, i1, j0, j1;
  bool empty() const { return i1 <= i0 || j1 <= j0; }
};

inline Rect full_rect(const Block& b) { return {0, b.nx, 0, b.ny}; }

// Points whose stencil does not read the ghost layer.
inline Rect interior_rect(const Block& b) { return {1, b.nx - 1, 1, b.ny - 1}; }

// The remaining one-point-wide frame as at most four disjoint rectangles.
inline int frame_rects(const Block& b, Rect out[4]) {
  int k = 0;
  out[k++] = {0, 1, 0, b.ny};
  if (b.nx > 1) out[k++] = {b.nx - 1, b.nx, 0, b.ny};
  if (b.nx > 2) {
    out[k++] = {1, b.nx - 1, 0, 1};
    if (b.ny > 1) out[k++] = {1, b.nx - 1, b.ny - 1, b.ny};
  }
  return k;
}

inline double now() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Wall time spent inside halo exchanges and global reductions during a solve.
struct Timers {
  double halo = 0.0, reduce = 0.0;
};

// Local contributions to the final accuracy check.
struct LocalStats {
  double res2 = 0.0, err2 = 0.0, err_max = 0.0, b2 = 0.0;
};

using Row = std::vector<std::pair<std::string, std::string>>;

template <class T>
std::string str(const T& v) {
  std::ostringstream s;
  s.precision(10);
  s << v;
  return s.str();
}
inline std::string str(const std::string& v) {
  std::string s = v;
  std::replace(s.begin(), s.end(), ',', ';');
  return s.empty() ? "-" : s;
}
inline std::string str(const char* v) { return str(std::string(v)); }
// Exact representation of a double, to check bit-for-bit reproducibility.
inline std::string hexfloat(double v) {
  std::ostringstream s;
  s << std::hexfloat << v;
  return s.str();
}

inline void print_row(const Row& row) {
  for (const auto& kv : row) std::printf("%-14s %s\n", kv.first.c_str(), kv.second.c_str());
  std::fflush(stdout);
}

inline void append_csv(const std::string& path, const Row& row) {
  bool header = true;
  {
    std::ifstream in(path);
    header = !in.good() || in.peek() == std::ifstream::traits_type::eof();
  }
  std::ofstream out(path, std::ios::app);
  if (!out) {
    std::fprintf(stderr, "cannot open %s for writing\n", path.c_str());
    return;
  }
  if (header) {
    for (size_t k = 0; k < row.size(); ++k) out << (k ? "," : "") << row[k].first;
    out << "\n";
  }
  for (size_t k = 0; k < row.size(); ++k) out << (k ? "," : "") << row[k].second;
  out << "\n";
}

}  // namespace cg
