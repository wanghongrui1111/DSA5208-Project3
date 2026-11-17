// mpi_main.cpp -- MPI + OpenMP driver: one configuration -> (repeated) distributed CG solves -> one CSV row.
// Stages S0, S2-S10; experiments V3, V4 and every performance run later on.
//
//   mpirun -n P ./poisson --dim 2|3 --n 1024 [--decomp 1d|2d|3d | --px a --py b --pz c]
//        [--halo block|blockseq|nonblock|overlap|thread|rma|allgather]  (blockseq = old chained order) [--cg classic|cgcg|pipelined]
//        [--problem bump|exp|sin] [--threads T]
//        [--tol 1e-8 --maxit 100000 | --iters K|auto]       (--iters => tol=0, exactly K iterations)
//        [--repeat 3] [--barrier-reduce 1] [--true-res 1]
//        [--check-serial 1 [--check-tol 1e-6] [--check-iters 2]]   serial CG on every rank, compare solutions
//        [--monitor 1]                                              per-rank table + invariant checks
//        [--history file.csv [--true-every K]]                      residual history (V4)
//        [--csv results.csv] [--tag name] [--quiet 1]
//
// Exit code 1 if --check-serial or --monitor finds a failure.
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "common.hpp"
#include "csv.hpp"
#include "serial.hpp"
#include "solver.hpp"

#define NV_STR2(x) #x
#define NV_STR(x) NV_STR2(x)

static int g_fail = 0;
static void check(bool ok, const char* what, int rank) {
  if (rank == 0) printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) g_fail++;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, P;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &P);

  Args args(argc, argv);
  Config cfg;
  cfg.kind = parse_problem(args.str("problem", "bump"));
  cfg.dim = args.integer("dim", 2);
  cfg.n = args.integer("n", 256);
  cfg.decomp = args.str("decomp", "");
  cfg.px = args.integer("px", 0);
  cfg.py = args.integer("py", 0);
  cfg.pz = args.integer("pz", 0);
  cfg.halo = args.str("halo", "block");
  cfg.cg = args.str("cg", "classic");
  cfg.tol = args.real("tol", 1e-8);
  cfg.maxit = args.integer("maxit", 100000);
  cfg.barrier_reduce = args.integer("barrier-reduce", 0) != 0;
  cfg.true_res = args.integer("true-res", 0) != 0;
  cfg.true_every = args.integer("true-every", 0);
  const std::string iters = args.str("iters", "");
  const int threads = args.integer("threads", 0);
  const int repeat = std::max(1, args.integer("repeat", 1));
  const bool monitor = args.integer("monitor", 0) != 0;
  const bool check_serial = args.integer("check-serial", 0) != 0;
  const double check_tol = args.real("check-tol", 1e-6);
  const int check_iters = args.integer("check-iters", 2);
  const std::string hist_file = args.str("history", "");
  const std::string csv = args.str("csv", "");
  const std::string tag = args.str("tag", "run");
  const bool quiet = args.integer("quiet", 0) != 0;
  cfg.history = !hist_file.empty();

  auto bad_arg = [&](const char* msg) {
    if (rank == 0) fprintf(stderr, "error: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 2);
  };
  if (rank == 0)
    for (auto& u : args.unused()) {
      fprintf(stderr, "unknown argument --%s\n", u.c_str());
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
  MPI_Barrier(MPI_COMM_WORLD);
  if (cfg.dim != 2 && cfg.dim != 3) bad_arg("--dim must be 2 or 3");
  const std::string& H = cfg.halo;
  if (H != "block" && H != "blockseq" && H != "nonblock" && H != "overlap" && H != "thread" && H != "rma" && H != "allgather")
    bad_arg("--halo must be block|blockseq|nonblock|overlap|thread|rma|allgather");
  if (cfg.cg != "classic" && cfg.cg != "cgcg" && cfg.cg != "pipelined") bad_arg("--cg must be classic|cgcg|pipelined");
  if (threads > 0) omp_set_num_threads(threads);
  if (!iters.empty()) {  // performance mode: tol = 0, fixed iteration count
    cfg.maxit = iters == "auto" ? auto_iters(cfg.dim, cfg.n) : std::atoi(iters.c_str());
    cfg.tol = 0;
  }
  if (cfg.maxit < 1) bad_arg("iteration count must be >= 1");

  // number of distinct nodes
  MPI_Comm node;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
  int lrank, nodes = 0, one;
  MPI_Comm_rank(node, &lrank);
  one = lrank == 0;
  MPI_Allreduce(&one, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Comm_free(&node);
  const int nthreads = omp_get_max_threads();
  int omp_team = 0;  // threads that really run in a parallel region (proves OpenMP is compiled in and honoured)
#pragma omp parallel
  {
#pragma omp single
    omp_team = omp_get_num_threads();
  }
#if defined(__NVCOMPILER)
  const char* compiler = "nvc++ " NV_STR(__NVCOMPILER_MAJOR__) "." NV_STR(__NVCOMPILER_MINOR__);
#elif defined(__clang__)
  const char* compiler = "clang " __VERSION__;
#elif defined(__GNUC__)
  const char* compiler = "gcc " __VERSION__;
#else
  const char* compiler = "unknown";
#endif
#ifdef _OPENMP
  const int omp_version = _OPENMP;
#else
  const int omp_version = 0;
#endif

  // repeated solves; report the median-time run (solution is identical every time)
  std::vector<Result> runs;
  for (int rep = 0; rep < repeat; ++rep) runs.push_back(solve(MPI_COMM_WORLD, cfg, check_serial && rep == repeat - 1));
  std::vector<int> order(repeat);
  for (int i = 0; i < repeat; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) { return runs[a].t_total < runs[b].t_total; });
  const Result& R = runs[order[repeat / 2]];
  const std::vector<double>& u_local = runs[repeat - 1].u_local;
  // all ranks must agree on which run is the median (max-over-ranks times are identical on every rank)

  const double h = 1.0 / (cfg.n + 1);
  const double per_iter_ms = R.iters ? 1e3 * R.t_total / R.iters : 0.0;
  if (rank == 0 && !quiet) {
    printf("poisson dim=%d n=%d problem=%s ranks=%d nodes=%d threads=%d grid=%dx%dx%d halo=%s cg=%s\n", cfg.dim, cfg.n,
           problem_name(cfg.kind), P, nodes, nthreads, R.px, R.py, R.pz, H.c_str(), cfg.cg.c_str());
    printf("  build: %s  _OPENMP=%d  omp_max_threads=%d  omp_team=%d\n", compiler, omp_version, nthreads, omp_team);
    printf("  iters=%d converged=%d relres=%.3e relres_true=%.3e err_inf=%.4e err_l2=%.4e\n", R.iters, (int)R.converged,
           R.relres, R.relres_true, R.err_inf, R.err_l2);
    printf("  time=%.4fs per_iter=%.4f ms  compute=%.4f halo=%.4f reduce=%.4f (wait %.4f pure %.4f)  repeat=%d\n",
           R.t_total, per_iter_ms, R.t_compute, R.t_halo, R.t_reduce, R.t_reduce_wait, R.t_reduce_pure, repeat);
    printf("  halo: %.0f B and %.0f msgs per matvec (max rank), %.0f B total over the run\n", R.halo_bytes_per_matvec,
           R.halo_msgs_per_matvec, R.halo_bytes_total);
  }

  // ------------------------------------------------------------------ monitor (S3, S7)
  if (monitor) {
    if (rank == 0) printf("--- monitor: per rank ---\n%4s %14s %14s\n", "rank", "local size", "global offset");
    for (int r = 0; r < P; ++r) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (r == rank) printf("%4d %4dx%4dx%-4d %4d,%4d,%-4d\n", rank, R.l[0], R.l[1], R.l[2], R.o[0], R.o[1], R.o[2]);
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("--- monitor: invariants ---\n");
    long long dof = (long long)R.l[0] * R.l[1] * R.l[2], gdof = 0;
    MPI_Allreduce(&dof, &gdof, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    check(gdof == (cfg.dim == 3 ? (long long)cfg.n * cfg.n * cfg.n : (long long)cfg.n * cfg.n), "sum of local dof == n^dim", rank);
    check(R.px * R.py * R.pz == P, "px*py*pz == number of ranks", rank);
    if (cfg.decomp == "1d") check(cfg.dim == 2 ? R.px == 1 : R.px == 1 && R.py == 1, "1d decomposition cuts only the slowest axis", rank);
    // S7: measured halo bytes (sum over ranks, whole run) == closed form  sum_a 2 (p_a-1) n^(dim-1) * 8 B * matvecs
    check(std::fabs(R.halo_bytes_total - R.halo_bytes_theory) < 0.5, "measured halo bytes == theoretical halo bytes", rank);
    check(R.halo_msgs_per_matvec <= 2 * cfg.dim, "<= 2*dim halo messages per matvec per rank", rank);
    const long long exp_red = cfg.cg == "classic" ? 2LL * R.iters : (long long)R.iters + 1;
    if (cfg.cg == "classic")
      check(R.n_reduce == exp_red, "classic CG: 2 reductions per iteration", rank);
    else if (cfg.cg == "cgcg")
      check(R.n_reduce == exp_red, "cgcg: 1 reduction per iteration (+1 initial)", rank);
    else
      check(R.n_reduce >= R.iters && R.n_reduce <= R.iters + 1, "pipelined: 1 reduction per iteration", rank);
    if (cfg.tol > 0) check(R.converged, "CG converged (relres < tol)", rank);
    if (cfg.tol > 0) check(R.err_inf < 50.0 * h * h, "max error vs exact solution is O(h^2)", rank);
  }

  // ------------------------------------------------------------------ serial comparison (V3)
  double ser_diff = -1;
  int ser_iters = -1;
  if (check_serial) {
    std::vector<double> us;
    SerialResult S = serial_cg(cfg.kind, cfg.dim, cfg.n, cfg.tol, cfg.maxit, &us);
    ser_iters = S.iters;
    const bool d3 = cfg.dim == 3;
    int ex, ey, ez, kb, gex, gey, gez, gkb;
    padded_extents(cfg.dim, R.l, ex, ey, ez, kb);
    const int gl[3] = {cfg.n, cfg.n, d3 ? cfg.n : 1};
    padded_extents(cfg.dim, gl, gex, gey, gez, gkb);
    double dmax = 0, umax = 0;
    for (int kk = 0; kk < R.l[2]; ++kk)
      for (int j = 1; j <= R.l[1]; ++j)
        for (int i = 1; i <= R.l[0]; ++i) {
          double a = u_local[((size_t)(kb + kk) * ey + j) * ex + i];
          double b = us[((size_t)(gkb + R.o[2] + kk) * gey + R.o[1] + j) * gex + R.o[0] + i];
          dmax = std::fmax(dmax, std::fabs(a - b));
          umax = std::fmax(umax, std::fabs(b));
        }
    MPI_Allreduce(MPI_IN_PLACE, &dmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    ser_diff = umax > 0 ? dmax / umax : dmax;
    if (rank == 0)
      printf("--- serial comparison: iters serial=%d parallel=%d  max|u_par-u_ser|/max|u| = %.3e  (err_inf serial %.4e par %.4e)\n",
             S.iters, R.iters, ser_diff, S.err_inf, R.err_inf);
    check(ser_diff <= check_tol, "parallel solution == serial solution within --check-tol", rank);
    if (cfg.cg != "pipelined")
      check(std::abs(S.iters - R.iters) <= check_iters, "iteration count matches serial (within --check-iters)", rank);
    check(R.converged || cfg.tol <= 0, "parallel CG converged", rank);
  }

  // ------------------------------------------------------------------ residual history (V4)
  if (cfg.history && rank == 0) {
    FILE* f = fopen(hist_file.c_str(), "w");
    if (!f) {
      fprintf(stderr, "cannot open %s\n", hist_file.c_str());
    } else {
      fprintf(f, "iter,relres_rec,relres_true\n");
      for (size_t k = 0; k < R.hist_it.size(); ++k)
        fprintf(f, "%d,%.9e,%.9e\n", (int)R.hist_it[k], R.hist_rec[k], R.hist_true[k]);
      fclose(f);
    }
  }

  // ------------------------------------------------------------------ CSV
  if (rank == 0 && !csv.empty()) {
    Rec rc;
    rc.dim = cfg.dim;
    rc.n = cfg.n;
    rc.N = cfg.dim == 3 ? (long long)cfg.n * cfg.n * cfg.n : (long long)cfg.n * cfg.n;
    rc.nnz = nnz_of(cfg.dim, cfg.n);
    rc.ranks = P;
    rc.nodes = nodes;
    rc.threads = nthreads;
    rc.cores = P * nthreads;
    rc.px = R.px;
    rc.py = R.py;
    rc.pz = R.pz;
    rc.decomp = cfg.decomp.empty() ? (cfg.dim == 2 ? "2d" : "3d") : cfg.decomp;
    if (cfg.px > 0 || cfg.py > 0 || cfg.pz > 0) rc.decomp = "explicit";
    rc.halo = H;
    rc.cg = cfg.cg;
    rc.tol = cfg.tol;
    rc.maxit = cfg.maxit;
    rc.repeat = repeat;
    rc.iters = R.iters;
    rc.time_s = R.t_total;
    rc.time_per_iter_ms = per_iter_ms;
    rc.t_halo = R.t_halo;
    rc.t_reduce = R.t_reduce;
    rc.t_reduce_wait = R.t_reduce_wait;
    rc.t_reduce_pure = R.t_reduce_pure;
    rc.halo_bytes = R.halo_bytes_per_matvec;
    rc.halo_msgs = R.halo_msgs_per_matvec;
    rc.relres = R.relres;
    rc.err_inf = R.err_inf;
    rc.err_l2 = R.err_l2;
    rc.tag = tag;
    rc.t_compute = R.t_compute;
    rc.problem = problem_name(cfg.kind);
    rc.converged = R.converged;
    rc.relres_true = R.relres_true;
    rc.halo_bytes_total = R.halo_bytes_total;
    rc.ser_iters = ser_iters;
    rc.ser_diff = ser_diff;
    csv_append(csv, rc);
  }

  int fail = g_fail;
  MPI_Finalize();
  return fail ? 1 : 0;
}
