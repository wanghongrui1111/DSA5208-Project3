// csv.hpp -- one CSV row per run (Experiment_Plan section 0 field list, extras appended at the end).
#pragma once
#include <cstdio>
#include <string>

struct Rec {
  int dim = 2, n = 0;
  long long N = 0, nnz = 0;
  int ranks = 1, nodes = 1, threads = 1, cores = 1, px = 1, py = 1, pz = 1;
  std::string decomp = "serial", halo = "none", cg = "classic";
  double tol = 0;
  int maxit = 0, repeat = 1, iters = 0;
  double time_s = 0, time_per_iter_ms = 0;
  double t_halo = 0, t_reduce = 0, t_reduce_wait = 0, t_reduce_pure = 0;
  double halo_bytes = 0, halo_msgs = 0;  // per rank (max over ranks), per matvec
  double relres = 0, err_inf = 0, err_l2 = 0;
  std::string tag;
  // extras
  double t_compute = 0, relres_true = -1, halo_bytes_total = 0, ser_diff = -1;
  std::string problem;
  std::string rev = "2";  // timing revision: 2 = pre-loop ||b|| reduction excluded from t_reduce* (CSVs without this column: rev 1)
  int converged = 0, ser_iters = -1;
};

inline void csv_append(const std::string& path, const Rec& r) {
  if (path.empty()) return;
  FILE* f = std::fopen(path.c_str(), "a");
  if (!f) {
    std::fprintf(stderr, "cannot open csv %s\n", path.c_str());
    return;
  }
  std::fseek(f, 0, SEEK_END);
  if (std::ftell(f) == 0)
    std::fprintf(f,
                 "dim,n,N,nnz,ranks,nodes,threads,cores,px,py,pz,decomp,halo,cg,tol,maxit,repeat,iters,time_s,"
                 "time_per_iter_ms,t_halo,t_reduce,t_reduce_wait,t_reduce_pure,halo_bytes,halo_msgs,relres,err_inf,"
                 "err_l2,tag,t_compute,problem,converged,relres_true,halo_bytes_total,ser_iters,ser_diff,solver_rev\n");
  std::fprintf(f, "%d,%d,%lld,%lld,%d,%d,%d,%d,%d,%d,%d,%s,%s,%s,%g,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.0f,%.0f,%.6e,%.6e,%.6e,%s,%.9g,%s,%d,%.6e,%.0f,%d,%.6e,%s\n",
               r.dim, r.n, r.N, r.nnz, r.ranks, r.nodes, r.threads, r.cores, r.px, r.py, r.pz, r.decomp.c_str(),
               r.halo.c_str(), r.cg.c_str(), r.tol, r.maxit, r.repeat, r.iters, r.time_s, r.time_per_iter_ms, r.t_halo,
               r.t_reduce, r.t_reduce_wait, r.t_reduce_pure, r.halo_bytes, r.halo_msgs, r.relres, r.err_inf, r.err_l2,
               r.tag.c_str(), r.t_compute, r.problem.c_str(), r.converged, r.relres_true, r.halo_bytes_total,
               r.ser_iters, r.ser_diff, r.rev.c_str());
  std::fclose(f);
}

inline long long nnz_of(int dim, int n) {
  long long m = n;
  return dim == 3 ? 7 * m * m * m - 6 * m * m : 5 * m * m - 4 * m;
}
