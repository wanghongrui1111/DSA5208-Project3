#pragma once
// The three CG variants, written once against a backend interface. The serial,
// MPI+OpenMP and MPI+CUDA programs differ only in the backend they plug in, so
// every implementation executes exactly the same arithmetic.
//
// Backend interface (all vector operations act on the local block only):
//   void   reset()                      x = 0, r = b, every other vector = 0
//   double dot(Vec a, Vec b)            local partial dot product
//   void   matvec(Vec dst, Vec src, double* sd, double* ss)
//                                       halo exchange of src, then dst = A src;
//                                       non-null sd/ss receive the local
//                                       (src, dst) and (src, src)
//   double classic_xr(double alpha)     x += alpha p, r -= alpha q; local (r, r)
//   void   classic_p(double beta)       p = r + beta p
//   void   cgcg_update(double alpha, double beta)
//   void   pipe_update(double alpha, double beta, double& rr, double& wr)
//   void   allreduce(double* v, int n)  global sums in place
//   void   allreduce_max(double* v, int n)
//   void   iallreduce_begin(double* v, int n), iallreduce_end()
//   void   barrier(), sync()            sync() waits for outstanding device work
//   LocalStats final_stats()            true residual and error of the current x
//   Timers& timers(); int rank(); int nranks();

#include "common.hpp"

namespace cg {

enum Vec { X, B, R, P, S, W, Z, Q, kNumVec };

struct SolveResult {
  int iters = 0;
  double relres = 0.0;  // ||r|| / ||b|| from the recurrence
  double time = 0.0, t_halo = 0.0, t_reduce = 0.0;
  std::vector<double> history;  // history[k - 1] = ||r_k|| / ||b||
};

// Textbook CG: one stencil sweep, two blocking reductions per iteration.
template <class BK>
void classic_cg(BK& bk, const Options& o, SolveResult& res) {
  double rr = bk.dot(R, R);
  bk.allreduce(&rr, 1);
  const double bnorm = std::sqrt(rr);  // x0 = 0, so r0 = b
  bk.classic_p(0.0);                   // p = r
  for (int it = 0; it < o.maxit && rr > 0.0; ++it) {
    double pq = 0.0;
    bk.matvec(Q, P, &pq, nullptr);
    bk.allreduce(&pq, 1);
    const double alpha = rr / pq;
    double rr_new = bk.classic_xr(alpha);
    bk.allreduce(&rr_new, 1);
    res.iters = it + 1;
    res.relres = std::sqrt(rr_new) / bnorm;
    res.history.push_back(res.relres);
    if (res.relres < o.tol) break;
    bk.classic_p(rr_new / rr);
    rr = rr_new;
  }
}

// Chronopoulos-Gear CG: w = A r is carried along so that (r, r) and (w, r)
// can be reduced together, one blocking reduction per iteration.
template <class BK>
void chronopoulos_gear_cg(BK& bk, const Options& o, SolveResult& res) {
  double v[2] = {0.0, 0.0};  // {(r, r), (w, r)}
  bk.matvec(W, R, &v[1], &v[0]);
  bk.allreduce(v, 2);
  const double bnorm = std::sqrt(v[0]);
  double gamma = v[0], alpha = v[0] / v[1], beta = 0.0;
  for (int it = 0; it < o.maxit && gamma > 0.0; ++it) {
    bk.cgcg_update(alpha, beta);
    bk.matvec(W, R, &v[1], &v[0]);
    bk.allreduce(v, 2);
    res.iters = it + 1;
    res.relres = std::sqrt(v[0]) / bnorm;
    res.history.push_back(res.relres);
    if (res.relres < o.tol) break;
    beta = v[0] / gamma;
    alpha = v[0] / (v[1] - beta * v[0] / alpha);
    gamma = v[0];
  }
}

// Pipelined CG (Ghysels & Vanroose 2014): the single reduction per iteration
// is non-blocking and overlaps with the halo exchange and stencil q = A w.
template <class BK>
void pipelined_cg(BK& bk, const Options& o, SolveResult& res) {
  double loc[2] = {0.0, 0.0};  // local {(r, r), (w, r)} of the current iterate
  bk.matvec(W, R, &loc[1], &loc[0]);
  double bnorm = 0.0, gamma_old = 0.0, alpha = 0.0;
  for (int it = 0;; ++it) {
    double v[2] = {loc[0], loc[1]};
    bk.iallreduce_begin(v, 2);
    if (it < o.maxit) bk.matvec(Q, W, nullptr, nullptr);
    bk.iallreduce_end();
    const double gamma = v[0], delta = v[1];
    if (it == 0) {
      bnorm = std::sqrt(gamma);
    } else {
      res.iters = it;
      res.relres = std::sqrt(gamma) / bnorm;
      res.history.push_back(res.relres);
    }
    if (it == o.maxit || !(gamma > 0.0) || (it > 0 && res.relres < o.tol)) break;
    const double beta = it == 0 ? 0.0 : gamma / gamma_old;
    alpha = it == 0 ? gamma / delta : gamma / (delta - beta * gamma / alpha);
    bk.pipe_update(alpha, beta, loc[0], loc[1]);
    gamma_old = gamma;
  }
}

template <class BK>
SolveResult solve(BK& bk, const Options& o) {
  bk.reset();
  bk.timers() = Timers{};
  bk.barrier();
  SolveResult res;
  const double t0 = now();
  switch (o.cg) {
    case Variant::Classic: classic_cg(bk, o, res); break;
    case Variant::ChronopoulosGear: chronopoulos_gear_cg(bk, o, res); break;
    case Variant::Pipelined: pipelined_cg(bk, o, res); break;
  }
  bk.sync();
  res.time = now() - t0;
  res.t_halo = bk.timers().halo;
  res.t_reduce = bk.timers().reduce;
  return res;
}

// Describes how a run was parallelised; copied into the result row.
struct RunInfo {
  std::string impl;
  int ranks = 1, nodes = 1, threads = 1, px = 1, py = 1, gpus = 0;
  std::string decomp = "-", comm = "-", gpu_mpi = "-", reduce = "-", device = "-";
};

// Runs o.repeat timed solves, checks the last solution and reports on rank 0.
template <class BK>
int run(BK& bk, const Options& o, const RunInfo& info) {
  std::vector<double> times;
  double halo = 0.0, reduce = 0.0;
  SolveResult res;
  for (int rep = 0; rep < o.repeat; ++rep) {
    res = solve(bk, o);
    times.push_back(res.time);
    halo += res.t_halo / o.repeat;
    reduce += res.t_reduce / o.repeat;
  }
  const LocalStats st = bk.final_stats();
  double sums[5] = {st.res2, st.err2, st.b2, halo, reduce};
  double maxs[3] = {st.err_max, halo, reduce};
  bk.allreduce(sums, 5);
  bk.allreduce_max(maxs, 3);
  if (bk.rank() != 0) return 0;

  std::sort(times.begin(), times.end());
  const size_t m = times.size();
  const double t_med = m % 2 ? times[m / 2] : 0.5 * (times[m / 2 - 1] + times[m / 2]);
  const double h = 1.0 / static_cast<double>(o.n + 1);
  const double points = static_cast<double>(o.n) * static_cast<double>(o.n);
  const int its = std::max(res.iters, 1);

  Row row = {{"impl", str(info.impl)},
             {"n", str(o.n)},
             {"ranks", str(info.ranks)},
             {"nodes", str(info.nodes)},
             {"threads", str(info.threads)},
             {"cores", str(info.ranks * info.threads)},
             {"gpus", str(info.gpus)},
             {"px", str(info.px)},
             {"py", str(info.py)},
             {"decomp", str(info.decomp)},
             {"comm", str(info.comm)},
             {"gpu_mpi", str(info.gpu_mpi)},
             {"reduce", str(info.reduce)},
             {"device", str(info.device)},
             {"cg", str(to_string(o.cg))},
             {"tol", str(o.tol)},
             {"maxit", str(o.maxit)},
             {"repeat", str(o.repeat)},
             {"tag", str(o.tag)},
             {"iters", str(res.iters)},
             {"time_s", str(t_med)},
             {"time_min_s", str(times.front())},
             {"time_per_iter_ms", str(1e3 * t_med / its)},
             {"t_halo_avg_s", str(sums[3] / bk.nranks())},
             {"t_halo_max_s", str(maxs[1])},
             {"t_reduce_avg_s", str(sums[4] / bk.nranks())},
             {"t_reduce_max_s", str(maxs[2])},
             {"mlups", str(points * its / t_med / 1e6)},
             {"relres", str(res.relres)},
             {"relres_bits", hexfloat(res.relres)},
             {"true_relres", str(std::sqrt(sums[0] / sums[2]))},
             {"err_inf", str(maxs[0])},
             {"err_l2", str(std::sqrt(h * h * sums[1]))}};
  print_row(row);
  if (!o.csv.empty()) append_csv(o.csv, row);
  if (!o.history.empty()) {
    std::ofstream f(o.history);
    f << "# iteration relres\n";
    f.precision(12);
    for (size_t k = 0; k < res.history.size(); ++k) f << k + 1 << " " << res.history[k] << "\n";
  }
  return 0;
}

}  // namespace cg
