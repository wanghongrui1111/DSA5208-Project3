// serial_main.cpp -- serial CG driver (stages S1, S8, S9, S10; experiments V1, V2, E1).
//   serial --dim 2|3 --n 31,63,127 [--problem bump|exp|sin] [--tol 1e-12 | --iters K|auto] [--maxit 100000]
//          [--repeat 1] [--sweep 1] [--order-tol 0.01] [--csv f] [--tag t]
//
// --sweep 1 : n list is a convergence study (h halves each step); prints errors, observed orders and
//             "ORDER PASS/FAIL" (|order-2| <= order-tol for every consecutive pair); exit code 1 on FAIL.
// --iters   : performance mode, tol=0 and exactly K iterations (K = 'auto' -> clamp(100*4096^2/N,100,1000)).
//             time_per_iter_ms is the median over --repeat runs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "common.hpp"
#include "csv.hpp"
#include "serial.hpp"

int main(int argc, char** argv) {
  Args args(argc, argv);
  const int dim = args.integer("dim", 2);
  const std::vector<int> ns = args.int_list("n", {63});
  const ProblemKind kind = parse_problem(args.str("problem", "bump"));
  double tol = args.real("tol", 1e-8);
  int maxit = args.integer("maxit", 100000);
  const std::string iters = args.str("iters", "");
  const int repeat = args.integer("repeat", 1);
  const bool sweep = args.integer("sweep", 0) != 0;
  const double order_tol = args.real("order-tol", 0.01);
  const std::string csv = args.str("csv", "");
  const std::string tag = args.str("tag", "serial");
  for (auto& u : args.unused()) {
    fprintf(stderr, "unknown argument --%s\n", u.c_str());
    return 2;
  }
  if (dim != 2 && dim != 3) {
    fprintf(stderr, "--dim must be 2 or 3\n");
    return 2;
  }

  bool ok = true;
  double prev_einf = 0, prev_el2 = 0, prev_h = 0;
  if (sweep)
    printf("# convergence study dim=%d problem=%s tol=%g\n#%7s %12s %12s %7s %12s %7s %7s %9s\n", dim, problem_name(kind),
           tol, "n", "h", "err_inf", "ord_inf", "err_l2", "ord_l2", "iters", "t_solve");
  for (int n : ns) {
    int it_fixed = 0;
    if (!iters.empty()) {
      it_fixed = iters == "auto" ? auto_iters(dim, n) : std::atoi(iters.c_str());
      tol = 0;
      maxit = it_fixed;
    }
    std::vector<double> times;
    SerialResult R;
    for (int rep = 0; rep < std::max(1, repeat); ++rep) {
      R = serial_cg(kind, dim, n, tol, maxit, nullptr);
      times.push_back(R.t_solve);
    }
    std::sort(times.begin(), times.end());
    const double tmed = times[times.size() / 2];
    const double h = 1.0 / (n + 1);

    if (sweep) {
      double oi = 0, o2 = 0;
      if (prev_h > 0) {
        oi = std::log(prev_einf / R.err_inf) / std::log(prev_h / h);
        o2 = std::log(prev_el2 / R.err_l2) / std::log(prev_h / h);
        if (std::fabs(oi - 2.0) > order_tol || std::fabs(o2 - 2.0) > order_tol) ok = false;
      }
      if (prev_h > 0)
        printf("%8d %12.5e %12.5e %7.4f %12.5e %7.4f %7d %9.3f\n", n, h, R.err_inf, oi, R.err_l2, o2, R.iters, tmed);
      else
        printf("%8d %12.5e %12.5e %7s %12.5e %7s %7d %9.3f\n", n, h, R.err_inf, "-", R.err_l2, "-", R.iters, tmed);
      if (!R.converged) {
        ok = false;
        printf("  (n=%d did not reach tol)\n", n);
      }
      prev_einf = R.err_inf;
      prev_el2 = R.err_l2;
      prev_h = h;
    } else {
      printf("serial dim=%d n=%d N=%lld nnz=%lld iters=%d t_solve=%.6f s  per_iter=%.4f ms relres=%.3e err_inf=%.3e\n", dim,
             n, R.dof, R.nnz, R.iters, tmed, R.iters ? 1e3 * tmed / R.iters : 0.0, R.relres, R.err_inf);
    }

    Rec rc;
    rc.dim = dim;
    rc.n = n;
    rc.N = R.dof;
    rc.nnz = R.nnz;
    rc.tol = tol;
    rc.maxit = maxit;
    rc.repeat = std::max(1, repeat);
    rc.iters = R.iters;
    rc.time_s = tmed;
    rc.time_per_iter_ms = R.iters ? 1e3 * tmed / R.iters : 0.0;
    rc.relres = R.relres;
    rc.err_inf = R.err_inf;
    rc.err_l2 = R.err_l2;
    rc.tag = tag;
    rc.t_compute = tmed;
    rc.problem = problem_name(kind);
    rc.converged = R.converged;
    csv_append(csv, rc);
  }
  if (sweep) printf("ORDER %s (dim=%d problem=%s, |order-2| <= %g)\n", ok ? "PASS" : "FAIL", dim, problem_name(kind), order_tol);
  return ok ? 0 : 1;
}
