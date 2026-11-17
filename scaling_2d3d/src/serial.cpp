#include "serial.hpp"

#include <chrono>
#include <cmath>

namespace {
double now() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}
}  // namespace

SerialResult serial_cg(ProblemKind kind, int dim, int n, double tol, int maxit, std::vector<double>* u_out) {
  SerialResult res;
  const bool d3 = dim == 3;
  res.dof = d3 ? (long long)n * n * n : (long long)n * n;
  {
    long long m = n;
    res.nnz = d3 ? 7 * m * m * m - 6 * m * m : 5 * m * m - 4 * m;
  }
  const int o[3] = {0, 0, 0}, l[3] = {n, n, d3 ? n : 1};
  int ex, ey, ez, kb;
  padded_extents(dim, l, ex, ey, ez, kb);
  const size_t sy = ex, sz = (size_t)ex * ey, tot = sz * ez;
  const double h = 1.0 / (n + 1), ih2 = 1.0 / (h * h), c0 = d3 ? 6.0 : 4.0;
  const int k0 = kb, k1 = kb + l[2] - 1;

  double t0 = now();
  std::vector<double> b;
  build_rhs(kind, dim, n, o, l, b);
  std::vector<double> x(tot, 0.0), r(b), p(b), Ap(tot, 0.0);
  res.t_setup = now() - t0;

  auto dot = [&](const std::vector<double>& a, const std::vector<double>& c) {
    double s = 0;
    for (int k = k0; k <= k1; ++k)
      for (int j = 1; j <= n; ++j) {
        size_t base = k * sz + j * sy;
        for (int i = 1; i <= n; ++i) s += a[base + i] * c[base + i];
      }
    return s;
  };

  t0 = now();
  double rr = dot(r, r);
  const double bnorm = std::sqrt(rr);
  int it = 0;
  if (bnorm > 0) {
    for (; it < maxit; ++it) {
      if (tol > 0 && std::sqrt(rr) / bnorm < tol) break;
      for (int k = k0; k <= k1; ++k)
        for (int j = 1; j <= n; ++j) {
          size_t base = k * sz + j * sy;
          for (int i = 1; i <= n; ++i) {
            size_t q = base + i;
            double s = p[q - 1] + p[q + 1] + p[q - sy] + p[q + sy];
            if (d3) s += p[q - sz] + p[q + sz];
            Ap[q] = (c0 * p[q] - s) * ih2;
          }
        }
      double pAp = dot(p, Ap);
      if (!(pAp > 0)) break;
      double alpha = rr / pAp;
      for (int k = k0; k <= k1; ++k)
        for (int j = 1; j <= n; ++j) {
          size_t base = k * sz + j * sy;
          for (int i = 1; i <= n; ++i) {
            size_t q = base + i;
            x[q] += alpha * p[q];
            r[q] -= alpha * Ap[q];
          }
        }
      double rrn = dot(r, r);
      double beta = rrn / rr;
      rr = rrn;
      for (int k = k0; k <= k1; ++k)
        for (int j = 1; j <= n; ++j) {
          size_t base = k * sz + j * sy;
          for (int i = 1; i <= n; ++i) {
            size_t q = base + i;
            p[q] = r[q] + beta * p[q];
          }
        }
    }
  }
  res.t_solve = now() - t0;
  res.iters = it;
  res.relres = bnorm > 0 ? std::sqrt(rr) / bnorm : 0.0;
  res.converged = tol > 0 && res.relres < tol;

  double einf, esq;
  error_norms(kind, dim, n, o, l, x.data(), einf, esq);
  res.err_inf = einf;
  res.err_l2 = std::sqrt(esq * std::pow(h, dim));
  if (u_out) *u_out = x;
  return res;
}
