// problem.hpp -- 2D / 3D Poisson test problems:  -Laplace(u) = f on (0,1)^dim,  u = g on the boundary.
//
// Grid: n^dim interior unknowns, h = 1/(n+1), unknown (gi,gj,gk) (1-based) sits at (gi*h, gj*h, gk*h).
// Local storage is "padded" x-fastest:  index = (k*ey + j)*ex + i,  ex = lx+2, ey = ly+2,
//   interior i in [1,lx], j in [1,ly]; k in [1,lz] (3D, ez = lz+2)  or  k = 0 only (2D, ez = 1).
// Halo / physical-boundary cells sit at i,j,(k) = 0 and l+1 and always stay 0 on the physical boundary:
// Dirichlet data is folded into the right-hand side (matrix-free operator always sees zero boundary):
//   b = f + g(boundary neighbour)/h^2   for points adjacent to the boundary.
#pragma once
#include <cmath>
#include <string>
#include <vector>

// bump : u = a(x)a(y)[a(z)], a(t) = t(1-t)e^t, g = 0, NOT a discrete eigenvector -> realistic CG iteration count
// sin  : u = sin(pi x) sin(pi y) [sin(pi z)], g = 0.  Discrete eigenvector: CG converges in 1 step (sanity only).
// exp  : u = e^(x+y[+z]), non-zero g (boundary-term check, stage S8/S10)
enum ProblemKind { PROB_SIN = 0, PROB_EXP = 1, PROB_BUMP = 2 };
inline double bump_a(double t) { return t * (1 - t) * std::exp(t); }
inline double bump_a2(double t) { return -(3 * t + t * t) * std::exp(t); }  // a''

inline ProblemKind parse_problem(const std::string& s) { return s == "exp" ? PROB_EXP : s == "sin" ? PROB_SIN : PROB_BUMP; }
inline const char* problem_name(ProblemKind k) { return k == PROB_EXP ? "exp" : k == PROB_SIN ? "sin" : "bump"; }

inline double u_exact(ProblemKind k, int dim, double x, double y, double z) {
  const bool d3 = dim == 3;
  if (k == PROB_BUMP) return bump_a(x) * bump_a(y) * (d3 ? bump_a(z) : 1.0);
  if (k == PROB_SIN) return std::sin(M_PI * x) * std::sin(M_PI * y) * (d3 ? std::sin(M_PI * z) : 1.0);
  return std::exp(x + y + (d3 ? z : 0.0));
}
inline double f_rhs(ProblemKind k, int dim, double x, double y, double z) {  // f = -Laplace(u_exact)
  const bool d3 = dim == 3;
  if (k == PROB_BUMP) {
    if (!d3) return -(bump_a2(x) * bump_a(y) + bump_a(x) * bump_a2(y));
    return -(bump_a2(x) * bump_a(y) * bump_a(z) + bump_a(x) * bump_a2(y) * bump_a(z) + bump_a(x) * bump_a(y) * bump_a2(z));
  }
  const double u = u_exact(k, dim, x, y, z);
  return k == PROB_SIN ? dim * M_PI * M_PI * u : -dim * u;
}
inline double g_bc(ProblemKind k, int dim, double x, double y, double z) {  // boundary values (exactly 0 unless exp)
  return k == PROB_EXP ? u_exact(k, dim, x, y, z) : 0.0;
}

// padded local extents for a block of interior sizes l[3] (l[2] = 1 for 2D)
inline void padded_extents(int dim, const int l[3], int& ex, int& ey, int& ez, int& kb) {
  ex = l[0] + 2;
  ey = l[1] + 2;
  ez = dim == 3 ? l[2] + 2 : 1;
  kb = dim == 3 ? 1 : 0;
}

// Right-hand side of the block [o0,o0+l0) x [o1,o1+l1) x [o2,o2+l2) (global 0-based offsets), padded layout.
inline void build_rhs(ProblemKind k, int dim, int n, const int o[3], const int l[3], std::vector<double>& b) {
  const double h = 1.0 / (n + 1), ih2 = 1.0 / (h * h);
  int ex, ey, ez, kb;
  padded_extents(dim, l, ex, ey, ez, kb);
  b.assign((size_t)ex * ey * ez, 0.0);
  const bool d3 = dim == 3;
  for (int kk = 0; kk < l[2]; ++kk) {
    const int gk = o[2] + kk + 1;
    const double z = d3 ? gk * h : 0.0;
    for (int j = 1; j <= l[1]; ++j) {
      const int gj = o[1] + j;
      const double y = gj * h;
      for (int i = 1; i <= l[0]; ++i) {
        const int gi = o[0] + i;
        const double x = gi * h;
        double v = f_rhs(k, dim, x, y, z);
        if (gi == 1) v += g_bc(k, dim, 0.0, y, z) * ih2;
        if (gi == n) v += g_bc(k, dim, 1.0, y, z) * ih2;
        if (gj == 1) v += g_bc(k, dim, x, 0.0, z) * ih2;
        if (gj == n) v += g_bc(k, dim, x, 1.0, z) * ih2;
        if (d3 && gk == 1) v += g_bc(k, dim, x, y, 0.0) * ih2;
        if (d3 && gk == n) v += g_bc(k, dim, x, y, 1.0) * ih2;
        b[((size_t)(kb + kk) * ey + j) * ex + i] = v;
      }
    }
  }
}

// max abs / sum of squared error of a padded local solution vs the exact solution.
inline void error_norms(ProblemKind k, int dim, int n, const int o[3], const int l[3], const double* u, double& einf,
                        double& esq) {
  const double h = 1.0 / (n + 1);
  int ex, ey, ez, kb;
  padded_extents(dim, l, ex, ey, ez, kb);
  const bool d3 = dim == 3;
  einf = 0.0;
  esq = 0.0;
  for (int kk = 0; kk < l[2]; ++kk)
    for (int j = 1; j <= l[1]; ++j)
      for (int i = 1; i <= l[0]; ++i) {
        double e = u[((size_t)(kb + kk) * ey + j) * ex + i] -
                   u_exact(k, dim, (o[0] + i) * h, (o[1] + j) * h, d3 ? (o[2] + kk + 1) * h : 0.0);
        einf = std::fmax(einf, std::fabs(e));
        esq += e * e;
      }
}

// fixed-iteration rule of the performance runs (Experiment_Plan section 0):
//   iters = 100 * 4096^2 / N  clamped to [100, 1000],  N = n^dim (3D uses the same ratio with the point count)
inline int auto_iters(int dim, int n) {
  double N = std::pow((double)n, dim);
  double it = 100.0 * 4096.0 * 4096.0 / N;
  if (it < 100) it = 100;
  if (it > 1000) it = 1000;
  return (int)std::llround(it);
}
