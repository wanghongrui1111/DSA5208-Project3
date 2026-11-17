#pragma once
// Host kernels on one Block (see common.hpp). The OpenMP pragmas thread the
// loops in the MPI+OpenMP build; the serial build is compiled with
// -fopenmp-simd, which keeps the SIMD hints and drops the threading.
//
// All loops run over rows with schedule(static), so a given row is always
// touched by the same thread. Together with the parallel first touch in fill()
// this keeps every row on the NUMA node of the thread that works on it. The
// one exception is stencil_shared(), which trades that locality for dynamic
// load balance.

#include "common.hpp"

namespace cg {

// 64-byte aligned, uninitialised array of doubles. Uninitialised on purpose:
// the first parallel write decides on which NUMA node each page lives.
class HostArray {
 public:
  HostArray() = default;
  explicit HostArray(long n) : n_(n) {
    const size_t bytes = ((static_cast<size_t>(n) * sizeof(double) + 63) / 64) * 64;
    p_ = static_cast<double*>(std::aligned_alloc(64, bytes));
    if (!p_) throw std::bad_alloc();
  }
  HostArray(HostArray&& o) noexcept : n_(o.n_), p_(o.p_) {
    o.n_ = 0;
    o.p_ = nullptr;
  }
  HostArray& operator=(HostArray&& o) noexcept {
    std::swap(n_, o.n_);
    std::swap(p_, o.p_);
    return *this;
  }
  HostArray(const HostArray&) = delete;
  HostArray& operator=(const HostArray&) = delete;
  ~HostArray() { std::free(p_); }

  double* data() { return p_; }
  const double* data() const { return p_; }
  long size() const { return n_; }

 private:
  long n_ = 0;
  double* p_ = nullptr;
};

namespace cpu {

// Sets every entry, ghost layer included.
inline void fill(const Block& bl, double* v, double value) {
  const long ld = bl.ld(), rows = bl.nx + 2;
#pragma omp parallel for schedule(static)
  for (long i = 0; i < rows; ++i)
    for (long j = 0; j < ld; ++j) v[i * ld + j] = value;
}

inline void copy(const Block& bl, double* __restrict dst, const double* __restrict src) {
  const long ld = bl.ld(), rows = bl.nx + 2;
#pragma omp parallel for schedule(static)
  for (long i = 0; i < rows; ++i)
    for (long j = 0; j < ld; ++j) dst[i * ld + j] = src[i * ld + j];
}

// b = h^2 f at the local interior points, zero in the ghost layer.
inline void init_rhs(const Block& bl, double* b) {
  fill(bl, b, 0.0);
  const long ld = bl.ld();
  const double h = bl.h();
#pragma omp parallel for schedule(static)
  for (long i = 0; i < bl.nx; ++i)
    for (long j = 0; j < bl.ny; ++j)
      b[(i + 1) * ld + j + 1] = h * h * source_term((bl.i0 + i + 1) * h, (bl.j0 + j + 1) * h);
}

// Reductions. Every kernel that returns sums first forms one partial sum per
// row. With rows == nullptr the row partials are combined by an OpenMP
// reduction, whose order depends on the thread count and may change from run
// to run. Otherwise row i's partials are stored at rows[NC * i + c] and added
// in row order, which makes the result independent of threads and timing.
template <int NC>
inline void add_rows(const double* rows, long nrows, double* out) {
  double s[NC] = {};
  for (long i = 0; i < nrows; ++i)
    for (int c = 0; c < NC; ++c) s[c] += rows[NC * i + c];
  for (int c = 0; c < NC; ++c) out[c] += s[c];
}

inline double dot(const Block& bl, const double* __restrict a, const double* __restrict b, double* rows = nullptr) {
  const long ld = bl.ld(), nx = bl.nx, ny = bl.ny;
  double s = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : s)
  for (long i = 0; i < nx; ++i) {
    const long row = (i + 1) * ld + 1;
    double rs = 0.0;
#pragma omp simd reduction(+ : rs)
    for (long j = 0; j < ny; ++j) rs += a[row + j] * b[row + j];
    if (rows) rows[i] = rs;
    else s += rs;
  }
  if (rows) add_rows<1>(rows, nx, &s);
  return s;
}

// One row of dst = A src, where A is the 5-point Laplacian scaled by h^2;
// with Dots also the row's sum(src * dst) and sum(src * src).
template <bool Dots>
inline void stencil_row(long ld, double* __restrict dst, const double* __restrict src, long i, long j0, long j1,
                        double& sd, double& ss) {
  const long row = (i + 1) * ld + 1;
  double a = 0.0, c = 0.0;
#pragma omp simd reduction(+ : a, c)
  for (long j = j0; j < j1; ++j) {
    const long k = row + j;
    const double s = src[k];
    const double v = 4.0 * s - src[k - ld] - src[k + ld] - src[k - 1] - src[k + 1];
    dst[k] = v;
    if constexpr (Dots) {
      a += s * v;
      c += s * s;
    }
  }
  sd = a;
  ss = c;
}

// dst = A src on r. With Dots, also accumulates sd += sum(src * dst) and
// ss += sum(src * src) over r (rows: 2 per row of r, see add_rows).
template <bool Dots>
inline void stencil(const Block& bl, double* __restrict dst, const double* __restrict src, const Rect& r,
                    double& sd, double& ss, double* rows = nullptr) {
  if (r.empty()) return;
  const long ld = bl.ld();
  double a = 0.0, c = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : a, c)
  for (long i = r.i0; i < r.i1; ++i) {
    double ra, rc;
    stencil_row<Dots>(ld, dst, src, i, r.j0, r.j1, ra, rc);
    if constexpr (Dots) {
      if (rows) {
        rows[2 * (i - r.i0)] = ra;
        rows[2 * (i - r.i0) + 1] = rc;
      } else {
        a += ra;
        c += rc;
      }
    }
  }
  if constexpr (Dots) {
    double s[2] = {a, c};
    if (rows) add_rows<2>(rows, r.i1 - r.i0, s);
    sd += s[0];
    ss += s[1];
  }
}

// Worksharing variant of stencil() for use inside an enclosing parallel
// region. Rows are handed out dynamically because the master thread joins late
// (it does the MPI calls first); a and c are the calling thread's private
// accumulators. With rows, the caller adds the row partials after the region.
template <bool Dots>
inline void stencil_shared(const Block& bl, double* __restrict dst, const double* __restrict src, const Rect& r,
                           double& a, double& c, double* rows) {
  if (r.empty()) return;
  const long ld = bl.ld();
#pragma omp for schedule(dynamic, 8) nowait
  for (long i = r.i0; i < r.i1; ++i) {
    double ra, rc;
    stencil_row<Dots>(ld, dst, src, i, r.j0, r.j1, ra, rc);
    if constexpr (Dots) {
      if (rows) {
        rows[2 * (i - r.i0)] = ra;
        rows[2 * (i - r.i0) + 1] = rc;
      } else {
        a += ra;
        c += rc;
      }
    }
  }
}

// Classic CG: x += alpha p, r -= alpha q; returns the local (r, r).
inline double classic_xr(const Block& bl, double* __restrict x, double* __restrict r, const double* __restrict p,
                         const double* __restrict q, double alpha, double* rows = nullptr) {
  const long ld = bl.ld(), nx = bl.nx, ny = bl.ny;
  double rr = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : rr)
  for (long i = 0; i < nx; ++i) {
    const long row = (i + 1) * ld + 1;
    double rs = 0.0;
#pragma omp simd reduction(+ : rs)
    for (long j = 0; j < ny; ++j) {
      const long k = row + j;
      x[k] += alpha * p[k];
      const double rk = r[k] - alpha * q[k];
      r[k] = rk;
      rs += rk * rk;
    }
    if (rows) rows[i] = rs;
    else rr += rs;
  }
  if (rows) add_rows<1>(rows, nx, &rr);
  return rr;
}

// Classic CG: p = r + beta p.
inline void classic_p(const Block& bl, double* __restrict p, const double* __restrict r, double beta) {
  const long ld = bl.ld(), nx = bl.nx, ny = bl.ny;
#pragma omp parallel for schedule(static)
  for (long i = 0; i < nx; ++i) {
    const long row = (i + 1) * ld + 1;
#pragma omp simd
    for (long j = 0; j < ny; ++j) p[row + j] = r[row + j] + beta * p[row + j];
  }
}

// Chronopoulos-Gear CG: p = r + beta p, s = w + beta s, x += alpha p, r -= alpha s.
inline void cgcg_update(const Block& bl, double* __restrict x, double* __restrict r, double* __restrict p,
                        double* __restrict s, const double* __restrict w, double alpha, double beta) {
  const long ld = bl.ld(), nx = bl.nx, ny = bl.ny;
#pragma omp parallel for schedule(static)
  for (long i = 0; i < nx; ++i) {
    const long row = (i + 1) * ld + 1;
#pragma omp simd
    for (long j = 0; j < ny; ++j) {
      const long k = row + j;
      const double pk = r[k] + beta * p[k];
      const double sk = w[k] + beta * s[k];
      p[k] = pk;
      s[k] = sk;
      x[k] += alpha * pk;
      r[k] -= alpha * sk;
    }
  }
}

// Pipelined CG (Ghysels & Vanroose): z = q + beta z, s = w + beta s,
// p = r + beta p, x += alpha p, r -= alpha s, w -= alpha z, and the local
// (r, r) and (w, r) of the new iterate, all in a single sweep.
inline void pipe_update(const Block& bl, double* __restrict x, double* __restrict r, double* __restrict p,
                        double* __restrict s, double* __restrict w, double* __restrict z,
                        const double* __restrict q, double alpha, double beta, double& rr, double& wr,
                        double* rows = nullptr) {
  const long ld = bl.ld(), nx = bl.nx, ny = bl.ny;
  double a = 0.0, c = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : a, c)
  for (long i = 0; i < nx; ++i) {
    const long row = (i + 1) * ld + 1;
    double ra = 0.0, rc = 0.0;
#pragma omp simd reduction(+ : ra, rc)
    for (long j = 0; j < ny; ++j) {
      const long k = row + j;
      const double zk = q[k] + beta * z[k];
      const double sk = w[k] + beta * s[k];
      const double pk = r[k] + beta * p[k];
      const double rk = r[k] - alpha * sk;
      const double wk = w[k] - alpha * zk;
      z[k] = zk;
      s[k] = sk;
      p[k] = pk;
      x[k] += alpha * pk;
      r[k] = rk;
      w[k] = wk;
      ra += rk * rk;
      rc += wk * rk;
    }
    if (rows) {
      rows[2 * i] = ra;
      rows[2 * i + 1] = rc;
    } else {
      a += ra;
      c += rc;
    }
  }
  double out[2] = {a, c};
  if (rows) add_rows<2>(rows, nx, out);
  rr = out[0];
  wr = out[1];
}

// Residual and error contributions of the local block; q must hold A x.
inline LocalStats local_stats(const Block& bl, const double* x, const double* q, const double* b) {
  const long ld = bl.ld(), nx = bl.nx, ny = bl.ny;
  const double h = bl.h();
  double res2 = 0.0, err2 = 0.0, b2 = 0.0, emax = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : res2, err2, b2) reduction(max : emax)
  for (long i = 0; i < nx; ++i) {
    for (long j = 0; j < ny; ++j) {
      const long k = (i + 1) * ld + j + 1;
      const double r = b[k] - q[k];
      const double e = x[k] - exact_solution((bl.i0 + i + 1) * h, (bl.j0 + j + 1) * h);
      res2 += r * r;
      b2 += b[k] * b[k];
      err2 += e * e;
      emax = std::max(emax, std::fabs(e));
    }
  }
  LocalStats st;
  st.res2 = res2;
  st.err2 = err2;
  st.err_max = emax;
  st.b2 = b2;
  return st;
}

}  // namespace cpu
}  // namespace cg
