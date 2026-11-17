#include "solver.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

inline double tic() { return MPI_Wtime(); }

inline void split1d(int N, int p, int r, int& n, int& off) {
  int base = N / p, rem = N % p;
  n = base + (r < rem ? 1 : 0);
  off = r * base + std::min(r, rem);
}

// aligned array, first-touched by the same static schedule the kernels use
struct Vec {
  double* p = nullptr;
  size_t n = 0;
  void alloc(size_t n_) {
    n = n_;
    size_t bytes = ((n * sizeof(double) + 63) / 64) * 64;
    p = static_cast<double*>(std::aligned_alloc(64, bytes ? bytes : 64));
  }
  ~Vec() { std::free(p); }
  operator double*() const { return p; }
};

struct Domain {
  int dim, n;
  bool d3;
  int l[3], o[3];
  int ex, ey, ez, kb, k0, k1;
  size_t sy, sz, tot;
  double ih2, c0;
};

#define ROW_LOOP _Pragma("omp parallel for collapse(2) schedule(static)")

inline void row_stencil(const Domain& D, const double* __restrict v, double* __restrict out, size_t base, int ia, int ib) {
  const size_t sy = D.sy, sz = D.sz;
  const double c0 = D.c0, ih2 = D.ih2;
  if (D.d3) {
#pragma omp simd
    for (int i = ia; i <= ib; ++i) {
      size_t q = base + i;
      out[q] = (c0 * v[q] - (v[q - 1] + v[q + 1] + v[q - sy] + v[q + sy] + v[q - sz] + v[q + sz])) * ih2;
    }
  } else {
#pragma omp simd
    for (int i = ia; i <= ib; ++i) {
      size_t q = base + i;
      out[q] = (c0 * v[q] - (v[q - 1] + v[q + 1] + v[q - sy] + v[q + sy])) * ih2;
    }
  }
}

inline void pt_stencil(const Domain& D, const double* v, double* out, size_t q) {
  double s = v[q - 1] + v[q + 1] + v[q - D.sy] + v[q + D.sy];
  if (D.d3) s += v[q - D.sz] + v[q + D.sz];
  out[q] = (D.c0 * v[q] - s) * D.ih2;
}

struct Solver {
  MPI_Comm comm, cart;
  int rank, P;
  Config cfg;
  Domain D;
  Result R;
  int nthreads;

  // halo
  int nbr[6], flen[6], nfaces;
  int cpar[3] = {0, 0, 0};  // parity of this rank's Cartesian coordinate per axis (pairing order of blocking halo)
  size_t S = 0;  // slot stride (same on every rank, needed for RMA displacements)
  Vec sbuf_all, rbuf_all, agsend, agrecv;
  MPI_Win win = MPI_WIN_NULL;
  bool counting = true;
  long long msgs_total = 0, bytes_total = 0;
  int ag_axis = -1;

  // timers
  double T_comp = 0, T_halo = 0, T_red = 0, T_wait = 0, T_pure = 0;
  long long n_matvec = 0, n_reduce = 0;

  double* sbuf(int f) { return sbuf_all.p + (size_t)f * S; }
  double* rbuf(int f) { return rbuf_all.p + (size_t)f * S; }

  // ---------------------------------------------------------------- setup
  void setup() {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);
    const int dim = cfg.dim;
    D.dim = dim;
    D.n = cfg.n;
    D.d3 = dim == 3;
    nthreads = omp_get_max_threads();

    int px = cfg.px, py = cfg.py, pz = cfg.pz;
    if (px > 0 || py > 0 || pz > 0) {
      px = std::max(px, 1);
      py = std::max(py, 1);
      pz = dim == 3 ? std::max(pz, 1) : 1;
    } else {
      choose_grid(P, dim, cfg.decomp, px, py, pz);
    }
    if (px * py * pz != P) fail("px*py*pz != number of ranks");
    R.px = px;
    R.py = py;
    R.pz = pz;
    const int pp[3] = {px, py, pz};
    // Cartesian communicator with x varying fastest => reversed dims order
    int cd[3], per[3] = {0, 0, 0};
    for (int a = 0; a < dim; ++a) cd[a] = pp[dim - 1 - a];
    MPI_Cart_create(comm, dim, cd, per, 0, &cart);
    int cc[3];
    MPI_Cart_coords(cart, rank, dim, cc);
    int coord[3] = {0, 0, 0};
    for (int a = 0; a < dim; ++a) coord[a] = cc[dim - 1 - a];
    for (int a = 0; a < dim; ++a) cpar[a] = coord[a] & 1;
    for (int a = 0; a < 3; ++a) {
      if (a >= dim) {
        D.l[a] = 1;
        D.o[a] = 0;
        continue;
      }
      split1d(cfg.n, pp[a], coord[a], D.l[a], D.o[a]);
      if (D.l[a] < 1) fail("grid too small for the process grid");
    }
    for (int a = 0; a < 3; ++a) {
      R.l[a] = D.l[a];
      R.o[a] = D.o[a];
    }
    padded_extents(dim, D.l, D.ex, D.ey, D.ez, D.kb);
    D.k0 = D.kb;
    D.k1 = D.kb + D.l[2] - 1;
    D.sy = D.ex;
    D.sz = (size_t)D.ex * D.ey;
    D.tot = D.sz * D.ez;
    const double h = 1.0 / (cfg.n + 1);
    D.ih2 = 1.0 / (h * h);
    D.c0 = D.d3 ? 6.0 : 4.0;

    nfaces = 2 * dim;
    for (int f = 0; f < 6; ++f) {
      nbr[f] = MPI_PROC_NULL;
      flen[f] = 0;
    }
    for (int a = 0; a < dim; ++a) {
      int lo, hi;
      MPI_Cart_shift(cart, dim - 1 - a, 1, &lo, &hi);
      nbr[2 * a] = lo;
      nbr[2 * a + 1] = hi;
      flen[2 * a] = flen[2 * a + 1] = face_size(a);
    }
    // fixed slot stride: largest face any rank can have
    int lmax[3] = {1, 1, 1};
    for (int a = 0; a < dim; ++a) lmax[a] = (cfg.n + pp[a] - 1) / pp[a];
    S = 1;
    for (int a = 0; a < dim; ++a) {
      size_t s = 1;
      for (int b = 0; b < dim; ++b)
        if (b != a) s *= lmax[b];
      S = std::max(S, s);
    }
    sbuf_all.alloc(S * 6);
    rbuf_all.alloc(S * 6);
    std::memset(sbuf_all.p, 0, S * 6 * sizeof(double));
    std::memset(rbuf_all.p, 0, S * 6 * sizeof(double));

    if (cfg.halo == "rma") {
      MPI_Win_create(rbuf_all.p, S * 6 * sizeof(double), sizeof(double), MPI_INFO_NULL, comm, &win);
    }
    if (cfg.halo == "allgather") {
      ag_axis = dim - 1;
      for (int a = 0; a < dim - 1; ++a)
        if (pp[a] != 1) fail("halo=allgather needs a 1D decomposition (use --decomp 1d)");
      agsend.alloc(2 * (size_t)flen[2 * ag_axis]);
      agrecv.alloc(2 * (size_t)flen[2 * ag_axis] * P);
    }
  }

  void fail(const char* msg) {
    if (rank == 0) fprintf(stderr, "error: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  int face_size(int a) const {
    int s = 1;
    for (int b = 0; b < D.dim; ++b)
      if (b != a) s *= D.l[b];
    return s;
  }

  void teardown() {
    if (win != MPI_WIN_NULL) MPI_Win_free(&win);
    MPI_Comm_free(&cart);
  }

  // ---------------------------------------------------------------- vectors / kernels
  void vzero(double* v) {
    ROW_LOOP
    for (int k = 0; k < D.ez; ++k)
      for (int j = 0; j < D.ey; ++j) std::memset(v + ((size_t)k * D.ey + j) * D.ex, 0, D.ex * sizeof(double));
  }
  void vcopy(double* dst, const double* src) {
    ROW_LOOP
    for (int k = 0; k < D.ez; ++k)
      for (int j = 0; j < D.ey; ++j) {
        size_t base = ((size_t)k * D.ey + j) * D.ex;
        std::memcpy(dst + base, src + base, D.ex * sizeof(double));
      }
  }
  inline size_t row0(int k, int j) const { return ((size_t)k * D.ey + j) * D.ex; }

  double dot(const double* a, const double* b) {
    double s = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : s)
    for (int k = D.k0; k <= D.k1; ++k)
      for (int j = 1; j <= D.l[1]; ++j) {
        size_t base = row0(k, j);
        double rs = 0;
#pragma omp simd reduction(+ : rs)
        for (int i = 1; i <= D.l[0]; ++i) rs += a[base + i] * b[base + i];
        s += rs;
      }
    return s;
  }
  void dot2(const double* a, const double* b, const double* c, const double* d, double& s1, double& s2) {
    double x1 = 0, x2 = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : x1, x2)
    for (int k = D.k0; k <= D.k1; ++k)
      for (int j = 1; j <= D.l[1]; ++j) {
        size_t base = row0(k, j);
        double r1 = 0, r2 = 0;
#pragma omp simd reduction(+ : r1, r2)
        for (int i = 1; i <= D.l[0]; ++i) {
          r1 += a[base + i] * b[base + i];
          r2 += c[base + i] * d[base + i];
        }
        x1 += r1;
        x2 += r2;
      }
    s1 = x1;
    s2 = x2;
  }
  // classic: x += a p ; r -= a Ap ; return local r.r
  double upd_xr(double* x, double* r, const double* p, const double* Ap, double a) {
    double s = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : s)
    for (int k = D.k0; k <= D.k1; ++k)
      for (int j = 1; j <= D.l[1]; ++j) {
        size_t base = row0(k, j);
        double rs = 0;
#pragma omp simd reduction(+ : rs)
        for (int i = 1; i <= D.l[0]; ++i) {
          size_t q = base + i;
          x[q] += a * p[q];
          double rn = r[q] - a * Ap[q];
          r[q] = rn;
          rs += rn * rn;
        }
        s += rs;
      }
    return s;
  }
  void upd_p(double* p, const double* r, double beta) {
    ROW_LOOP
    for (int k = D.k0; k <= D.k1; ++k)
      for (int j = 1; j <= D.l[1]; ++j) {
        size_t base = row0(k, j);
#pragma omp simd
        for (int i = 1; i <= D.l[0]; ++i) p[base + i] = r[base + i] + beta * p[base + i];
      }
  }
  // cgcg: p=r+b p ; s=w+b s ; x+=a p ; r-=a s
  void upd_cgcg(double* x, double* r, double* p, double* s, const double* w, double a, double b) {
    ROW_LOOP
    for (int k = D.k0; k <= D.k1; ++k)
      for (int j = 1; j <= D.l[1]; ++j) {
        size_t base = row0(k, j);
#pragma omp simd
        for (int i = 1; i <= D.l[0]; ++i) {
          size_t q = base + i;
          double pn = r[q] + b * p[q];
          double sn = w[q] + b * s[q];
          p[q] = pn;
          s[q] = sn;
          x[q] += a * pn;
          r[q] -= a * sn;
        }
      }
  }
  // pipelined: z=q+b z ; s=w+b s ; p=r+b p ; x+=a p ; r-=a s ; w-=a z ; local (r.r, w.r)
  void upd_pipe(double* x, double* r, double* w, double* p, double* s, double* z, const double* q_, double a, double b,
                double& g, double& d) {
    double g_ = 0, d_ = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : g_, d_)
    for (int k = D.k0; k <= D.k1; ++k)
      for (int j = 1; j <= D.l[1]; ++j) {
        size_t base = row0(k, j);
        double rg = 0, rd = 0;
#pragma omp simd reduction(+ : rg, rd)
        for (int i = 1; i <= D.l[0]; ++i) {
          size_t q = base + i;
          double zn = q_[q] + b * z[q];
          double sn = w[q] + b * s[q];
          double pn = r[q] + b * p[q];
          z[q] = zn;
          s[q] = sn;
          p[q] = pn;
          x[q] += a * pn;
          double rn = r[q] - a * sn;
          double wn = w[q] - a * zn;
          r[q] = rn;
          w[q] = wn;
          rg += rn * rn;
          rd += wn * rn;
        }
        g_ += rg;
        d_ += rd;
      }
    g = g_;
    d = d_;
  }

  // stencil over the whole block
  void stencil_all(const double* v, double* out) {
    ROW_LOOP
    for (int k = D.k0; k <= D.k1; ++k)
      for (int j = 1; j <= D.l[1]; ++j) row_stencil(D, v, out, row0(k, j), 1, D.l[0]);
  }
  // interior = points that do not touch a halo cell
  void interior_range(int& kA, int& kB, int& jA, int& jB, int& iA, int& iB) const {
    kA = D.k0 + (D.d3 ? 1 : 0);
    kB = D.k1 - (D.d3 ? 1 : 0);
    jA = 2;
    jB = D.l[1] - 1;
    iA = 2;
    iB = D.l[0] - 1;
  }
  void stencil_interior(const double* v, double* out) {
    int kA, kB, jA, jB, iA, iB;
    interior_range(kA, kB, jA, jB, iA, iB);
    if (iB < iA) return;
    ROW_LOOP
    for (int k = kA; k <= kB; ++k)
      for (int j = jA; j <= jB; ++j) row_stencil(D, v, out, row0(k, j), iA, iB);
  }
  void stencil_interior_rows(const double* v, double* out, long r0, long r1) {  // flattened (k,j) rows, serial
    int kA, kB, jA, jB, iA, iB;
    interior_range(kA, kB, jA, jB, iA, iB);
    if (iB < iA || jB < jA || kB < kA) return;
    const long nj = jB - jA + 1;
    for (long r = r0; r < r1; ++r) row_stencil(D, v, out, row0(kA + (int)(r / nj), jA + (int)(r % nj)), iA, iB);
  }
  long interior_rows() const {
    int kA, kB, jA, jB, iA, iB;
    interior_range(kA, kB, jA, jB, iA, iB);
    if (iB < iA || jB < jA || kB < kA) return 0;
    return (long)(kB - kA + 1) * (jB - jA + 1);
  }
  void stencil_shell(const double* v, double* out) {
    ROW_LOOP
    for (int k = D.k0; k <= D.k1; ++k)
      for (int j = 1; j <= D.l[1]; ++j) {
        const bool full = (j == 1 || j == D.l[1] || (D.d3 && (k == D.k0 || k == D.k1)));
        size_t base = row0(k, j);
        if (full) {
          row_stencil(D, v, out, base, 1, D.l[0]);
        } else {
          pt_stencil(D, v, out, base + 1);
          if (D.l[0] > 1) pt_stencil(D, v, out, base + D.l[0]);
        }
      }
  }

  // ---------------------------------------------------------------- halo
  // plane `idx` along axis a <-> contiguous buffer
  void face_op(double* v, int a, int idx, double* buf, bool pack) {
    const int lx = D.l[0], ly = D.l[1], lz = D.l[2];
    if (a == 0) {
#pragma omp parallel for collapse(2) schedule(static)
      for (int kk = 0; kk < lz; ++kk)
        for (int j = 0; j < ly; ++j) {
          double* vp = v + row0(D.kb + kk, j + 1) + idx;
          double* bp = buf + (size_t)kk * ly + j;
          if (pack) *bp = *vp; else *vp = *bp;
        }
    } else if (a == 1) {
#pragma omp parallel for collapse(2) schedule(static)
      for (int kk = 0; kk < lz; ++kk)
        for (int i = 0; i < lx; ++i) {
          double* vp = v + row0(D.kb + kk, idx) + 1 + i;
          double* bp = buf + (size_t)kk * lx + i;
          if (pack) *bp = *vp; else *vp = *bp;
        }
    } else {
#pragma omp parallel for collapse(2) schedule(static)
      for (int j = 0; j < ly; ++j)
        for (int i = 0; i < lx; ++i) {
          double* vp = v + row0(idx, j + 1) + 1 + i;
          double* bp = buf + (size_t)j * lx + i;
          if (pack) *bp = *vp; else *vp = *bp;
        }
    }
  }
  void pack_all(double* v) {
    for (int f = 0; f < nfaces; ++f)
      if (nbr[f] != MPI_PROC_NULL) face_op(v, f / 2, (f & 1) ? D.l[f / 2] : 1, sbuf(f), true);
  }
  void unpack_all(double* v) {
    for (int f = 0; f < nfaces; ++f)
      if (nbr[f] != MPI_PROC_NULL) face_op(v, f / 2, (f & 1) ? D.l[f / 2] + 1 : 0, rbuf(f), false);
  }
  void count_msgs() {
    if (!counting) return;
    for (int f = 0; f < nfaces; ++f)
      if (nbr[f] != MPI_PROC_NULL) {
        msgs_total++;
        bytes_total += (long long)flen[f] * sizeof(double);
      }
  }
  // message from my face f carries tag f; the receive for halo f expects tag f^1 (neighbour's opposite face)
  void post(MPI_Request* req) {
    for (int f = 0; f < nfaces; ++f) {
      MPI_Irecv(rbuf(f), flen[f], MPI_DOUBLE, nbr[f], f ^ 1, cart, &req[f]);
      MPI_Isend(sbuf(f), flen[f], MPI_DOUBLE, nbr[f], f, cart, &req[nfaces + f]);
    }
  }

  // complete halo update of v (modes without overlap).  `mode` may be forced (true-residual uses nonblock)
  void exchange_full(double* v, const std::string& mode) {
    count_msgs();
    pack_all(v);
    if (mode == "block") {
      // Paired two-phase blocking exchange, per axis: phase 0 pairs (even,odd) coordinates, phase 1 pairs (odd,even).
      // Even coordinate talks to its hi neighbour first, odd to its lo neighbour first => every MPI_Sendrecv finds its
      // partner already waiting, no rank waits for a neighbour's other exchange (no chain of length P).
      for (int a = 0; a < D.dim; ++a)
        for (int ph = 0; ph < 2; ++ph) {
          const int f = 2 * a + ((cpar[a] + ph) & 1 ? 0 : 1);  // odd+phase0 / even+phase1 -> lo face, else hi face
          MPI_Sendrecv(sbuf(f), flen[f], MPI_DOUBLE, nbr[f], f, rbuf(f), flen[f], MPI_DOUBLE, nbr[f], f ^ 1, cart,
                       MPI_STATUS_IGNORE);
        }
    } else if (mode == "blockseq") {  // original order lo,hi,...: rank k's 2nd exchange waits for k-1's 1st => chain of length P (1D)
      for (int f = 0; f < nfaces; ++f)
        MPI_Sendrecv(sbuf(f), flen[f], MPI_DOUBLE, nbr[f], f, rbuf(f), flen[f], MPI_DOUBLE, nbr[f], f ^ 1, cart,
                     MPI_STATUS_IGNORE);
    } else if (mode == "rma") {
      MPI_Win_fence(0, win);
      for (int f = 0; f < nfaces; ++f)
        if (nbr[f] != MPI_PROC_NULL)
          MPI_Put(sbuf(f), flen[f], MPI_DOUBLE, nbr[f], (MPI_Aint)((size_t)(f ^ 1) * S), flen[f], MPI_DOUBLE, win);
      MPI_Win_fence(0, win);
    } else if (mode == "allgather") {
      const int f0 = 2 * ag_axis, F = flen[f0];
      std::memcpy(agsend.p, sbuf(f0), F * sizeof(double));
      std::memcpy(agsend.p + F, sbuf(f0 + 1), F * sizeof(double));
      MPI_Allgather(agsend.p, 2 * F, MPI_DOUBLE, agrecv.p, 2 * F, MPI_DOUBLE, cart);
      if (nbr[f0] != MPI_PROC_NULL) std::memcpy(rbuf(f0), agrecv.p + (size_t)nbr[f0] * 2 * F + F, F * sizeof(double));
      if (nbr[f0 + 1] != MPI_PROC_NULL)
        std::memcpy(rbuf(f0 + 1), agrecv.p + (size_t)nbr[f0 + 1] * 2 * F, F * sizeof(double));
    } else {  // nonblock
      MPI_Request req[12];
      post(req);
      MPI_Waitall(2 * nfaces, req, MPI_STATUSES_IGNORE);
    }
    unpack_all(v);
  }

  // out = A v
  void matvec(double* v, double* out) {
    ++n_matvec;
    const std::string& m = cfg.halo;
    double t;
    if (m == "overlap") {
      t = tic();
      count_msgs();
      pack_all(v);
      MPI_Request req[12];
      post(req);
      T_halo += tic() - t;
      t = tic();
      stencil_interior(v, out);
      T_comp += tic() - t;
      t = tic();
      MPI_Waitall(2 * nfaces, req, MPI_STATUSES_IGNORE);
      unpack_all(v);
      T_halo += tic() - t;
      t = tic();
      stencil_shell(v, out);
      T_comp += tic() - t;
    } else if (m == "thread") {
      t = tic();
      count_msgs();
      pack_all(v);
      T_halo += tic() - t;
      MPI_Request req[12];
      const long nr = interior_rows();
      double tcomm = 0, wmax = 0;
      std::vector<double> wt(nthreads, 0.0);
      const double treg = tic();
#pragma omp parallel num_threads(nthreads)
      {
        const int tid = omp_get_thread_num(), nth = omp_get_num_threads();
        if (nth == 1) {
          double a = tic();
          post(req);
          MPI_Waitall(2 * nfaces, req, MPI_STATUSES_IGNORE);
          tcomm = tic() - a;
          a = tic();
          stencil_interior_rows(v, out, 0, nr);
          wt[0] = tic() - a;
        } else if (tid == 0) {
          double a = tic();
          post(req);
          MPI_Waitall(2 * nfaces, req, MPI_STATUSES_IGNORE);
          tcomm = tic() - a;
        } else {
          double a = tic();
          const int W = nth - 1, w = tid - 1;
          stencil_interior_rows(v, out, nr * w / W, nr * (w + 1) / W);
          wt[tid] = tic() - a;
        }
      }
      const double wall = tic() - treg;
      for (double x : wt) wmax = std::max(wmax, x);
      if (nthreads == 1) {
        T_halo += tcomm;
      } else {
        T_halo += std::max(0.0, wall - wmax);  // exposed communication
      }
      T_comp += wmax;
      t = tic();
      unpack_all(v);
      T_halo += tic() - t;
      t = tic();
      stencil_shell(v, out);
      T_comp += tic() - t;
    } else {  // block, nonblock, rma, allgather
      t = tic();
      exchange_full(v, m);
      T_halo += tic() - t;
      t = tic();
      stencil_all(v, out);
      T_comp += tic() - t;
    }
  }

  // ---------------------------------------------------------------- reductions
  void reduce(double* vals, int cnt) {
    double in[4], out[4];
    for (int i = 0; i < cnt; ++i) in[i] = vals[i];
    double t = tic();
    if (cfg.barrier_reduce) {
      MPI_Barrier(comm);
      double t2 = tic();
      T_wait += t2 - t;
      MPI_Allreduce(in, out, cnt, MPI_DOUBLE, MPI_SUM, comm);
      T_pure += tic() - t2;
    } else {
      MPI_Allreduce(in, out, cnt, MPI_DOUBLE, MPI_SUM, comm);
    }
    T_red += tic() - t;
    ++n_reduce;
    for (int i = 0; i < cnt; ++i) vals[i] = out[i];
  }
  double reduce1(double v) {
    reduce(&v, 1);
    return v;
  }

  // ---------------------------------------------------------------- true residual (untimed, uncounted)
  double true_relres(double* x, const double* b, double* tmp, double bnorm) {
    const double sc = T_comp, sh = T_halo, sr = T_red, sw = T_wait, sp = T_pure;
    const long long sm = n_matvec, smsg = msgs_total, sbytes = bytes_total;
    const bool sc_count = counting;
    counting = false;
    exchange_full(x, "nonblock");
    stencil_all(x, tmp);
    double s = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : s)
    for (int k = D.k0; k <= D.k1; ++k)
      for (int j = 1; j <= D.l[1]; ++j) {
        size_t base = row0(k, j);
        for (int i = 1; i <= D.l[0]; ++i) {
          double e = b[base + i] - tmp[base + i];
          s += e * e;
        }
      }
    double g;
    MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, comm);
    counting = sc_count;
    T_comp = sc;
    T_halo = sh;
    T_red = sr;
    T_wait = sw;
    T_pure = sp;
    n_matvec = sm;
    msgs_total = smsg;
    bytes_total = sbytes;
    return std::sqrt(g) / bnorm;
  }

  // ---------------------------------------------------------------- the solve
  void run(bool keep_solution) {
    const std::string& cg = cfg.cg;
    Vec x, r, b, w, p, q, s, z;
    const bool need_b = cfg.true_res || cfg.history;
    x.alloc(D.tot);
    r.alloc(D.tot);
    p.alloc(D.tot);
    w.alloc(D.tot);  // classic: Ap
    vzero(x);
    vzero(r);
    vzero(p);
    vzero(w);
    if (cg != "classic") {
      s.alloc(D.tot);
      vzero(s);
    }
    if (cg == "pipelined") {
      z.alloc(D.tot);
      q.alloc(D.tot);
      vzero(z);
      vzero(q);
    }
    Vec tmp;
    if (need_b) {
      b.alloc(D.tot);
      tmp.alloc(D.tot);
      vzero(b);
      vzero(tmp);
    }
    {
      std::vector<double> hb;
      build_rhs(cfg.kind, cfg.dim, cfg.n, D.o, D.l, hb);
#pragma omp parallel for collapse(2) schedule(static)
      for (int k = D.k0; k <= D.k1; ++k)
        for (int j = 1; j <= D.l[1]; ++j) {
          size_t base = row0(k, j);
          for (int i = 1; i <= D.l[0]; ++i) {
            r[base + i] = hb[base + i];
            if (cg == "classic") p[base + i] = hb[base + i];
            if (need_b) b[base + i] = hb[base + i];
          }
        }
    }
    const double bn2 = reduce1(dot(r, r));
    n_reduce = 0;
    T_red = T_wait = T_pure = 0;  // the initial ||b|| reduction runs before the timed loop and waits for setup imbalance: not part of the per-iteration reduce time
    const double bnorm = std::sqrt(bn2);

    const bool classic = cg == "classic", cgcg = cg == "cgcg";
    double excluded = 0;
    int it = 0;
    double gamma = bn2;  // (r,r) of the current state
    auto record = [&](int itn, double gam) {
      if (!cfg.history) return;
      double t0 = tic();
      double tr = -1;
      if (cfg.true_every > 0 && (itn % cfg.true_every == 0)) tr = true_relres(x, b, tmp, bnorm);
      if (rank == 0) {
        R.hist_it.push_back(itn);
        R.hist_rec.push_back(bnorm > 0 ? std::sqrt(gam) / bnorm : 0.0);
        R.hist_true.push_back(tr);
      }
      excluded += tic() - t0;
    };
    auto stop = [&](double gam) { return cfg.tol > 0 && std::sqrt(gam) / bnorm < cfg.tol; };

    MPI_Barrier(comm);
    const double t_start = tic();
    if (bnorm > 0) {
      if (classic) {
        double rr = bn2;
        for (; it < cfg.maxit; ++it) {
          record(it, rr);
          if (stop(rr)) break;
          matvec(p, w);
          double t = tic();
          double l = dot(p, w);
          T_comp += tic() - t;
          double pAp = reduce1(l);
          if (!(pAp > 0)) break;
          double alpha = rr / pAp;
          t = tic();
          double ln = upd_xr(x, r, p, w, alpha);
          T_comp += tic() - t;
          double rrn = reduce1(ln);
          double beta = rrn / rr;
          t = tic();
          upd_p(p, r, beta);
          T_comp += tic() - t;
          rr = rrn;
        }
        gamma = rr;
      } else if (cgcg) {
        matvec(r, w);
        double t = tic(), v[2];
        dot2(r, r, w, r, v[0], v[1]);
        T_comp += tic() - t;
        reduce(v, 2);
        gamma = v[0];
        double alpha = gamma / v[1], beta = 0;
        for (; it < cfg.maxit; ++it) {
          record(it, gamma);
          if (stop(gamma)) break;
          t = tic();
          upd_cgcg(x, r, p, s, w, alpha, beta);
          T_comp += tic() - t;
          matvec(r, w);
          t = tic();
          dot2(r, r, w, r, v[0], v[1]);
          T_comp += tic() - t;
          reduce(v, 2);
          double gn = v[0], delta = v[1];
          beta = gn / gamma;
          alpha = gn / (delta - beta * gn / alpha);
          gamma = gn;
        }
      } else {  // pipelined
        matvec(r, w);
        double loc[2], red[2];
        double t = tic();
        dot2(r, r, w, r, loc[0], loc[1]);
        T_comp += tic() - t;
        double gprev = 0, aprev = 0;
        for (;; ++it) {
          if (it >= cfg.maxit) {  // finish without the (wasted) matvec
            t = tic();
            MPI_Allreduce(loc, red, 2, MPI_DOUBLE, MPI_SUM, comm);
            T_red += tic() - t;
            ++n_reduce;
            gamma = red[0];
            record(it, gamma);
            break;
          }
          MPI_Request rq;
          t = tic();
          MPI_Iallreduce(loc, red, 2, MPI_DOUBLE, MPI_SUM, comm, &rq);
          T_red += tic() - t;
          matvec(w, q);
          t = tic();
          MPI_Wait(&rq, MPI_STATUS_IGNORE);
          T_red += tic() - t;
          ++n_reduce;
          gamma = red[0];
          const double delta = red[1];
          record(it, gamma);
          if (stop(gamma)) break;
          double alpha, beta;
          if (it == 0) {
            beta = 0;
            alpha = gamma / delta;
          } else {
            beta = gamma / gprev;
            alpha = gamma / (delta - beta * gamma / aprev);
          }
          t = tic();
          upd_pipe(x, r, w, p, s, z, q, alpha, beta, loc[0], loc[1]);
          T_comp += tic() - t;
          gprev = gamma;
          aprev = alpha;
        }
      }
    }
    const double t_loop = tic() - t_start - excluded;

    R.iters = it;
    R.relres = bnorm > 0 ? std::sqrt(gamma) / bnorm : 0.0;
    R.converged = cfg.tol > 0 && R.relres < cfg.tol;
    if (cfg.true_res && bnorm > 0) R.relres_true = true_relres(x, b, tmp, bnorm);

    double einf, esq;
    error_norms(cfg.kind, cfg.dim, cfg.n, D.o, D.l, x.p, einf, esq);
    double gi, gs;
    MPI_Allreduce(&einf, &gi, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&esq, &gs, 1, MPI_DOUBLE, MPI_SUM, comm);
    R.err_inf = gi;
    R.err_l2 = std::sqrt(gs * std::pow(1.0 / (cfg.n + 1), cfg.dim));

    // timers: max over ranks
    double loc[6] = {t_loop, T_comp, T_halo, T_red, T_wait, T_pure}, mx[6];
    MPI_Allreduce(loc, mx, 6, MPI_DOUBLE, MPI_MAX, comm);
    R.t_total = mx[0];
    R.t_compute = mx[1];
    R.t_halo = mx[2];
    R.t_reduce = mx[3];
    R.t_reduce_wait = mx[4];
    R.t_reduce_pure = mx[5];
    R.n_matvec = n_matvec;
    R.n_reduce = n_reduce;
    double hm[2] = {n_matvec ? (double)bytes_total / n_matvec : 0.0, n_matvec ? (double)msgs_total / n_matvec : 0.0}, hmx[2];
    MPI_Allreduce(hm, hmx, 2, MPI_DOUBLE, MPI_MAX, comm);
    R.halo_bytes_per_matvec = hmx[0];
    R.halo_msgs_per_matvec = hmx[1];
    double bt = (double)bytes_total, bts;
    MPI_Allreduce(&bt, &bts, 1, MPI_DOUBLE, MPI_SUM, comm);
    R.halo_bytes_total = bts;
    const int pp[3] = {R.px, R.py, R.pz};
    double per = 0;
    for (int a = 0; a < cfg.dim; ++a) per += 2.0 * (pp[a] - 1) * std::pow((double)cfg.n, cfg.dim - 1) * sizeof(double);
    R.halo_bytes_theory = per * n_matvec;
    if (keep_solution) R.u_local.assign(x.p, x.p + D.tot);
  }
};

}  // namespace

void choose_grid(int P, int dim, const std::string& decomp_in, int& px, int& py, int& pz) {
  std::string dc = decomp_in.empty() ? (dim == 2 ? "2d" : "3d") : decomp_in;
  // Cart dims order (slowest first): 2D (y,x), 3D (z,y,x)
  int cd[3] = {0, 0, 0};
  if (dim == 2) {
    if (dc == "1d") { cd[0] = 0; cd[1] = 1; }
  } else {
    if (dc == "1d") { cd[0] = 0; cd[1] = 1; cd[2] = 1; }
    else if (dc == "2d") { cd[0] = 0; cd[1] = 0; cd[2] = 1; }
  }
  MPI_Dims_create(P, dim, cd);
  if (dim == 2) {
    py = cd[0];
    px = cd[1];
    pz = 1;
  } else {
    pz = cd[0];
    py = cd[1];
    px = cd[2];
  }
}

Result solve(MPI_Comm comm, const Config& cfg, bool keep_solution) {
  Solver S;
  S.comm = comm;
  S.cfg = cfg;
  S.R.dim = cfg.dim;
  S.R.n = cfg.n;
  S.setup();
  S.run(keep_solution);
  S.teardown();
  return S.R;
}
