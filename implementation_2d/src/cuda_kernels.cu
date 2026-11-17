// Device kernels of the MPI+CUDA solver (see cuda_kernels.hpp). C++14 so that
// older toolkits (CUDA 10.x) compile it too.
//
// 2D kernels use 32 x 8 thread blocks with threadIdx.x along j, the contiguous
// direction, so warps access memory coalesced. Grids are capped and the
// threads stride over the rectangle, which keeps the number of partial sums
// per launch (one per block) small. Reductions are two-stage and
// deterministic: block partials first, then one block sums the partials.

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "cuda_kernels.hpp"

namespace gpu {
namespace {

constexpr int kBX = 32, kBY = 8, kThreads = kBX * kBY;
constexpr int kMaxGX = 32, kMaxGY = 64;

dim3 grid_for(long rows, long cols) {
  const long gx = std::min<long>((cols + kBX - 1) / kBX, kMaxGX);
  const long gy = std::min<long>((rows + kBY - 1) / kBY, kMaxGY);
  return dim3(static_cast<unsigned>(std::max<long>(gx, 1)), static_cast<unsigned>(std::max<long>(gy, 1)));
}

dim3 grid_1d(long n) { return dim3(static_cast<unsigned>(std::min<long>(std::max<long>((n + 255) / 256, 1), 1024))); }

// Claims one partial slot per block of a launch with grid g.
int claim(Partials* p, dim3 g) {
  const int off = p->used;
  p->used += static_cast<int>(g.x * g.y);
  if (p->used > p->capacity) {
    std::fprintf(stderr, "gpu::Partials capacity %d exceeded\n", p->capacity);
    std::abort();
  }
  return off;
}

// Sums NC per-thread values over the block and stores them in the block's slot.
template <int NC>
__device__ void block_store(const double (&v)[NC], double* partials, int cap, int off) {
  __shared__ double sh[NC][kThreads];
  const int t = threadIdx.y * blockDim.x + threadIdx.x;
  for (int c = 0; c < NC; ++c) sh[c][t] = v[c];
  __syncthreads();
  for (int w = kThreads / 2; w > 0; w >>= 1) {
    if (t < w)
      for (int c = 0; c < NC; ++c) sh[c][t] += sh[c][t + w];
    __syncthreads();
  }
  if (t == 0) {
    const int slot = off + blockIdx.y * gridDim.x + blockIdx.x;
    for (int c = 0; c < NC; ++c) partials[c * cap + slot] = sh[c][0];
  }
}

// Grid-stride loop over the local points [i0, i1) x [j0, j1); defines i, j.
#define FOR_RECT(i0, i1, j0, j1)                                                                       \
  for (int i = (i0) + blockIdx.y * blockDim.y + threadIdx.y; i < (i1); i += gridDim.y * blockDim.y) \
    for (int j = (j0) + blockIdx.x * blockDim.x + threadIdx.x; j < (j1); j += gridDim.x * blockDim.x)

__device__ inline long idx(int i, int j, long ld) { return (i + 1) * ld + j + 1; }

template <bool Dots>
__global__ void k_stencil(double* __restrict__ dst, const double* __restrict__ src, long ld, int i0, int i1, int j0,
                          int j1, double* partials, int cap, int off) {
  double sd = 0.0, ss = 0.0;
  FOR_RECT(i0, i1, j0, j1) {
    const long k = idx(i, j, ld);
    const double s = src[k];
    const double v = 4.0 * s - src[k - ld] - src[k + ld] - src[k - 1] - src[k + 1];
    dst[k] = v;
    if (Dots) {
      sd += s * v;
      ss += s * s;
    }
  }
  if (Dots) {
    const double v[2] = {sd, ss};
    block_store<2>(v, partials, cap, off);
  }
}

__global__ void k_dot(const double* __restrict__ a, const double* __restrict__ b, int nx, int ny, long ld,
                      double* partials, int cap, int off) {
  double s = 0.0;
  FOR_RECT(0, nx, 0, ny) {
    const long k = idx(i, j, ld);
    s += a[k] * b[k];
  }
  const double v[1] = {s};
  block_store<1>(v, partials, cap, off);
}

__global__ void k_classic_xr(double* __restrict__ x, double* __restrict__ r, const double* __restrict__ p,
                             const double* __restrict__ q, double alpha, int nx, int ny, long ld, double* partials,
                             int cap, int off) {
  double rr = 0.0;
  FOR_RECT(0, nx, 0, ny) {
    const long k = idx(i, j, ld);
    x[k] += alpha * p[k];
    const double rk = r[k] - alpha * q[k];
    r[k] = rk;
    rr += rk * rk;
  }
  const double v[1] = {rr};
  block_store<1>(v, partials, cap, off);
}

__global__ void k_classic_p(double* __restrict__ p, const double* __restrict__ r, double beta, int nx, int ny,
                            long ld) {
  FOR_RECT(0, nx, 0, ny) {
    const long k = idx(i, j, ld);
    p[k] = r[k] + beta * p[k];
  }
}

__global__ void k_cgcg(double* __restrict__ x, double* __restrict__ r, double* __restrict__ p, double* __restrict__ s,
                       const double* __restrict__ w, double alpha, double beta, int nx, int ny, long ld) {
  FOR_RECT(0, nx, 0, ny) {
    const long k = idx(i, j, ld);
    const double pk = r[k] + beta * p[k];
    const double sk = w[k] + beta * s[k];
    p[k] = pk;
    s[k] = sk;
    x[k] += alpha * pk;
    r[k] -= alpha * sk;
  }
}

__global__ void k_pipe(double* __restrict__ x, double* __restrict__ r, double* __restrict__ p, double* __restrict__ s,
                       double* __restrict__ w, double* __restrict__ z, const double* __restrict__ q, double alpha,
                       double beta, int nx, int ny, long ld, double* partials, int cap, int off) {
  double rr = 0.0, wr = 0.0;
  FOR_RECT(0, nx, 0, ny) {
    const long k = idx(i, j, ld);
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
    rr += rk * rk;
    wr += wk * rk;
  }
  const double v[2] = {rr, wr};
  block_store<2>(v, partials, cap, off);
}

__global__ void k_reduce(const double* __restrict__ partials, int cap, int used, int nc, double* __restrict__ out) {
  __shared__ double sh[kThreads];
  const int t = threadIdx.x;
  for (int c = 0; c < nc; ++c) {
    double s = 0.0;
    for (int k = t; k < used; k += blockDim.x) s += partials[c * cap + k];
    sh[t] = s;
    __syncthreads();
    for (int w = kThreads / 2; w > 0; w >>= 1) {
      if (t < w) sh[t] += sh[t + w];
      __syncthreads();
    }
    if (t == 0) out[c] = sh[0];
    __syncthreads();
  }
}

__global__ void k_pack(double* __restrict__ buf, const double* __restrict__ u, int nx, int ny, long ld) {
  const int total = 2 * (nx + ny);
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += gridDim.x * blockDim.x) {
    long k;
    if (t < ny)
      k = idx(0, t, ld);  // first interior row -> north
    else if (t < 2 * ny)
      k = idx(nx - 1, t - ny, ld);  // last interior row -> south
    else if (t < 2 * ny + nx)
      k = idx(t - 2 * ny, 0, ld);  // first interior column -> west
    else
      k = idx(t - 2 * ny - nx, ny - 1, ld);  // last interior column -> east
    buf[t] = u[k];
  }
}

__global__ void k_unpack(double* __restrict__ u, const double* __restrict__ buf, int nx, int ny, long ld, int mask) {
  const int total = 2 * (nx + ny);
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += gridDim.x * blockDim.x) {
    long k;
    int side;
    if (t < ny) {
      side = kNorth;
      k = idx(-1, t, ld);
    } else if (t < 2 * ny) {
      side = kSouth;
      k = idx(nx, t - ny, ld);
    } else if (t < 2 * ny + nx) {
      side = kWest;
      k = idx(t - 2 * ny, -1, ld);
    } else {
      side = kEast;
      k = idx(t - 2 * ny - nx, ny, ld);
    }
    if (mask & side) u[k] = buf[t];
  }
}

}  // namespace

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
    std::abort();
  }
}

int max_blocks_per_launch() { return kMaxGX * kMaxGY; }

void stencil(double* dst, const double* src, long ld, Rect r, Partials* p, cudaStream_t s) {
  if (r.i1 <= r.i0 || r.j1 <= r.j0) return;
  const dim3 g = grid_for(r.i1 - r.i0, r.j1 - r.j0), b(kBX, kBY);
  const int i0 = static_cast<int>(r.i0), i1 = static_cast<int>(r.i1);
  const int j0 = static_cast<int>(r.j0), j1 = static_cast<int>(r.j1);
  if (p)
    k_stencil<true><<<g, b, 0, s>>>(dst, src, ld, i0, i1, j0, j1, p->data, p->capacity, claim(p, g));
  else
    k_stencil<false><<<g, b, 0, s>>>(dst, src, ld, i0, i1, j0, j1, nullptr, 0, 0);
  check(cudaGetLastError(), "k_stencil");
}

void dot(const double* a, const double* b, Shape sh, Partials* p, cudaStream_t s) {
  const dim3 g = grid_for(sh.nx, sh.ny), blk(kBX, kBY);
  k_dot<<<g, blk, 0, s>>>(a, b, static_cast<int>(sh.nx), static_cast<int>(sh.ny), sh.ld, p->data, p->capacity,
                          claim(p, g));
  check(cudaGetLastError(), "k_dot");
}

void classic_xr(double* x, double* r, const double* p, const double* q, double alpha, Shape sh, Partials* part,
                cudaStream_t s) {
  const dim3 g = grid_for(sh.nx, sh.ny), b(kBX, kBY);
  k_classic_xr<<<g, b, 0, s>>>(x, r, p, q, alpha, static_cast<int>(sh.nx), static_cast<int>(sh.ny), sh.ld,
                               part->data, part->capacity, claim(part, g));
  check(cudaGetLastError(), "k_classic_xr");
}

void classic_p(double* p, const double* r, double beta, Shape sh, cudaStream_t s) {
  const dim3 g = grid_for(sh.nx, sh.ny), b(kBX, kBY);
  k_classic_p<<<g, b, 0, s>>>(p, r, beta, static_cast<int>(sh.nx), static_cast<int>(sh.ny), sh.ld);
  check(cudaGetLastError(), "k_classic_p");
}

void cgcg_update(double* x, double* r, double* p, double* s, const double* w, double alpha, double beta, Shape sh,
                 cudaStream_t stream) {
  const dim3 g = grid_for(sh.nx, sh.ny), b(kBX, kBY);
  k_cgcg<<<g, b, 0, stream>>>(x, r, p, s, w, alpha, beta, static_cast<int>(sh.nx), static_cast<int>(sh.ny), sh.ld);
  check(cudaGetLastError(), "k_cgcg");
}

void pipe_update(double* x, double* r, double* p, double* s, double* w, double* z, const double* q, double alpha,
                 double beta, Shape sh, Partials* part, cudaStream_t stream) {
  const dim3 g = grid_for(sh.nx, sh.ny), b(kBX, kBY);
  k_pipe<<<g, b, 0, stream>>>(x, r, p, s, w, z, q, alpha, beta, static_cast<int>(sh.nx), static_cast<int>(sh.ny),
                              sh.ld, part->data, part->capacity, claim(part, g));
  check(cudaGetLastError(), "k_pipe");
}

void reduce(const Partials& p, int nc, double* out, cudaStream_t s) {
  k_reduce<<<1, kThreads, 0, s>>>(p.data, p.capacity, p.used, nc, out);
  check(cudaGetLastError(), "k_reduce");
}

void pack(double* buf, const double* u, Shape sh, cudaStream_t s) {
  k_pack<<<grid_1d(2 * (sh.nx + sh.ny)), 256, 0, s>>>(buf, u, static_cast<int>(sh.nx), static_cast<int>(sh.ny),
                                                       sh.ld);
  check(cudaGetLastError(), "k_pack");
}

void unpack(double* u, const double* buf, Shape sh, int mask, cudaStream_t s) {
  k_unpack<<<grid_1d(2 * (sh.nx + sh.ny)), 256, 0, s>>>(u, buf, static_cast<int>(sh.nx), static_cast<int>(sh.ny),
                                                         sh.ld, mask);
  check(cudaGetLastError(), "k_unpack");
}

}  // namespace gpu
