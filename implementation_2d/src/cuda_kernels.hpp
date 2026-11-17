#pragma once
// Device kernels of the MPI+CUDA solver behind a plain C++ interface, so the
// host driver (cg_mpi_cuda.cpp) is compiled by the ordinary C++ compiler and
// only this file's implementation needs nvcc. Index conventions match
// cg::Block: local point (i, j) lives at (i + 1) * ld + (j + 1).

#include <cuda_runtime_api.h>

namespace gpu {

struct Shape {
  long nx, ny, ld;
};

// Half-open rectangle of local interior points, as cg::Rect.
struct Rect {
  long i0, i1, j0, j1;
};

// Scratch space for deterministic two-stage reductions. Each block of a launch
// writes its partial sums to its own slot; component c of slot k lives at
// data[c * capacity + k]. Launches claim consecutive slots via `used`.
struct Partials {
  double* data = nullptr;
  int capacity = 0;
  int used = 0;
};

enum HaloSide { kNorth = 1, kSouth = 2, kWest = 4, kEast = 8 };

void check(cudaError_t err, const char* what);
int max_blocks_per_launch();

// dst = A src on r. With p: partial (src, dst) and (src, src), components 0, 1.
void stencil(double* dst, const double* src, long ld, Rect r, Partials* p, cudaStream_t s);
// Partial (a, b), component 0.
void dot(const double* a, const double* b, Shape sh, Partials* p, cudaStream_t s);
// x += alpha p, r -= alpha q; partial (r, r).
void classic_xr(double* x, double* r, const double* p, const double* q, double alpha, Shape sh, Partials* part,
                cudaStream_t s);
// p = r + beta p.
void classic_p(double* p, const double* r, double beta, Shape sh, cudaStream_t s);
// p = r + beta p, s = w + beta s, x += alpha p, r -= alpha s.
void cgcg_update(double* x, double* r, double* p, double* s, const double* w, double alpha, double beta, Shape sh,
                 cudaStream_t stream);
// z = q + beta z, s = w + beta s, p = r + beta p, x += alpha p, r -= alpha s,
// w -= alpha z; partial (r, r) and (w, r).
void pipe_update(double* x, double* r, double* p, double* s, double* w, double* z, const double* q, double alpha,
                 double beta, Shape sh, Partials* part, cudaStream_t stream);
// out[c] = sum over the used slots of component c, for c < nc.
void reduce(const Partials& p, int nc, double* out, cudaStream_t s);

// Halo strips in one buffer: [north row (ny) | south row (ny) | west column (nx) | east column (nx)].
// pack copies the outermost interior strips of u into buf; unpack copies buf
// into the ghost layer for the sides set in mask.
void pack(double* buf, const double* u, Shape sh, cudaStream_t s);
void unpack(double* u, const double* buf, Shape sh, int mask, cudaStream_t s);

}  // namespace gpu
