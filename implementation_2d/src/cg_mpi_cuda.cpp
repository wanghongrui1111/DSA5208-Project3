// MPI + CUDA solver: one GPU per MPI rank, all vectors resident on the device.
// The CG driver is the same as for the CPU builds; this backend maps every
// vector operation to a kernel and only moves scalars and halo strips.
//
//   --gpu-mpi staged: pack on the GPU, copy to pinned host memory, MPI between
//                     host buffers, copy back, unpack.
//   --gpu-mpi aware:  pack on the GPU and hand the device buffers to a
//                     CUDA-aware MPI directly.
//   --comm blocking:  halo exchange, then one stencil kernel over the block.
//   --comm overlap:   the interior stencil runs on a second stream while the
//                     host does the halo exchange; the frame follows.
//   --reduce repro:   the device partial sums are already deterministic (fixed
//                     launch geometry, two-stage reduction); repro also fixes
//                     the order of the MPI sum (see Reducer).

#include <cuda_runtime.h>

#include "cpu_kernels.hpp"
#include "cuda_kernels.hpp"
#include "mpi_common.hpp"

#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>
#endif

#define CK(call) gpu::check((call), #call)

namespace cg {

namespace {

gpu::Rect to_gpu(const Rect& r) { return {r.i0, r.i1, r.j0, r.j1}; }

// Node-local rank from the launcher's environment, so the GPU can be selected
// before MPI_Init (some CUDA-aware MPI libraries require that). -1 if unknown.
int env_local_rank() {
  for (const char* name : {"OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
                           "PALS_LOCAL_RANKID", "PMI_LOCAL_RANK", "SLURM_LOCALID"})
    if (const char* v = std::getenv(name)) return std::atoi(v);
  return -1;
}

// 1 or 0 if the MPI library can tell whether it is CUDA-aware, -1 otherwise.
int cuda_aware_mpi() {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  return MPIX_Query_cuda_support();
#else
  return -1;
#endif
}

}  // namespace

class MpiCudaBackend {
 public:
  MpiCudaBackend(const ProcGrid& g, const Block& bl, const Options& o)
      : g_(g), bl_(bl), o_(o), sh_{bl.nx, bl.ny, bl.ld()}, red_(g.comm, g.size, o.reduce, &tm_) {
    bytes_ = sizeof(double) * static_cast<size_t>(bl_.size());
    for (auto& p : d_) {
      CK(cudaMalloc(&p, bytes_));
      CK(cudaMemset(p, 0, bytes_));
    }
    halo_bytes_ = sizeof(double) * 2 * static_cast<size_t>(bl_.nx + bl_.ny);
    CK(cudaMalloc(&d_sbuf_, halo_bytes_));
    CK(cudaMalloc(&d_rbuf_, halo_bytes_));
    CK(cudaMallocHost(&h_sbuf_, halo_bytes_));
    CK(cudaMallocHost(&h_rbuf_, halo_bytes_));
    part_.capacity = 6 * gpu::max_blocks_per_launch();
    CK(cudaMalloc(&part_.data, 2 * sizeof(double) * part_.capacity));
    CK(cudaMalloc(&d_out_, 2 * sizeof(double)));
    CK(cudaMallocHost(&h_out_, 2 * sizeof(double)));
    CK(cudaStreamCreateWithFlags(&s_main_, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s_int_, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&ev_int_, cudaEventDisableTiming));

    mask_ = (g_.north != MPI_PROC_NULL ? gpu::kNorth : 0) | (g_.south != MPI_PROC_NULL ? gpu::kSouth : 0) |
            (g_.west != MPI_PROC_NULL ? gpu::kWest : 0) | (g_.east != MPI_PROC_NULL ? gpu::kEast : 0);
    nframe_ = frame_rects(bl_, frame_);

    h_b_ = HostArray(bl_.size());
    h_x_ = HostArray(bl_.size());
    h_q_ = HostArray(bl_.size());
    cpu::init_rhs(bl_, h_b_.data());
    CK(cudaMemcpy(d_[B], h_b_.data(), bytes_, cudaMemcpyHostToDevice));

    // Warm-up: the first launches load the kernels (and JIT-compile the PTX),
    // the first messages set up the MPI connections.
    double a = 0.0, c = 0.0;
    matvec(Q, B, &a, &c);
    allreduce(&a, 1);
  }

  ~MpiCudaBackend() {
    for (auto p : d_) cudaFree(p);
    cudaFree(d_sbuf_);
    cudaFree(d_rbuf_);
    cudaFreeHost(h_sbuf_);
    cudaFreeHost(h_rbuf_);
    cudaFree(part_.data);
    cudaFree(d_out_);
    cudaFreeHost(h_out_);
    cudaEventDestroy(ev_int_);
    cudaStreamDestroy(s_main_);
    cudaStreamDestroy(s_int_);
  }
  MpiCudaBackend(const MpiCudaBackend&) = delete;
  MpiCudaBackend& operator=(const MpiCudaBackend&) = delete;

  void reset() {
    for (Vec v : {X, P, S, W, Z, Q}) CK(cudaMemsetAsync(d_[v], 0, bytes_, s_main_));
    CK(cudaMemcpyAsync(d_[R], d_[B], bytes_, cudaMemcpyDeviceToDevice, s_main_));
    CK(cudaStreamSynchronize(s_main_));
  }

  double dot(Vec a, Vec b) {
    part_.used = 0;
    gpu::dot(d_[a], d_[b], sh_, &part_, s_main_);
    double out = 0.0;
    finish(1, &out);
    return out;
  }

  void matvec(Vec dst, Vec src, double* sd, double* ss) {
    double* q = d_[dst];
    double* u = d_[src];
    const long ld = bl_.ld();
    gpu::Partials* p = (sd || ss) ? &part_ : nullptr;
    part_.used = 0;
    // Let earlier kernels finish outside the timed halo exchange. This also
    // orders the interior launch on s_int_ after them.
    CK(cudaStreamSynchronize(s_main_));
    if (o_.comm == Comm::Blocking) {
      halo(u);
      gpu::stencil(q, u, ld, to_gpu(full_rect(bl_)), p, s_main_);
    } else {
      gpu::stencil(q, u, ld, to_gpu(interior_rect(bl_)), p, s_int_);
      CK(cudaEventRecord(ev_int_, s_int_));
      halo(u);
      for (int k = 0; k < nframe_; ++k) gpu::stencil(q, u, ld, to_gpu(frame_[k]), p, s_main_);
      CK(cudaStreamWaitEvent(s_main_, ev_int_, 0));
    }
    if (p) {
      double out[2];
      finish(2, out);
      if (sd) *sd = out[0];
      if (ss) *ss = out[1];
    }
  }

  double classic_xr(double alpha) {
    part_.used = 0;
    gpu::classic_xr(d_[X], d_[R], d_[P], d_[Q], alpha, sh_, &part_, s_main_);
    double rr = 0.0;
    finish(1, &rr);
    return rr;
  }
  void classic_p(double beta) { gpu::classic_p(d_[P], d_[R], beta, sh_, s_main_); }
  void cgcg_update(double alpha, double beta) {
    gpu::cgcg_update(d_[X], d_[R], d_[P], d_[S], d_[W], alpha, beta, sh_, s_main_);
  }
  void pipe_update(double alpha, double beta, double& rr, double& wr) {
    part_.used = 0;
    gpu::pipe_update(d_[X], d_[R], d_[P], d_[S], d_[W], d_[Z], d_[Q], alpha, beta, sh_, &part_, s_main_);
    double out[2];
    finish(2, out);
    rr = out[0];
    wr = out[1];
  }

  void allreduce(double* v, int n) { red_.sum(v, n); }
  void allreduce_max(double* v, int n) { red_.max(v, n); }
  void iallreduce_begin(double* v, int n) { red_.sum_begin(v, n); }
  void iallreduce_end() { red_.sum_end(); }
  void barrier() { MPI_Barrier(g_.comm); }
  void sync() {
    CK(cudaStreamSynchronize(s_main_));
    CK(cudaStreamSynchronize(s_int_));
  }

  LocalStats final_stats() {
    matvec(Q, X, nullptr, nullptr);
    sync();
    CK(cudaMemcpy(h_x_.data(), d_[X], bytes_, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_q_.data(), d_[Q], bytes_, cudaMemcpyDeviceToHost));
    return cpu::local_stats(bl_, h_x_.data(), h_q_.data(), h_b_.data());
  }

  Timers& timers() { return tm_; }
  int rank() const { return g_.rank; }
  int nranks() const { return g_.size; }

 private:
  // Sums the claimed partials on the device and brings nc scalars to the host.
  void finish(int nc, double* out) {
    gpu::reduce(part_, nc, d_out_, s_main_);
    CK(cudaMemcpyAsync(h_out_, d_out_, nc * sizeof(double), cudaMemcpyDeviceToHost, s_main_));
    CK(cudaStreamSynchronize(s_main_));
    for (int c = 0; c < nc; ++c) out[c] = h_out_[c];
  }

  void halo(double* u) {
    if (mask_ == 0) return;  // single rank: the ghost layer is the Dirichlet boundary
    const double t = now();
    const bool staged = o_.gpu_mpi == GpuMpi::Staged;
    gpu::pack(d_sbuf_, u, sh_, s_main_);
    if (staged) CK(cudaMemcpyAsync(h_sbuf_, d_sbuf_, halo_bytes_, cudaMemcpyDeviceToHost, s_main_));
    CK(cudaStreamSynchronize(s_main_));

    double* sb = staged ? h_sbuf_ : d_sbuf_;
    double* rb = staged ? h_rbuf_ : d_rbuf_;
    const int nx = static_cast<int>(bl_.nx), ny = static_cast<int>(bl_.ny);
    MPI_Request req[8];
    MPI_Irecv(rb, ny, MPI_DOUBLE, g_.north, kToSouth, g_.comm, &req[0]);
    MPI_Irecv(rb + ny, ny, MPI_DOUBLE, g_.south, kToNorth, g_.comm, &req[1]);
    MPI_Irecv(rb + 2 * ny, nx, MPI_DOUBLE, g_.west, kToEast, g_.comm, &req[2]);
    MPI_Irecv(rb + 2 * ny + nx, nx, MPI_DOUBLE, g_.east, kToWest, g_.comm, &req[3]);
    MPI_Isend(sb, ny, MPI_DOUBLE, g_.north, kToNorth, g_.comm, &req[4]);
    MPI_Isend(sb + ny, ny, MPI_DOUBLE, g_.south, kToSouth, g_.comm, &req[5]);
    MPI_Isend(sb + 2 * ny, nx, MPI_DOUBLE, g_.west, kToWest, g_.comm, &req[6]);
    MPI_Isend(sb + 2 * ny + nx, nx, MPI_DOUBLE, g_.east, kToEast, g_.comm, &req[7]);
    MPI_Waitall(8, req, MPI_STATUSES_IGNORE);

    if (staged) CK(cudaMemcpyAsync(d_rbuf_, h_rbuf_, halo_bytes_, cudaMemcpyHostToDevice, s_main_));
    gpu::unpack(u, d_rbuf_, sh_, mask_, s_main_);
    tm_.halo += now() - t;
  }

  ProcGrid g_;
  Block bl_;
  Options o_;
  gpu::Shape sh_;
  Timers tm_;
  Reducer red_;
  size_t bytes_ = 0, halo_bytes_ = 0;
  double* d_[kNumVec] = {};
  double *d_sbuf_ = nullptr, *d_rbuf_ = nullptr, *h_sbuf_ = nullptr, *h_rbuf_ = nullptr;
  gpu::Partials part_;
  double *d_out_ = nullptr, *h_out_ = nullptr;
  cudaStream_t s_main_ = nullptr, s_int_ = nullptr;
  cudaEvent_t ev_int_ = nullptr;
  HostArray h_b_, h_x_, h_q_;
  int mask_ = 0;
  Rect frame_[4];
  int nframe_ = 0;
};

}  // namespace cg

int main(int argc, char** argv) {
  int ndev = 0;
  const bool have_gpu = cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0;
  const int env_rank = cg::env_local_rank();
  if (have_gpu && env_rank >= 0) CK(cudaSetDevice(env_rank % ndev));

  MPI_Init(&argc, &argv);
  const cg::Options o = cg::parse_or_exit(argc, argv);
  if (!have_gpu) cg::fail("no CUDA device visible");
  if (o.comm != cg::Comm::Blocking && o.comm != cg::Comm::Overlap)
    cg::fail(std::string("--comm ") + cg::to_string(o.comm) + " is only implemented in cg_mpi_omp");
  if (o.gpu_mpi == cg::GpuMpi::Aware && cg::cuda_aware_mpi() == 0)
    cg::fail("--gpu-mpi aware, but this MPI is not CUDA-aware at run time (Open MPI: set OMPI_MCA_opal_cuda_support=true)");

  cg::ProcGrid g = cg::make_proc_grid(o.decomp);
  cg::check_grid(g, o.n);
  if (env_rank < 0) CK(cudaSetDevice(g.local_rank % ndev));
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));

  int gpus = g.local_rank == 0 ? std::min(g.local_size, ndev) : 0;
  MPI_Allreduce(MPI_IN_PLACE, &gpus, 1, MPI_INT, MPI_SUM, g.comm);
  if (g.rank == 0 && g.local_size > ndev)
    std::fprintf(stderr, "warning: %d ranks share %d GPU(s) on a node\n", g.local_size, ndev);

  int rc = 0;
  {
    cg::MpiCudaBackend bk(g, cg::make_block(g, o.n), o);
    cg::RunInfo info;
    info.impl = "mpi_cuda";
    cg::fill_run_info(info, g, o);
    info.gpus = gpus;
    info.gpu_mpi = cg::to_string(o.gpu_mpi);
    info.device = prop.name;
    rc = cg::run(bk, o, info);
  }
  MPI_Comm_free(&g.comm);
  MPI_Finalize();
  return rc;
}
