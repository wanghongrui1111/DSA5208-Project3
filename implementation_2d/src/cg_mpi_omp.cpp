// MPI + OpenMP solver. MPI ranks own blocks of a 1d or 2d process grid; the
// OpenMP threads of a rank share its loops. Only the master thread calls MPI
// (MPI_THREAD_FUNNELED). The halo of the vector in each matrix-vector product
// is exchanged in one of five ways (--comm):
//
//   blocking:  four MPI_Sendrecv, then the stencil over the whole block.
//   overlap:   post Irecv/Isend, sweep the interior (which never reads ghosts)
//              in chunks with MPI_Testall in between to drive progress, wait,
//              then sweep the one-point frame.
//   thread:    one parallel region: the master thread posts and waits for the
//              messages while the other threads sweep the interior, then all
//              threads sweep the frame.
//   rma:       one-sided: MPI_Put of the boundary strips into the neighbours'
//              ghost layers inside an MPI_Win_fence epoch, the interior swept
//              before the closing fence.
//   allgather: MPI_Allgatherv of the whole vector to every rank (the design in
//              which every rank holds a full copy), then the stencil.

#include <omp.h>

#include <cstring>

#include "cpu_kernels.hpp"
#include "mpi_common.hpp"

namespace cg {

class MpiOmpBackend {
 public:
  MpiOmpBackend(const ProcGrid& g, const Block& bl, const Options& o)
      : g_(g), bl_(bl), o_(o), red_(g.comm, g.size, o.reduce, &tm_) {
    for (auto& v : v_) {
      v = HostArray(bl_.size());
      cpu::fill(bl_, v.data(), 0.0);
    }
    cpu::init_rhs(bl_, v_[B].data());
    MPI_Type_vector(static_cast<int>(bl_.nx), 1, static_cast<int>(bl_.ld()), MPI_DOUBLE, &column_);
    MPI_Type_commit(&column_);
    nframe_ = frame_rects(bl_, frame_);
    if (o_.reduce == Reduce::Repro) rows_ = HostArray(2 * (3 * bl_.nx + 8));
    if (o_.comm == Comm::Rma) setup_rma();
    if (o_.comm == Comm::Allgather) setup_allgather();
    // Warm-up: the first messages set up the MPI connections.
    double a = 0.0, c = 0.0;
    matvec(Q, B, &a, &c);
    allreduce(&a, 1);
  }

  ~MpiOmpBackend() {
    for (auto& w : win_)
      if (w != MPI_WIN_NULL) MPI_Win_free(&w);
    for (MPI_Datatype* t : {&column_, &column_west_, &column_east_})
      if (*t != MPI_DATATYPE_NULL) MPI_Type_free(t);
  }
  MpiOmpBackend(const MpiOmpBackend&) = delete;
  MpiOmpBackend& operator=(const MpiOmpBackend&) = delete;

  void reset() {
    for (Vec v : {X, P, S, W, Z, Q}) cpu::fill(bl_, d(v), 0.0);
    cpu::copy(bl_, d(R), d(B));
  }

  double dot(Vec a, Vec b) { return cpu::dot(bl_, d(a), d(b), rows()); }

  void matvec(Vec dst, Vec src, double* sd, double* ss) {
    double* q = d(dst);
    double* u = d(src);
    const bool dots = sd || ss;
    double a = 0.0, c = 0.0;
    double t = now();
    switch (o_.comm) {
      case Comm::Blocking:
        halo_blocking(u);
        tm_.halo += now() - t;
        sweep(q, u, full_rect(bl_), dots, a, c);
        break;
      case Comm::Allgather:
        halo_allgather(u);
        tm_.halo += now() - t;
        sweep(q, u, full_rect(bl_), dots, a, c);
        break;
      case Comm::Overlap: {
        MPI_Request req[8];
        halo_post(u, req);
        tm_.halo += now() - t;
        sweep_interior(q, u, dots, a, c, req);
        t = now();
        MPI_Waitall(8, req, MPI_STATUSES_IGNORE);
        tm_.halo += now() - t;
        for (int k = 0; k < nframe_; ++k) sweep(q, u, frame_[k], dots, a, c);
        break;
      }
      case Comm::Rma: {
        MPI_Win w = win_[src];
        MPI_Win_fence(MPI_MODE_NOPRECEDE, w);
        halo_put(u, w);
        tm_.halo += now() - t;
        sweep_interior(q, u, dots, a, c, nullptr);
        t = now();
        MPI_Win_fence(MPI_MODE_NOSUCCEED, w);
        tm_.halo += now() - t;
        for (int k = 0; k < nframe_; ++k) sweep(q, u, frame_[k], dots, a, c);
        break;
      }
      case Comm::Thread:
        if (dots)
          matvec_thread<true>(q, u, a, c);
        else
          matvec_thread<false>(q, u, a, c);
        break;
    }
    if (sd) *sd = a;
    if (ss) *ss = c;
  }

  double classic_xr(double alpha) { return cpu::classic_xr(bl_, d(X), d(R), d(P), d(Q), alpha, rows()); }
  void classic_p(double beta) { cpu::classic_p(bl_, d(P), d(R), beta); }
  void cgcg_update(double alpha, double beta) { cpu::cgcg_update(bl_, d(X), d(R), d(P), d(S), d(W), alpha, beta); }
  void pipe_update(double alpha, double beta, double& rr, double& wr) {
    cpu::pipe_update(bl_, d(X), d(R), d(P), d(S), d(W), d(Z), d(Q), alpha, beta, rr, wr, rows());
  }

  void allreduce(double* v, int n) { red_.sum(v, n); }
  void allreduce_max(double* v, int n) { red_.max(v, n); }
  void iallreduce_begin(double* v, int n) { red_.sum_begin(v, n); }
  void iallreduce_end() { red_.sum_end(); }
  void barrier() { MPI_Barrier(g_.comm); }
  void sync() {}

  LocalStats final_stats() {
    matvec(Q, X, nullptr, nullptr);
    return cpu::local_stats(bl_, d(X), d(Q), d(B));
  }

  Timers& timers() { return tm_; }
  int rank() const { return g_.rank; }
  int nranks() const { return g_.size; }

 private:
  double* d(Vec v) { return v_[v].data(); }
  // Scratch for row partial sums in repro mode, nullptr in fast mode.
  double* rows() { return o_.reduce == Reduce::Repro ? rows_.data() : nullptr; }

  void sweep(double* q, const double* u, const Rect& r, bool dots, double& a, double& c) {
    if (dots)
      cpu::stencil<true>(bl_, q, u, r, a, c, rows());
    else
      cpu::stencil<false>(bl_, q, u, r, a, c, rows());
  }

  // The interior in o_.chunks row blocks; with req, MPI_Testall after each
  // block so that the library can progress the messages.
  void sweep_interior(double* q, const double* u, bool dots, double& a, double& c, MPI_Request* req) {
    const Rect in = interior_rect(bl_);
    if (in.empty()) return;
    const long step = (in.i1 - in.i0 + o_.chunks - 1) / o_.chunks;
    for (long i = in.i0; i < in.i1; i += step) {
      sweep(q, u, Rect{i, std::min(i + step, in.i1), in.j0, in.j1}, dots, a, c);
      if (req) {
        const double t = now();
        int done = 0;
        MPI_Testall(8, req, &done, MPI_STATUSES_IGNORE);
        tm_.halo += now() - t;
      }
    }
  }

  // --comm thread. tm_.halo gets the master's time in MPI, which here runs
  // concurrently with the other threads' work.
  template <bool Dots>
  void matvec_thread(double* q, double* u, [[maybe_unused]] double& sd, [[maybe_unused]] double& ss) {
    const Rect in = interior_rect(bl_);
    // Repro scratch: the interior's rows first, then each frame rectangle's.
    double* scratch = rows();
    double* part[5] = {};
    long nrows[5] = {in.empty() ? 0 : in.i1 - in.i0};
    for (int k = 0; k < nframe_; ++k) nrows[k + 1] = frame_[k].i1 - frame_[k].i0;
    if (scratch)
      for (int k = 0, off = 0; k <= nframe_; off += 2 * nrows[k], ++k) part[k] = scratch + off;
    double a = 0.0, c = 0.0, t_mpi = 0.0;
#pragma omp parallel reduction(+ : a, c)
    {
#pragma omp master
      {
        const double t = now();
        MPI_Request req[8];
        halo_post(u, req);
        MPI_Waitall(8, req, MPI_STATUSES_IGNORE);
        t_mpi = now() - t;
      }
      cpu::stencil_shared<Dots>(bl_, q, u, in, a, c, part[0]);
#pragma omp barrier
      for (int k = 0; k < nframe_; ++k) cpu::stencil_shared<Dots>(bl_, q, u, frame_[k], a, c, part[k + 1]);
    }
    tm_.halo += t_mpi;
    if constexpr (Dots) {
      double s[2] = {a, c};
      if (scratch)
        for (int k = 0; k <= nframe_; ++k) cpu::add_rows<2>(part[k], nrows[k], s);
      sd += s[0];
      ss += s[1];
    }
  }

  // Storage offsets of the strips: rows are contiguous, columns use column_.
  long top_row() const { return bl_.ld() + 1; }
  long bottom_row() const { return bl_.nx * bl_.ld() + 1; }
  long top_ghost() const { return 1; }
  long bottom_ghost() const { return (bl_.nx + 1) * bl_.ld() + 1; }
  long left_col() const { return bl_.ld() + 1; }
  long right_col() const { return bl_.ld() + bl_.ny; }
  long left_ghost() const { return bl_.ld(); }
  long right_ghost() const { return bl_.ld() + bl_.ny + 1; }

  void halo_blocking(double* u) {
    const int ny = static_cast<int>(bl_.ny);
    MPI_Sendrecv(u + top_row(), ny, MPI_DOUBLE, g_.north, kToNorth, u + bottom_ghost(), ny, MPI_DOUBLE, g_.south,
                 kToNorth, g_.comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(u + bottom_row(), ny, MPI_DOUBLE, g_.south, kToSouth, u + top_ghost(), ny, MPI_DOUBLE, g_.north,
                 kToSouth, g_.comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(u + left_col(), 1, column_, g_.west, kToWest, u + right_ghost(), 1, column_, g_.east, kToWest,
                 g_.comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(u + right_col(), 1, column_, g_.east, kToEast, u + left_ghost(), 1, column_, g_.west, kToEast,
                 g_.comm, MPI_STATUS_IGNORE);
  }

  void halo_post(double* u, MPI_Request req[8]) {
    const int ny = static_cast<int>(bl_.ny);
    MPI_Irecv(u + top_ghost(), ny, MPI_DOUBLE, g_.north, kToSouth, g_.comm, &req[0]);
    MPI_Irecv(u + bottom_ghost(), ny, MPI_DOUBLE, g_.south, kToNorth, g_.comm, &req[1]);
    MPI_Irecv(u + left_ghost(), 1, column_, g_.west, kToEast, g_.comm, &req[2]);
    MPI_Irecv(u + right_ghost(), 1, column_, g_.east, kToWest, g_.comm, &req[3]);
    MPI_Isend(u + top_row(), ny, MPI_DOUBLE, g_.north, kToNorth, g_.comm, &req[4]);
    MPI_Isend(u + bottom_row(), ny, MPI_DOUBLE, g_.south, kToSouth, g_.comm, &req[5]);
    MPI_Isend(u + left_col(), 1, column_, g_.west, kToWest, g_.comm, &req[6]);
    MPI_Isend(u + right_col(), 1, column_, g_.east, kToEast, g_.comm, &req[7]);
  }

  // --comm rma: one window per vector. A put needs the target's offsets, so
  // the shapes of the neighbouring blocks are worked out here: north/south
  // neighbours share our columns (same ld), west/east neighbours our rows.
  void setup_rma() {
    const MPI_Aint bytes = static_cast<MPI_Aint>(sizeof(double)) * bl_.size();
    for (int v = 0; v < kNumVec; ++v)
      MPI_Win_create(d(static_cast<Vec>(v)), bytes, sizeof(double), MPI_INFO_NULL, g_.comm, &win_[v]);
    long start = 0, count = 0;
    if (g_.north != MPI_PROC_NULL) {
      partition(bl_.n, g_.dims[0], g_.coords[0] - 1, start, count);
      north_bottom_ghost_ = (count + 1) * bl_.ld() + 1;
    }
    if (g_.west != MPI_PROC_NULL) {
      partition(bl_.n, g_.dims[1], g_.coords[1] - 1, start, count);
      west_right_ghost_ = (count + 2) + count + 1;  // row 1, column count + 1 of a block with ld = count + 2
      MPI_Type_vector(static_cast<int>(bl_.nx), 1, static_cast<int>(count + 2), MPI_DOUBLE, &column_west_);
      MPI_Type_commit(&column_west_);
    }
    if (g_.east != MPI_PROC_NULL) {
      partition(bl_.n, g_.dims[1], g_.coords[1] + 1, start, count);
      east_left_ghost_ = count + 2;  // row 1, column 0
      MPI_Type_vector(static_cast<int>(bl_.nx), 1, static_cast<int>(count + 2), MPI_DOUBLE, &column_east_);
      MPI_Type_commit(&column_east_);
    }
  }

  void halo_put(const double* u, MPI_Win w) {
    const int ny = static_cast<int>(bl_.ny);
    if (g_.north != MPI_PROC_NULL)
      MPI_Put(u + top_row(), ny, MPI_DOUBLE, g_.north, north_bottom_ghost_, ny, MPI_DOUBLE, w);
    if (g_.south != MPI_PROC_NULL) MPI_Put(u + bottom_row(), ny, MPI_DOUBLE, g_.south, top_ghost(), ny, MPI_DOUBLE, w);
    if (g_.west != MPI_PROC_NULL) MPI_Put(u + left_col(), 1, column_, g_.west, west_right_ghost_, 1, column_west_, w);
    if (g_.east != MPI_PROC_NULL) MPI_Put(u + right_col(), 1, column_, g_.east, east_left_ghost_, 1, column_east_, w);
  }

  // --comm allgather: rank r's rows sit at storage rows i0_r + 1 ... of the
  // full (n + 2) x (n + 2) array, whose boundary rows stay zero.
  void setup_allgather() {
    if (o_.decomp != Decomp::OneD) fail("--comm allgather needs --decomp 1d");
    full_ = HostArray((bl_.n + 2) * bl_.ld());
    std::memset(full_.data(), 0, sizeof(double) * full_.size());
    counts_.resize(g_.size);
    displs_.resize(g_.size);
    for (int r = 0; r < g_.size; ++r) {
      long start = 0, count = 0;
      partition(bl_.n, g_.size, r, start, count);
      counts_[r] = static_cast<int>(count * bl_.ld());
      displs_[r] = static_cast<int>((start + 1) * bl_.ld());
    }
  }

  void halo_allgather(double* u) {
    const long ld = bl_.ld();
    MPI_Allgatherv(u + ld, counts_[g_.rank], MPI_DOUBLE, full_.data(), counts_.data(), displs_.data(), MPI_DOUBLE,
                   g_.comm);
    // Of the whole vector, the stencil only needs the rows next to ours.
    if (g_.north != MPI_PROC_NULL) std::memcpy(u, full_.data() + bl_.i0 * ld, sizeof(double) * ld);
    if (g_.south != MPI_PROC_NULL)
      std::memcpy(u + (bl_.nx + 1) * ld, full_.data() + (bl_.i0 + bl_.nx + 1) * ld, sizeof(double) * ld);
  }

  ProcGrid g_;
  Block bl_;
  Options o_;
  Timers tm_;
  Reducer red_;
  HostArray v_[kNumVec];
  HostArray rows_, full_;
  MPI_Datatype column_ = MPI_DATATYPE_NULL, column_west_ = MPI_DATATYPE_NULL, column_east_ = MPI_DATATYPE_NULL;
  MPI_Win win_[kNumVec] = {MPI_WIN_NULL, MPI_WIN_NULL, MPI_WIN_NULL, MPI_WIN_NULL,
                           MPI_WIN_NULL, MPI_WIN_NULL, MPI_WIN_NULL, MPI_WIN_NULL};
  MPI_Aint north_bottom_ghost_ = 0, west_right_ghost_ = 0, east_left_ghost_ = 0;
  std::vector<int> counts_, displs_;
  Rect frame_[4];
  int nframe_ = 0;
};

}  // namespace cg

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  const cg::Options o = cg::parse_or_exit(argc, argv);
  if (provided < MPI_THREAD_FUNNELED) cg::fail("MPI library does not provide MPI_THREAD_FUNNELED");

  cg::ProcGrid g = cg::make_proc_grid(o.decomp);
  cg::check_grid(g, o.n);
  int rc = 0;
  {
    cg::MpiOmpBackend bk(g, cg::make_block(g, o.n), o);
    cg::RunInfo info;
    info.impl = "mpi_omp";
    cg::fill_run_info(info, g, o);
    info.threads = omp_get_max_threads();
    rc = cg::run(bk, o, info);
  }
  MPI_Comm_free(&g.comm);
  MPI_Finalize();
  return rc;
}
