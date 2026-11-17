// Serial reference solver: the whole n x n grid is one block, no MPI and no
// threads. This is the baseline T_s for speedup and efficiency.

#include "cg_driver.hpp"
#include "cpu_kernels.hpp"

namespace cg {

class SerialBackend {
 public:
  explicit SerialBackend(long n) {
    bl_.n = n;
    bl_.nx = n;
    bl_.ny = n;
    for (auto& v : v_) {
      v = HostArray(bl_.size());
      cpu::fill(bl_, v.data(), 0.0);
    }
    cpu::init_rhs(bl_, v_[B].data());
  }

  void reset() {
    for (Vec v : {X, P, S, W, Z, Q}) cpu::fill(bl_, v_[v].data(), 0.0);
    cpu::copy(bl_, v_[R].data(), v_[B].data());
  }

  double dot(Vec a, Vec b) { return cpu::dot(bl_, v_[a].data(), v_[b].data()); }

  void matvec(Vec dst, Vec src, double* sd, double* ss) {
    double a = 0.0, c = 0.0;
    if (sd || ss)
      cpu::stencil<true>(bl_, v_[dst].data(), v_[src].data(), full_rect(bl_), a, c);
    else
      cpu::stencil<false>(bl_, v_[dst].data(), v_[src].data(), full_rect(bl_), a, c);
    if (sd) *sd = a;
    if (ss) *ss = c;
  }

  double classic_xr(double alpha) { return cpu::classic_xr(bl_, d(X), d(R), d(P), d(Q), alpha); }
  void classic_p(double beta) { cpu::classic_p(bl_, d(P), d(R), beta); }
  void cgcg_update(double alpha, double beta) { cpu::cgcg_update(bl_, d(X), d(R), d(P), d(S), d(W), alpha, beta); }
  void pipe_update(double alpha, double beta, double& rr, double& wr) {
    cpu::pipe_update(bl_, d(X), d(R), d(P), d(S), d(W), d(Z), d(Q), alpha, beta, rr, wr);
  }

  void allreduce(double*, int) {}
  void allreduce_max(double*, int) {}
  void iallreduce_begin(double*, int) {}
  void iallreduce_end() {}
  void barrier() {}
  void sync() {}

  LocalStats final_stats() {
    matvec(Q, X, nullptr, nullptr);
    return cpu::local_stats(bl_, d(X), d(Q), d(B));
  }

  Timers& timers() { return tm_; }
  int rank() const { return 0; }
  int nranks() const { return 1; }

 private:
  double* d(Vec v) { return v_[v].data(); }

  Block bl_;
  HostArray v_[kNumVec];
  Timers tm_;
};

}  // namespace cg

int main(int argc, char** argv) {
  cg::Options o;
  try {
    o = cg::parse_options(argc, argv);
  } catch (const cg::HelpRequested&) {
    std::printf("usage: %s [options]\n%s", argv[0], cg::usage_text());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n%s", e.what(), cg::usage_text());
    return 1;
  }
  cg::SerialBackend bk(o.n);
  cg::RunInfo info;
  info.impl = "serial";
  return cg::run(bk, o, info);
}
