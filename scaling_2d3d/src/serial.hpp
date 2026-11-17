// serial.hpp -- reference serial classic CG for the 2D / 3D Poisson problem (no MPI, no OpenMP).
// (a) serial baseline for runtime-vs-size (E1), (b) convergence-order tests (V1/V2), (c) reference solution for V3.
#pragma once
#include <vector>

#include "problem.hpp"

struct SerialResult {
  int iters = 0;
  double relres = 0;  // ||r|| / ||b|| from the CG recurrence
  bool converged = false;
  double t_setup = 0, t_solve = 0;  // [s]
  double err_inf = 0, err_l2 = 0;
  long long dof = 0, nnz = 0;
};

// Solves on the full n^dim grid.  If u_out != nullptr it receives the padded global solution
// (layout of problem.hpp with l = {n,n,n or 1}).  tol <= 0 => run exactly maxit iterations.
SerialResult serial_cg(ProblemKind kind, int dim, int n, double tol, int maxit, std::vector<double>* u_out);
