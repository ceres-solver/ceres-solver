// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2026 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Author: Sergiu Deitsch

#ifndef CERES_INTERNAL_MKL_SPARSE_QR_SOLVER_H_
#define CERES_INTERNAL_MKL_SPARSE_QR_SOLVER_H_

#include "ceres/internal/config.h"
#include "ceres/internal/export.h"
#include "ceres/linear_solver.h"

namespace ceres::internal {

class CERES_NO_EXPORT MKLSparseQRSolver final
    : public CompressedRowSparseMatrixSolver {
 public:
  explicit MKLSparseQRSolver(const LinearSolver::Options& options);

 private:
  LinearSolver::Options options_;

  LinearSolver::Summary SolveImpl(
      CompressedRowSparseMatrix* A,
      const double* b,
      const LinearSolver::PerSolveOptions& per_solve_options,
      double* x) final;
};

}  // namespace ceres::internal

#endif  // CERES_INTERNAL_MKL_SPARSE_QR_SOLVER_H_
