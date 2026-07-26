// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2026 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Author: Sergiu Deitsch

#include "ceres/mkl_sparse_qr_solver.h"

#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/context_impl.h"
#include "ceres/internal/config.h"
#include "ceres/types.h"
#include "gtest/gtest.h"

namespace ceres::internal {

TEST(MKLSparseQRSolver, SolvesOverdeterminedSystem) {
#ifdef CERES_NO_MKL
  GTEST_SKIP() << "MKL Sparse QR support is not enabled.";
#else
  CompressedRowSparseMatrix matrix(3, 2, 6);
  matrix.mutable_rows()[0] = 0;
  matrix.mutable_rows()[1] = 2;
  matrix.mutable_rows()[2] = 4;
  matrix.mutable_rows()[3] = 6;
  matrix.mutable_cols()[0] = 0;
  matrix.mutable_cols()[1] = 1;
  matrix.mutable_cols()[2] = 0;
  matrix.mutable_cols()[3] = 1;
  matrix.mutable_cols()[4] = 0;
  matrix.mutable_cols()[5] = 1;
  matrix.mutable_values()[0] = 1.0;
  matrix.mutable_values()[1] = 0.0;
  matrix.mutable_values()[2] = 0.0;
  matrix.mutable_values()[3] = 1.0;
  matrix.mutable_values()[4] = 1.0;
  matrix.mutable_values()[5] = 1.0;

  LinearSolver::Options options;
  options.type = MKL_SPARSE_QR;
  options.sparse_linear_algebra_library_type = MKL_SPARSE;
  options.dynamic_sparsity = true;
  options.num_threads = 2;
  ContextImpl context;
  options.context = &context;
  MKLSparseQRSolver solver(options);

  const double rhs[] = {2.0, -1.0, 1.0};
  double solution[] = {0.0, 0.0};
  const LinearSolver::Summary summary =
      solver.Solve(&matrix, rhs, LinearSolver::PerSolveOptions(), solution);

  EXPECT_EQ(summary.termination_type, LinearSolverTerminationType::SUCCESS);
  EXPECT_NEAR(solution[0], 2.0, 1e-12);
  EXPECT_NEAR(solution[1], -1.0, 1e-12);
#endif
}

}  // namespace ceres::internal
