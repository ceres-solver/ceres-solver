
#include <iostream>

#include "bundle_adjustment_test_util.h"
#include "ceres/internal/config.h"

#ifndef CERES_NO_THREADS

namespace ceres::internal {

TEST_F(BundleAdjustmentTest,
       ReuseContextWithThreads) {  // NOLINT
  BundleAdjustmentProblem bundle_adjustment_problem;
  Solver::Options* options = bundle_adjustment_problem.mutable_solver_options();
  options->num_threads = 4;
  options->linear_solver_type = DENSE_SCHUR;
  options->dense_linear_algebra_library_type = EIGEN;
  options->sparse_linear_algebra_library_type = NO_SPARSE;
  options->preconditioner_type = IDENTITY;
  if (kUserOrdering) {
    options->linear_solver_ordering = nullptr;
  }
  Problem* problem = bundle_adjustment_problem.mutable_problem();
  RunSolverForConfigAndExpectResidualsMatch(*options, problem);

  // Solve a second time re-using the context but use a smaller number of
  // threads. This will catch accessing the parallel for temp space out of range
  // if we don't handle it appropriately.
  options->num_threads = 2;
  RunSolverForConfigAndExpectResidualsMatch(*options, problem);
}

}  // namespace ceres::internal

#endif  // CERES_NO_THREADS
