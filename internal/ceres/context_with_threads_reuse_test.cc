// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2018 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: vitus@google.com (Michael Vitus)

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
