// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
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
// Author: markshachkov@gmail.com (Mark Shachkov)

#include "ceres/power_series_expansion_solver.h"

#include "ceres/implicit_schur_complement.h"
#include "ceres/internal/eigen.h"
#include "ceres/stringprintf.h"
#include "ceres/types.h"
#include "glog/logging.h"
namespace ceres::internal {

PowerSeriesExpansionSolver::PowerSeriesExpansionSolver(
    LinearSolver::Options options)
    : options_(std::move(options)) {}

LinearSolver::Summary PowerSeriesExpansionSolver::Solve(
    LinearOperator* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double* x) {
  CHECK(A != nullptr);
  CHECK(x != nullptr);
  CHECK(b != nullptr);
  CHECK_EQ(A->num_rows(), A->num_cols());
  auto A_schur = down_cast<ImplicitSchurComplement*>(A);

  LinearSolver::Summary summary;
  summary.termination_type = LinearSolverTerminationType::NO_CONVERGENCE;
  summary.message = "Maximum number of iterations reached.";
  summary.num_iterations = 0;

  const int num_cols = A->num_cols();
  VectorRef xref(x, num_cols);
  ConstVectorRef bref(b, num_cols);

  const double norm_b = bref.norm();
  if (norm_b == 0.0) {
    xref.setZero();
    summary.termination_type = LinearSolverTerminationType::SUCCESS;
    summary.message = "Convergence. |b| = 0.";
    return summary;
  }

  Matrix temp_storage;
  temp_storage.resize(3, num_cols);
  VectorRef b_init(temp_storage.row(0).data(), num_cols);
  VectorRef b_temp(temp_storage.row(1).data(), num_cols);
  VectorRef b_temp_previous(temp_storage.row(2).data(), num_cols);

  b_init.setZero();
  A_schur->block_diagonal_FtF_inverse()->RightMultiply(b, b_init.data());
  b_temp_previous = b_init;
  xref = b_init;

  double norm_b_temp;
  const double norm_b_init = b_init.norm();
  const double norm_threshold = per_solve_options.e_tolerance * norm_b_init;
  for (summary.num_iterations = 1;; ++summary.num_iterations) {
    b_temp.setZero();
    A_schur->RightMultiply_Z(b_temp_previous.data(), b_temp.data());
    xref += b_temp;
    norm_b_temp = b_temp.norm();
    if (norm_b_temp < norm_threshold &&
        summary.num_iterations >= options_.min_num_iterations) {
      summary.termination_type = LinearSolverTerminationType::SUCCESS;
      summary.message = StringPrintf(
          "Iteration: %d Convergence: |b_temp| < e_tolerance * |b_init|. "
          "e_tolerance = %e, |b_temp| = %e, "
          "|b_init| = %e",
          summary.num_iterations,
          per_solve_options.e_tolerance,
          norm_b_temp,
          norm_b_init);
      break;
    }

    if (summary.num_iterations >= options_.max_num_iterations) {
      break;
    }
    b_temp_previous = b_temp;
  }

  return summary;
}

}  // namespace ceres::internal
