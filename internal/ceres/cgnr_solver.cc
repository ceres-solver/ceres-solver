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
// Author: keir@google.com (Keir Mierle)

#include "ceres/cgnr_solver.h"

#include <memory>
#include <utility>

#include "ceres/block_jacobi_preconditioner.h"
#include "ceres/cgnr_linear_operator.h"
#include "ceres/conjugate_gradients_solver.h"
#include "ceres/internal/eigen.h"
#include "ceres/linear_solver.h"
#include "ceres/subset_preconditioner.h"
#include "ceres/wall_time.h"
#include "glog/logging.h"

namespace ceres::internal {

CgnrSolver::CgnrSolver(LinearSolver::Options options)
    : options_(std::move(options)) {
  if (options_.preconditioner_type != JACOBI &&
      options_.preconditioner_type != IDENTITY &&
      options_.preconditioner_type != SUBSET) {
    LOG(FATAL)
        << "Preconditioner = "
        << PreconditionerTypeToString(options_.preconditioner_type) << ". "
        << "Congratulations, you found a bug in Ceres. Please report it.";
  }
}

CgnrSolver::~CgnrSolver() = default;

LinearSolver::Summary CgnrSolver::SolveImpl(
    BlockSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double* x) {
  EventLogger event_logger("CgnrSolver::Solve");

  // Form z = Atb.
  Vector rhs(A->num_cols());
  rhs.setZero();
  A->LeftMultiply(b, rhs.data());

  if (!preconditioner_) {
    if (options_.preconditioner_type == JACOBI) {
      preconditioner_ = std::make_unique<BlockJacobiPreconditioner>(*A);
    } else if (options_.preconditioner_type == SUBSET) {
      Preconditioner::Options preconditioner_options;
      preconditioner_options.type = SUBSET;
      preconditioner_options.subset_preconditioner_start_row_block =
          options_.subset_preconditioner_start_row_block;
      preconditioner_options.sparse_linear_algebra_library_type =
          options_.sparse_linear_algebra_library_type;
      preconditioner_options.ordering_type = options_.ordering_type;
      preconditioner_options.num_threads = options_.num_threads;
      preconditioner_options.context = options_.context;
      preconditioner_ =
          std::make_unique<SubsetPreconditioner>(preconditioner_options, *A);
    } else {
      preconditioner_ = std::make_unique<IdentityPreconditioner>(A->num_cols());
    }
  }

  preconditioner_->Update(*A, per_solve_options.D);

  ConjugateGradientsSolverOptions cg_options;
  cg_options.min_num_iterations = options_.min_num_iterations;
  cg_options.max_num_iterations = options_.max_num_iterations;
  cg_options.residual_reset_period = options_.residual_reset_period;
  cg_options.q_tolerance = per_solve_options.q_tolerance;
  cg_options.r_tolerance = per_solve_options.r_tolerance;

  // Solve (AtA + DtD)x = Atb.
  CgnrLinearOperator cgnr_lhs(*A, per_solve_options.D);
  LinearOperatorToEigenVectorAdapter lhs(cgnr_lhs);
  LinearOperatorToEigenVectorAdapter preconditioner(*preconditioner_);

  Vector cg_solution = Vector::Zero(A->num_cols());
  Vector scratch[4];
  for (int i = 0; i < 4; ++i) {
    scratch[i] = cg_solution;
  }
  event_logger.AddEvent("Setup");

  auto summary = ConjugateGradientsSolver(
      cg_options, lhs, rhs, preconditioner, scratch, cg_solution);
  VectorRef(x, A->num_cols()) = cg_solution;
  event_logger.AddEvent("Solve");
  return summary;
}

}  // namespace ceres::internal
