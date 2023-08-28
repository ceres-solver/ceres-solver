// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
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
// Authors: dmitriy.korchemkin@gmail.com (Dmitriy Korchemkin)

#include "ceres/cuda_iterative_schur_complement_solver.h"

#ifndef CERES_NO_CUDA
#include "ceres/cuda_implicit_schur_complement.h"
#include "ceres/cuda_preconditioner_wrapper.h"
#include "ceres/implicit_schur_complement.h"
#include "ceres/power_series_expansion_preconditioner.h"
#include "ceres/schur_jacobi_preconditioner.h"
#include "ceres/solver.h"
#include "ceres/visibility_based_preconditioner.h"

namespace ceres::internal {

namespace {
SparseLinearAlgebraLibraryType DefaultSparseLinearAlgebraType() {
  const Solver::Options options;
  return options.sparse_linear_algebra_library_type;
}
}  // namespace

CudaIterativeSchurComplementSolver::CudaIterativeSchurComplementSolver(
    LinearSolver::Options options)
    : IterativeSchurComplementSolverBase(options),
      reduced_linear_system_solution_(options.context, 0),
      x_(options.context, 0) {}
double* CudaIterativeSchurComplementSolver::reduced_linear_system_solution() {
  return reduced_linear_system_solution_.mutable_data();
}

void CudaIterativeSchurComplementSolver::Initialize() {
  const int num_rows = schur_complement_->num_rows();
  reduced_linear_system_solution_.Resize(num_rows);
  reduced_linear_system_solution_.SetZero();
  const int num_cols = schur_complement_->num_cols_total();
  x_.Resize(num_cols);
  for (int i = 0; i < 4; ++i) {
    if (scratch_[i]) break;
    scratch_[i] = std::make_unique<CudaVector>(options_.context,
                                               schur_complement_->num_rows());
  }
}

class CudaLinearOperatorAdapter
    : public ConjugateGradientsLinearOperator<CudaVector> {
 public:
  CudaLinearOperatorAdapter(LinearOperator& linear_operator)
      : linear_operator_(linear_operator) {}
  void RightMultiplyAndAccumulate(const CudaVector& x, CudaVector& y) {
    linear_operator_.RightMultiplyAndAccumulate(x.data(), y.mutable_data());
  }

 private:
  LinearOperator& linear_operator_;
};

LinearSolver::Summary CudaIterativeSchurComplementSolver::ReducedSolve(
    const ConjugateGradientsSolverOptions& cg_options) {
  CudaVector* scratch_ptr[4] = {scratch_[0].get(),
                                scratch_[1].get(),
                                scratch_[2].get(),
                                scratch_[3].get()};

  CudaLinearOperatorAdapter lhs(*schur_complement_);
  CudaLinearOperatorAdapter preconditioner(*preconditioner_);
  return ConjugateGradientsSolver(
      cg_options,
      lhs,
      down_cast<CudaImplicitSchurComplement*>(schur_complement_.get())->rhs(),
      preconditioner,
      scratch_ptr,
      reduced_linear_system_solution_);
}

void CudaIterativeSchurComplementSolver::BackSubstitute(
    const double* reduced_system_solution, double* x) {
  schur_complement_->BackSubstitute(reduced_system_solution, x_.mutable_data());
  x_.CopyTo(x);
}

void CudaIterativeSchurComplementSolver::CreatePreSolver(
    const int max_num_spse_iterations, const double spse_tolerance) {
  auto pre_solver = std::make_unique<PowerSeriesExpansionPreconditioner>(
      &down_cast<CudaImplicitSchurComplement*>(schur_complement_.get())
           ->isc_cpu(),
      max_num_spse_iterations,
      spse_tolerance);
  pre_solver_ = std::make_unique<CudaPreconditionerWrapper>(
      std::move(pre_solver), options_.context);
}

void CudaIterativeSchurComplementSolver::CreatePreconditioner(
    const BlockSparseMatrix* A) {
  if (preconditioner_ != nullptr) {
    return;
  }

  Preconditioner::Options preconditioner_options;
  preconditioner_options.type = options_.preconditioner_type;
  preconditioner_options.visibility_clustering_type =
      options_.visibility_clustering_type;
  preconditioner_options.sparse_linear_algebra_library_type =
      DefaultSparseLinearAlgebraType();
  preconditioner_options.num_threads = options_.num_threads;
  preconditioner_options.row_block_size = options_.row_block_size;
  preconditioner_options.e_block_size = options_.e_block_size;
  preconditioner_options.f_block_size = options_.f_block_size;
  preconditioner_options.elimination_groups = options_.elimination_groups;
  CHECK(options_.context != nullptr);
  preconditioner_options.context = options_.context;

  std::unique_ptr<Preconditioner> preconditioner;
  const auto& schur_complement_gpu =
      down_cast<CudaImplicitSchurComplement*>(schur_complement_.get());
  const auto& schur_complement_cpu = schur_complement_gpu->isc_cpu();
  switch (options_.preconditioner_type) {
    case IDENTITY:
      preconditioner_ = std::make_unique<CudaIdentityPreconditionerBlockSparse>(
          schur_complement_->num_cols(), options_.context);
      return;
    case JACOBI:
      preconditioner_ = std::make_unique<SparseMatrixPreconditionerWrapper>(
          schur_complement_gpu->block_diagonal_FtF_inverse(),
          preconditioner_options);
      return;
    case SCHUR_POWER_SERIES_EXPANSION:
      // Ignoring the value of spse_tolerance to ensure preconditioner stays
      // fixed during the iterations of cg.
      preconditioner = std::make_unique<PowerSeriesExpansionPreconditioner>(
          &schur_complement_cpu, options_.max_num_spse_iterations, 0);
      break;
    case SCHUR_JACOBI: {
      auto schur_jacobi = std::make_unique<SchurJacobiPreconditioner>(
          *A->block_structure(), preconditioner_options);
      preconditioner_ = std::make_unique<CudaSchurJacobiPreconditioner>(
          std::move(schur_jacobi), options_.context);
      return;
    }
    case CLUSTER_JACOBI:
    case CLUSTER_TRIDIAGONAL:
      preconditioner = std::make_unique<VisibilityBasedPreconditioner>(
          *A->block_structure(), preconditioner_options);
      break;
    default:
      LOG(FATAL) << "Unknown Preconditioner Type";
  }
  preconditioner_ = std::make_unique<CudaPreconditionerWrapper>(
      std::move(preconditioner), options_.context);
};

}  // namespace ceres::internal

#endif
