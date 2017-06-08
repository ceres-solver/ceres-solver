// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2017 Google Inc. All rights reserved.
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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/casts.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/linear_solver.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/types.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// TODO(sameeragarwal): These tests needs to be re-written, since
// SparseNormalCholeskySolver is a composition of two classes now,
// OuterProduct and SparseCholesky.
//
// So the test should exercise the composition, rather than the
// numerics of the solver, which are well covered by tests for those
// classes.
class SparseNormalCholeskyLinearSolverTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    scoped_ptr<LinearLeastSquaresProblem> problem(
        CreateLinearLeastSquaresProblemFromId(0));

    CHECK_NOTNULL(problem.get());
    A_.reset(down_cast<TripletSparseMatrix*>(problem->A.release()));
    b_.reset(problem->b.release());
    D_.reset(problem->D.release());
    sol_unregularized_.reset(problem->x.release());
    sol_regularized_.reset(problem->x_D.release());
  }

  void TestSolver(const LinearSolver::Options& options) {
    LinearSolver::PerSolveOptions per_solve_options;
    LinearSolver::Summary unregularized_solve_summary;
    LinearSolver::Summary regularized_solve_summary;
    Vector x_unregularized(A_->num_cols());
    Vector x_regularized(A_->num_cols());

    scoped_ptr<SparseMatrix> transformed_A;

    CompressedRowSparseMatrix* crsm =
        CompressedRowSparseMatrix::FromTripletSparseMatrix(*A_);
    // Add row/column blocks structure.
    for (int i = 0; i < A_->num_rows(); ++i) {
      crsm->mutable_row_blocks()->push_back(1);
    }

    for (int i = 0; i < A_->num_cols(); ++i) {
      crsm->mutable_col_blocks()->push_back(1);
    }

    // With all blocks of size 1, crsb_rows and crsb_cols are equivalent to
    // rows and cols.
    std::copy(crsm->rows(),
              crsm->rows() + crsm->num_rows() + 1,
              std::back_inserter(*crsm->mutable_crsb_rows()));

    std::copy(crsm->cols(),
              crsm->cols() + crsm->num_nonzeros(),
              std::back_inserter(*crsm->mutable_crsb_cols()));

    transformed_A.reset(crsm);

    // Unregularized
    scoped_ptr<LinearSolver> solver(LinearSolver::Create(options));
    unregularized_solve_summary = solver->Solve(transformed_A.get(),
                                                b_.get(),
                                                per_solve_options,
                                                x_unregularized.data());

    // Sparsity structure is changing, reset the solver.
    solver.reset(LinearSolver::Create(options));
    // Regularized solution
    per_solve_options.D = D_.get();
    regularized_solve_summary = solver->Solve(
        transformed_A.get(), b_.get(), per_solve_options, x_regularized.data());

    EXPECT_EQ(unregularized_solve_summary.termination_type,
              LINEAR_SOLVER_SUCCESS);

    for (int i = 0; i < A_->num_cols(); ++i) {
      EXPECT_NEAR(sol_unregularized_[i], x_unregularized[i], 1e-8)
          << "\nExpected: "
          << ConstVectorRef(sol_unregularized_.get(), A_->num_cols())
                 .transpose()
          << "\nActual: " << x_unregularized.transpose();
    }

    EXPECT_EQ(regularized_solve_summary.termination_type,
              LINEAR_SOLVER_SUCCESS);
    for (int i = 0; i < A_->num_cols(); ++i) {
      EXPECT_NEAR(sol_regularized_[i], x_regularized[i], 1e-8)
          << "\nExpected: "
          << ConstVectorRef(sol_regularized_.get(), A_->num_cols()).transpose()
          << "\nActual: " << x_regularized.transpose();
    }
  }

  scoped_ptr<TripletSparseMatrix> A_;
  scoped_array<double> b_;
  scoped_array<double> D_;
  scoped_array<double> sol_unregularized_;
  scoped_array<double> sol_regularized_;
};

#ifndef CERES_NO_SUITESPARSE
TEST_F(SparseNormalCholeskyLinearSolverTest,
       SparseNormalCholeskyUsingSuiteSparsePreOrdering) {
  LinearSolver::Options options;
  options.sparse_linear_algebra_library_type = SUITE_SPARSE;
  options.type = SPARSE_NORMAL_CHOLESKY;
  options.use_postordering = false;
  TestSolver(options);
}

TEST_F(SparseNormalCholeskyLinearSolverTest,
       SparseNormalCholeskyUsingSuiteSparsePostOrdering) {
  LinearSolver::Options options;
  options.sparse_linear_algebra_library_type = SUITE_SPARSE;
  options.type = SPARSE_NORMAL_CHOLESKY;
  options.use_postordering = true;
  TestSolver(options);
}

TEST_F(SparseNormalCholeskyLinearSolverTest,
       SparseNormalCholeskyUsingSuiteSparseDynamicSparsity) {
  LinearSolver::Options options;
  options.sparse_linear_algebra_library_type = SUITE_SPARSE;
  options.type = SPARSE_NORMAL_CHOLESKY;
  options.dynamic_sparsity = true;
  TestSolver(options);
}
#endif

#ifndef CERES_NO_CXSPARSE
TEST_F(SparseNormalCholeskyLinearSolverTest,
       SparseNormalCholeskyUsingCXSparsePreOrdering) {
  LinearSolver::Options options;
  options.sparse_linear_algebra_library_type = CX_SPARSE;
  options.type = SPARSE_NORMAL_CHOLESKY;
  options.use_postordering = false;
  TestSolver(options);
}

TEST_F(SparseNormalCholeskyLinearSolverTest,
       SparseNormalCholeskyUsingCXSparsePostOrdering) {
  LinearSolver::Options options;
  options.sparse_linear_algebra_library_type = CX_SPARSE;
  options.type = SPARSE_NORMAL_CHOLESKY;
  options.use_postordering = true;
  TestSolver(options);
}

TEST_F(SparseNormalCholeskyLinearSolverTest,
       SparseNormalCholeskyUsingCXSparseDynamicSparsity) {
  LinearSolver::Options options;
  options.sparse_linear_algebra_library_type = CX_SPARSE;
  options.type = SPARSE_NORMAL_CHOLESKY;
  options.dynamic_sparsity = true;
  TestSolver(options);
}
#endif

#ifdef CERES_USE_EIGEN_SPARSE
TEST_F(SparseNormalCholeskyLinearSolverTest,
       SparseNormalCholeskyUsingEigenPreOrdering) {
  LinearSolver::Options options;
  options.sparse_linear_algebra_library_type = EIGEN_SPARSE;
  options.type = SPARSE_NORMAL_CHOLESKY;
  options.use_postordering = false;
  TestSolver(options);
}

TEST_F(SparseNormalCholeskyLinearSolverTest,
       SparseNormalCholeskyUsingEigenPostOrdering) {
  LinearSolver::Options options;
  options.sparse_linear_algebra_library_type = EIGEN_SPARSE;
  options.type = SPARSE_NORMAL_CHOLESKY;
  options.use_postordering = true;
  TestSolver(options);
}

TEST_F(SparseNormalCholeskyLinearSolverTest,
       SparseNormalCholeskyUsingEigenDynamicSparsity) {
  LinearSolver::Options options;
  options.sparse_linear_algebra_library_type = EIGEN_SPARSE;
  options.type = SPARSE_NORMAL_CHOLESKY;
  options.dynamic_sparsity = true;
  TestSolver(options);
}
#endif  // CERES_USE_EIGEN_SPARSE

}  // namespace internal
}  // namespace ceres
