// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
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

#include <glog/logging.h>
#include "gtest/gtest.h"
#include "ceres/casts.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/linear_solver.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/types.h"


namespace ceres {
namespace internal {

class UnsymmetricLinearSolverTest : public ::testing::Test {
 protected :
  virtual void SetUp() {
    scoped_ptr<LinearLeastSquaresProblem> problem(
        CreateLinearLeastSquaresProblemFromId(0));

    CHECK_NOTNULL(problem.get());
    A_.reset(down_cast<TripletSparseMatrix*>(problem->A.release()));
    b_.reset(problem->b.release());
    D_.reset(problem->D.release());
    sol1_.reset(problem->x.release());
    sol2_.reset(problem->x_D.release());
    x_.reset(new double[A_->num_cols()]);
  }

  void TestSolver(LinearSolverType linear_solver_type) {
    LinearSolver::Options options;
    options.type = linear_solver_type;
    scoped_ptr<LinearSolver> solver(LinearSolver::Create(options));

    LinearSolver::PerSolveOptions per_solve_options;

    // Unregularized
    LinearSolver::Summary summary =
        solver->Solve(A_.get(), b_.get(), per_solve_options, x_.get());

    EXPECT_EQ(summary.termination_type, TOLERANCE);

    for (int i = 0; i < A_->num_cols(); ++i) {
      EXPECT_NEAR(sol1_[i], x_[i], 1e-8);
    }

    // Regularized solution
    per_solve_options.D = D_.get();
    summary = solver->Solve(A_.get(), b_.get(), per_solve_options, x_.get());

    EXPECT_EQ(summary.termination_type, TOLERANCE);

    for (int i = 0; i < A_->num_cols(); ++i) {
      EXPECT_NEAR(sol2_[i], x_[i], 1e-8);
    }
  }

  scoped_ptr<TripletSparseMatrix> A_;
  scoped_array<double> b_;
  scoped_array<double> D_;
  scoped_array<double> sol1_;
  scoped_array<double> sol2_;

  scoped_array<double> x_;
};

// TODO(keir): Reduce duplication.
TEST_F(UnsymmetricLinearSolverTest, DenseQR) {
  LinearSolver::Options options;
  options.type = DENSE_QR;
  scoped_ptr<LinearSolver> solver(LinearSolver::Create(options));

  LinearSolver::PerSolveOptions per_solve_options;
  DenseSparseMatrix A(*A_);

  // Unregularized
  LinearSolver::Summary summary =
      solver->Solve(&A, b_.get(), per_solve_options, x_.get());

  EXPECT_EQ(summary.termination_type, TOLERANCE);
  for (int i = 0; i < A_->num_cols(); ++i) {
    EXPECT_NEAR(sol1_[i], x_[i], 1e-8);
  }

  VectorRef x(x_.get(), A_->num_cols());
  VectorRef b(b_.get(), A_->num_rows());
  Vector r = A.matrix()*x - b;
  LOG(INFO) << "r = A*x - b: \n" << r;

  // Regularized solution
  per_solve_options.D = D_.get();
  summary = solver->Solve(&A, b_.get(), per_solve_options, x_.get());

  EXPECT_EQ(summary.termination_type, TOLERANCE);
  for (int i = 0; i < A_->num_cols(); ++i) {
    EXPECT_NEAR(sol2_[i], x_[i], 1e-8);
  }
}

#ifndef CERES_NO_SUITESPARSE
TEST_F(UnsymmetricLinearSolverTest, SparseNormalCholeskyUsingSuiteSparse) {
  LinearSolver::Options options;
  options.type = SPARSE_NORMAL_CHOLESKY;
  options.sparse_linear_algebra_library = SUITESPARSE;
  scoped_ptr<LinearSolver>solver(LinearSolver::Create(options));

  LinearSolver::PerSolveOptions per_solve_options;
  CompressedRowSparseMatrix A(*A_);

  // Unregularized
  LinearSolver::Summary summary =
      solver->Solve(&A, b_.get(), per_solve_options, x_.get());

  EXPECT_EQ(summary.termination_type, TOLERANCE);
  for (int i = 0; i < A_->num_cols(); ++i) {
    EXPECT_NEAR(sol1_[i], x_[i], 1e-8);
  }

  // Regularized solution
  per_solve_options.D = D_.get();
  summary = solver->Solve(&A, b_.get(), per_solve_options, x_.get());

  EXPECT_EQ(summary.termination_type, TOLERANCE);
  for (int i = 0; i < A_->num_cols(); ++i) {
    EXPECT_NEAR(sol2_[i], x_[i], 1e-8);
  }
}
#endif  // CERES_NO_SUITESPARSE

#ifndef CERES_NO_CXSPARSE
TEST_F(UnsymmetricLinearSolverTest, SparseNormalCholeskyUsingCXSparse) {
  LinearSolver::Options options;
  options.type = SPARSE_NORMAL_CHOLESKY;
  options.sparse_linear_algebra_library = CXSPARSE;
  scoped_ptr<LinearSolver>solver(LinearSolver::Create(options));

  LinearSolver::PerSolveOptions per_solve_options;
  CompressedRowSparseMatrix A(*A_);

  // Unregularized
  LinearSolver::Summary summary =
      solver->Solve(&A, b_.get(), per_solve_options, x_.get());

  EXPECT_EQ(summary.termination_type, TOLERANCE);
  for (int i = 0; i < A_->num_cols(); ++i) {
    EXPECT_NEAR(sol1_[i], x_[i], 1e-8);
  }

  // Regularized solution
  per_solve_options.D = D_.get();
  summary = solver->Solve(&A, b_.get(), per_solve_options, x_.get());

  EXPECT_EQ(summary.termination_type, TOLERANCE);
  for (int i = 0; i < A_->num_cols(); ++i) {
    EXPECT_NEAR(sol2_[i], x_[i], 1e-8);
  }
}
#endif  // CERES_NO_CXSPARSE

}  // namespace internal
}  // namespace ceres
