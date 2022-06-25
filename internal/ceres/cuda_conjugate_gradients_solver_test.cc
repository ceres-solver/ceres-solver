// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2022 Google Inc. All rights reserved.
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
// Author: joydeepb@cs.utexas.edu (Joydeep Biswas)

#include <string>

#include "ceres/internal/config.h"
#include "ceres/internal/eigen.h"
#include "ceres/cuda_conjugate_gradients_solver.h"
#include "ceres/cuda_sparse_matrix.h"
#include "ceres/cuda_vector.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

#ifndef CERES_NO_CUDA

namespace {

class IdentityPreconditioner : public CudaLinearOperator {
 public:
  explicit IdentityPreconditioner(int num_rows)
      : num_rows_(num_rows) {}
  ~IdentityPreconditioner() {}
  int num_rows() const override { return num_rows_; }
  int num_cols() const override { return num_rows_; }

  void RightMultiply(const CudaVector& x, CudaVector* y) override {
    y->CopyFrom(x);
  }

  void LeftMultiply(const CudaVector& x, CudaVector* y) override {
    y->CopyFrom(x);
  }

 private:
  int num_rows_;
};

}  // namespace

TEST(CudaConjugateGradientsSolver, InvalidOptionOnInit) {
  LinearSolver::Options options;
  auto solver = CudaConjugateGradientsSolver::Create(options);
  ContextImpl* context = nullptr;
  std::string message;
  EXPECT_FALSE(solver->Init(context, &message));
}

TEST(CudaConjugateGradientsSolver, Solves3x3IdentitySystem) {
  LinearSolver::Options options;
  auto solver = CudaConjugateGradientsSolver::Create(options);
  ContextImpl context;
  std::string message;
  EXPECT_TRUE(solver->Init(&context, &message));

  CudaSparseMatrix A;
  CudaVector b;
  CudaVector x;
  EXPECT_TRUE(A.Init(&context, &message));
  EXPECT_TRUE(b.Init(&context, &message));
  EXPECT_TRUE(x.Init(&context, &message));

  TripletSparseMatrix triplet_matrix(
      3,
      3,
      {0, 1, 2},
      {0, 1, 2},
      {1.0, 1.0, 1.0}
  );
  A.CopyFrom(triplet_matrix);
  Vector b_cpu(3);
  b_cpu.setConstant(2.0);
  b.CopyFrom(b_cpu);

  LinearSolver::PerSolveOptions per_solve_options;

  IdentityPreconditioner preconditioner(3);
  LinearSolver::Summary summary = solver->Solve(
      &A, &preconditioner, b, per_solve_options, &x);
  EXPECT_EQ(summary.termination_type, LinearSolverTerminationType::SUCCESS);

  Vector x_cpu;
  x.CopyTo(&x_cpu);
  EXPECT_NEAR(x_cpu[0], 2.0, std::numeric_limits<double>::epsilon() * 1e4);
  EXPECT_NEAR(x_cpu[1], 2.0, std::numeric_limits<double>::epsilon() * 1e4);
  EXPECT_NEAR(x_cpu[2], 2.0, std::numeric_limits<double>::epsilon() * 1e4);
}

TEST(CudaConjugateGradientsSolver, Solves3x3SymmetricSystem) {
  LinearSolver::Options options;
  options.max_num_iterations = 10;
  auto solver = CudaConjugateGradientsSolver::Create(options);
  ContextImpl context;
  std::string message;
  EXPECT_TRUE(solver->Init(&context, &message));

  CudaSparseMatrix A;
  CudaVector b;
  CudaVector x;
  EXPECT_TRUE(A.Init(&context, &message));
  EXPECT_TRUE(b.Init(&context, &message));
  EXPECT_TRUE(x.Init(&context, &message));

  //      | 2  -1  0|
  //  A = |-1   2 -1| is symmetric positive definite.
  //      | 0  -1  2|
  TripletSparseMatrix triplet_matrix(
      3,
      3,
      {0, 0, 1, 1, 1, 2, 2},
      {0, 1, 0, 1, 2, 1, 2},
      {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0}
  );
  A.CopyFrom(triplet_matrix);
  Vector b_cpu(3);
  b_cpu(0) = -1;
  b_cpu(1) = 0;
  b_cpu(2) = 3;
  b.CopyFrom(b_cpu);

  LinearSolver::PerSolveOptions per_solve_options;
  per_solve_options.r_tolerance = 1e-9;
  IdentityPreconditioner preconditioner(3);
  LinearSolver::Summary summary = solver->Solve(
      &A, &preconditioner, b, per_solve_options, &x);

  EXPECT_EQ(summary.termination_type, LinearSolverTerminationType::SUCCESS);

  Vector x_cpu;
  x.CopyTo(&x_cpu);
  EXPECT_NEAR(x_cpu[0], 0.0, std::numeric_limits<double>::epsilon() * 1e4);
  EXPECT_NEAR(x_cpu[1], 1.0, std::numeric_limits<double>::epsilon() * 1e4);
  EXPECT_NEAR(x_cpu[2], 2.0, std::numeric_limits<double>::epsilon() * 1e4);
}

#endif  // CERES_NO_CUDA

}  // namespace internal
}  // namespace ceres
