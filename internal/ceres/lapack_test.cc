// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2014 Google Inc. All rights reserved.
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

#include <limits>

#include "Eigen/Dense"
#include "ceres/internal/eigen.h"
#include "ceres/lapack.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

const double kTolerance = 10 * std::numeric_limits<double>::epsilon();

TEST(MaybeTruncateAndPseudoInvert, DiagonalMatrix) {
  Matrix A = Eigen::MatrixXd::Zero(3,3);
  A(0, 0) = 1.0;
  A(1, 1) = 1.0;
  A(2, 2) = -1.0;

  MaybeTruncateAndPseudoInvert(A.data(), 3);
  EXPECT_NEAR(A(0, 0), 1.0, kTolerance);
  EXPECT_NEAR(A(1, 1), 1.0, kTolerance);
  EXPECT_NEAR(A(2, 2), 0, kTolerance);
}

TEST(MaybeTruncateAndPseudoInvert, RandomMatrix) {
  Matrix A = Matrix::Random(5,5);
  A +=  A.transpose().eval(); // Can't add a matrix to its transpose
                              // without creating a temporary.
  Matrix B = A;
  MaybeTruncateAndPseudoInvert(B.data(), 5);
  const Vector eigenvalues =
      Eigen::SelfAdjointEigenSolver<Matrix>(B * A).eigenvalues().transpose();
  EXPECT_NEAR(((eigenvalues.array() * eigenvalues.array()).matrix() -
               eigenvalues).norm(), 0.0, kTolerance);
}

}  // namespace internal
}  // namespace ceres
