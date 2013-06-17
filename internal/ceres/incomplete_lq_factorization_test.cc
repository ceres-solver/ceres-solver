// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
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

#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/incomplete_lq_factorization.h"
#include "ceres/internal/scoped_ptr.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

void ExpectMatricesAreEqual(const CompressedRowSparseMatrix& expected,
                            const CompressedRowSparseMatrix& actual,
                            const double tolerance) {
  EXPECT_EQ(expected.num_rows(), actual.num_rows());
  EXPECT_EQ(expected.num_cols(), actual.num_cols());
  EXPECT_EQ(expected.storage().rows, actual.storage().rows);
  for (int i = 0; i < actual.num_nonzeros(); ++i) {
    EXPECT_EQ(expected.cols()[i], actual.cols()[i]);
    EXPECT_NEAR(expected.values()[i], actual.values()[i], tolerance);
  }
}

TEST(IncompleteQRFactorization, OneByOneMatrix) {
  CompressedRowSparseMatrix matrix(1, 1, 1);
  matrix.mutable_rows()[0] = 0;
  matrix.mutable_rows()[1] = 1;
  matrix.mutable_cols()[0] = 0;
  matrix.mutable_values()[0] = 2;

  scoped_ptr<CompressedRowSparseMatrix> l(
      IncompleteLQFactorization(matrix, 1, 0.0, 1, 0.0));
  ExpectMatricesAreEqual(matrix, *l, 1e-16);
}

TEST(IncompleteQRFactorization, CompleteFactorization) {
  CompressedRowSparseMatrix matrix(3, 4, 12);

  matrix.mutable_rows()[0] = 0;
  matrix.mutable_rows()[1] = 4;
  matrix.mutable_rows()[2] = 8;
  matrix.mutable_rows()[3] = 12;

  matrix.mutable_cols()[0] = 0;
  matrix.mutable_cols()[1] = 1;
  matrix.mutable_cols()[2] = 2;
  matrix.mutable_cols()[3] = 3;
  matrix.mutable_cols()[4] = 0;
  matrix.mutable_cols()[5] = 1;
  matrix.mutable_cols()[6] = 2;
  matrix.mutable_cols()[7] = 3;
  matrix.mutable_cols()[8] = 0;
  matrix.mutable_cols()[9] = 1;
  matrix.mutable_cols()[10] = 2;
  matrix.mutable_cols()[11] = 3;

  matrix.mutable_values()[0] = 1;
  matrix.mutable_values()[1] = 4;
  matrix.mutable_values()[2] = 3;
  matrix.mutable_values()[3] = 10;
  matrix.mutable_values()[4] = 2;
  matrix.mutable_values()[5] = 8;
  matrix.mutable_values()[6] = 9;
  matrix.mutable_values()[7] = 12;
  matrix.mutable_values()[8] = 3;
  matrix.mutable_values()[9] = 16;
  matrix.mutable_values()[10] = 27;
  matrix.mutable_values()[11] = 16;

  CompressedRowSparseMatrix expected(3, 3, 6);
  expected.mutable_rows()[0] = 0;
  expected.mutable_rows()[1] = 1;
  expected.mutable_rows()[2] = 3;
  expected.mutable_rows()[3] = 6;
  expected.mutable_cols()[0] = 0;
  expected.mutable_values()[0] = 11.22497;
  expected.mutable_cols()[1] = 0;
  expected.mutable_values()[1] = 16.12476;
  expected.mutable_cols()[2] = 1;
  expected.mutable_values()[2] = 5.74387;
  expected.mutable_cols()[3] = 0;
  expected.mutable_values()[3] = 27.43882;
  expected.mutable_cols()[4] = 1;
  expected.mutable_values()[4] = 22.03314;
  expected.mutable_cols()[5] = 2;
  expected.mutable_values()[5] = 3.41345;

  scoped_ptr<CompressedRowSparseMatrix> l(
      IncompleteLQFactorization(matrix, 3, 0.0, 4, 0.0));
  ExpectMatricesAreEqual(expected, *l, 1e-4);
}

}  // namespace internal
}  // namespace ceres
