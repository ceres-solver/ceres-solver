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

// This include must come before any #ifndef check on Ceres compile options.
#include "ceres/internal/port.h"

#include "ceres/cxsparse.h"

#ifndef CERES_NO_CXSPARSE

#include <vector>
#include "ceres/compressed_col_sparse_matrix_utils.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/random.h"
#include "ceres/triplet_sparse_matrix.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// TODO(sameeragarwal): Use the factory class in TripletSparseMatrix.
Eigen::SparseMatrix<double> CreateRandomSparseMatrix(const int num_rows,
                                                     const int num_cols,
                                                     const double p) {
  std::vector<Eigen::Triplet<double> > triplet_list;
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      const double v_ij = RandDouble();
      if (v_ij < p) {
        triplet_list.push_back(Eigen::Triplet<double>(i, j, v_ij));
      }
    }
  }

  Eigen::SparseMatrix<double> mat(num_rows, num_cols);
  mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
  return mat;
}

// TODO(sameeragarwal): do we need this test?

TEST(CXSparse, CreateSparseMatrixView) {
  const int num_rows = 8;
  const int num_cols = 4;
  const double p = 0.5;
  Eigen::SparseMatrix<double> m =
      CreateRandomSparseMatrix(num_rows, num_cols, p);
  CXSparse cxsparse;
  cs_di cs_m = cxsparse.CreateSparseMatrixView(&m);

  Eigen::VectorXd x(num_cols);
  Eigen::VectorXd y(num_rows);
  for (int i = 0; i < num_cols; ++i) {
    x.setZero();
    x(i) = 1.0;
    y.setZero();
    cs_gaxpy(&cs_m, x.data(), y.data());
    Eigen::VectorXd z = m.col(i);
    EXPECT_NEAR(
        (y - z).norm() / z.norm(), 0.0, std::numeric_limits<double>::epsilon());
  }
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_NO_CXSPARSE
