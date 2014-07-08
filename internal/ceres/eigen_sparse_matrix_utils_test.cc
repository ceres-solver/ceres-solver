// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2013 Google Inc. All rights reserved.
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
// Author: mike@hidof.com (Michael Vitus)
// Adapted from compressed_col_sparse_matrix_utils_test.cc.

#include <algorithm>
#include "ceres/eigen_sparse_matrix_utils.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

class SolveUpperTriangularTest : public ::testing::Test {
 protected:
  void SetUp() {
    R.resize(4,4);
    std::vector<Eigen::Triplet<double> > triplet_list;
    triplet_list.push_back(Eigen::Triplet<double>(0, 0, 0.50754));
    triplet_list.push_back(Eigen::Triplet<double>(1, 1, 0.80483));
    triplet_list.push_back(Eigen::Triplet<double>(1, 2, 0.14120));
    triplet_list.push_back(Eigen::Triplet<double>(2, 2, 0.3));
    triplet_list.push_back(Eigen::Triplet<double>(0, 3, 0.77696));
    triplet_list.push_back(Eigen::Triplet<double>(1, 3, 0.41860));
    triplet_list.push_back(Eigen::Triplet<double>(3, 3, 0.88979));
    R.setFromTriplets(triplet_list.begin(), triplet_list.end());
  }

  Eigen::SparseMatrix<double, Eigen::ColMajor> R;
};

TEST_F(SolveUpperTriangularTest, SolveInPlace) {
  Eigen::VectorXd rhs_and_solution(4);
  rhs_and_solution << 1.0, 1.0, 2.0, 2.0;
  const double expected[] = { -1.4706, -1.0962, 6.6667, 2.2477};

  SolveUpperTriangularInPlace(R, &rhs_and_solution);

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(rhs_and_solution[i], expected[i], 1e-4) << i;
  }
}

TEST_F(SolveUpperTriangularTest, RTRSolveWithSparseRHS) {
  Eigen::VectorXd solution(4);
  double expected[] = { 6.8420e+00,   1.0057e+00,  -1.4907e-16,  -1.9335e+00,
                        1.0057e+00,   2.2275e+00,  -1.9493e+00,  -6.5693e-01,
                       -1.4907e-16,  -1.9493e+00,   1.1111e+01,   9.7381e-17,
                       -1.9335e+00,  -6.5693e-01,   9.7381e-17,   1.2631e+00 };

  for (int i = 0; i < 4; ++i) {
    SolveRTRWithSparseRHS(R, i, &solution);
    for (int j = 0; j < 4; ++j) {
      EXPECT_NEAR(solution[j], expected[4 * i + j], 1e-3) << i;
    }
  }
}

}  // namespace internal
}  // namespace ceres
