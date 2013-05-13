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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/covariance_impl.h"

#include <algorithm>
#include <cmath>
#include "ceres/problem_impl.h"
#include "gtest/gtest.h"
#include "ceres/covariance.h"
#include "ceres/cost_function.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/local_parameterization.h"

namespace ceres {
namespace internal {

TEST(CovarianceImpl, ComputeCovarianceSparsity) {
  double parameters[10];

  double* block1 = parameters;
  double* block2 = block1 + 1;
  double* block3 = block2 + 2;
  double* block4 = block3 + 3;

  ProblemImpl problem;

  // Add in random order
  problem.AddParameterBlock(block1, 1);
  problem.AddParameterBlock(block4, 4);
  problem.AddParameterBlock(block3, 3);
  problem.AddParameterBlock(block2, 2);

  // Sparsity pattern
  //
  //  x 0 0 0 0 0 x x x x
  //  0 x x x x x 0 0 0 0
  //  0 x x x x x 0 0 0 0
  //  0 0 0 x x x 0 0 0 0
  //  0 0 0 x x x 0 0 0 0
  //  0 0 0 x x x 0 0 0 0
  //  0 0 0 0 0 0 x x x x
  //  0 0 0 0 0 0 x x x x
  //  0 0 0 0 0 0 x x x x
  //  0 0 0 0 0 0 x x x x

  int expected_rows[] = {0, 5, 10, 15, 18, 21, 24, 28, 32, 36, 40};
  int expected_cols[] = {0, 6, 7, 8, 9,
                         1, 2, 3, 4, 5,
                         1, 2, 3, 4, 5,
                         3, 4, 5,
                         3, 4, 5,
                         3, 4, 5,
                         6, 7, 8, 9,
                         6, 7, 8, 9,
                         6, 7, 8, 9,
                         6, 7, 8, 9};


  vector<pair<const double*, const double*> > covariance_blocks;
  covariance_blocks.push_back(make_pair(block1, block1));
  covariance_blocks.push_back(make_pair(block4, block4));
  covariance_blocks.push_back(make_pair(block2, block2));
  covariance_blocks.push_back(make_pair(block3, block3));
  covariance_blocks.push_back(make_pair(block2, block3));
  covariance_blocks.push_back(make_pair(block4, block1));  // reversed

  Covariance::Options options;
  CovarianceImpl covariance_impl(options);
  EXPECT_TRUE(covariance_impl
              .ComputeCovarianceSparsity(covariance_blocks, &problem));

  const CompressedRowSparseMatrix* crsm = covariance_impl.covariance_matrix();

  EXPECT_EQ(crsm->num_rows(), 10);
  EXPECT_EQ(crsm->num_cols(), 10);
  EXPECT_EQ(crsm->num_nonzeros(), 40);

  const int* rows = crsm->rows();
  for (int r = 0; r < crsm->num_rows() + 1; ++r) {
    EXPECT_EQ(rows[r], expected_rows[r])
        << r << " "
        << rows[r] << " "
        << expected_rows[r];
  }

  const int* cols = crsm->cols();
  for (int c = 0; c < crsm->num_nonzeros(); ++c) {
    EXPECT_EQ(cols[c], expected_cols[c])
        << c << " "
        << cols[c] << " "
        << expected_cols[c];
  }
}


class UnaryCostFunction: public CostFunction {
 public:
  UnaryCostFunction(const int num_residuals,
                    const int16 parameter_block_size,
                    const double* jacobian) {
    set_num_residuals(num_residuals);
    mutable_parameter_block_sizes()->push_back(parameter_block_size);

    jacobian_.reset(new double[num_residuals * parameter_block_size]);
    copy(jacobian,
         jacobian + num_residuals * parameter_block_size,
         jacobian_.get());
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    for (int i = 0; i < num_residuals(); ++i) {
      residuals[i] = 1;
    }

    if (jacobians == NULL) {
      return true;
    }

    if (jacobians[0] != NULL) {
      copy(jacobian_.get(),
           jacobian_.get() + num_residuals() * parameter_block_sizes()[0],
           jacobians[0]);
    }

    return true;
  }

 private:
  scoped_array<double> jacobian_;
};


class BinaryCostFunction: public CostFunction {
 public:
  BinaryCostFunction(const int num_residuals,
                     const int16 parameter_block1_size,
                     const int16 parameter_block2_size,
                     const double* jacobian1,
                     const double* jacobian2) {
    set_num_residuals(num_residuals);
    mutable_parameter_block_sizes()->push_back(parameter_block1_size);
    mutable_parameter_block_sizes()->push_back(parameter_block2_size);

    jacobian1_.reset(new double[num_residuals * parameter_block1_size]);
    copy(jacobian1,
         jacobian1 + num_residuals * parameter_block1_size,
         jacobian1_.get());

    jacobian2_.reset(new double[num_residuals * parameter_block2_size]);
    copy(jacobian2,
         jacobian2 + num_residuals * parameter_block2_size,
         jacobian2_.get());
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    for (int i = 0; i < num_residuals(); ++i) {
      residuals[i] = 2;
    }

    if (jacobians == NULL) {
      return true;
    }

    if (jacobians[0] != NULL) {
      copy(jacobian1_.get(),
           jacobian1_.get() + num_residuals() * parameter_block_sizes()[0],
           jacobians[0]);
    }

    if (jacobians[1] != NULL) {
      copy(jacobian2_.get(),
           jacobian2_.get() + num_residuals() * parameter_block_sizes()[1],
           jacobians[1]);
    }

    return true;
  }

 private:
  scoped_array<double> jacobian1_;
  scoped_array<double> jacobian2_;
};

TEST(Covariance, ScalarMultiDimensionalCovariance) {
  Problem problem;
  double x = 1.0;
  double y = 1.0;
  double z = 1.0;

  {
    double jacobian = 2.0;
    problem.AddResidualBlock(new UnaryCostFunction(1, 1, &jacobian), NULL, &x);
  }

  {
    double jacobian = 3.0;
    problem.AddResidualBlock(new UnaryCostFunction(1, 1, &jacobian), NULL, &y);
  }

  {
    double jacobian = 5.0;
    problem.AddResidualBlock(new UnaryCostFunction(1, 1, &jacobian), NULL, &z);
  }

  {
    double jacobian1 = 1.0;
    double jacobian2 = 2.0;
    problem.AddResidualBlock(
        new BinaryCostFunction(1, 1, 1, &jacobian1, &jacobian2),
        NULL,
        &y,
        &x);
  }

  // Jacobian (J)
  //
  //  2 0 0
  //  0 3 0
  //  0 0 5
  //  2 1 0

  // J'J
  //
  // 8  2  0
  // 2 10  0
  // 0  0 25

  // covariance (J'J)^-1
  //
  //   0.13158  -0.02632  -0.00000
  //  -0.02632   0.10526  -0.00000
  //  -0.00000  -0.00000   0.04000

  vector<pair<const double*, const double*> > all_blocks;
  all_blocks.push_back(make_pair(&x, &x));
  all_blocks.push_back(make_pair(&y, &y));
  all_blocks.push_back(make_pair(&z, &z));
  all_blocks.push_back(make_pair(&x, &y));
  all_blocks.push_back(make_pair(&x, &z));
  all_blocks.push_back(make_pair(&y, &z));

  map<pair<const double*, const double*>, double> expected_covariance;
  expected_covariance[make_pair(&x, &x)] = 0.13158;
  expected_covariance[make_pair(&y, &y)] = 0.10526;
  expected_covariance[make_pair(&z, &z)] = 0.04;
  expected_covariance[make_pair(&x, &y)] = -0.02632;
  expected_covariance[make_pair(&x, &z)] = 0.0;
  expected_covariance[make_pair(&y, &z)] = 0.0;

  const double kTolerance = 1e-5;
  // Generate all possible combination of block pairs and check if the
  // covariance computation is correct.
  for (int i = 0; i < 64; ++i) {
    vector<pair<const double*, const double*> > covariance_blocks;
    if (i & 1) {
      covariance_blocks.push_back(all_blocks[0]);
    }

    if (i & 2) {
      covariance_blocks.push_back(all_blocks[1]);
    }

    if (i & 4) {
      covariance_blocks.push_back(all_blocks[2]);
    }

    if (i & 8) {
      covariance_blocks.push_back(all_blocks[3]);
    }

    if (i & 16) {
      covariance_blocks.push_back(all_blocks[4]);
    }

    if (i & 32) {
      covariance_blocks.push_back(all_blocks[5]);
    }

    Covariance::Options options;
    Covariance covariance(options);
    EXPECT_TRUE(covariance.Compute(covariance_blocks, &problem));

    for (int i = 0; i < covariance_blocks.size(); ++i) {
      const double* block1 = covariance_blocks[i].first;
      const double* block2 = covariance_blocks[i].second;
      double actual = -10.0;
      const double expected = expected_covariance[covariance_blocks[i]];
      EXPECT_TRUE(covariance.GetCovarianceBlock(block1, block2, &actual));
      EXPECT_NEAR(actual, expected, kTolerance);
    }
  }
}

TEST(Covariance, BlockMultiDimensionalCovariance) {
  Problem problem;
  double parameters[6];

  double* x = parameters;
  double* y = x + 2;
  double* z = y + 3;

  {
    double jacobian[] = { 1.0, 0.0, 0.0, 1.0};
    problem.AddResidualBlock(new UnaryCostFunction(2, 2, jacobian), NULL, x);
  }

  {
    double jacobian[] = { 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0 };
    problem.AddResidualBlock(new UnaryCostFunction(3, 3, jacobian), NULL, y);
  }

  {
    double jacobian = 5.0;
    problem.AddResidualBlock(new UnaryCostFunction(1, 1, &jacobian), NULL, z);
  }

  {
    double jacobian1[] = { 1.0, 2.0, 3.0 };
    double jacobian2[] = { -5.0, -6.0 };
    problem.AddResidualBlock(
        new BinaryCostFunction(1, 3, 2, jacobian1, jacobian2),
        NULL,
        y,
        x);
  }

  {
    double jacobian1[] = {2.0 };
    double jacobian2[] = { 3.0, -2.0 };
    problem.AddResidualBlock(
        new BinaryCostFunction(1, 1, 2, jacobian1, jacobian2),
        NULL,
        z,
        x);
  }

  // J
  //  1  0  0  0  0  0
  //  0  1  0  0  0  0
  //  0  0  2  0  0  0
  //  0  0  0  2  0  0
  //  0  0  0  0  2  0
  //  0  0  0  0  0  5
  // -5 -6  1  2  3  0
  //  3 -2  0  0  0  2

  // J'J
  //
  //  35   24   -5  -10  -15    6
  //  24   41   -6  -12  -18   -4
  //  -5   -6    5    2    3    0
  // -10  -12    2    8    6    0
  // -15  -18    3    6   13    0
  //   6   -4    0    0    0   29

  double expected_covariance[] = {
     7.0747e-02,  -8.4923e-03,   1.6821e-02,   3.3643e-02,   5.0464e-02,  -1.5809e-02,  // NOLINT
    -8.4923e-03,   8.1352e-02,   2.4758e-02,   4.9517e-02,   7.4275e-02,   1.2978e-02,  // NOLINT
     1.6821e-02,   2.4758e-02,   2.4904e-01,  -1.9271e-03,  -2.8906e-03,  -6.5325e-05,  // NOLINT
     3.3643e-02,   4.9517e-02,  -1.9271e-03,   2.4615e-01,  -5.7813e-03,  -1.3065e-04,  // NOLINT
     5.0464e-02,   7.4275e-02,  -2.8906e-03,  -5.7813e-03,   2.4133e-01,  -1.9598e-04,  // NOLINT
    -1.5809e-02,   1.2978e-02,  -6.5325e-05,  -1.3065e-04,  -1.9598e-04,   3.9544e-02,  // NOLINT
  };

  vector<pair<const double*, const double*> > all_blocks;
  all_blocks.push_back(make_pair(x, x));
  all_blocks.push_back(make_pair(y, y));
  all_blocks.push_back(make_pair(z, z));
  all_blocks.push_back(make_pair(x, y));
  all_blocks.push_back(make_pair(x, z));
  all_blocks.push_back(make_pair(y, z));

  const double kTolerance = 1e-5;
  // Generate all possible combination of block pairs and check if the
  // covariance computation is correct.
  for (int i = 1; i <= 64; ++i) {
    vector<pair<const double*, const double*> > covariance_blocks;
    if (i & 1) {
      covariance_blocks.push_back(all_blocks[0]);
    }

    if (i & 2) {
      covariance_blocks.push_back(all_blocks[1]);
    }

    if (i & 4) {
      covariance_blocks.push_back(all_blocks[2]);
    }

    if (i & 8) {
      covariance_blocks.push_back(all_blocks[3]);
    }

    if (i & 16) {
      covariance_blocks.push_back(all_blocks[4]);
    }

    if (i & 32) {
      covariance_blocks.push_back(all_blocks[5]);
    }

    Covariance::Options options;
    Covariance covariance(options);
    EXPECT_TRUE(covariance.Compute(covariance_blocks, &problem));

    for (int i = 0; i < covariance_blocks.size(); ++i) {
      const double* block1 = covariance_blocks[i].first;
      const double* block2 = covariance_blocks[i].second;
      double actual_covariance[10];
      EXPECT_TRUE(covariance.GetCovarianceBlock(block1,
                                                block2,
                                                actual_covariance));

      int row_begin = 0;
      int row_end = 0;
      if (block1 == x) {
        row_begin = 0;
        row_end = 2;
      } else if (block1 == y) {
        row_begin = 2;
        row_end = 5;
      } else if (block1 == z) {
        row_begin = 5;
        row_end = 6;
      } else {
        LOG(FATAL) << "Matched nothing for block1 " << block1;
      }

      int col_begin = 0;
      int col_end = 0;
      if (block2 == x) {
        col_begin = 0;
        col_end = 2;
      } else if (block2 == y) {
        col_begin = 2;
        col_end = 5;
      } else if (block2 == z) {
        col_begin = 5;
        col_end = 6;
      } else {
        LOG(FATAL) << "Matched nothing for block2 " << block2;
      }

      MatrixRef expected(expected_covariance, 6, 6);
      MatrixRef actual(actual_covariance,
                       row_end - row_begin,
                       col_end - col_begin);
      double diff_norm = (expected.block(row_begin,
                                         col_begin,
                                         row_end - row_begin,
                                         col_end - col_begin) - actual).norm();
      diff_norm /= (row_end - row_begin) * (col_end - col_begin);
      EXPECT_NEAR(diff_norm, 0.0, kTolerance)
          << "\n\n" << expected.block(row_begin,
                                      col_begin,
                                      row_end - row_begin,
                                      col_end - col_begin)
          << "\n\n" << actual;
    }
  }
  expected_covariance[0] = 1.0;
}


TEST(Covariance, BlockMultiDimensionalCovarianceWithConstantParameterBlock) {
  Problem problem;
  double parameters[6];

  double* x = parameters;
  double* y = x + 2;
  double* z = y + 3;

  {
    double jacobian[] = { 1.0, 0.0, 0.0, 1.0};
    problem.AddResidualBlock(new UnaryCostFunction(2, 2, jacobian), NULL, x);
  }

  {
    double jacobian[] = { 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0 };
    problem.AddResidualBlock(new UnaryCostFunction(3, 3, jacobian), NULL, y);
  }

  {
    double jacobian = 5.0;
    problem.AddResidualBlock(new UnaryCostFunction(1, 1, &jacobian), NULL, z);
  }

  {
    double jacobian1[] = { 1.0, 2.0, 3.0 };
    double jacobian2[] = { -5.0, -6.0 };
    problem.AddResidualBlock(
        new BinaryCostFunction(1, 3, 2, jacobian1, jacobian2),
        NULL,
        y,
        x);
  }

  {
    double jacobian1[] = {2.0 };
    double jacobian2[] = { 3.0, -2.0 };
    problem.AddResidualBlock(
        new BinaryCostFunction(1, 1, 2, jacobian1, jacobian2),
        NULL,
        z,
        x);
  }

  problem.SetParameterBlockConstant(x);

  // J
  //  0  0  0  0  0  0
  //  0  0  0  0  0  0
  //  0  0  2  0  0  0
  //  0  0  0  2  0  0
  //  0  0  0  0  2  0
  //  0  0  0  0  0  5
  //  0  0  1  2  3  0
  //  0  0  0  0  0  2

  // J'J
  //
  //   0    0    0    0    0    0
  //   0    0    0    0    0    0
  //   0    0    5    2    3    0
  //   0    0    2    8    6    0
  //   0    0    3    6   13    0
  //   0    0    0    0    0   29

  double expected_covariance[] = {
              0,            0,            0,            0,            0,            0,  // NOLINT
              0,            0,            0,            0,            0,            0,  // NOLINT
              0,            0,      0.23611,     -0.02778,     -0.04167,     -0.00000,  // NOLINT
              0,            0,     -0.02778,      0.19444,     -0.08333,     -0.00000,  // NOLINT
              0,            0,     -0.04167,     -0.08333,      0.12500,     -0.00000,  // NOLINT
              0,            0,     -0.00000,     -0.00000,     -0.00000,      0.03448   // NOLINT
  };

  vector<pair<const double*, const double*> > all_blocks;
  all_blocks.push_back(make_pair(x, x));
  all_blocks.push_back(make_pair(y, y));
  all_blocks.push_back(make_pair(z, z));
  all_blocks.push_back(make_pair(x, y));
  all_blocks.push_back(make_pair(x, z));
  all_blocks.push_back(make_pair(y, z));

  const double kTolerance = 1e-5;
  // Generate all possible combination of block pairs and check if the
  // covariance computation is correct.
  for (int i = 1; i <= 64; ++i) {
    vector<pair<const double*, const double*> > covariance_blocks;
    if (i & 1) {
      covariance_blocks.push_back(all_blocks[0]);
    }

    if (i & 2) {
      covariance_blocks.push_back(all_blocks[1]);
    }

    if (i & 4) {
      covariance_blocks.push_back(all_blocks[2]);
    }

    if (i & 8) {
      covariance_blocks.push_back(all_blocks[3]);
    }

    if (i & 16) {
      covariance_blocks.push_back(all_blocks[4]);
    }

    if (i & 32) {
      covariance_blocks.push_back(all_blocks[5]);
    }

    Covariance::Options options;
    Covariance covariance(options);
    EXPECT_TRUE(covariance.Compute(covariance_blocks, &problem));

    for (int i = 0; i < covariance_blocks.size(); ++i) {
      const double* block1 = covariance_blocks[i].first;
      const double* block2 = covariance_blocks[i].second;
      double actual_covariance[10];
      EXPECT_TRUE(covariance.GetCovarianceBlock(block1,
                                                block2,
                                                actual_covariance));

      int row_begin = 0;
      int row_end = 0;
      if (block1 == x) {
        row_begin = 0;
        row_end = 2;
      } else if (block1 == y) {
        row_begin = 2;
        row_end = 5;
      } else if (block1 == z) {
        row_begin = 5;
        row_end = 6;
      } else {
        LOG(FATAL) << "Matched nothing for block1 " << block1;
      }

      int col_begin = 0;
      int col_end = 0;
      if (block2 == x) {
        col_begin = 0;
        col_end = 2;
      } else if (block2 == y) {
        col_begin = 2;
        col_end = 5;
      } else if (block2 == z) {
        col_begin = 5;
        col_end = 6;
      } else {
        LOG(FATAL) << "Matched nothing for block2 " << block2;
      }

      MatrixRef expected(expected_covariance, 6, 6);
      MatrixRef actual(actual_covariance,
                       row_end - row_begin,
                       col_end - col_begin);
      double diff_norm = (expected.block(row_begin,
                                         col_begin,
                                         row_end - row_begin,
                                         col_end - col_begin) - actual).norm();
      diff_norm /= (row_end - row_begin) * (col_end - col_begin);
      EXPECT_NEAR(diff_norm, 0.0, kTolerance)
          << "\n\n" << expected.block(row_begin,
                                      col_begin,
                                      row_end - row_begin,
                                      col_end - col_begin)
          << "\n\n" << actual;
    }
  }
  expected_covariance[0] = 1.0;
}


TEST(Covariance, BlockMultiDimensionalCovarianceWithLocalParameterization) {
  Problem problem;
  double parameters[6];

  double* x = parameters;
  double* y = x + 2;
  double* z = y + 3;

  {
    double jacobian[] = { 1.0, 0.0, 0.0, 1.0};
    problem.AddResidualBlock(new UnaryCostFunction(2, 2, jacobian), NULL, x);
  }

  {
    double jacobian[] = { 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0 };
    problem.AddResidualBlock(new UnaryCostFunction(3, 3, jacobian), NULL, y);
  }

  {
    double jacobian = 5.0;
    problem.AddResidualBlock(new UnaryCostFunction(1, 1, &jacobian), NULL, z);
  }

  {
    double jacobian1[] = { 1.0, 2.0, 3.0 };
    double jacobian2[] = { -5.0, -6.0 };
    problem.AddResidualBlock(
        new BinaryCostFunction(1, 3, 2, jacobian1, jacobian2),
        NULL,
        y,
        x);
  }

  {
    double jacobian1[] = {2.0 };
    double jacobian2[] = { 3.0, -2.0 };
    problem.AddResidualBlock(
        new BinaryCostFunction(1, 1, 2, jacobian1, jacobian2),
        NULL,
        z,
        x);
  }

  vector<int> subset;
  subset.push_back(2);

  problem.SetParameterization(y, new SubsetParameterization(3, subset));

  // J
  //  1  0  0  0  0  0
  //  0  1  0  0  0  0
  //  0  0  2  0  0  0
  //  0  0  0  2  0  0
  //  0  0  0  0  0  0
  //  0  0  0  0  0  5
  // -5 -6  1  2  0  0
  //  3 -2  0  0  0  2

  // J'J
  //
  //  35   24   -5  -10    0    6
  //  24   41   -6  -12    0   -4
  //  -5   -6    5    2    0    0
  // -10  -12    2    8    0    0
  //   0    0    0    0    0    0
  //   6   -4    0    0    0   29

  double expected_covariance[] = {
     0.06019,  -0.02402,   0.01743,   0.03485,   0.00000,  -0.01577,
    -0.02402,   0.05849,   0.02565,   0.05130,   0.00000,   0.01304,
     0.01743,   0.02565,   0.24900,  -0.00200,   0.00000,  -0.00007,
     0.03485,   0.05130,  -0.00200,   0.24601,   0.00000,  -0.00014,
     0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,
    -0.01577,   0.01304,  -0.00007,  -0.00014,   0.00000,   0.03954
  };

  vector<pair<const double*, const double*> > all_blocks;
  all_blocks.push_back(make_pair(x, x));
  all_blocks.push_back(make_pair(y, y));
  all_blocks.push_back(make_pair(z, z));
  all_blocks.push_back(make_pair(x, y));
  all_blocks.push_back(make_pair(x, z));
  all_blocks.push_back(make_pair(y, z));

  const double kTolerance = 1e-5;
  // Generate all possible combination of block pairs and check if the
  // covariance computation is correct.
  for (int i = 1; i <= 1; ++i) {
    vector<pair<const double*, const double*> > covariance_blocks;
    if (i & 1) {
      covariance_blocks.push_back(all_blocks[0]);
    }

    if (i & 2) {
      covariance_blocks.push_back(all_blocks[1]);
    }

    if (i & 4) {
      covariance_blocks.push_back(all_blocks[2]);
    }

    if (i & 8) {
      covariance_blocks.push_back(all_blocks[3]);
    }

    if (i & 16) {
      covariance_blocks.push_back(all_blocks[4]);
    }

    if (i & 32) {
      covariance_blocks.push_back(all_blocks[5]);
    }

    Covariance::Options options;
    Covariance covariance(options);
    EXPECT_TRUE(covariance.Compute(covariance_blocks, &problem));

    for (int i = 0; i < covariance_blocks.size(); ++i) {
      const double* block1 = covariance_blocks[i].first;
      const double* block2 = covariance_blocks[i].second;
      double actual_covariance[10];
      EXPECT_TRUE(covariance.GetCovarianceBlock(block1,
                                                block2,
                                                actual_covariance));

      int row_begin = 0;
      int row_end = 0;
      if (block1 == x) {
        row_begin = 0;
        row_end = 2;
      } else if (block1 == y) {
        row_begin = 2;
        row_end = 5;
      } else if (block1 == z) {
        row_begin = 5;
        row_end = 6;
      } else {
        LOG(FATAL) << "Matched nothing for block1 " << block1;
      }

      int col_begin = 0;
      int col_end = 0;
      if (block2 == x) {
        col_begin = 0;
        col_end = 2;
      } else if (block2 == y) {
        col_begin = 2;
        col_end = 5;
      } else if (block2 == z) {
        col_begin = 5;
        col_end = 6;
      } else {
        LOG(FATAL) << "Matched nothing for block2 " << block2;
      }

      MatrixRef expected(expected_covariance, 6, 6);
      MatrixRef actual(actual_covariance,
                       row_end - row_begin,
                       col_end - col_begin);
      double diff_norm = (expected.block(row_begin,
                                         col_begin,
                                         row_end - row_begin,
                                         col_end - col_begin) - actual).norm();
      diff_norm /= (row_end - row_begin) * (col_end - col_begin);
      EXPECT_NEAR(diff_norm, 0.0, kTolerance)
          << "\n\n" << expected.block(row_begin,
                                         col_begin,
                                         row_end - row_begin,
                                         col_end - col_begin)
          << "\n\n" << actual;
    }
  }
  expected_covariance[0] = 1.0;
}

}  // namespace internal
}  // namespace ceres
