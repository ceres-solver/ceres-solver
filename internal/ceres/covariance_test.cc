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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/covariance.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/cost_function.h"
#include "ceres/covariance_impl.h"
#include "ceres/local_parameterization.h"
#include "ceres/map_util.h"
#include "ceres/problem_impl.h"
#include "ceres/autodiff_cost_function.h"
#include "ceres/loss_function.h"
#include "ceres/solver.h"
#include "ceres/random.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

using std::make_pair;
using std::map;
using std::pair;
using std::vector;

class UnaryCostFunction: public CostFunction {
 public:
  UnaryCostFunction(const int num_residuals,
                    const int32 parameter_block_size,
                    const double* jacobian)
      : jacobian_(jacobian, jacobian + num_residuals * parameter_block_size) {
    set_num_residuals(num_residuals);
    mutable_parameter_block_sizes()->push_back(parameter_block_size);
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
      copy(jacobian_.begin(), jacobian_.end(), jacobians[0]);
    }

    return true;
  }

 private:
  vector<double> jacobian_;
};


class BinaryCostFunction: public CostFunction {
 public:
  BinaryCostFunction(const int num_residuals,
                     const int32 parameter_block1_size,
                     const int32 parameter_block2_size,
                     const double* jacobian1,
                     const double* jacobian2)
      : jacobian1_(jacobian1,
                   jacobian1 + num_residuals * parameter_block1_size),
        jacobian2_(jacobian2,
                   jacobian2 + num_residuals * parameter_block2_size) {
    set_num_residuals(num_residuals);
    mutable_parameter_block_sizes()->push_back(parameter_block1_size);
    mutable_parameter_block_sizes()->push_back(parameter_block2_size);
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
      copy(jacobian1_.begin(), jacobian1_.end(), jacobians[0]);
    }

    if (jacobians[1] != NULL) {
      copy(jacobian2_.begin(), jacobian2_.end(), jacobians[1]);
    }

    return true;
  }

 private:
  vector<double> jacobian1_;
  vector<double> jacobian2_;
};

// x_plus_delta = delta * x;
class PolynomialParameterization : public LocalParameterization {
 public:
  virtual ~PolynomialParameterization() {}

  virtual bool Plus(const double* x,
                    const double* delta,
                    double* x_plus_delta) const {
    x_plus_delta[0] = delta[0] * x[0];
    x_plus_delta[1] = delta[0] * x[1];
    return true;
  }

  virtual bool ComputeJacobian(const double* x, double* jacobian) const {
    jacobian[0] = x[0];
    jacobian[1] = x[1];
    return true;
  }

  virtual int GlobalSize() const { return 2; }
  virtual int LocalSize() const { return 1; }
};

TEST(CovarianceImpl, ComputeCovarianceSparsity) {
  double parameters[10];

  double* block1 = parameters;
  double* block2 = block1 + 1;
  double* block3 = block2 + 2;
  double* block4 = block3 + 3;

  ProblemImpl problem;

  // Add in random order
  Vector junk_jacobian = Vector::Zero(10);
  problem.AddResidualBlock(
      new UnaryCostFunction(1, 1, junk_jacobian.data()), NULL, block1);
  problem.AddResidualBlock(
      new UnaryCostFunction(1, 4, junk_jacobian.data()), NULL, block4);
  problem.AddResidualBlock(
      new UnaryCostFunction(1, 3, junk_jacobian.data()), NULL, block3);
  problem.AddResidualBlock(
      new UnaryCostFunction(1, 2, junk_jacobian.data()), NULL, block2);

  // Sparsity pattern
  //
  // Note that the problem structure does not imply this sparsity
  // pattern since all the residual blocks are unary. But the
  // ComputeCovarianceSparsity function in its current incarnation
  // does not pay attention to this fact and only looks at the
  // parameter block pairs that the user provides.
  //
  //  X . . . . . X X X X
  //  . X X X X X . . . .
  //  . X X X X X . . . .
  //  . . . X X X . . . .
  //  . . . X X X . . . .
  //  . . . X X X . . . .
  //  . . . . . . X X X X
  //  . . . . . . X X X X
  //  . . . . . . X X X X
  //  . . . . . . X X X X

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

TEST(CovarianceImpl, ComputeCovarianceSparsityWithConstantParameterBlock) {
  double parameters[10];

  double* block1 = parameters;
  double* block2 = block1 + 1;
  double* block3 = block2 + 2;
  double* block4 = block3 + 3;

  ProblemImpl problem;

  // Add in random order
  Vector junk_jacobian = Vector::Zero(10);
  problem.AddResidualBlock(
      new UnaryCostFunction(1, 1, junk_jacobian.data()), NULL, block1);
  problem.AddResidualBlock(
      new UnaryCostFunction(1, 4, junk_jacobian.data()), NULL, block4);
  problem.AddResidualBlock(
      new UnaryCostFunction(1, 3, junk_jacobian.data()), NULL, block3);
  problem.AddResidualBlock(
      new UnaryCostFunction(1, 2, junk_jacobian.data()), NULL, block2);
  problem.SetParameterBlockConstant(block3);

  // Sparsity pattern
  //
  // Note that the problem structure does not imply this sparsity
  // pattern since all the residual blocks are unary. But the
  // ComputeCovarianceSparsity function in its current incarnation
  // does not pay attention to this fact and only looks at the
  // parameter block pairs that the user provides.
  //
  //  X . . X X X X
  //  . X X . . . .
  //  . X X . . . .
  //  . . . X X X X
  //  . . . X X X X
  //  . . . X X X X
  //  . . . X X X X

  int expected_rows[] = {0, 5, 7, 9, 13, 17, 21, 25};
  int expected_cols[] = {0, 3, 4, 5, 6,
                         1, 2,
                         1, 2,
                         3, 4, 5, 6,
                         3, 4, 5, 6,
                         3, 4, 5, 6,
                         3, 4, 5, 6};

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

  EXPECT_EQ(crsm->num_rows(), 7);
  EXPECT_EQ(crsm->num_cols(), 7);
  EXPECT_EQ(crsm->num_nonzeros(), 25);

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

TEST(CovarianceImpl, ComputeCovarianceSparsityWithFreeParameterBlock) {
  double parameters[10];

  double* block1 = parameters;
  double* block2 = block1 + 1;
  double* block3 = block2 + 2;
  double* block4 = block3 + 3;

  ProblemImpl problem;

  // Add in random order
  Vector junk_jacobian = Vector::Zero(10);
  problem.AddResidualBlock(
      new UnaryCostFunction(1, 1, junk_jacobian.data()), NULL, block1);
  problem.AddResidualBlock(
      new UnaryCostFunction(1, 4, junk_jacobian.data()), NULL, block4);
  problem.AddParameterBlock(block3, 3);
  problem.AddResidualBlock(
      new UnaryCostFunction(1, 2, junk_jacobian.data()), NULL, block2);

  // Sparsity pattern
  //
  // Note that the problem structure does not imply this sparsity
  // pattern since all the residual blocks are unary. But the
  // ComputeCovarianceSparsity function in its current incarnation
  // does not pay attention to this fact and only looks at the
  // parameter block pairs that the user provides.
  //
  //  X . . X X X X
  //  . X X . . . .
  //  . X X . . . .
  //  . . . X X X X
  //  . . . X X X X
  //  . . . X X X X
  //  . . . X X X X

  int expected_rows[] = {0, 5, 7, 9, 13, 17, 21, 25};
  int expected_cols[] = {0, 3, 4, 5, 6,
                         1, 2,
                         1, 2,
                         3, 4, 5, 6,
                         3, 4, 5, 6,
                         3, 4, 5, 6,
                         3, 4, 5, 6};

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

  EXPECT_EQ(crsm->num_rows(), 7);
  EXPECT_EQ(crsm->num_cols(), 7);
  EXPECT_EQ(crsm->num_nonzeros(), 25);

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

class CovarianceTest : public ::testing::Test {
 protected:
  typedef map<const double*, pair<int, int> > BoundsMap;

  virtual void SetUp() {
    double* x = parameters_;
    double* y = x + 2;
    double* z = y + 3;

    x[0] = 1;
    x[1] = 1;
    y[0] = 2;
    y[1] = 2;
    y[2] = 2;
    z[0] = 3;

    {
      double jacobian[] = { 1.0, 0.0, 0.0, 1.0};
      problem_.AddResidualBlock(new UnaryCostFunction(2, 2, jacobian), NULL, x);
    }

    {
      double jacobian[] = { 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0 };
      problem_.AddResidualBlock(new UnaryCostFunction(3, 3, jacobian), NULL, y);
    }

    {
      double jacobian = 5.0;
      problem_.AddResidualBlock(new UnaryCostFunction(1, 1, &jacobian),
                                NULL,
                                z);
    }

    {
      double jacobian1[] = { 1.0, 2.0, 3.0 };
      double jacobian2[] = { -5.0, -6.0 };
      problem_.AddResidualBlock(
          new BinaryCostFunction(1, 3, 2, jacobian1, jacobian2),
          NULL,
          y,
          x);
    }

    {
      double jacobian1[] = {2.0 };
      double jacobian2[] = { 3.0, -2.0 };
      problem_.AddResidualBlock(
          new BinaryCostFunction(1, 1, 2, jacobian1, jacobian2),
          NULL,
          z,
          x);
    }

    all_covariance_blocks_.push_back(make_pair(x, x));
    all_covariance_blocks_.push_back(make_pair(y, y));
    all_covariance_blocks_.push_back(make_pair(z, z));
    all_covariance_blocks_.push_back(make_pair(x, y));
    all_covariance_blocks_.push_back(make_pair(x, z));
    all_covariance_blocks_.push_back(make_pair(y, z));

    column_bounds_[x] = make_pair(0, 2);
    column_bounds_[y] = make_pair(2, 5);
    column_bounds_[z] = make_pair(5, 6);
  }

  // Computes covariance in ambient space.
  void ComputeAndCompareCovarianceBlocks(const Covariance::Options& options,
                                         const double* expected_covariance) {
    ComputeAndCompareCovarianceBlocksInTangentOrAmbientSpace(
        options,
        true,  // ambient
        expected_covariance);
  }

  // Computes covariance in tangent space.
  void ComputeAndCompareCovarianceBlocksInTangentSpace(
                                         const Covariance::Options& options,
                                         const double* expected_covariance) {
    ComputeAndCompareCovarianceBlocksInTangentOrAmbientSpace(
        options,
        false,  // tangent
        expected_covariance);
  }

  void ComputeAndCompareCovarianceBlocksInTangentOrAmbientSpace(
      const Covariance::Options& options,
      bool lift_covariance_to_ambient_space,
      const double* expected_covariance) {
    // Generate all possible combination of block pairs and check if the
    // covariance computation is correct.
    for (int i = 0; i <= 64; ++i) {
      vector<pair<const double*, const double*> > covariance_blocks;
      if (i & 1) {
        covariance_blocks.push_back(all_covariance_blocks_[0]);
      }

      if (i & 2) {
        covariance_blocks.push_back(all_covariance_blocks_[1]);
      }

      if (i & 4) {
        covariance_blocks.push_back(all_covariance_blocks_[2]);
      }

      if (i & 8) {
        covariance_blocks.push_back(all_covariance_blocks_[3]);
      }

      if (i & 16) {
        covariance_blocks.push_back(all_covariance_blocks_[4]);
      }

      if (i & 32) {
        covariance_blocks.push_back(all_covariance_blocks_[5]);
      }

      Covariance covariance(options);
      EXPECT_TRUE(covariance.Compute(covariance_blocks, &problem_));

      for (int i = 0; i < covariance_blocks.size(); ++i) {
        const double* block1 = covariance_blocks[i].first;
        const double* block2 = covariance_blocks[i].second;
        // block1, block2
        GetCovarianceBlockAndCompare(block1,
                                     block2,
                                     lift_covariance_to_ambient_space,
                                     covariance,
                                     expected_covariance);
        // block2, block1
        GetCovarianceBlockAndCompare(block2,
                                     block1,
                                     lift_covariance_to_ambient_space,
                                     covariance,
                                     expected_covariance);
      }
    }
  }

  void GetCovarianceBlockAndCompare(const double* block1,
                                    const double* block2,
                                    bool lift_covariance_to_ambient_space,
                                    const Covariance& covariance,
                                    const double* expected_covariance) {
    const BoundsMap& column_bounds = lift_covariance_to_ambient_space ?
        column_bounds_ : local_column_bounds_;
    const int row_begin = FindOrDie(column_bounds, block1).first;
    const int row_end = FindOrDie(column_bounds, block1).second;
    const int col_begin = FindOrDie(column_bounds, block2).first;
    const int col_end = FindOrDie(column_bounds, block2).second;

    Matrix actual(row_end - row_begin, col_end - col_begin);
    if (lift_covariance_to_ambient_space) {
      EXPECT_TRUE(covariance.GetCovarianceBlock(block1,
                                                block2,
                                                actual.data()));
    } else {
      EXPECT_TRUE(covariance.GetCovarianceBlockInTangentSpace(block1,
                                                              block2,
                                                              actual.data()));
    }

    int dof = 0;  // degrees of freedom = sum of LocalSize()s
    for (BoundsMap::const_iterator iter = column_bounds.begin();
         iter != column_bounds.end(); ++iter) {
      dof = std::max(dof, iter->second.second);
    }
    ConstMatrixRef expected(expected_covariance, dof, dof);
    double diff_norm = (expected.block(row_begin,
                                       col_begin,
                                       row_end - row_begin,
                                       col_end - col_begin) - actual).norm();
    diff_norm /= (row_end - row_begin) * (col_end - col_begin);

    const double kTolerance = 1e-5;
    EXPECT_NEAR(diff_norm, 0.0, kTolerance)
        << "rows: " << row_begin << " " << row_end << "  "
        << "cols: " << col_begin << " " << col_end << "  "
        << "\n\n expected: \n " << expected.block(row_begin,
                                                  col_begin,
                                                  row_end - row_begin,
                                                  col_end - col_begin)
        << "\n\n actual: \n " << actual
        << "\n\n full expected: \n" << expected;
  }

  double parameters_[6];
  Problem problem_;
  vector<pair<const double*, const double*> > all_covariance_blocks_;
  BoundsMap column_bounds_;
  BoundsMap local_column_bounds_;
};


TEST_F(CovarianceTest, NormalBehavior) {
  // J
  //
  //   1  0  0  0  0  0
  //   0  1  0  0  0  0
  //   0  0  2  0  0  0
  //   0  0  0  2  0  0
  //   0  0  0  0  2  0
  //   0  0  0  0  0  5
  //  -5 -6  1  2  3  0
  //   3 -2  0  0  0  2

  // J'J
  //
  //   35  24 -5 -10 -15  6
  //   24  41 -6 -12 -18 -4
  //   -5  -6  5   2   3  0
  //  -10 -12  2   8   6  0
  //  -15 -18  3   6  13  0
  //    6  -4  0   0   0 29

  // inv(J'J) computed using octave.
  double expected_covariance[] = {
     7.0747e-02,  -8.4923e-03,   1.6821e-02,   3.3643e-02,   5.0464e-02,  -1.5809e-02,  // NOLINT
    -8.4923e-03,   8.1352e-02,   2.4758e-02,   4.9517e-02,   7.4275e-02,   1.2978e-02,  // NOLINT
     1.6821e-02,   2.4758e-02,   2.4904e-01,  -1.9271e-03,  -2.8906e-03,  -6.5325e-05,  // NOLINT
     3.3643e-02,   4.9517e-02,  -1.9271e-03,   2.4615e-01,  -5.7813e-03,  -1.3065e-04,  // NOLINT
     5.0464e-02,   7.4275e-02,  -2.8906e-03,  -5.7813e-03,   2.4133e-01,  -1.9598e-04,  // NOLINT
    -1.5809e-02,   1.2978e-02,  -6.5325e-05,  -1.3065e-04,  -1.9598e-04,   3.9544e-02,  // NOLINT
  };

  Covariance::Options options;

#ifndef CERES_NO_SUITESPARSE
  options.algorithm_type = SUITE_SPARSE_QR;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);
#endif

  options.algorithm_type = DENSE_SVD;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);

  options.algorithm_type = EIGEN_SPARSE_QR;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);
}

#ifdef CERES_USE_OPENMP

TEST_F(CovarianceTest, ThreadedNormalBehavior) {
  // J
  //
  //   1  0  0  0  0  0
  //   0  1  0  0  0  0
  //   0  0  2  0  0  0
  //   0  0  0  2  0  0
  //   0  0  0  0  2  0
  //   0  0  0  0  0  5
  //  -5 -6  1  2  3  0
  //   3 -2  0  0  0  2

  // J'J
  //
  //   35  24 -5 -10 -15  6
  //   24  41 -6 -12 -18 -4
  //   -5  -6  5   2   3  0
  //  -10 -12  2   8   6  0
  //  -15 -18  3   6  13  0
  //    6  -4  0   0   0 29

  // inv(J'J) computed using octave.
  double expected_covariance[] = {
     7.0747e-02,  -8.4923e-03,   1.6821e-02,   3.3643e-02,   5.0464e-02,  -1.5809e-02,  // NOLINT
    -8.4923e-03,   8.1352e-02,   2.4758e-02,   4.9517e-02,   7.4275e-02,   1.2978e-02,  // NOLINT
     1.6821e-02,   2.4758e-02,   2.4904e-01,  -1.9271e-03,  -2.8906e-03,  -6.5325e-05,  // NOLINT
     3.3643e-02,   4.9517e-02,  -1.9271e-03,   2.4615e-01,  -5.7813e-03,  -1.3065e-04,  // NOLINT
     5.0464e-02,   7.4275e-02,  -2.8906e-03,  -5.7813e-03,   2.4133e-01,  -1.9598e-04,  // NOLINT
    -1.5809e-02,   1.2978e-02,  -6.5325e-05,  -1.3065e-04,  -1.9598e-04,   3.9544e-02,  // NOLINT
  };

  Covariance::Options options;
  options.num_threads = 4;

#ifndef CERES_NO_SUITESPARSE
  options.algorithm_type = SUITE_SPARSE_QR;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);
#endif

  options.algorithm_type = DENSE_SVD;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);

  options.algorithm_type = EIGEN_SPARSE_QR;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);
}

#endif  // CERES_USE_OPENMP

TEST_F(CovarianceTest, ConstantParameterBlock) {
  problem_.SetParameterBlockConstant(parameters_);

  // J
  //
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
  //  0  0  0  0  0  0
  //  0  0  0  0  0  0
  //  0  0  5  2  3  0
  //  0  0  2  8  6  0
  //  0  0  3  6 13  0
  //  0  0  0  0  0 29

  // pinv(J'J) computed using octave.
  double expected_covariance[] = {
              0,            0,            0,            0,            0,            0,  // NOLINT
              0,            0,            0,            0,            0,            0,  // NOLINT
              0,            0,      0.23611,     -0.02778,     -0.04167,     -0.00000,  // NOLINT
              0,            0,     -0.02778,      0.19444,     -0.08333,     -0.00000,  // NOLINT
              0,            0,     -0.04167,     -0.08333,      0.12500,     -0.00000,  // NOLINT
              0,            0,     -0.00000,     -0.00000,     -0.00000,      0.03448   // NOLINT
  };

  Covariance::Options options;

#ifndef CERES_NO_SUITESPARSE
  options.algorithm_type = SUITE_SPARSE_QR;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);
#endif

  options.algorithm_type = DENSE_SVD;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);

  options.algorithm_type = EIGEN_SPARSE_QR;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);
}

TEST_F(CovarianceTest, LocalParameterization) {
  double* x = parameters_;
  double* y = x + 2;

  problem_.SetParameterization(x, new PolynomialParameterization);

  vector<int> subset;
  subset.push_back(2);
  problem_.SetParameterization(y, new SubsetParameterization(3, subset));

  // Raw Jacobian: J
  //
  //   1   0  0  0  0  0
  //   0   1  0  0  0  0
  //   0   0  2  0  0  0
  //   0   0  0  2  0  0
  //   0   0  0  0  2  0
  //   0   0  0  0  0  5
  //  -5  -6  1  2  3  0
  //   3  -2  0  0  0  2

  // Local to global jacobian: A
  //
  //  1   0   0   0
  //  1   0   0   0
  //  0   1   0   0
  //  0   0   1   0
  //  0   0   0   0
  //  0   0   0   1

  // A * inv((J*A)'*(J*A)) * A'
  // Computed using octave.
  double expected_covariance[] = {
    0.01766,   0.01766,   0.02158,   0.04316,   0.00000,  -0.00122,
    0.01766,   0.01766,   0.02158,   0.04316,   0.00000,  -0.00122,
    0.02158,   0.02158,   0.24860,  -0.00281,   0.00000,  -0.00149,
    0.04316,   0.04316,  -0.00281,   0.24439,   0.00000,  -0.00298,
    0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,
   -0.00122,  -0.00122,  -0.00149,  -0.00298,   0.00000,   0.03457
  };

  Covariance::Options options;

#ifndef CERES_NO_SUITESPARSE
  options.algorithm_type = SUITE_SPARSE_QR;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);
#endif

  options.algorithm_type = DENSE_SVD;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);

  options.algorithm_type = EIGEN_SPARSE_QR;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);
}

TEST_F(CovarianceTest, LocalParameterizationInTangentSpace) {
  double* x = parameters_;
  double* y = x + 2;
  double* z = y + 3;

  problem_.SetParameterization(x, new PolynomialParameterization);

  vector<int> subset;
  subset.push_back(2);
  problem_.SetParameterization(y, new SubsetParameterization(3, subset));

  local_column_bounds_[x] = make_pair(0, 1);
  local_column_bounds_[y] = make_pair(1, 3);
  local_column_bounds_[z] = make_pair(3, 4);

  // Raw Jacobian: J
  //
  //   1   0  0  0  0  0
  //   0   1  0  0  0  0
  //   0   0  2  0  0  0
  //   0   0  0  2  0  0
  //   0   0  0  0  2  0
  //   0   0  0  0  0  5
  //  -5  -6  1  2  3  0
  //   3  -2  0  0  0  2

  // Local to global jacobian: A
  //
  //  1   0   0   0
  //  1   0   0   0
  //  0   1   0   0
  //  0   0   1   0
  //  0   0   0   0
  //  0   0   0   1

  // inv((J*A)'*(J*A))
  // Computed using octave.
  double expected_covariance[] = {
    0.01766,   0.02158,   0.04316,   -0.00122,
    0.02158,   0.24860,  -0.00281,   -0.00149,
    0.04316,  -0.00281,   0.24439,   -0.00298,
   -0.00122,  -0.00149,  -0.00298,    0.03457  // NOLINT
  };

  Covariance::Options options;

#ifndef CERES_NO_SUITESPARSE
  options.algorithm_type = SUITE_SPARSE_QR;
  ComputeAndCompareCovarianceBlocksInTangentSpace(options, expected_covariance);
#endif

  options.algorithm_type = DENSE_SVD;
  ComputeAndCompareCovarianceBlocksInTangentSpace(options, expected_covariance);

  options.algorithm_type = EIGEN_SPARSE_QR;
  ComputeAndCompareCovarianceBlocksInTangentSpace(options, expected_covariance);
}

TEST_F(CovarianceTest, LocalParameterizationInTangentSpaceWithConstantBlocks) {
  double* x = parameters_;
  double* y = x + 2;
  double* z = y + 3;

  problem_.SetParameterization(x, new PolynomialParameterization);
  problem_.SetParameterBlockConstant(x);

  vector<int> subset;
  subset.push_back(2);
  problem_.SetParameterization(y, new SubsetParameterization(3, subset));
  problem_.SetParameterBlockConstant(y);

  local_column_bounds_[x] = make_pair(0, 1);
  local_column_bounds_[y] = make_pair(1, 3);
  local_column_bounds_[z] = make_pair(3, 4);

  // Raw Jacobian: J
  //
  //   1   0  0  0  0  0
  //   0   1  0  0  0  0
  //   0   0  2  0  0  0
  //   0   0  0  2  0  0
  //   0   0  0  0  2  0
  //   0   0  0  0  0  5
  //  -5  -6  1  2  3  0
  //   3  -2  0  0  0  2

  // Local to global jacobian: A
  //
  //  0   0   0   0
  //  0   0   0   0
  //  0   0   0   0
  //  0   0   0   0
  //  0   0   0   0
  //  0   0   0   1

  // pinv((J*A)'*(J*A))
  // Computed using octave.
  double expected_covariance[] = {
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.034482 // NOLINT
  };

  Covariance::Options options;

#ifndef CERES_NO_SUITESPARSE
  options.algorithm_type = SUITE_SPARSE_QR;
  ComputeAndCompareCovarianceBlocksInTangentSpace(options, expected_covariance);
#endif

  options.algorithm_type = DENSE_SVD;
  ComputeAndCompareCovarianceBlocksInTangentSpace(options, expected_covariance);

  options.algorithm_type = EIGEN_SPARSE_QR;
  ComputeAndCompareCovarianceBlocksInTangentSpace(options, expected_covariance);
}

TEST_F(CovarianceTest, TruncatedRank) {
  // J
  //
  //   1  0  0  0  0  0
  //   0  1  0  0  0  0
  //   0  0  2  0  0  0
  //   0  0  0  2  0  0
  //   0  0  0  0  2  0
  //   0  0  0  0  0  5
  //  -5 -6  1  2  3  0
  //   3 -2  0  0  0  2

  // J'J
  //
  //   35  24 -5 -10 -15  6
  //   24  41 -6 -12 -18 -4
  //   -5  -6  5   2   3  0
  //  -10 -12  2   8   6  0
  //  -15 -18  3   6  13  0
  //    6  -4  0   0   0 29

  // 3.4142 is the smallest eigen value of J'J. The following matrix
  // was obtained by dropping the eigenvector corresponding to this
  // eigenvalue.
  double expected_covariance[] = {
     5.4135e-02,  -3.5121e-02,   1.7257e-04,   3.4514e-04,   5.1771e-04,  -1.6076e-02,  // NOLINT
    -3.5121e-02,   3.8667e-02,  -1.9288e-03,  -3.8576e-03,  -5.7864e-03,   1.2549e-02,  // NOLINT
     1.7257e-04,  -1.9288e-03,   2.3235e-01,  -3.5297e-02,  -5.2946e-02,  -3.3329e-04,  // NOLINT
     3.4514e-04,  -3.8576e-03,  -3.5297e-02,   1.7941e-01,  -1.0589e-01,  -6.6659e-04,  // NOLINT
     5.1771e-04,  -5.7864e-03,  -5.2946e-02,  -1.0589e-01,   9.1162e-02,  -9.9988e-04,  // NOLINT
    -1.6076e-02,   1.2549e-02,  -3.3329e-04,  -6.6659e-04,  -9.9988e-04,   3.9539e-02   // NOLINT
  };


  {
    Covariance::Options options;
    options.algorithm_type = DENSE_SVD;
    // Force dropping of the smallest eigenvector.
    options.null_space_rank = 1;
    ComputeAndCompareCovarianceBlocks(options, expected_covariance);
  }

  {
    Covariance::Options options;
    options.algorithm_type = DENSE_SVD;
    // Force dropping of the smallest eigenvector via the ratio but
    // automatic truncation.
    options.min_reciprocal_condition_number = 0.044494;
    options.null_space_rank = -1;
    ComputeAndCompareCovarianceBlocks(options, expected_covariance);
  }
}

TEST_F(CovarianceTest, DenseCovarianceMatrixFromSetOfParameters) {
  Covariance::Options options;
  Covariance covariance(options);
  double* x = parameters_;
  double* y = x + 2;
  double* z = y + 3;
  vector<const double*> parameter_blocks;
  parameter_blocks.push_back(x);
  parameter_blocks.push_back(y);
  parameter_blocks.push_back(z);
  covariance.Compute(parameter_blocks, &problem_);
  double expected_covariance[36];
  covariance.GetCovarianceMatrix(parameter_blocks, expected_covariance);

#ifndef CERES_NO_SUITESPARSE
  options.algorithm_type = SUITE_SPARSE_QR;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);
#endif

  options.algorithm_type = DENSE_SVD;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);

  options.algorithm_type = EIGEN_SPARSE_QR;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);
}

TEST_F(CovarianceTest, DenseCovarianceMatrixFromSetOfParametersThreaded) {
  Covariance::Options options;
  options.num_threads = 4;
  Covariance covariance(options);
  double* x = parameters_;
  double* y = x + 2;
  double* z = y + 3;
  vector<const double*> parameter_blocks;
  parameter_blocks.push_back(x);
  parameter_blocks.push_back(y);
  parameter_blocks.push_back(z);
  covariance.Compute(parameter_blocks, &problem_);
  double expected_covariance[36];
  covariance.GetCovarianceMatrix(parameter_blocks, expected_covariance);

#ifndef CERES_NO_SUITESPARSE
  options.algorithm_type = SUITE_SPARSE_QR;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);
#endif

  options.algorithm_type = DENSE_SVD;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);

  options.algorithm_type = EIGEN_SPARSE_QR;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);
}

TEST_F(CovarianceTest, DenseCovarianceMatrixFromSetOfParametersInTangentSpace) {
  Covariance::Options options;
  Covariance covariance(options);
  double* x = parameters_;
  double* y = x + 2;
  double* z = y + 3;

  problem_.SetParameterization(x, new PolynomialParameterization);

  vector<int> subset;
  subset.push_back(2);
  problem_.SetParameterization(y, new SubsetParameterization(3, subset));

  local_column_bounds_[x] = make_pair(0, 1);
  local_column_bounds_[y] = make_pair(1, 3);
  local_column_bounds_[z] = make_pair(3, 4);

  vector<const double*> parameter_blocks;
  parameter_blocks.push_back(x);
  parameter_blocks.push_back(y);
  parameter_blocks.push_back(z);
  covariance.Compute(parameter_blocks, &problem_);
  double expected_covariance[16];
  covariance.GetCovarianceMatrixInTangentSpace(parameter_blocks,
                                               expected_covariance);

#ifndef CERES_NO_SUITESPARSE
  options.algorithm_type = SUITE_SPARSE_QR;
  ComputeAndCompareCovarianceBlocksInTangentSpace(options, expected_covariance);
#endif

  options.algorithm_type = DENSE_SVD;
  ComputeAndCompareCovarianceBlocksInTangentSpace(options, expected_covariance);

  options.algorithm_type = EIGEN_SPARSE_QR;
  ComputeAndCompareCovarianceBlocksInTangentSpace(options, expected_covariance);
}

TEST_F(CovarianceTest, ComputeCovarianceFailure) {
  Covariance::Options options;
  Covariance covariance(options);
  double* x = parameters_;
  double* y = x + 2;
  vector<const double*> parameter_blocks;
  parameter_blocks.push_back(x);
  parameter_blocks.push_back(x);
  parameter_blocks.push_back(y);
  parameter_blocks.push_back(y);
  EXPECT_DEATH_IF_SUPPORTED(covariance.Compute(parameter_blocks, &problem_),
                            "Covariance::Compute called with duplicate blocks "
                            "at indices \\(0, 1\\) and \\(2, 3\\)");
  vector<pair<const double*, const double*> > covariance_blocks;
  covariance_blocks.push_back(make_pair(x, x));
  covariance_blocks.push_back(make_pair(x, x));
  covariance_blocks.push_back(make_pair(y, y));
  covariance_blocks.push_back(make_pair(y, y));
  EXPECT_DEATH_IF_SUPPORTED(covariance.Compute(covariance_blocks, &problem_),
                            "Covariance::Compute called with duplicate blocks "
                            "at indices \\(0, 1\\) and \\(2, 3\\)");
}

class RankDeficientCovarianceTest : public CovarianceTest {
 protected:
  virtual void SetUp() {
    double* x = parameters_;
    double* y = x + 2;
    double* z = y + 3;

    {
      double jacobian[] = { 1.0, 0.0, 0.0, 1.0};
      problem_.AddResidualBlock(new UnaryCostFunction(2, 2, jacobian), NULL, x);
    }

    {
      double jacobian[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
      problem_.AddResidualBlock(new UnaryCostFunction(3, 3, jacobian), NULL, y);
    }

    {
      double jacobian = 5.0;
      problem_.AddResidualBlock(new UnaryCostFunction(1, 1, &jacobian),
                                NULL,
                                z);
    }

    {
      double jacobian1[] = { 0.0, 0.0, 0.0 };
      double jacobian2[] = { -5.0, -6.0 };
      problem_.AddResidualBlock(
          new BinaryCostFunction(1, 3, 2, jacobian1, jacobian2),
          NULL,
          y,
          x);
    }

    {
      double jacobian1[] = {2.0 };
      double jacobian2[] = { 3.0, -2.0 };
      problem_.AddResidualBlock(
          new BinaryCostFunction(1, 1, 2, jacobian1, jacobian2),
          NULL,
          z,
          x);
    }

    all_covariance_blocks_.push_back(make_pair(x, x));
    all_covariance_blocks_.push_back(make_pair(y, y));
    all_covariance_blocks_.push_back(make_pair(z, z));
    all_covariance_blocks_.push_back(make_pair(x, y));
    all_covariance_blocks_.push_back(make_pair(x, z));
    all_covariance_blocks_.push_back(make_pair(y, z));

    column_bounds_[x] = make_pair(0, 2);
    column_bounds_[y] = make_pair(2, 5);
    column_bounds_[z] = make_pair(5, 6);
  }
};

TEST_F(RankDeficientCovarianceTest, AutomaticTruncation) {
  // J
  //
  //   1  0  0  0  0  0
  //   0  1  0  0  0  0
  //   0  0  0  0  0  0
  //   0  0  0  0  0  0
  //   0  0  0  0  0  0
  //   0  0  0  0  0  5
  //  -5 -6  0  0  0  0
  //   3 -2  0  0  0  2

  // J'J
  //
  //  35 24  0  0  0  6
  //  24 41  0  0  0 -4
  //   0  0  0  0  0  0
  //   0  0  0  0  0  0
  //   0  0  0  0  0  0
  //   6 -4  0  0  0 29

  // pinv(J'J) computed using octave.
  double expected_covariance[] = {
     0.053998,  -0.033145,   0.000000,   0.000000,   0.000000,  -0.015744,
    -0.033145,   0.045067,   0.000000,   0.000000,   0.000000,   0.013074,
     0.000000,   0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
     0.000000,   0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
     0.000000,   0.000000,   0.000000,   0.000000,   0.000000,   0.000000,
    -0.015744,   0.013074,   0.000000,   0.000000,   0.000000,   0.039543
  };

  Covariance::Options options;
  options.algorithm_type = DENSE_SVD;
  options.null_space_rank = -1;
  ComputeAndCompareCovarianceBlocks(options, expected_covariance);
}

class LargeScaleCovarianceTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    num_parameter_blocks_ = 2000;
    parameter_block_size_ = 5;
    parameters_.reset(
        new double[parameter_block_size_ * num_parameter_blocks_]);

    Matrix jacobian(parameter_block_size_, parameter_block_size_);
    for (int i = 0; i < num_parameter_blocks_; ++i) {
      jacobian.setIdentity();
      jacobian *= (i + 1);

      double* block_i = parameters_.get() + i * parameter_block_size_;
      problem_.AddResidualBlock(new UnaryCostFunction(parameter_block_size_,
                                                      parameter_block_size_,
                                                      jacobian.data()),
                                NULL,
                                block_i);
      for (int j = i; j < num_parameter_blocks_; ++j) {
        double* block_j = parameters_.get() + j * parameter_block_size_;
        all_covariance_blocks_.push_back(make_pair(block_i, block_j));
      }
    }
  }

  void ComputeAndCompare(CovarianceAlgorithmType algorithm_type,
                         int num_threads) {
    Covariance::Options options;
    options.algorithm_type = algorithm_type;
    options.num_threads = num_threads;
    Covariance covariance(options);
    EXPECT_TRUE(covariance.Compute(all_covariance_blocks_, &problem_));

    Matrix expected(parameter_block_size_, parameter_block_size_);
    Matrix actual(parameter_block_size_, parameter_block_size_);
    const double kTolerance = 1e-16;

    for (int i = 0; i < num_parameter_blocks_; ++i) {
      expected.setIdentity();
      expected /= (i + 1.0) * (i + 1.0);

      double* block_i = parameters_.get() + i * parameter_block_size_;
      covariance.GetCovarianceBlock(block_i, block_i, actual.data());
      EXPECT_NEAR((expected - actual).norm(), 0.0, kTolerance)
          << "block: " << i << ", " << i << "\n"
          << "expected: \n" << expected << "\n"
          << "actual: \n" << actual;

      expected.setZero();
      for (int j = i + 1; j < num_parameter_blocks_; ++j) {
        double* block_j = parameters_.get() + j * parameter_block_size_;
        covariance.GetCovarianceBlock(block_i, block_j, actual.data());
        EXPECT_NEAR((expected - actual).norm(), 0.0, kTolerance)
            << "block: " << i << ", " << j << "\n"
            << "expected: \n" << expected << "\n"
            << "actual: \n" << actual;
      }
    }
  }

  scoped_array<double> parameters_;
  int parameter_block_size_;
  int num_parameter_blocks_;

  Problem problem_;
  vector<pair<const double*, const double*> > all_covariance_blocks_;
};

#if !defined(CERES_NO_SUITESPARSE) && defined(CERES_USE_OPENMP)

TEST_F(LargeScaleCovarianceTest, Parallel) {
  ComputeAndCompare(SUITE_SPARSE_QR, 4);
}

#endif  // !defined(CERES_NO_SUITESPARSE) && defined(CERES_USE_OPENMP)


#ifndef CERES_NO_SUITESPARSE

/* Simple bundle adjust problem with cameras and points to test covariance
 * on Schur reduced system.
 *
 * A pinhole camera model with a three intrinsic parameters (f, cx, cy) is used.
 * Projection is simply im = f * cam + c.
 *
 * The points line on the XY plane (with some 3DOF noise). The cameras are
 * situated pointing down at the XY plane, giving them positive Z values.
 * Each camera is parameterized only as a 3DOF camera center and focal length.
 *
 * In this scene, f is designed to be unconstrained.
 */

typedef Eigen::Vector3d Vector3;
typedef Eigen::Vector2d Vector2;

struct Intrinsics {
  enum { NumParameters = 3 };

  union {
    double params[NumParameters];
    struct {
      double f, cx, cy;
    };
  };

  inline Vector2 Project(const Vector3& cam) const {
    return f * cam.head<2>() + Vector2(cx, cy);
  }
};

struct Point {
  Vector3 P;
};

struct Camera {
  Vector3 C;
};

struct Observation {
  Observation(const Vector2& _im, const size_t _camera, const size_t _point)
    : im(_im), camera(_camera), point(_point) {}
  Vector2 im;
  size_t camera;
  size_t point;
};

typedef std::vector<Camera> Cameras;
typedef std::vector<Point> Points;
typedef std::vector<Observation> Observations;

struct Scene {
  Intrinsics intrinsics;
  Cameras cameras;
  Points points;
  Observations observations;
  std::vector<Vector3> gps;
};

Vector3 RandomVector3() {
  Vector3 v(RandDoubleUniform(1), RandDoubleUniform(1), RandDoubleUniform(1));
  v.normalize();
  return v;
}

Vector2 RandomVector2() {
  Vector2 v(RandDoubleUniform(1), RandDoubleUniform(1));
  v.normalize();
  return v;
}

inline Scene CreateScene(int num_cameras, int num_points) {
  SetRandomState(1);

  // XY plane size.
  const double width = 100;
  const double height = 50;

  Scene s;

  // 45 deg field of view, sensor size of 1.
  const double fov = 60;
  const double f = 0.5 / std::tan(.5 * M_PI/180 * fov);
  s.intrinsics.f = f;
  s.intrinsics.cx = s.intrinsics.cy = 0;

  // Create points on XY plane.
  s.points.resize(num_points);
  for (size_t i = 0; i < s.points.size(); ++i) {
    Point& p = s.points[i];
    p.P.x() = RandDouble() * width;
    p.P.y() = RandDouble() * height;
    p.P.z() = 0;
  }

  // For sufficient overlap, each camera should see ~20% of the scene
  const double desired_area = .2 * width * height;
  const double Z = std::sqrt(desired_area) * s.intrinsics.f;

  // Create a border so camera frustum is inside scene
  const double border = .5 / s.intrinsics.f * Z;
  const double cam_width = width - 2 * border;
  const double cam_height = height - 2 * border;

  // Create cameras looking at XY plane
  s.cameras.resize(num_cameras);
  for (size_t i = 0; i < s.cameras.size(); ++i) {
    Camera& c = s.cameras[i];
    c.C.x() = RandDouble() * cam_width + border;
    c.C.y() = RandDouble() * cam_height + border;
    c.C.z() = Z;
  }

  // Create observations
  s.observations.reserve(num_cameras * num_points * .2);
  const double im_noise = 2 / 4000.0;
  for (size_t pi = 0; pi < s.points.size(); ++pi) {
    const Point& p = s.points[pi];
    size_t start = s.observations.size();
    for (size_t ci = 0; ci < s.cameras.size(); ++ci) {
      Camera& c = s.cameras[ci];

      Vector3 cam = c.C - p.P;
      cam /= cam.z();

      Vector2 im = s.intrinsics.Project(cam);
      if (im.x() < -.5 || im.x() > .5
          || im.y() < -.5 || im.y() > .5) {
        // outside image
        continue;
      }

      // add some noise to image measurements
      Vector2 nvec = RandomVector2() * RandNormal() * im_noise;
      im += nvec;

      s.observations.push_back(Observation(im, ci, pi));
    }

    if (s.observations.size() - start < 3) {
      // point needs at least two cameras to constrain
      s.observations.erase(s.observations.begin() + start, s.observations.end());
    }
  }

  // Create "GPS" measurements near the cameras but increase the Z
  s.gps.reserve(s.cameras.size());
  for (size_t i = 0; i < s.cameras.size(); ++i) {
    Camera& c = s.cameras[i];
    Vector3 gps = c.C;
    gps.z() += 1;
    s.gps.push_back(gps);
  }

  // Perturb parameters to allow for optimization to do something
  const double point_noise = 1;
  for (size_t i = 0; i < s.points.size(); ++i) {
    Point& p = s.points[i];
    p.P += RandomVector3() * RandNormal() * point_noise;
  }
  const double cam_noise = 1;
  for (size_t i = 0; i < s.cameras.size(); ++i) {
    Camera& c = s.cameras[i];
    c.C += RandomVector3() * RandNormal() * cam_noise;
  }
  const double gps_noise = 1;
  for (size_t i = 0; i < s.gps.size(); ++i) {
    Vector3& g = s.gps[i];
    g += RandomVector3() * RandNormal() * gps_noise;
  }
  s.intrinsics.f += .5 / std::atan(.5 * M_PI/180.0 * (fov - 10.0));

  return s;
}

struct ObservationPointIC  {
  enum {
    NumResiduals = 2,
    NumParams = 3
  };

  // the observation coordinates in image plane
  inline ObservationPointIC(const Vector2& _pos) : pos(_pos) {}

  template <typename T>
  void Project(const T _cx[2], const T* intrinsics, T x[2]) const {
    // compute projection in image coordinates
    const T& f = intrinsics[0];
    const T& cx = intrinsics[1];
    const T& cy = intrinsics[2];
    x[0] = f * _cx[0] + cx;
    x[1] = f * _cx[1] + cy;
  }

  // cost function for regular cameras
  template <typename T>
  bool operator()(const T* const intrinsics, const T* const center,
                  const T* const point, T* residuals) const
  {
    // translate point to camera position
    T x[3] = {
      center[0] - point[0],
      center[1] - point[1],
      center[2] - point[2]
    };

    // transform the point in camera space to inhomogeneous coordinates
    x[0] /= x[2];
    x[1] /= x[2];

    // transform the point to image plane
    T m[2] = { T(pos.x()), T(pos.y()) };
    Project(x, intrinsics, x);

    // the reprojection error is the difference between the predicted and observed position
    residuals[0] = x[0] - m[0];
    residuals[1] = x[1] - m[1];

    return true;
  }

  Vector2 pos;
};

struct APrioriCameraPosition  {
  enum { NumResiduals = 3 };

  // the observation coordinates in image plane
  inline APrioriCameraPosition(const Vector3& _pos) : pos(_pos) {}

  // cost function for regular cameras
  template <typename T>
  bool operator()(const T* const center, T* residuals) const
  {
    residuals[0] = center[0] - pos[0];
    residuals[1] = center[1] - pos[1];
    residuals[2] = center[2] - pos[2];

    return true;
  }

  Vector3 pos;
};

typedef AutoDiffCostFunction<
  ObservationPointIC,
  ObservationPointIC::NumResiduals,
  ObservationPointIC::NumParams,
  3,
  3> PointCostFunction;

typedef AutoDiffCostFunction<
  APrioriCameraPosition,
  APrioriCameraPosition::NumResiduals,
  3> GPSCostFunction;

class FastSchurCovarianceTest : public ::testing::Test {
 protected:

  ParameterBlockOrdering* AddResidualBlocks(Problem& problem, Scene& s)
  {
    std::set<size_t> point_indices, camera_indices;
    for (size_t i = 0; i < s.observations.size(); ++i) {
      const Observation& ob = s.observations[i];
      problem.AddResidualBlock(new PointCostFunction(new ObservationPointIC(ob.im)),
                               0, /* loss function == squared error */
                               s.intrinsics.params,
                               &s.cameras[ob.camera].C.x(),
          &s.points[ob.point].P.x());
      point_indices.insert(ob.point);
      camera_indices.insert(ob.camera);
    }

    for (size_t c = 0; c < s.gps.size(); ++c) {
      problem.AddResidualBlock(new GPSCostFunction(new APrioriCameraPosition(s.gps[c])),
                               0, /* loss function == squared error */
                               &s.cameras[c].C.x());
    }

    // Create explicit parameter blocks
    // points, cameras, then intrinsics
    ParameterBlockOrdering* ordering = new ParameterBlockOrdering;
    for (std::set<size_t>::iterator it = point_indices.begin();
         it != point_indices.end(); ++it) {
      ordering->AddElementToGroup(&s.points[*it].P.x(), 0);
    }
    for (std::set<size_t>::iterator it = camera_indices.begin();
         it != camera_indices.end(); ++it) {
      ordering->AddElementToGroup(&s.cameras[*it].C.x(), 1);
    }
    ordering->AddElementToGroup(s.intrinsics.params, 2);

    return ordering;
  }

  Scene scene;
  shared_ptr<Problem> problem;
  shared_ptr<ParameterBlockOrdering> ordering;
  Solver::Options options;

  typedef std::pair<const double*, const double*> covblock_t;
  std::vector<covblock_t> covariance_blocks;

  virtual void SetUp() {
    scene = CreateScene(20, 500);

    problem.reset(new Problem);

    ordering.reset(AddResidualBlocks(*problem, scene));

    options.linear_solver_type = SPARSE_SCHUR;
    options.preconditioner_type = SCHUR_JACOBI;
    options.max_num_iterations = 1;
    options.num_threads = 1;
    options.num_linear_solver_threads = options.num_threads;
    options.linear_solver_ordering.reset(new ParameterBlockOrdering(*ordering));

    // we want covariance for cameras and focal length
    std::set<double*> elements = ordering->group_to_elements().at(1);
    for (std::set<double*>::const_iterator it = elements.begin();
         it != elements.end(); ++it) {
      covariance_blocks.push_back(std::make_pair(*it, *it));
    }
    elements = ordering->group_to_elements().at(2);
    for (std::set<double*>::const_iterator it = elements.begin();
         it != elements.end(); ++it) {
      covariance_blocks.push_back(std::make_pair(*it, *it));
    }
  }

  void CompareCovariance() {
    // get covariance using sparse schur method
    Covariance::Options coptions_schur;
    coptions_schur.linear_solver_ordering.reset(new ParameterBlockOrdering(*ordering));
    if (IsSchurType(options.linear_solver_type)) {
      coptions_schur.algorithm_type = SUITE_SPARSE_SCHUR_CHOLESKY;
    }
    Covariance cov_schur(coptions_schur);
    ASSERT_TRUE(cov_schur.Compute(covariance_blocks, problem.get()));

    Covariance::Options coptions;
    coptions.num_threads = options.num_threads;
    Covariance cov(coptions);
    ASSERT_TRUE(cov.Compute(covariance_blocks, problem.get()));

    double camCov[3*3], camCovSchur[3*3];
    std::set<double*> elements = ordering->group_to_elements().at(1);
    for (std::set<double*>::const_iterator it = elements.begin();
         it != elements.end(); ++it) {
      double* cam = *it;
      cov.GetCovarianceBlock(cam, cam, camCov);
      cov_schur.GetCovarianceBlock(cam, cam, camCovSchur);
      for (int i = 0; i < 9; ++i) {
        ASSERT_NEAR(camCov[i], camCovSchur[i], 1e-10);
      }
    }

    const unsigned iCovSize = Intrinsics::NumParameters * Intrinsics::NumParameters;
    double fCov[iCovSize], fCovSchur[iCovSize];
    elements = ordering->group_to_elements().at(2);
    for (std::set<double*>::const_iterator it = elements.begin();
         it != elements.end(); ++it) {
      double* i = *it;
      cov.GetCovarianceBlock(i, i, fCov);
      cov_schur.GetCovarianceBlock(i, i, fCovSchur);
      for (unsigned i = 0; i < iCovSize; ++i) {
        ASSERT_NEAR(fCov[i], fCovSchur[i], 1e-10);
      }
    }
  }
};

TEST_F(FastSchurCovarianceTest, Basic)
{
  Solver::Summary summary;
  Solve(options, problem.get(), &summary);
  CompareCovariance();

  // Ensure that constant parameters work.
  problem->SetParameterBlockConstant(scene.intrinsics.params);
  CompareCovariance();

  // Ensure that covariance computation doesn't break the problem structure
  problem->SetParameterBlockVariable(scene.intrinsics.params);
  CompareCovariance();

  // Ensure that partial constant parameters work.
  std::vector<int> consts;
  consts.push_back(1);
  problem->SetParameterization(scene.intrinsics.params,
                              new SubsetParameterization(scene.intrinsics.NumParameters,
                                                         consts));
  CompareCovariance();
}
#endif // CERES_NO_SUITESPARSE

}  // namespace internal
}  // namespace ceres
