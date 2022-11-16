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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/partitioned_matrix_view.h"

#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "ceres/block_structure.h"
#include "ceres/casts.h"
#include "ceres/internal/eigen.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/sparse_matrix.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

const double kEpsilon = 1e-14;

class PartitionedMatrixViewTest : public ::testing::Test {
 protected:
  void SetUp() final {
    std::unique_ptr<LinearLeastSquaresProblem> problem =
        CreateLinearLeastSquaresProblemFromId(2);
    CHECK(problem != nullptr);
    A_ = std::move(problem->A);

    num_cols_ = A_->num_cols();
    num_rows_ = A_->num_rows();
    num_eliminate_blocks_ = problem->num_eliminate_blocks;
    LinearSolver::Options options;
    options.elimination_groups.push_back(num_eliminate_blocks_);
    pmv_ = PartitionedMatrixViewBase::Create(
        options, *down_cast<BlockSparseMatrix*>(A_.get()));
  }

  double RandDouble() { return distribution_(prng_); }

  int num_rows_;
  int num_cols_;
  int num_eliminate_blocks_;
  std::unique_ptr<SparseMatrix> A_;
  std::unique_ptr<PartitionedMatrixViewBase> pmv_;
  std::mt19937 prng_;
  std::uniform_real_distribution<double> distribution_ =
      std::uniform_real_distribution<double>(0.0, 1.0);
};

TEST_F(PartitionedMatrixViewTest, BlockDiagonalEtE) {
  std::unique_ptr<BlockSparseMatrix> block_diagonal_ee(
      pmv_->CreateBlockDiagonalEtE());
  const CompressedRowBlockStructure* bs = block_diagonal_ee->block_structure();

  EXPECT_EQ(block_diagonal_ee->num_rows(), 2);
  EXPECT_EQ(block_diagonal_ee->num_cols(), 2);
  EXPECT_EQ(bs->cols.size(), 2);
  EXPECT_EQ(bs->rows.size(), 2);

  EXPECT_NEAR(block_diagonal_ee->values()[0], 10.0, kEpsilon);
  EXPECT_NEAR(block_diagonal_ee->values()[1], 155.0, kEpsilon);
}

TEST_F(PartitionedMatrixViewTest, BlockDiagonalFtF) {
  std::unique_ptr<BlockSparseMatrix> block_diagonal_ff(
      pmv_->CreateBlockDiagonalFtF());
  const CompressedRowBlockStructure* bs = block_diagonal_ff->block_structure();

  EXPECT_EQ(block_diagonal_ff->num_rows(), 3);
  EXPECT_EQ(block_diagonal_ff->num_cols(), 3);
  EXPECT_EQ(bs->cols.size(), 3);
  EXPECT_EQ(bs->rows.size(), 3);
  EXPECT_NEAR(block_diagonal_ff->values()[0], 70.0, kEpsilon);
  EXPECT_NEAR(block_diagonal_ff->values()[1], 17.0, kEpsilon);
  EXPECT_NEAR(block_diagonal_ff->values()[2], 37.0, kEpsilon);
}

// Param = <problem_id, num_threads>
using Param = ::testing::tuple<int, int>;

static std::string ParamInfoToString(testing::TestParamInfo<Param> info) {
  Param param = info.param;
  std::stringstream ss;
  ss << ::testing::get<0>(param) << "_" << ::testing::get<1>(param);
  return ss.str();
}

class PartitionedMatrixViewSpMVTest : public ::testing::TestWithParam<Param> {
 protected:
  void SetUp() final {
    const int problem_id = ::testing::get<0>(GetParam());
    const int num_threads = ::testing::get<1>(GetParam());
    auto problem = CreateLinearLeastSquaresProblemFromId(problem_id);
    CHECK(problem != nullptr);
    A_ = std::move(problem->A);
    auto block_sparse = down_cast<BlockSparseMatrix*>(A_.get());
    block_sparse->AddTransposeBlockStructure();

    options_.num_threads = num_threads;
    options_.context = &context_;
    options_.elimination_groups.push_back(problem->num_eliminate_blocks);
    pmv_ = PartitionedMatrixViewBase::Create(options_, *block_sparse);

    EXPECT_EQ(pmv_->num_col_blocks_e(), problem->num_eliminate_blocks);
    EXPECT_EQ(pmv_->num_col_blocks_f(),
              block_sparse->block_structure()->cols.size() -
                  problem->num_eliminate_blocks);
    EXPECT_EQ(pmv_->num_cols(), A_->num_cols());
    EXPECT_EQ(pmv_->num_rows(), A_->num_rows());
  }

  double RandDouble() { return distribution_(prng_); }

  LinearSolver::Options options_;
  ContextImpl context_;
  std::unique_ptr<LinearLeastSquaresProblem> problem_;
  std::unique_ptr<SparseMatrix> A_;
  std::unique_ptr<PartitionedMatrixViewBase> pmv_;
  std::mt19937 prng_;
  std::uniform_real_distribution<double> distribution_ =
      std::uniform_real_distribution<double>(0.0, 1.0);
};

TEST_P(PartitionedMatrixViewSpMVTest, RightMultiplyAndAccumulateE) {
  Vector x1(pmv_->num_cols_e());
  Vector x2(pmv_->num_cols());
  x2.setZero();

  for (int i = 0; i < pmv_->num_cols_e(); ++i) {
    x1(i) = x2(i) = RandDouble();
  }

  Vector expected = Vector::Zero(pmv_->num_rows());
  A_->RightMultiplyAndAccumulate(x2.data(), expected.data());

  Vector actual = Vector::Zero(pmv_->num_rows());
  pmv_->RightMultiplyAndAccumulateE(x1.data(), actual.data());

  for (int i = 0; i < pmv_->num_rows(); ++i) {
    EXPECT_NEAR(actual(i), expected(i), kEpsilon);
  }
}

TEST_P(PartitionedMatrixViewSpMVTest, RightMultiplyAndAccumulateF) {
  Vector x1(pmv_->num_cols_f());
  Vector x2(pmv_->num_cols());
  x2.setZero();

  for (int i = 0; i < pmv_->num_cols_f(); ++i) {
    x1(i) = x2(i + pmv_->num_cols_e()) = RandDouble();
  }

  Vector actual = Vector::Zero(pmv_->num_rows());
  pmv_->RightMultiplyAndAccumulateF(x1.data(), actual.data());

  Vector expected = Vector::Zero(pmv_->num_rows());
  A_->RightMultiplyAndAccumulate(x2.data(), expected.data());

  for (int i = 0; i < pmv_->num_rows(); ++i) {
    EXPECT_NEAR(actual(i), expected(i), kEpsilon);
  }
}

TEST_P(PartitionedMatrixViewSpMVTest, LeftMultiplyAndAccumulate) {
  Vector x = Vector::Zero(pmv_->num_rows());
  for (int i = 0; i < pmv_->num_rows(); ++i) {
    x(i) = RandDouble();
  }
  Vector x_pre = x;

  Vector expected = Vector::Zero(pmv_->num_cols());
  Vector e_actual = Vector::Zero(pmv_->num_cols_e());
  Vector f_actual = Vector::Zero(pmv_->num_cols_f());

  A_->LeftMultiplyAndAccumulate(x.data(), expected.data());
  pmv_->LeftMultiplyAndAccumulateE(x.data(), e_actual.data());
  pmv_->LeftMultiplyAndAccumulateF(x.data(), f_actual.data());

  for (int i = 0; i < pmv_->num_cols(); ++i) {
    EXPECT_NEAR(expected(i),
                (i < pmv_->num_cols_e()) ? e_actual(i)
                                         : f_actual(i - pmv_->num_cols_e()),
                kEpsilon);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ParallelProducts,
    PartitionedMatrixViewSpMVTest,
    ::testing::Combine(::testing::Values(2, 4, 6),
                       ::testing::Values(1, 2, 3, 4, 5, 6, 7, 8)),
    ParamInfoToString);

}  // namespace internal
}  // namespace ceres
