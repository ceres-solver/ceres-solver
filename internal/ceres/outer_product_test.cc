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

#include "ceres/outer_product.h"

#include <numeric>
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/random.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

#include "Eigen/SparseCore"

namespace ceres {
namespace internal {

template <enum Eigen::UpLoType T>
void CompareTriangularPartOfMatrices(const Matrix& expected,
                                     const Matrix& actual) {
  EXPECT_EQ(actual.rows(), actual.cols());
  EXPECT_EQ(expected.rows(), expected.cols());
  EXPECT_EQ(actual.rows(), expected.rows());

  const Matrix expected_t = expected.triangularView<T>();
  const Matrix actual_t = actual.triangularView<T>();
  LOG(INFO) << "\n" << expected_t;

  EXPECT_NEAR((expected_t - actual_t).norm() / actual_t.norm(),
              0.0,
              std::numeric_limits<double>::epsilon())
      << "expected: \n"
      << expected_t << "\nactual: \n"
      << actual_t;
}

TEST(OuterProduct, NormalOperation) {
  // "Randomly generated seed."
  SetRandomState(29823);
  const int kMaxNumRowBlocks = 10;
  const int kMaxNumColBlocks = 10;
  const int kNumTrials = 10;

  // Create a random matrix, compute its outer product using Eigen and
  // ComputeOuterProduct. Convert both matrices to dense matrices and
  // compare their upper triangular parts.
  for (int num_row_blocks = 1; num_row_blocks < kMaxNumRowBlocks;
       ++num_row_blocks) {
    for (int num_col_blocks = 1; num_col_blocks < kMaxNumColBlocks;
         ++num_col_blocks) {
      for (int trial = 0; trial < kNumTrials; ++trial) {
        CompressedRowSparseMatrix::RandomMatrixOptions options;
        options.num_row_blocks = num_row_blocks;
        options.num_col_blocks = num_col_blocks;
        options.min_row_block_size = 1;
        options.max_row_block_size = 5;
        options.min_col_block_size = 1;
        options.max_col_block_size = 10;
        options.block_density = std::max(0.1, RandDouble());

        VLOG(2) << "num row blocks: " << options.num_row_blocks;
        VLOG(2) << "num col blocks: " << options.num_col_blocks;
        VLOG(2) << "min row block size: " << options.min_row_block_size;
        VLOG(2) << "max row block size: " << options.max_row_block_size;
        VLOG(2) << "min col block size: " << options.min_col_block_size;
        VLOG(2) << "max col block size: " << options.max_col_block_size;
        VLOG(2) << "block density: " << options.block_density;

        scoped_ptr<CompressedRowSparseMatrix> random_matrix(
            CompressedRowSparseMatrix::CreateRandomMatrix(options));

        Eigen::MappedSparseMatrix<double, Eigen::RowMajor> mapped_random_matrix(
            random_matrix->num_rows(),
            random_matrix->num_cols(),
            random_matrix->num_nonzeros(),
            random_matrix->mutable_rows(),
            random_matrix->mutable_cols(),
            random_matrix->mutable_values());

        Matrix expected_outer_product =
            mapped_random_matrix.transpose() * mapped_random_matrix;

        // Lower triangular
        {
          scoped_ptr<OuterProduct> outer_product(OuterProduct::Create(
              *random_matrix, CompressedRowSparseMatrix::LOWER_TRIANGULAR));
          outer_product->ComputeProduct();
          CompressedRowSparseMatrix* actual_product_crsm =
              outer_product->mutable_matrix();

          EXPECT_EQ(actual_product_crsm->row_blocks(),
                    random_matrix->col_blocks());
          EXPECT_EQ(actual_product_crsm->col_blocks(),
                    random_matrix->col_blocks());

          Matrix actual_outer_product =
              Eigen::MappedSparseMatrix<double, Eigen::ColMajor>(
                  actual_product_crsm->num_rows(),
                  actual_product_crsm->num_rows(),
                  actual_product_crsm->num_nonzeros(),
                  actual_product_crsm->mutable_rows(),
                  actual_product_crsm->mutable_cols(),
                  actual_product_crsm->mutable_values());
          CompareTriangularPartOfMatrices<Eigen::Upper>(expected_outer_product,
                                                        actual_outer_product);
        }

        // Upper triangular
        {
          scoped_ptr<OuterProduct> outer_product(OuterProduct::Create(
              *random_matrix, CompressedRowSparseMatrix::UPPER_TRIANGULAR));
          outer_product->ComputeProduct();
          CompressedRowSparseMatrix* actual_product_crsm =
              outer_product->mutable_matrix();

          EXPECT_EQ(actual_product_crsm->row_blocks(),
                    random_matrix->col_blocks());
          EXPECT_EQ(actual_product_crsm->col_blocks(),
                    random_matrix->col_blocks());

          Matrix actual_outer_product =
              Eigen::MappedSparseMatrix<double, Eigen::ColMajor>(
                  actual_product_crsm->num_rows(),
                  actual_product_crsm->num_rows(),
                  actual_product_crsm->num_nonzeros(),
                  actual_product_crsm->mutable_rows(),
                  actual_product_crsm->mutable_cols(),
                  actual_product_crsm->mutable_values());
          CompareTriangularPartOfMatrices<Eigen::Lower>(expected_outer_product,
                                                        actual_outer_product);
        }
      }
    }
  }
}

TEST(OuterProduct, SubMatrix) {
  // "Randomly generated seed."
  SetRandomState(29823);
  const int kNumRowBlocks = 10;
  const int kNumColBlocks = 20;
  const int kNumTrials = 5;

  // Create a random matrix, compute its outer product using Eigen and
  // ComputeOuterProduct. Convert both matrices to dense matrices and
  // compare their upper triangular parts.
  for (int trial = 0; trial < kNumTrials; ++trial) {
    CompressedRowSparseMatrix::RandomMatrixOptions options;
    options.num_row_blocks = kNumRowBlocks;
    options.num_col_blocks = kNumColBlocks;
    options.min_row_block_size = 1;
    options.max_row_block_size = 5;
    options.min_col_block_size = 1;
    options.max_col_block_size = 10;
    options.block_density = std::max(0.1, RandDouble());

    VLOG(2) << "num row blocks: " << options.num_row_blocks;
    VLOG(2) << "num col blocks: " << options.num_col_blocks;
    VLOG(2) << "min row block size: " << options.min_row_block_size;
    VLOG(2) << "max row block size: " << options.max_row_block_size;
    VLOG(2) << "min col block size: " << options.min_col_block_size;
    VLOG(2) << "max col block size: " << options.max_col_block_size;
    VLOG(2) << "block density: " << options.block_density;

    scoped_ptr<CompressedRowSparseMatrix> random_matrix(
        CompressedRowSparseMatrix::CreateRandomMatrix(options));

    const std::vector<int>& row_blocks = random_matrix->row_blocks();
    const int num_row_blocks = row_blocks.size();

    for (int start_row_block = 0; start_row_block < num_row_blocks - 1;
         ++start_row_block) {
      for (int end_row_block = start_row_block + 1;
           end_row_block < num_row_blocks;
           ++end_row_block) {
        const int start_row =
            std::accumulate(&row_blocks[0], &row_blocks[start_row_block], 0);
        const int end_row =
            std::accumulate(&row_blocks[0], &row_blocks[end_row_block], 0);

        Eigen::MappedSparseMatrix<double, Eigen::RowMajor> mapped_random_matrix(
            end_row - start_row,
            random_matrix->num_cols(),
            random_matrix->num_nonzeros(),
            random_matrix->mutable_rows() + start_row,
            random_matrix->mutable_cols(),
            random_matrix->mutable_values());

        Matrix expected_outer_product =
            mapped_random_matrix.transpose() * mapped_random_matrix;

        // Lower triangular
        {
          scoped_ptr<OuterProduct> outer_product(OuterProduct::Create(
              *random_matrix,
              start_row_block,
              end_row_block,
              CompressedRowSparseMatrix::LOWER_TRIANGULAR));
          outer_product->ComputeProduct();
          CompressedRowSparseMatrix* actual_product_crsm =
              outer_product->mutable_matrix();

          EXPECT_EQ(actual_product_crsm->row_blocks(),
                    random_matrix->col_blocks());
          EXPECT_EQ(actual_product_crsm->col_blocks(),
                    random_matrix->col_blocks());

          Matrix actual_outer_product =
              Eigen::MappedSparseMatrix<double, Eigen::ColMajor>(
                  actual_product_crsm->num_rows(),
                  actual_product_crsm->num_rows(),
                  actual_product_crsm->num_nonzeros(),
                  actual_product_crsm->mutable_rows(),
                  actual_product_crsm->mutable_cols(),
                  actual_product_crsm->mutable_values());
          CompareTriangularPartOfMatrices<Eigen::Upper>(expected_outer_product,
                                                        actual_outer_product);
        }

        // Upper triangular
        {
          scoped_ptr<OuterProduct> outer_product(OuterProduct::Create(
              *random_matrix,
              start_row_block,
              end_row_block,
              CompressedRowSparseMatrix::UPPER_TRIANGULAR));
          outer_product->ComputeProduct();
          CompressedRowSparseMatrix* actual_product_crsm =
              outer_product->mutable_matrix();

          EXPECT_EQ(actual_product_crsm->row_blocks(),
                    random_matrix->col_blocks());
          EXPECT_EQ(actual_product_crsm->col_blocks(),
                    random_matrix->col_blocks());

          Matrix actual_outer_product =
              Eigen::MappedSparseMatrix<double, Eigen::ColMajor>(
                  actual_product_crsm->num_rows(),
                  actual_product_crsm->num_rows(),
                  actual_product_crsm->num_nonzeros(),
                  actual_product_crsm->mutable_rows(),
                  actual_product_crsm->mutable_cols(),
                  actual_product_crsm->mutable_values());
          CompareTriangularPartOfMatrices<Eigen::Lower>(expected_outer_product,
                                                        actual_outer_product);
        }
      }
    }
  }
}

}  // namespace internal
}  // namespace ceres
