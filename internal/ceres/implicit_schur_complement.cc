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

#include "ceres/implicit_schur_complement.h"

#include "Eigen/Dense"
#include "ceres/block_sparse_matrix.h"
#include "ceres/block_structure.h"
#include "ceres/internal/eigen.h"
#include "ceres/linear_solver.h"
#include "ceres/parallel_for.h"
#include "ceres/types.h"
#include "glog/logging.h"

namespace ceres::internal {

namespace {
// Dense operations on vectors might use vector instructions, thus we
// split the whole vector into blocks of size being a multiple of power of two.
const int kBlockSize = 1024 * 128;
// y = D^2 * x
void D2X(const double* D,
         const double* x,
         double* y,
         int num_values,
         ContextImpl* context,
         int num_threads) {
  const int NumBlocks = (num_values + kBlockSize - 1) / kBlockSize;
  ParallelFor(context,
              0,
              NumBlocks,
              num_threads,
              [D, x, y, num_values](const int block_id) {
                const int block_start = kBlockSize * block_id;
                const int block_size =
                    std::min(num_values - block_start, kBlockSize);
                ConstVectorRef block_D(D + block_start, block_size);
                ConstVectorRef block_x(x + block_start, block_size);
                VectorRef block_y(y + block_start, block_size);
                block_y = block_D.array().square() * block_x.array();
              });
}
// Set vector to zero
void SetZero(double* data,
             int num_values,
             ContextImpl* context,
             int num_threads) {
  const int NumBlocks = (num_values + kBlockSize - 1) / kBlockSize;
  ParallelFor(context,
              0,
              NumBlocks,
              num_threads,
              [data, num_values](const int block_id) {
                const int block_start = kBlockSize * block_id;
                const int block_size =
                    std::min(num_values - block_start, kBlockSize);
                VectorRef(data + block_start, block_size).setZero();
              });
}
// y = x
void Assign(const double* x,
            double* y,
            int num_values,
            ContextImpl* context,
            int num_threads) {
  const int NumBlocks = (num_values + kBlockSize - 1) / kBlockSize;
  ParallelFor(context,
              0,
              NumBlocks,
              num_threads,
              [x, y, num_values](const int block_id) {
                const int block_start = kBlockSize * block_id;
                const int block_size =
                    std::min(num_values - block_start, kBlockSize);
                ConstVectorRef block_x(x + block_start, block_size);
                VectorRef block_y(y + block_start, block_size);
                block_y = block_x;
              });
}
}  // namespace

ImplicitSchurComplement::ImplicitSchurComplement(
    const LinearSolver::Options& options)
    : options_(options) {}

void ImplicitSchurComplement::Init(const BlockSparseMatrix& A,
                                   const double* D,
                                   const double* b) {
  // Since initialization is reasonably heavy, perhaps we can save on
  // constructing a new object everytime.
  if (A_ == nullptr) {
    A_ = PartitionedMatrixViewBase::Create(options_, A);
  }

  D_ = D;
  b_ = b;

  compute_ftf_inverse_ =
      options_.use_spse_initialization ||
      options_.preconditioner_type == JACOBI ||
      options_.preconditioner_type == SCHUR_POWER_SERIES_EXPANSION;

  // Initialize temporary storage and compute the block diagonals of
  // E'E and F'E.
  if (block_diagonal_EtE_inverse_ == nullptr) {
    block_diagonal_EtE_inverse_ = A_->CreateBlockDiagonalEtE();
    if (compute_ftf_inverse_) {
      block_diagonal_FtF_inverse_ = A_->CreateBlockDiagonalFtF();
    }
    rhs_.resize(A_->num_cols_f());
    rhs_.setZero();
    tmp_rows_.resize(A_->num_rows());
    tmp_e_cols_.resize(A_->num_cols_e());
    tmp_e_cols_2_.resize(A_->num_cols_e());
    tmp_f_cols_.resize(A_->num_cols_f());
  } else {
    A_->UpdateBlockDiagonalEtE(block_diagonal_EtE_inverse_.get());
    if (compute_ftf_inverse_) {
      A_->UpdateBlockDiagonalFtF(block_diagonal_FtF_inverse_.get());
    }
  }

  // The block diagonals of the augmented linear system contain
  // contributions from the diagonal D if it is non-null. Add that to
  // the block diagonals and invert them.
  AddDiagonalAndInvert(D_, block_diagonal_EtE_inverse_.get());
  if (compute_ftf_inverse_) {
    AddDiagonalAndInvert((D_ == nullptr) ? nullptr : D_ + A_->num_cols_e(),
                         block_diagonal_FtF_inverse_.get());
  }

  // Compute the RHS of the Schur complement system.
  UpdateRhs();
}

// Evaluate the product
//
//   Sx = [F'F - F'E (E'E)^-1 E'F]x
//
// By breaking it down into individual matrix vector products
// involving the matrices E and F. This is implemented using a
// PartitionedMatrixView of the input matrix A.
void ImplicitSchurComplement::RightMultiplyAndAccumulate(const double* x,
                                                         double* y) const {
  // y1 = F x
  A_->RightMultiplyAndAccumulateF(x, tmp_rows_.data(), 0);

  // y2 = E' y1
  A_->LeftMultiplyAndAccumulateE(tmp_rows_.data(), tmp_e_cols_.data(), 0);

  // y3 = (E'E)^-1 y2
  SetZero(tmp_e_cols_2_.data(),
          tmp_e_cols_2_.size(),
          options_.context,
          options_.num_threads);
  block_diagonal_EtE_inverse_->RightMultiplyAndAccumulate(tmp_e_cols_.data(),
                                                          tmp_e_cols_2_.data(),
                                                          options_.context,
                                                          options_.num_threads);

  // y1 = y1 - E y3
  A_->RightMultiplyAndAccumulateE(tmp_e_cols_2_.data(), tmp_rows_.data(), -1);

  if (D_ != nullptr) {
    // y5 = D * x
    D2X(D_ + A_->num_cols_e(),
        x,
        y,
        num_cols(),
        options_.context,
        options_.num_threads);
    // y = y5 + F' y1
  }
  // y = y5 + F' y1
  A_->LeftMultiplyAndAccumulateF(tmp_rows_.data(), y, D_ == nullptr ? 0 : 1);
}

void ImplicitSchurComplement::InversePowerSeriesOperatorRightMultiplyAccumulate(
    const double* x, double* y) const {
  CHECK(compute_ftf_inverse_);
  // y1 = F x
  A_->RightMultiplyAndAccumulateF(x, tmp_rows_.data(), 0);

  // y2 = E' y1
  A_->LeftMultiplyAndAccumulateE(tmp_rows_.data(), tmp_e_cols_.data(), 0);

  // y3 = (E'E)^-1 y2
  SetZero(tmp_e_cols_2_.data(),
          tmp_e_cols_2_.size(),
          options_.context,
          options_.num_threads);
  block_diagonal_EtE_inverse_->RightMultiplyAndAccumulate(tmp_e_cols_.data(),
                                                          tmp_e_cols_2_.data(),
                                                          options_.context,
                                                          options_.num_threads);
  // y1 = E y3
  A_->RightMultiplyAndAccumulateE(tmp_e_cols_2_.data(), tmp_rows_.data(), 0);

  // y4 = F' y1
  A_->LeftMultiplyAndAccumulateF(tmp_rows_.data(), tmp_f_cols_.data(), 0);

  // y += (F'F)^-1 y4
  block_diagonal_FtF_inverse_->RightMultiplyAndAccumulate(
      tmp_f_cols_.data(), y, options_.context, options_.num_threads);
}

// Given a block diagonal matrix and an optional array of diagonal
// entries D, add them to the diagonal of the matrix and compute the
// inverse of each diagonal block.
void ImplicitSchurComplement::AddDiagonalAndInvert(
    const double* D, BlockSparseMatrix* block_diagonal) {
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();
  for (const auto& row : block_diagonal_structure->rows) {
    const int row_block_pos = row.block.position;
    const int row_block_size = row.block.size;
    const Cell& cell = row.cells[0];
    MatrixRef m(block_diagonal->mutable_values() + cell.position,
                row_block_size,
                row_block_size);

    if (D != nullptr) {
      ConstVectorRef d(D + row_block_pos, row_block_size);
      m += d.array().square().matrix().asDiagonal();
    }

    m = m.selfadjointView<Eigen::Upper>().llt().solve(
        Matrix::Identity(row_block_size, row_block_size));
  }
}

// Similar to RightMultiplyAndAccumulate, use the block structure of the matrix
// A to compute y = (E'E)^-1 (E'b - E'F x).
void ImplicitSchurComplement::BackSubstitute(const double* x, double* y) {
  const int num_cols_e = A_->num_cols_e();
  const int num_cols_f = A_->num_cols_f();
  const int num_cols = A_->num_cols();
  const int num_rows = A_->num_rows();

  // y2 = b - F x
  tmp_rows_ = ConstVectorRef(b_, num_rows);
  A_->RightMultiplyAndAccumulateF(x, tmp_rows_.data(), -1);

  // y3 = E' y2
  A_->LeftMultiplyAndAccumulateE(tmp_rows_.data(), tmp_e_cols_.data(), 0);

  // y = (E'E)^-1 y3
  SetZero(y, num_cols, options_.context, options_.num_threads);
  block_diagonal_EtE_inverse_->RightMultiplyAndAccumulate(
      tmp_e_cols_.data(), y, options_.context, options_.num_threads);

  // The full solution vector y has two blocks. The first block of
  // variables corresponds to the eliminated variables, which we just
  // computed via back substitution. The second block of variables
  // corresponds to the Schur complement system, so we just copy those
  // values from the solution to the Schur complement.
  Assign(x, y + num_cols_e, num_cols_f, options_.context, options_.num_threads);
}

// Compute the RHS of the Schur complement system.
//
// rhs = F'b - F'E (E'E)^-1 E'b
//
// Like BackSubstitute, we use the block structure of A to implement
// this using a series of matrix vector products.
void ImplicitSchurComplement::UpdateRhs() {
  // y1 = E'b
  A_->LeftMultiplyAndAccumulateE(b_, tmp_e_cols_.data(), 0);

  // y2 = (E'E)^-1 y1
  SetZero(tmp_e_cols_2_.data(),
          tmp_e_cols_2_.size(),
          options_.context,
          options_.num_threads);
  block_diagonal_EtE_inverse_->RightMultiplyAndAccumulate(tmp_e_cols_.data(),
                                                          tmp_e_cols_2_.data(),
                                                          options_.context,
                                                          options_.num_threads);

  // y3 = b - E y2
  Assign(b_,
         tmp_rows_.data(),
         A_->num_rows(),
         options_.context,
         options_.num_threads);
  A_->RightMultiplyAndAccumulateE(tmp_e_cols_2_.data(), tmp_rows_.data(), -1);

  // rhs = F' y3
  A_->LeftMultiplyAndAccumulateF(tmp_rows_.data(), rhs_.data(), 0);
}

}  // namespace ceres::internal
