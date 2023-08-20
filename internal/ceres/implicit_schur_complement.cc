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
#include "ceres/cuda_implicit_schur_complement.h"
#include "ceres/internal/eigen.h"
#include "ceres/linear_solver.h"
#include "ceres/parallel_for.h"
#include "ceres/parallel_vector_ops.h"
#include "ceres/types.h"
#include "glog/logging.h"

namespace ceres::internal {

std::unique_ptr<ImplicitSchurComplementBase>
ImplicitSchurComplementBase::Create(const LinearSolver::Options& options) {
#ifndef CERES_NO_CUDA
  if (options.sparse_linear_algebra_library_type ==
      SparseLinearAlgebraLibraryType::CUDA_SPARSE) {
    return std::make_unique<CudaImplicitSchurComplement>(options);
  }
#endif
  return std::make_unique<ImplicitSchurComplement>(options);
}

int ImplicitSchurComplementBase::num_cols() const { return num_rows(); }

int ImplicitSchurComplementBase::num_rows() const {
  return PartitionedOperator()->num_cols_f();
}

int ImplicitSchurComplementBase::num_cols_total() const {
  return PartitionedOperator()->num_cols();
}

bool ImplicitSchurComplementBase::IsFtFRequired() const {
  return options_.use_spse_initialization ||
         options_.preconditioner_type == JACOBI ||
         options_.preconditioner_type == SCHUR_POWER_SERIES_EXPANSION;
}

ImplicitSchurComplementBase::ImplicitSchurComplementBase(
    const LinearSolver::Options& options)
    : options_(options) {}

// Evaluate the product
//
//   Sx = [F'F - F'E (E'E)^-1 E'F]x
//
// By breaking it down into individual matrix vector products
// involving the matrices E and F. This is implemented using a
// PartitionedMatrixView of the input matrix A.
void ImplicitSchurComplementBase::RightMultiplyAndAccumulate(const double* x,
                                                             double* y) const {
  const auto& A = PartitionedOperator();

  // y1 = F x
  SetZero(tmp_rows(), A->num_rows());
  A->RightMultiplyAndAccumulateF(x, tmp_rows());

  // y2 = E' y1
  SetZero(tmp_e_cols(), A->num_cols_e());
  A->LeftMultiplyAndAccumulateE(tmp_rows(), tmp_e_cols());

  // y3 = -(E'E)^-1 y2
  SetZero(tmp_e_cols_2(), A->num_cols_e());
  BlockDiagonalEtEInverseRightMultiplyAndAccumulate(tmp_e_cols(),
                                                    tmp_e_cols_2());

  Negate(tmp_e_cols_2(), A->num_cols_e());

  // y1 = y1 + E y3
  A->RightMultiplyAndAccumulateE(tmp_e_cols_2(), tmp_rows());

  // y5 = D * x
  if (Diag() != nullptr) {
    D2x(y, Diag() + A->num_cols_e(), x, num_cols());
  } else {
    SetZero(y, num_cols());
  }

  // y = y5 + F' y1
  A->LeftMultiplyAndAccumulateF(tmp_rows(), y);
}

void ImplicitSchurComplementBase::
    InversePowerSeriesOperatorRightMultiplyAccumulate(const double* x,
                                                      double* y) const {
  const auto A = PartitionedOperator();
  CHECK(compute_ftf_inverse_);
  // y1 = F x
  SetZero(tmp_rows(), A->num_rows());
  A->RightMultiplyAndAccumulateF(x, tmp_rows());

  // y2 = E' y1
  SetZero(tmp_e_cols(), A->num_cols_e());
  A->LeftMultiplyAndAccumulateE(tmp_rows(), tmp_e_cols());

  // y3 = (E'E)^-1 y2
  SetZero(tmp_e_cols_2(), A->num_cols_e());
  BlockDiagonalEtEInverseRightMultiplyAndAccumulate(tmp_e_cols(),
                                                    tmp_e_cols_2());
  // y1 = E y3
  SetZero(tmp_rows(), A->num_rows());
  A->RightMultiplyAndAccumulateE(tmp_e_cols_2(), tmp_rows());

  // y4 = F' y1
  SetZero(tmp_f_cols(), A->num_cols_f());
  A->LeftMultiplyAndAccumulateF(tmp_rows(), tmp_f_cols());

  // y += (F'F)^-1 y4
  BlockDiagonalFtFInverseRightMultiplyAndAccumulate(tmp_f_cols(), y);
}

// Similar to RightMultiplyAndAccumulate, use the block structure of the matrix
// A to compute y = (E'E)^-1 (E'b - E'F x).
void ImplicitSchurComplementBase::BackSubstitute(const double* x,
                                                 double* y) const {
  const auto A = PartitionedOperator();
  const int num_cols_e = A->num_cols_e();
  const int num_cols_f = A->num_cols_f();
  const int num_cols = A->num_cols();
  const int num_rows = A->num_rows();

  // y1 = F x
  SetZero(tmp_rows(), num_rows);
  A->RightMultiplyAndAccumulateF(x, tmp_rows());

  // y2 = b - y1
  YXmY(tmp_rows(), b(), num_rows);

  // y3 = E' y2
  SetZero(tmp_e_cols(), num_cols_e);
  A->LeftMultiplyAndAccumulateE(tmp_rows(), tmp_e_cols());

  // y = (E'E)^-1 y3
  SetZero(y, num_cols);
  BlockDiagonalEtEInverseRightMultiplyAndAccumulate(tmp_e_cols(), y);

  // The full solution vector y has two blocks. The first block of
  // variables corresponds to the eliminated variables, which we just
  // computed via back substitution. The second block of variables
  // corresponds to the Schur complement system, so we just copy those
  // values from the solution to the Schur complement.
  Assign(y + num_cols_e, x, num_cols_f);
}

// Compute the RHS of the Schur complement system.
//
// rhs = F'b - F'E (E'E)^-1 E'b
//
// Like BackSubstitute, we use the block structure of A to implement
// this using a series of matrix vector products.
void ImplicitSchurComplementBase::UpdateRhs() {
  const auto A = PartitionedOperator();

  // y1 = E'b
  SetZero(tmp_e_cols(), A->num_cols_e());
  A->LeftMultiplyAndAccumulateE(b(), tmp_e_cols());

  // y2 = (E'E)^-1 y1
  SetZero(tmp_e_cols_2(), A->num_cols_e());
  BlockDiagonalEtEInverseRightMultiplyAndAccumulate(tmp_e_cols(),
                                                    tmp_e_cols_2());

  // y3 = E y2
  SetZero(tmp_rows(), A->num_rows());
  A->RightMultiplyAndAccumulateE(tmp_e_cols_2(), tmp_rows());

  // y3 = b - y3
  YXmY(tmp_rows(), b(), A->num_rows());

  // rhs = F' y3
  SetZero(mutable_rhs(), A->num_cols_f());
  A->LeftMultiplyAndAccumulateF(tmp_rows(), mutable_rhs());
}

void ImplicitSchurComplementBase::Init(const BlockSparseMatrix& A,
                                       const double* D,
                                       const double* b) {
  compute_ftf_inverse_ = IsFtFRequired();
  InitImpl(A, D, b, true);
}

ImplicitSchurComplement::ImplicitSchurComplement(
    const LinearSolver::Options& options)
    : ImplicitSchurComplementBase(options) {}

void ImplicitSchurComplement::InitImpl(const BlockSparseMatrix& A,
                                       const double* D,
                                       const double* b,
                                       bool update_rhs) {
  compute_ftf_inverse_ = IsFtFRequired();
  // Since initialization is reasonably heavy, perhaps we can save on
  // constructing a new object everytime.
  if (A_ == nullptr) {
    A_ = PartitionedMatrixViewBase::Create(options_, A);
  }

  D_ = D;
  b_ = b;

  // Initialize temporary storage and compute the block diagonals of
  // E'E and F'F.
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

  if (update_rhs) {
    // Compute the RHS of the Schur complement system.
    UpdateRhs();
  }
}

// Given a block diagonal matrix and an optional array of diagonal
// entries D, add them to the diagonal of the matrix and compute the
// inverse of each diagonal block.
void ImplicitSchurComplement::AddDiagonalAndInvert(
    const double* D, BlockSparseMatrix* block_diagonal) {
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();
  ParallelFor(options_.context,
              0,
              block_diagonal_structure->rows.size(),
              options_.num_threads,
              [block_diagonal_structure, D, block_diagonal](int row_block_id) {
                auto& row = block_diagonal_structure->rows[row_block_id];
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
              });
}

void ImplicitSchurComplement::BlockDiagonalEtEInverseRightMultiplyAndAccumulate(
    const double* x, double* y) const {
  block_diagonal_EtE_inverse_->RightMultiplyAndAccumulate(
      x, y, options_.context, options_.num_threads);
}

void ImplicitSchurComplement::BlockDiagonalFtFInverseRightMultiplyAndAccumulate(
    const double* x, double* y) const {
  block_diagonal_FtF_inverse_->RightMultiplyAndAccumulate(
      x, y, options_.context, options_.num_threads);
}

const PartitionedLinearOperator* ImplicitSchurComplement::PartitionedOperator()
    const {
  return A_.get();
}

void ImplicitSchurComplement::SetZero(double* ptr, int size) const {
  VectorRef vector(ptr, size);
  ParallelSetZero(options_.context, options_.num_threads, vector);
}

void ImplicitSchurComplement::Negate(double* ptr, int size) const {
  VectorRef vector(ptr, size);
  ParallelAssign(options_.context, options_.num_threads, vector, -vector);
}

void ImplicitSchurComplement::YXmY(double* y, const double* x, int size) const {
  VectorRef vector_y(y, size);
  ConstVectorRef vector_x(x, size);
  ParallelAssign(
      options_.context, options_.num_threads, vector_y, vector_x - vector_y);
}

void ImplicitSchurComplement::D2x(double* y,
                                  const double* D,
                                  const double* x,
                                  int size) const {
  VectorRef vector_y(y, size);
  ConstVectorRef vector_D(D, size);
  ConstVectorRef vector_x(x, size);
  ParallelAssign(options_.context,
                 options_.num_threads,
                 vector_y,
                 vector_D.array().square() * vector_x.array());
}

void ImplicitSchurComplement::Assign(double* to,
                                     const double* from,
                                     int size) const {
  VectorRef vector_to(to, size);
  ConstVectorRef vector_from(from, size);
  ParallelAssign(
      options_.context, options_.num_threads, vector_to, vector_from);
}

double* ImplicitSchurComplement::tmp_rows() const { return tmp_rows_.data(); }

double* ImplicitSchurComplement::tmp_e_cols() const {
  return tmp_e_cols_.data();
}

double* ImplicitSchurComplement::tmp_e_cols_2() const {
  return tmp_e_cols_2_.data();
}

double* ImplicitSchurComplement::tmp_f_cols() const {
  return tmp_f_cols_.data();
}

const double* ImplicitSchurComplement::Diag() const { return D_; }
const double* ImplicitSchurComplement::b() const { return b_; }
double* ImplicitSchurComplement::mutable_rhs() { return rhs_.data(); }

}  // namespace ceres::internal
