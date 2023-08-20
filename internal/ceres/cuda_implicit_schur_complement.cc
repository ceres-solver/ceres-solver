// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
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
// Authors: dmitriy.korchemkin@gmail.com (Dmitriy Korchemkin)
//
// An iterative solver for solving the Schur complement/reduced camera
// linear system that arise in SfM problems.

#include "ceres/cuda_implicit_schur_complement.h"

#ifndef CERES_NO_CUDA

namespace ceres::internal {

CudaImplicitSchurComplement::CudaImplicitSchurComplement(
    const LinearSolver::Options& options)
    : ImplicitSchurComplementBase(options),
      isc_cpu_(options),
      b_(options.context, 0),
      rhs_(options.context, 0),
      tmp_rows_(options.context, 0),
      tmp_e_cols_(options.context, 0),
      tmp_e_cols_2_(options.context, 0),
      tmp_f_cols_(options.context, 0) {}

void CudaImplicitSchurComplement::InitImpl(const BlockSparseMatrix& A,
                                           const double* D,
                                           const double* b,
                                           bool update_rhs) {
  if (!A_) {
    A_ = std::make_unique<CudaPartitionedBlockSparseCRSView>(
        A, options_.elimination_groups[0], options_.context);
  } else {
    A_->UpdateValues(A);
  }
  isc_cpu_.InitImpl(A, D, b, false);

  if (!block_diagonal_EtE_inverse_) {
    block_diagonal_EtE_inverse_ = std::make_unique<CudaBlockSparseCRSView>(
        *isc_cpu_.block_diagonal_EtE_inverse(), options_.context);
    if (compute_ftf_inverse_) {
      block_diagonal_FtF_inverse_ = std::make_unique<CudaBlockSparseCRSView>(
          *isc_cpu_.block_diagonal_FtF_inverse(), options_.context);
    }
    rhs_.Resize(A_->num_cols_f());
    rhs_.SetZero();
    tmp_rows_.Resize(A_->num_rows());
    tmp_e_cols_.Resize(A_->num_cols_e());
    tmp_e_cols_2_.Resize(A_->num_cols_e());
    tmp_f_cols_.Resize(A_->num_cols_f());
  } else {
    block_diagonal_EtE_inverse_->UpdateValues(
        *isc_cpu_.block_diagonal_EtE_inverse());
    if (compute_ftf_inverse_) {
      block_diagonal_FtF_inverse_->UpdateValues(
          *isc_cpu_.block_diagonal_FtF_inverse());
    }
  }

  if (D) {
    if (!D_) {
      D_ = std::make_unique<CudaVector>(options_.context, A.num_cols());
    }
    D_->CopyFromCpu(D);
  }
  if (!b_.num_rows()) {
    b_.Resize(A_->num_rows());
  }
  b_.CopyFromCpu(b);

  if (update_rhs) {
    UpdateRhs();
  }
}

void CudaImplicitSchurComplement::
    BlockDiagonalEtEInverseRightMultiplyAndAccumulate(const double* x,
                                                      double* y) const {
  block_diagonal_EtE_inverse_->RightMultiplyAndAccumulate(x, y);
}

void CudaImplicitSchurComplement::
    BlockDiagonalFtFInverseRightMultiplyAndAccumulate(const double* x,
                                                      double* y) const {
  block_diagonal_FtF_inverse_->RightMultiplyAndAccumulate(x, y);
}

const PartitionedLinearOperator*
CudaImplicitSchurComplement::PartitionedOperator() const {
  return A_.get();
}

void CudaImplicitSchurComplement::SetZero(double* ptr, int size) const {
  CudaSetZeroFP64(ptr, size, options_.context->DefaultStream());
}
void CudaImplicitSchurComplement::Negate(double* ptr, int size) const {
  CudaNegate(ptr, size, options_.context->DefaultStream());
}
void CudaImplicitSchurComplement::YXmY(double* y,
                                       const double* x,
                                       int size) const {
  CudaYXmY(y, x, size, options_.context->DefaultStream());
}
void CudaImplicitSchurComplement::D2x(double* y,
                                      const double* D,
                                      const double* x,
                                      int size) const {
  CudaD2x(y, D, x, size, options_.context->DefaultStream());
}

void CudaImplicitSchurComplement::Assign(double* to,
                                         const double* from,
                                         int size) const {
  CHECK_EQ(cudaSuccess,
           cudaMemcpyAsync(to,
                           from,
                           sizeof(double) * size,
                           cudaMemcpyDeviceToDevice,
                           options_.context->DefaultStream()));
}

double* CudaImplicitSchurComplement::tmp_rows() const {
  return tmp_rows_.mutable_data();
}

double* CudaImplicitSchurComplement::tmp_e_cols() const {
  return tmp_e_cols_.mutable_data();
}

double* CudaImplicitSchurComplement::tmp_e_cols_2() const {
  return tmp_e_cols_2_.mutable_data();
}

double* CudaImplicitSchurComplement::tmp_f_cols() const {
  return tmp_f_cols_.mutable_data();
}

const double* CudaImplicitSchurComplement::Diag() const {
  return D_ ? D_->data() : nullptr;
}

const double* CudaImplicitSchurComplement::b() const { return b_.data(); }

double* CudaImplicitSchurComplement::mutable_rhs() {
  return rhs_.mutable_data();
}

}  // namespace ceres::internal
#endif
