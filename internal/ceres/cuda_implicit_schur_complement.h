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

#ifndef CERES_INTERNAL_CUDA_IMPLICIT_SCHUR_COMPLEMENT_H_
#define CERES_INTERNAL_CUDA_IMPLICIT_SCHUR_COMPLEMENT_H_
#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA
#include "ceres/cuda_block_sparse_crs_view.h"
#include "ceres/cuda_partitioned_block_sparse_crs_view.h"
#include "ceres/cuda_vector.h"
#include "ceres/implicit_schur_complement.h"

namespace ceres::internal {

// Implementation of ImplicitSchurComplementBase interface with storage &
// operations performed on gpu
class CERES_NO_EXPORT CudaImplicitSchurComplement final
    : public ImplicitSchurComplementBase {
 public:
  explicit CudaImplicitSchurComplement(const LinearSolver::Options& options);

  const CudaSparseMatrix* block_diagonal_FtF_inverse() const {
    CHECK(compute_ftf_inverse_);
    return block_diagonal_FtF_inverse_->crs_matrix();
  }

  const CudaVector& rhs() const { return rhs_; }

  const double* RhsData() const override { return rhs_.data(); }

  const ImplicitSchurComplement& isc_cpu() const { return isc_cpu_; }

 private:
  void BlockDiagonalEtEInverseRightMultiplyAndAccumulate(
      const double* x, double* y) const override;
  void BlockDiagonalFtFInverseRightMultiplyAndAccumulate(
      const double* x, double* y) const override;

  const PartitionedLinearOperator* PartitionedOperator() const override;
  void SetZero(double* ptr, int size) const override;
  void Negate(double* ptr, int size) const override;
  void YXmY(double* y, const double* x, int size) const override;
  void D2x(double* y,
           const double* D,
           const double* x,
           int size) const override;
  void Assign(double* to, const double* from, int size) const override;

  double* tmp_rows() const override;
  double* tmp_e_cols() const override;
  double* tmp_e_cols_2() const override;
  double* tmp_f_cols() const override;
  const double* Diag() const override;
  const double* b() const override;
  double* mutable_rhs() override;

  void InitImpl(const BlockSparseMatrix& A,
                const double* D,
                const double* b,
                bool update_rhs) override;

  ImplicitSchurComplement isc_cpu_;
  std::unique_ptr<CudaPartitionedBlockSparseCRSView> A_;

  std::unique_ptr<CudaVector> D_;
  CudaVector b_;
  CudaVector rhs_;

  std::unique_ptr<CudaBlockSparseCRSView> block_diagonal_EtE_inverse_;
  std::unique_ptr<CudaBlockSparseCRSView> block_diagonal_FtF_inverse_;

  // Temporary storage vectors used to implement RightMultiplyAndAccumulate.
  mutable CudaVector tmp_rows_;
  mutable CudaVector tmp_e_cols_;
  mutable CudaVector tmp_e_cols_2_;
  mutable CudaVector tmp_f_cols_;
};

}  // namespace ceres::internal
#endif

#endif
