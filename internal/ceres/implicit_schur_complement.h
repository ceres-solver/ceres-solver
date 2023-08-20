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
//
// An iterative solver for solving the Schur complement/reduced camera
// linear system that arise in SfM problems.

#ifndef CERES_INTERNAL_IMPLICIT_SCHUR_COMPLEMENT_H_
#define CERES_INTERNAL_IMPLICIT_SCHUR_COMPLEMENT_H_

#include <memory>

#include "ceres/internal/disable_warnings.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/export.h"
#include "ceres/linear_operator.h"
#include "ceres/linear_solver.h"
#include "ceres/partitioned_matrix_view.h"
#include "ceres/types.h"

namespace ceres::internal {

class BlockSparseMatrix;

// This class implements various linear algebraic operations related
// to the Schur complement without explicitly forming it.
//
//
// Given a reactangular linear system Ax = b, where
//
//   A = [E F]
//
// The normal equations are given by
//
//   A'Ax = A'b
//
//  |E'E E'F||y| = |E'b|
//  |F'E F'F||z|   |F'b|
//
// and the Schur complement system is given by
//
//  [F'F - F'E (E'E)^-1 E'F] z = F'b - F'E (E'E)^-1 E'b
//
// Now if we wish to solve Ax = b in the least squares sense, one way
// is to form this Schur complement system and solve it using
// Preconditioned Conjugate Gradients.
//
// The key operation in a conjugate gradient solver is the evaluation of the
// matrix vector product with the Schur complement
//
//   S = F'F - F'E (E'E)^-1 E'F
//
// It is straightforward to see that matrix vector products with S can
// be evaluated without storing S in memory. Instead, given (E'E)^-1
// (which for our purposes is an easily inverted block diagonal
// matrix), it can be done in terms of matrix vector products with E,
// F and (E'E)^-1. This class implements this functionality and other
// auxiliary bits needed to implement a CG solver on the Schur
// complement using the PartitionedMatrixView object.
//
// THREAD SAFETY: This class is not thread safe. In particular, the
// RightMultiplyAndAccumulate (and the LeftMultiplyAndAccumulate) methods are
// not thread safe as they depend on mutable arrays used for the temporaries
// needed to compute the product y += Sx;
//
// Base class ImplicitSchurComplementBase implements Schur-complement operations
// using data access methods provided by derived class (for example -
// ImplicitSchurComplement)
//
//

class CERES_NO_EXPORT ImplicitSchurComplementBase : public LinearOperator {
 public:
  virtual ~ImplicitSchurComplementBase() = default;
  ImplicitSchurComplementBase(const LinearSolver::Options& options);

  // Initialize the Schur complement for a linear least squares
  // problem of the form
  //
  //   |A      | x = |b|
  //   |diag(D)|     |0|
  //
  // If D is null, then it is treated as a zero dimensional matrix. It
  // is important that the matrix A have a BlockStructure object
  // associated with it and has a block structure that is compatible
  // with the SchurComplement solver.
  // D and b pointers are always pointers to host memory
  void Init(const BlockSparseMatrix& A, const double* D, const double* b);

  // LinearOperator implementation; x and y might be either a pair of pointers
  // to host memory or a pair of pointers to gpu memory, depending on particular
  // implementation
  virtual void RightMultiplyAndAccumulate(const double* x, double* y) const;
  virtual void LeftMultiplyAndAccumulate(const double* x, double* y) const {
    RightMultiplyAndAccumulate(x, y);
  }

  // Following is useful for approximation of S^-1 via power series expansion.
  // Z = (F'F)^-1 F'E (E'E)^-1 E'F
  // y += Zx
  void InversePowerSeriesOperatorRightMultiplyAccumulate(const double* x,
                                                         double* y) const;

  // y = (E'E)^-1 (E'b - E'F x). Given an estimate of the solution to
  // the Schur complement system, this method computes the value of
  // the e_block variables that were eliminated to form the Schur
  // complement.
  void BackSubstitute(const double* x, double* y) const;

  int num_rows() const override;
  int num_cols() const override;
  int num_cols_total() const;

  // Data pointer to rhs of  schur-complement (reduced) linear system
  virtual const double* RhsData() const = 0;

  // Instantiates implementation corresponding to
  // options.sparse_linear_algebra_library_type:
  //  - CUDA_SPARSE: creates gpu-accelerated implementation (expecting pointers
  //  to device memory as input vectors)
  //  - Otherwise: cpu implementation (expecting pointers to host memory as
  //  input vectors)
  static std::unique_ptr<ImplicitSchurComplementBase> Create(
      const LinearSolver::Options& options);

 protected:
  // Initialization implementation, setting initialize_rhs skips the process of
  // obtaining Schur-complement rhs
  virtual void InitImpl(const BlockSparseMatrix& A,
                        const double* D,
                        const double* b,
                        bool initialize_rhs) = 0;
  // Returns true if block-diagonal FtF inverse has to be computed
  bool IsFtFRequired() const;
  void UpdateRhs();

  // Matrix-vector product with block-diagonal EtE and FtF inverses
  virtual void BlockDiagonalEtEInverseRightMultiplyAndAccumulate(
      const double* x, double* y) const = 0;
  virtual void BlockDiagonalFtFInverseRightMultiplyAndAccumulate(
      const double* x, double* y) const = 0;

  // Access to PartitionedLinearOperator (used for left and right products with
  // E and F sub-matrices)
  virtual const PartitionedLinearOperator* PartitionedOperator() const = 0;
  // Sets x[i] = 0
  virtual void SetZero(double* x, int size) const = 0;
  // Sets x[i] = -x[i]
  virtual void Negate(double* x, int size) const = 0;
  // Sets y[i] = x[i] - y[i]
  virtual void YXmY(double* y, const double* x, int size) const = 0;
  // Sets y[i] = D[i]^2 * x[i]
  virtual void D2x(double* y,
                   const double* D,
                   const double* x,
                   int size) const = 0;
  // Sets to[i] = from[i]
  virtual void Assign(double* to, const double* from, int size) const = 0;

  // Temporary array of size corresponding to rows of partitioned Jacobian
  virtual double* tmp_rows() const = 0;
  // Temporary array of num_cols_e size
  virtual double* tmp_e_cols() const = 0;
  // Temporary array of num_cols_e size
  virtual double* tmp_e_cols_2() const = 0;
  // Temporary array of num_cols_f size
  virtual double* tmp_f_cols() const = 0;
  // Pointer to values of diagonal
  virtual const double* Diag() const = 0;
  // Pointer to full rhs vector
  virtual const double* b() const = 0;
  // Pointer to reduced rhs vector
  virtual double* mutable_rhs() = 0;

  const LinearSolver::Options& options_;
  bool compute_ftf_inverse_ = false;
};

class CERES_NO_EXPORT ImplicitSchurComplement final
    : public ImplicitSchurComplementBase {
 public:
  // num_eliminate_blocks is the number of E blocks in the matrix
  // A.
  //
  // preconditioner indicates whether the inverse of the matrix F'F
  // should be computed or not as a preconditioner for the Schur
  // Complement.
  explicit ImplicitSchurComplement(const LinearSolver::Options& options);

  const Vector& rhs() const { return rhs_; }

  const BlockSparseMatrix* block_diagonal_FtF_inverse() const {
    CHECK(compute_ftf_inverse_);
    return block_diagonal_FtF_inverse_.get();
  }

  const BlockSparseMatrix* block_diagonal_EtE_inverse() const {
    return block_diagonal_EtE_inverse_.get();
  }

  const double* RhsData() const override { return rhs_.data(); }

  void InitImpl(const BlockSparseMatrix& A,
                const double* D,
                const double* b,
                bool update_rhs) override;

 protected:
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

 private:
  void AddDiagonalAndInvert(const double* D, BlockSparseMatrix* matrix);

  std::unique_ptr<PartitionedMatrixViewBase> A_;
  const double* D_ = nullptr;
  const double* b_ = nullptr;

  std::unique_ptr<BlockSparseMatrix> block_diagonal_EtE_inverse_;
  std::unique_ptr<BlockSparseMatrix> block_diagonal_FtF_inverse_;

  Vector rhs_;

  // Temporary storage vectors used to implement RightMultiplyAndAccumulate.
  mutable Vector tmp_rows_;
  mutable Vector tmp_e_cols_;
  mutable Vector tmp_e_cols_2_;
  mutable Vector tmp_f_cols_;
};

}  // namespace ceres::internal

#include "ceres/internal/reenable_warnings.h"

#endif  // CERES_INTERNAL_IMPLICIT_SCHUR_COMPLEMENT_H_
