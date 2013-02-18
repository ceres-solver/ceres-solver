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
//
// Detailed descriptions of these preconditions beyond what is
// documented here can be found in
//
// Bundle Adjustment in the Large
// S. Agarwal, N. Snavely, S. Seitz & R. Szeliski, ECCV 2010
// http://www.cs.washington.edu/homes/sagarwal/bal.pdf

#ifndef CERES_INTERNAL_SCHUR_JACOBI_PRECONDITIONER_H_
#define CERES_INTERNAL_SCHUR_JACOBI_PRECONDITIONER_H_

#include <set>
#include <vector>
#include <utility>
#include "ceres/collections_port.h"
#include "ceres/internal/macros.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/linear_operator.h"
#include "ceres/linear_solver.h"

namespace ceres {
namespace internal {

class BlockRandomAccessSparseMatrix;
class BlockSparseMatrixBase;
struct CompressedRowBlockStructure;
class SchurEliminatorBase;

// This class implements the SCHUR_JACOBI preconditioner for Structure
// from Motion/Bundle Adjustment problems. Full mathematical details
// can be found in
//
// Bundle Adjustment in the Large
// S. Agarwal, N. Snavely, S. Seitz & R. Szeliski, ECCV 2010
// http://www.cs.washington.edu/homes/sagarwal/bal.pdf
//
// Example usage:
//
//   LinearSolver::Options options;
//   options.preconditioner_type = SCHUR_JACOBI;
//   options.elimination_groups.push_back(num_points);
//   options.elimination_groups.push_back(num_cameras);
//   SchurJacobiPreconditioner preconditioner(
//      *A.block_structure(), options);
//   preconditioner.Update(A, NULL);
//   preconditioner.RightMultiply(x, y);
//
class SchurJacobiPreconditioner : public LinearOperator {
 public:
  // Initialize the symbolic structure of the preconditioner. bs is
  // the block structure of the linear system to be solved. It is used
  // to determine the sparsity structure of the preconditioner matrix.
  //
  // It has the same structural requirement as other Schur complement
  // based solvers. Please see schur_eliminator.h for more details.
  //
  //
  // TODO(sameeragarwal): The use of LinearSolver::Options should
  // ultimately be replaced with Preconditioner::Options and some sort
  // of preconditioner factory along the lines of
  // LinearSolver::CreateLinearSolver. I will wait to do this till I
  // create a general purpose block Jacobi preconditioner for general
  // sparse problems along with a CGLS solver.
  SchurJacobiPreconditioner(const CompressedRowBlockStructure& bs,
                            const LinearSolver::Options& options);
  virtual ~SchurJacobiPreconditioner();

  // Update the numerical value of the preconditioner for the linear
  // system:
  //
  //  |   A   | x = |b|
  //  |diag(D)|     |0|
  //
  // for some vector b. It is important that the matrix A have the
  // same block structure as the one used to construct this object.
  //
  // D can be NULL, in which case its interpreted as a diagonal matrix
  // of size zero.
  bool Update(const BlockSparseMatrixBase& A, const double* D);


  // LinearOperator interface. Since the operator is symmetric,
  // LeftMultiply and num_cols are just calls to RightMultiply and
  // num_rows respectively. Update() must be called before
  // RightMultiply can be called.
  virtual void RightMultiply(const double* x, double* y) const;
  virtual void LeftMultiply(const double* x, double* y) const {
    RightMultiply(x, y);
  }
  virtual int num_rows() const;
  virtual int num_cols() const { return num_rows(); }

  friend class SchurJacobiPreconditionerTest;
 private:
  void InitEliminator(const CompressedRowBlockStructure& bs);

  LinearSolver::Options options_;

  // Sizes of the blocks in the schur complement.
  vector<int> block_size_;
  scoped_ptr<SchurEliminatorBase> eliminator_;

  // Preconditioner matrix.
  scoped_ptr<BlockRandomAccessSparseMatrix> m_;

  CERES_DISALLOW_COPY_AND_ASSIGN(SchurJacobiPreconditioner);
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_SCHUR_JACOBI_PRECONDITIONER_H_
