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

#ifndef CERES_INTERNAL_OUTER_PRODUCT_H_
#define CERES_INTERNAL_OUTER_PRODUCT_H_

#include <vector>

#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/internal/scoped_ptr.h"

namespace ceres {
namespace internal {

// This class is used to repeatedly compute the product v = m' * m,
// where the sparsity structure of m remains constant across calls.
//
// Upon creation, the class computes and caches information needed to
// compute v, and then uses it to efficiently compute the product
// every time OuterProduct::ComputeProduct is called.
//
// See sparse_normal_cholesky_solver.cc for example usage.
class OuterProduct {
 public:
  // Factory
  //
  // m is the input matrix
  //
  // Since m' * m is a symmetric matrix, we only compute half of the
  // matrix and the value of storage_type which must be
  // UPPER_TRIANGULAR or LOWER_TRIANGULAR determines which half is
  // computed.
  //
  // The user must ensure that the matrix m is valid for the life time
  // of this object.
  static OuterProduct* Create(
      const BlockSparseMatrix& m,
      CompressedRowSparseMatrix::StorageType product_storage_type);

  static OuterProduct* Create(
      const BlockSparseMatrix& m,
      int start_row_block,
      int end_row_block,
      CompressedRowSparseMatrix::StorageType product_storage_type);

  // Update matrix_ to be numerically equal to m' * m.
  void ComputeProduct();

  // Accessors for the matrix containing the product.
  //
  // ComputeProduct must be called before accessing this matrix for
  // the first time.
  const CompressedRowSparseMatrix& matrix() const { return *matrix_; }
  CompressedRowSparseMatrix* mutable_matrix() const { return matrix_.get(); }

 private:
  // A ProductTerm is a term in the block outer product of a matrix with
  // itself.
  struct ProductTerm {
    ProductTerm(const int row, const int col, const int index);
    bool operator<(const ProductTerm& right) const;

    int row;
    int col;
    int index;
  };

  OuterProduct(const BlockSparseMatrix& m,
               int start_row_block,
               int end_row_block);

  void Init(CompressedRowSparseMatrix::StorageType product_storage_type);

  CompressedRowSparseMatrix* CreateOuterProductMatrix(
      const CompressedRowSparseMatrix::StorageType product_storage_type,
      const std::vector<ProductTerm>& product_terms,
      std::vector<int>* row_nnz);

  void CompressAndFillProgram(
      const CompressedRowSparseMatrix::StorageType product_storage_type,
      const std::vector<ProductTerm>& product_terms);

  const BlockSparseMatrix& m_;
  const int start_row_block_;
  const int end_row_block_;
  scoped_ptr<CompressedRowSparseMatrix> matrix_;
  std::vector<int> program_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_OUTER_PRODUCT_H_
