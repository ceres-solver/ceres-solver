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

#ifndef CERES_INTERNAL_COMPRESSED_ROW_SPARSE_MATRIX_H_
#define CERES_INTERNAL_COMPRESSED_ROW_SPARSE_MATRIX_H_

#include <vector>
#include "ceres/internal/macros.h"
#include "ceres/internal/port.h"
#include "ceres/sparse_matrix.h"
#include "ceres/types.h"
#include "glog/logging.h"

namespace ceres {

struct CRSMatrix;

namespace internal {

class TripletSparseMatrix;

class CompressedRowSparseMatrix : public SparseMatrix {
 public:
  enum StorageType {
    UNSYMMETRIC,
    // Matrix is assumed to be symmetric but only the lower triangular
    // part of the matrix is stored.
    LOWER_TRIANGULAR,
    // Matrix is assumed to be symmetric but only the upper triangular
    // part of the matrix is stored.
    UPPER_TRIANGULAR
  };

  // Create a matrix with the same content as the TripletSparseMatrix
  // input. We assume that input does not have any repeated
  // entries.
  //
  // The storage type of the matrix is set to UNSYMMETRIC.
  //
  // Caller owns the result.
  static CompressedRowSparseMatrix* FromTripletSparseMatrix(
      const TripletSparseMatrix& input);

  // Create a matrix with the same content as the TripletSparseMatrix
  // input transposed. We assume that input does not have any repeated
  // entries.
  //
  // The storage type of the matrix is set to UNSYMMETRIC.
  //
  // Caller owns the result.
  static CompressedRowSparseMatrix* FromTripletSparseMatrixTransposed(
      const TripletSparseMatrix& input);

  // Use this constructor only if you know what you are doing. This
  // creates a "blank" matrix with the appropriate amount of memory
  // allocated. However, the object itself is in an inconsistent state
  // as the rows and cols matrices do not match the values of
  // num_rows, num_cols and max_num_nonzeros.
  //
  // The use case for this constructor is that when the user knows the
  // size of the matrix to begin with and wants to update the layout
  // manually, instead of going via the indirect route of first
  // constructing a TripletSparseMatrix, which leads to more than
  // double the peak memory usage.
  //
  // The storage type is set to UNSYMMETRIC.
  CompressedRowSparseMatrix(int num_rows,
                            int num_cols,
                            int max_num_nonzeros);

  // Build a square sparse diagonal matrix with num_rows rows and
  // columns. The diagonal m(i,i) = diagonal(i);
  //
  // The storage type is set to UNSYMMETRIC
  CompressedRowSparseMatrix(const double* diagonal, int num_rows);

  // SparseMatrix interface.
  virtual ~CompressedRowSparseMatrix();
  virtual void SetZero();
  virtual void RightMultiply(const double* x, double* y) const;
  virtual void LeftMultiply(const double* x, double* y) const;
  virtual void SquaredColumnNorm(double* x) const;
  virtual void ScaleColumns(const double* scale);

  virtual void ToDenseMatrix(Matrix* dense_matrix) const;
  virtual void ToTextFile(FILE* file) const;
  virtual int num_rows() const { return num_rows_; }
  virtual int num_cols() const { return num_cols_; }
  virtual int num_nonzeros() const { return rows_[num_rows_]; }
  virtual const double* values() const { return &values_[0]; }
  virtual double* mutable_values() { return &values_[0]; }

  // Delete the bottom delta_rows.
  // num_rows -= delta_rows
  void DeleteRows(int delta_rows);

  // Append the contents of m to the bottom of this matrix. m must
  // have the same number of columns as this matrix.
  void AppendRows(const CompressedRowSparseMatrix& m);

  void ToCRSMatrix(CRSMatrix* matrix) const;

  CompressedRowSparseMatrix* Transpose() const;

  // Destructive array resizing method.
  void SetMaxNumNonZeros(int num_nonzeros);

  // Non-destructive array resizing method.
  void set_num_rows(const int num_rows) { num_rows_ = num_rows; }
  void set_num_cols(const int num_cols) { num_cols_ = num_cols; }

  // Low level access methods that expose the structure of the matrix.
  const int* cols() const { return &cols_[0]; }
  int* mutable_cols() { return &cols_[0]; }

  const int* rows() const { return &rows_[0]; }
  int* mutable_rows() { return &rows_[0]; }

  const StorageType storage_type() const { return storage_type_; }
  void set_storage_type(const StorageType storage_type) {
    storage_type_ = storage_type;
  }

  const std::vector<int>& row_blocks() const { return row_blocks_; }
  std::vector<int>* mutable_row_blocks() { return &row_blocks_; }

  const std::vector<int>& col_blocks() const { return col_blocks_; }
  std::vector<int>* mutable_col_blocks() { return &col_blocks_; }

  const std::vector<int>& block_offsets() const { return block_offsets_; }
  std::vector<int>* mutable_block_offsets() { return &block_offsets_; }

  const std::vector<int>& crsb_rows() const { return crsb_rows_; }
  std::vector<int>* mutable_crsb_rows() { return &crsb_rows_; }

  const std::vector<int>& crsb_cols() const { return crsb_cols_; }
  std::vector<int>* mutable_crsb_cols() { return &crsb_cols_; }

  // Create a block diagonal CompressedRowSparseMatrix with the given
  // block structure. The individual blocks are assumed to be laid out
  // contiguously in the diagonal array, one block at a time.
  //
  // Caller owns the result.
  static CompressedRowSparseMatrix* CreateBlockDiagonalMatrix(
      const double* diagonal,
      const std::vector<int>& blocks);

  // Options struct to control the generation of random block sparse
  // matrices in compressed row sparse format.
  //
  // The random matrix generation proceeds as follows.
  //
  // First the row and column block structure is determined by
  // generating random row and column block sizes that lie within the
  // given bounds.
  //
  // Then we walk the block structure of the resulting matrix, and with
  // probability block_density detemine whether they are structurally
  // zero or not. If the answer is no, then we generate entries for the
  // block which are distributed normally.
  struct RandomMatrixOptions {
    RandomMatrixOptions()
        : num_row_blocks(0),
          min_row_block_size(0),
          max_row_block_size(0),
          num_col_blocks(0),
          min_col_block_size(0),
          max_col_block_size(0),
          block_density(0.0) {
    }

    int num_row_blocks;
    int min_row_block_size;
    int max_row_block_size;
    int num_col_blocks;
    int min_col_block_size;
    int max_col_block_size;

    // 0 < block_density <= 1 is the probability of a block being
    // present in the matrix. A given random matrix will not have
    // precisely this density.
    double block_density;
  };

  // Create a random CompressedRowSparseMatrix whose entries are
  // normally distributed and whose structure is determined by
  // RandomMatrixOptions.
  //
  // Caller owns the result.
  static CompressedRowSparseMatrix* CreateRandomMatrix(
      const RandomMatrixOptions& options);

  // Compute the sparsity structure of the product m.transpose() * m
  // and create a CompressedRowSparseMatrix corresponding to it.
  //
  // Also compute a "program" vector, which for every term in the
  // block outer product provides the information for the entry
  // in the values array of the result matrix where it should be accumulated.
  //
  // This program is used by the ComputeOuterProduct function below to
  // compute the outer product.
  //
  // Since the entries of the program are the same for rows with the
  // same sparsity structure, the program only stores the result for
  // one row per row block. The ComputeOuterProduct function reuses
  // this information for each row in the row block.
  //
  // storage_type controls the form of the output matrix. It can be
  // LOWER_TRIANGULAR or UPPER_TRIANGULAR.
  static CompressedRowSparseMatrix* CreateOuterProductMatrixAndProgram(
      const CompressedRowSparseMatrix& m,
      const StorageType storage_type,
      std::vector<int>* program);

  // Compute the values array for the expression m.transpose() * m,
  // where the matrix used to store the result and a program have been
  // created using the CreateOuterProductMatrixAndProgram function
  // above.
  static void ComputeOuterProduct(const CompressedRowSparseMatrix& m,
                                  const std::vector<int>& program,
                                  CompressedRowSparseMatrix* result);

 private:

  static CompressedRowSparseMatrix* FromTripletSparseMatrix(
      const TripletSparseMatrix& input, bool transpose);

  int num_rows_;
  int num_cols_;
  std::vector<int> rows_;
  std::vector<int> cols_;
  std::vector<double> values_;
  StorageType storage_type_;

  // If the matrix has an underlying block structure, then it can also
  // carry with it row and column block sizes. This is auxilliary and
  // optional information for use by algorithms operating on the
  // matrix. The class itself does not make use of this information in
  // any way.
  std::vector<int> row_blocks_;
  std::vector<int> col_blocks_;

  // For outer product matrix (J' * J), we pre-compute its block
  // offsets information here for fast outer product computation in
  // block unit.  Since the outer product matrix is symmetric, we do
  // not need to distinguish row or col block. In another word, this
  // is the prefix sum of row_blocks_/col_blocks_.
  std::vector<int> block_offsets_;

  // If the matrix has an underlying block structure, then it can also
  // carry with it compressed row sparse block information.
  std::vector<int> crsb_rows_;
  std::vector<int> crsb_cols_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_COMPRESSED_ROW_SPARSE_MATRIX_H_
