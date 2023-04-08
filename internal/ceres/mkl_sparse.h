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
// Author: dmitriy.korchemkin@gmail.com (Dmitriy Korchemkin)
//
// A simple C++ interface to Intel MKL PARDISO solver

#ifndef CERES_INTERNAL_MKL_SPARSE_H_
#define CERES_INTERNAL_MKL_SPARSE_H_

// This include must come before any #ifndef check on Ceres compile options.
#include "ceres/internal/config.h"
#include "ceres/linear_solver.h"
#include "ceres/sparse_cholesky.h"

#ifdef CERES_USE_MKL
#include <mkl.h>

namespace ceres::internal {
class SparseCholesky;

namespace MKLUtils {
// Return a new handle that references structure of existing one, but stores
// values in a separate array. Callee is responsible for keeping array of values
// as long as it might be utilized via returned handle.
std::pair<sparse_matrix_t, std::unique_ptr<double[]>> AllocateValues(
    const sparse_matrix_t& handle);
// Convert MKL sparse matrix to CRS matrix
CompressedRowSparseMatrix FromMKLHandle(const sparse_matrix_t& handle,
                                        bool copy_values = true);
// Create MKL sparse matrix from CRS matrix
// Returned handle stores references to structure and values of the input
// matrix; callee is responsible for keeping matrix m as long as MKL handle is
// used.
// When object is no more needed, it should be destroyed with mkl_sparse_destroy
sparse_matrix_t ToMKLHandle(CompressedRowSparseMatrix& m);
// Compute AtA product
CompressedRowSparseMatrix AtA(const CompressedRowSparseMatrix& m);
// Compute structure of AtA product
sparse_matrix_t AtAStructure(const sparse_matrix_t& m);
}  // namespace MKLUtils

// Wrapper around direct solver interface of Intel mkl
class CERES_NO_EXPORT MKLPardiso {
 public:
  MKLPardiso();
  ~MKLPardiso();

  bool DefineStructure(
      const sparse_matrix_t& m,
      const CompressedRowSparseMatrix::StorageType storage_type,
      std::string* message = nullptr);
  bool DefineStructure(const CompressedRowSparseMatrix& m,
                       std::string* message = nullptr);
  // MKL direct solver interface requires symmetric matrices to be strictly
  // upper-triangular. In ceres-solver matrices are stored as
  // upper-block-triangular. A remap is stored in MKLPardiso wrapper from
  // uppper-block-triangular to upper-triangular matrix
  void AnalyzeStructure(const CompressedRowSparseMatrix& m);
  void AnalyzeStructure(
      const sparse_matrix_t& m,
      const CompressedRowSparseMatrix ::StorageType storage_type);
  // Compute permutation
  bool Reorder(const OrderingType ordering_type,
               int* permutation,
               std::string* message = nullptr);
  // Numeric factorization
  bool Factorize(const CompressedRowSparseMatrix& m,
                 bool positive_definite,
                 std::string* message = nullptr);
  bool Solve(const double* rhs,
             double* solution,
             std::string* message = nullptr);

 private:
  void AnalyzeStructure(
      int num_rows,
      int num_cols,
      int num_nonzeros,
      const int* rows,
      const int* cols,
      const CompressedRowSparseMatrix::StorageType storage_type);
  int CallPardiso(MKL_INT phase,
                  const double* values,
                  int* permutation,
                  const double* b,
                  double* x);
  bool DefineStructure(
      const CompressedRowSparseMatrix::StorageType storage_type,
      std::string* message);
  MKL_INT matrix_type_;
  MKL_INT message_level_ = 1;
  MKL_INT iparam[64] = {0};
  void* pparam[64] = {0};
  bool pardiso_initialized_ = false;

  int num_rows_;
  int num_cols_;
  int num_nonzeros_;
  bool requires_remap_;
  const int* rows_;
  const int* cols_;
  std::vector<int> order_;
  std::vector<int> rows_out_;
  std::vector<int> cols_out_;
  std::vector<std::pair<int, int>> permutation_;
  std::vector<double> values_prem_;
};

class CERES_NO_EXPORT MKLSparseCholesky final : public SparseCholesky {
 public:
  static std::unique_ptr<SparseCholesky> Create(
      const OrderingType ordering_type);

  ~MKLSparseCholesky() override = default;
  CompressedRowSparseMatrix::StorageType StorageType() const final;
  LinearSolverTerminationType Factorize(CompressedRowSparseMatrix* lhs,
                                        std::string* message) final;
  LinearSolverTerminationType Solve(const double* rhs,
                                    double* solution,
                                    std::string* message) final;

 private:
  explicit MKLSparseCholesky(const OrderingType ordering_type);

  const OrderingType ordering_type_;
  MKLPardiso mkl_;
  bool analyzed_ = false;
};

}  // namespace ceres::internal

#endif  // CERES_USE_MKL
#endif  // CERES_INTERNAL_MKL_SPARSE_H_
