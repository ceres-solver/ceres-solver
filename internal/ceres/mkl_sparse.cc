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

// This include must come before any #ifndef check on Ceres compile options.
#include "ceres/internal/config.h"

#ifndef CERES_NO_MKL

#include "ceres/compressed_col_sparse_matrix_utils.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/linear_solver.h"
#include "ceres/mkl_sparse.h"

namespace {
const char* PARDISOErrorToString(MKL_INT pardiso_error) {
  switch (pardiso_error) {
    case 0:
      return "No error";
    case -1:
      return "Input inconsistent";
    case -2:
      return "Not enough memory";
    case -3:
      return "Reordering problem";
    case -4:
      return "Zero pivot, numerical factorization or iterative refinement "
             "problem";
    case -5:
      return "Unclassified (internal) problem";
    case -6:
      return "Reordering failed (non-symmetric)";
    case -7:
      return "Diagonal matrix is singular";
    case -8:
      return "32-bit integer overflow problem";
    case -9:
      return "Not enough memory for Out-Of-Core solver";
    case -10:
      return "Problems with opening Out-Of-Core temporary files";
    case -11:
      return "Read/write problems with Out-Of-Core data file";
    default:
      return "Unrecognized error";
  }
}
}  // namespace

namespace ceres::internal {

MKLSparse::MKLSparse() {
  constexpr int kIParamIndexType = 34;
  constexpr int kIParamSetDefaults = 0;
  constexpr int kIParamMatrixChecker = 26;

  iparam[kIParamIndexType] = 1;
  iparam[kIParamSetDefaults] = 1;
#ifdef NDEBUG
  iparam[kIParamMatrixChecker] = 0;
#else
  iparam[kIParamMatrixChecker] = 1;
#endif
  message_level_ = VLOG_IS_ON(2) ? 1 : 0;
}

MKLSparse::~MKLSparse() {
  if (pardiso_initialized_) {
    CallPardiso(-1, nullptr, nullptr, nullptr, nullptr);
    pardiso_initialized_ = false;
  }
}

CompressedRowSparseMatrix MKLSparse::FromMKLHandle(
    const sparse_matrix_t& handle, bool copy_values) {
  int num_rows;
  int num_cols;
  sparse_index_base_t indexing;
  int *row_start, *row_end;
  int* col;
  double* values;
  mkl_sparse_d_export_csr(handle,
                          &indexing,
                          &num_rows,
                          &num_cols,
                          &row_start,
                          &row_end,
                          &col,
                          &values);
  CHECK_EQ(indexing, SPARSE_INDEX_BASE_ZERO);
  CHECK_EQ(row_end, row_start + 1);
  int num_nonzeros = row_start[num_rows];
  CompressedRowSparseMatrix crs_matrix(num_rows, num_cols, num_nonzeros);
  std::copy_n(row_start, num_rows + 1, crs_matrix.mutable_rows());
  std::copy_n(col, num_nonzeros, crs_matrix.mutable_cols());
  if (copy_values) {
    std::copy_n(values, num_nonzeros, crs_matrix.mutable_values());
  }
  return crs_matrix;
}

std::pair<sparse_matrix_t, std::unique_ptr<double[]>> MKLSparse::AllocateValues(
    const sparse_matrix_t& matrix) {
  int num_rows;
  int num_cols;
  sparse_index_base_t indexing;
  int *row_start, *row_end;
  int* col;
  double* values;
  mkl_sparse_d_export_csr(matrix,
                          &indexing,
                          &num_rows,
                          &num_cols,
                          &row_start,
                          &row_end,
                          &col,
                          &values);
  CHECK_EQ(indexing, SPARSE_INDEX_BASE_ZERO);
  CHECK_EQ(row_end, row_start + 1);
  int num_nonzeros = row_start[num_rows];
  auto new_values = std::make_unique<double[]>(num_nonzeros);
  sparse_matrix_t matrix_with_values;
  mkl_sparse_d_create_csr(&matrix_with_values,
                          SPARSE_INDEX_BASE_ZERO,
                          num_rows,
                          num_cols,
                          row_start,
                          row_end,
                          col,
                          new_values.get());
  return std::make_pair(matrix_with_values, std::move(new_values));
}

sparse_matrix_t MKLSparse::ToMKLHandle(TripletSparseMatrix& m) {
  sparse_matrix_t mkl_matrix;
  mkl_sparse_d_create_coo(&mkl_matrix,
                          SPARSE_INDEX_BASE_ZERO,
                          m.num_rows(),
                          m.num_cols(),
                          m.num_nonzeros(),
                          m.mutable_rows(),
                          m.mutable_cols(),
                          m.mutable_values());
  sparse_matrix_t mkl_csr_matrix;
  mkl_sparse_convert_csr(
      mkl_matrix, SPARSE_OPERATION_NON_TRANSPOSE, &mkl_csr_matrix);
  mkl_sparse_destroy(mkl_matrix);
  return mkl_csr_matrix;
}

sparse_matrix_t MKLSparse::ToMKLHandle(CompressedRowSparseMatrix& m) {
  sparse_matrix_t mkl_matrix;
  mkl_sparse_d_create_csr(&mkl_matrix,
                          SPARSE_INDEX_BASE_ZERO,
                          m.num_rows(),
                          m.num_cols(),
                          m.mutable_rows(),
                          m.mutable_rows() + 1,
                          m.mutable_cols(),
                          m.mutable_values());
  return mkl_matrix;
}

sparse_matrix_t MKLSparse::AtAStructure(const sparse_matrix_t& a) {
  sparse_matrix_t res;
  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  CHECK_EQ(SPARSE_STATUS_SUCCESS,
           mkl_sparse_sp2m(SPARSE_OPERATION_TRANSPOSE,
                           descr,
                           a,
                           SPARSE_OPERATION_NON_TRANSPOSE,
                           descr,
                           a,
                           SPARSE_STAGE_FULL_MULT_NO_VAL,
                           &res));
  return res;
}

CompressedRowSparseMatrix MKLSparse::AtA(const CompressedRowSparseMatrix& m) {
  auto mkl_m = ToMKLHandle(const_cast<CompressedRowSparseMatrix&>(m));
  sparse_matrix_t ata;
  mkl_sparse_syrk(SPARSE_OPERATION_TRANSPOSE, mkl_m, &ata);
  auto res = FromMKLHandle(ata);
  res.set_storage_type(
      CompressedRowSparseMatrix::StorageType::UPPER_TRIANGULAR);
  mkl_sparse_destroy(ata);
  mkl_sparse_destroy(mkl_m);
  return res;
}

#define MKL_HANDLE_ERROR(pardiso_status, text)                            \
  if (pardiso_status != MKL_DSS_SUCCESS) {                                \
    auto error_text =                                                     \
        StringPrintf("PARDISO call completed with error code %d (%s) %s", \
                     pardiso_status,                                      \
                     PARDISOErrorToString(pardiso_status),                \
                     text);                                               \
    if (message) {                                                        \
      *message = error_text;                                              \
    } else {                                                              \
      LOG(ERROR) << error_text;                                           \
    }                                                                     \
    return false;                                                         \
  }

bool MKLSparse::DefineStructure(
    const CompressedRowSparseMatrix::StorageType storage_type,
    std::string* message) {
  CHECK(storage_type !=
        CompressedRowSparseMatrix::StorageType::LOWER_TRIANGULAR);
  matrix_type_ =
      storage_type == CompressedRowSparseMatrix::StorageType::UNSYMMETRIC ? 11
                                                                          : 2;
  return true;
}

bool MKLSparse::DefineStructure(
    const sparse_matrix_t& m,
    const CompressedRowSparseMatrix::StorageType storage_type,
    std::string* message) {
  AnalyzeStructure(m, storage_type);
  return DefineStructure(storage_type, message);
}

bool MKLSparse::DefineStructure(const CompressedRowSparseMatrix& m,
                                std::string* message) {
  AnalyzeStructure(m);
  return DefineStructure(m.storage_type(), message);
}
void MKLSparse::AnalyzeStructure(
    int num_rows,
    int num_cols,
    int num_nonzeros,
    const int* rows,
    const int* cols,
    const CompressedRowSparseMatrix::StorageType storage_type) {
  num_rows_ = num_rows;
  num_cols_ = num_cols;
  num_nonzeros_ = num_nonzeros;
  rows_ = rows;
  cols_ = cols;
  if (storage_type == CompressedRowSparseMatrix::StorageType::UNSYMMETRIC) {
    requires_remap_ = false;
    return;
  }

  int modifications = 0;
  int value_offset = 0;
  rows_out_.clear();
  cols_out_.clear();
  cols_out_.reserve(num_nonzeros_);
  rows_out_.reserve(num_rows_ + 1);
  permutation_.reserve(num_nonzeros_);

  for (int row = 0; row < num_rows_; ++row) {
    rows_out_.push_back(value_offset);
    const auto row_end = cols_ + rows_[row + 1];
    const auto row_begin = cols_ + rows_[row];
    auto it = std::lower_bound(row_begin, row_end, row);
    if (it > row_begin) ++modifications;
    if (it == row_end || *it > row) {
      ++modifications;
      cols_out_.push_back(row);
      permutation_.emplace_back(-1, 0);
      ++value_offset;
    }

    const int num = row_end - it;
    permutation_.emplace_back(it - cols_, num);
    value_offset += num;
    std::copy(it, row_end, std::back_inserter(cols_out_));
  }
  rows_out_.push_back(value_offset);

  requires_remap_ = modifications != 0;
  if (!requires_remap_) {
    return;
  }

  num_nonzeros_ = value_offset;
  rows_ = rows_out_.data();
  cols_ = cols_out_.data();
}

void MKLSparse::AnalyzeStructure(const CompressedRowSparseMatrix& m) {
  AnalyzeStructure(m.num_rows(),
                   m.num_cols(),
                   m.num_nonzeros(),
                   m.rows(),
                   m.cols(),
                   m.storage_type());
}

void MKLSparse::AnalyzeStructure(
    const sparse_matrix_t& m,
    const CompressedRowSparseMatrix ::StorageType storage_type) {
  int num_rows;
  int num_cols;
  sparse_index_base_t indexing;
  int *row_start, *row_end;
  int* col;
  double* values;
  mkl_sparse_d_export_csr(
      m, &indexing, &num_rows, &num_cols, &row_start, &row_end, &col, &values);
  CHECK_EQ(indexing, SPARSE_INDEX_BASE_ZERO);
  CHECK_EQ(row_end, row_start + 1);
  int num_nonzeros = row_start[num_rows];
  AnalyzeStructure(
      num_rows, num_cols, num_nonzeros, row_start, col, storage_type);
}

bool MKLSparse::Reorder(const OrderingType ordering_type,
                        std::string* message) {
  return Reorder(ordering_type, nullptr, message);
}

int MKLSparse::CallPardiso(MKL_INT phase,
                           const double* values,
                           int* permutation,
                           const double* b,
                           double* x) {
  MKL_INT num_factors = 1;
  MKL_INT factor_id = 1;
  MKL_INT num_rhs = 1;
  MKL_INT error_code = 0;

  pardiso(pparam,
          &num_factors,
          &factor_id,
          &matrix_type_,
          &phase,
          &num_rows_,
          values,
          rows_,
          cols_,
          permutation,
          &num_rhs,
          iparam,
          &message_level_,
          const_cast<double*>(b),
          x,
          &error_code);
  pardiso_initialized_ = true;
  return error_code;
}

bool MKLSparse::Reorder(const OrderingType ordering_type,
                        int* permutation,
                        std::string* message) {
  const int kIParamUserPermutation = 4;
  const int kIParamFillInReducingPermutationAlgorithm = 1;
  iparam[kIParamUserPermutation] = 0;
  int* permutation_ptr = nullptr;
  switch (ordering_type) {
    case OrderingType::AMD:
      iparam[kIParamFillInReducingPermutationAlgorithm] = 0;
      break;
    case OrderingType::NESDIS:
      iparam[kIParamFillInReducingPermutationAlgorithm] = 3;
      break;
    case OrderingType::NATURAL:
      iparam[kIParamUserPermutation] = 1;
      order_.resize(num_rows_);
      std::iota(order_.begin(), order_.end(), 0);
      permutation_ptr = order_.data();
      break;
    default:
      LOG(FATAL)
          << "Congratulations, you found a Ceres bug! Please report this error "
          << "to the developers.";
      return false;
  }
  int res = CallPardiso(11, nullptr, permutation_ptr, nullptr, nullptr);
  MKL_HANDLE_ERROR(res, "when computing fill-in reducing reordering");
  // Retrieve permutation to user-provided pointer
  if (permutation) {
    if (ordering_type == OrderingType::NATURAL) {
      std::iota(permutation, permutation + num_rows_, 0);
      return true;
    }
    iparam[kIParamUserPermutation] = 2;
    int res = CallPardiso(11, nullptr, permutation, nullptr, nullptr);
    MKL_HANDLE_ERROR(res, "when exporting fill-in reducing reordering");
  }
  return true;
}

bool MKLSparse::Factorize(const CompressedRowSparseMatrix& m,
                          bool positive_definite,
                          std::string* message) {
  if (!positive_definite && matrix_type_ == 2) {
    matrix_type_ = -2;
  }

  EventLogger el("MKLSparse::Factorize");
  const double* values = m.values();
  if (requires_remap_) {
    int value_offset = 0;
    values_prem_.resize(num_nonzeros_);
    for (auto& pidx : permutation_) {
      auto [from, num] = pidx;
      if (from == -1) {
        // zeros don't move, std::vector is zero-initialized
        ++value_offset;
        continue;
      }
      std::copy(values + from,
                values + from + num,
                values_prem_.data() + value_offset);
      value_offset += num;
    }
    values = values_prem_.data();
  }
  el.AddEvent("Remap");

  int res = CallPardiso(22, values, nullptr, nullptr, nullptr);
  MKL_HANDLE_ERROR(res, "when computing numeric factorization");
  el.AddEvent("Factorize");
  return true;
}

bool MKLSparse::Solve(const double* rhs,
                      double* solution,
                      std::string* message) {
  int res = CallPardiso(33, nullptr, nullptr, rhs, solution);
  MKL_HANDLE_ERROR(res, "when omputing solution vector");
  return true;
}
std::unique_ptr<SparseCholesky> MKLSparseCholesky::Create(
    const OrderingType ordering_type) {
  return std::unique_ptr<SparseCholesky>(new MKLSparseCholesky(ordering_type));
}

MKLSparseCholesky::MKLSparseCholesky(const OrderingType ordering_type)
    : ordering_type_(ordering_type) {}

CompressedRowSparseMatrix::StorageType MKLSparseCholesky::StorageType() const {
  return CompressedRowSparseMatrix::StorageType::UPPER_TRIANGULAR;
}

LinearSolverTerminationType MKLSparseCholesky::Factorize(
    CompressedRowSparseMatrix* lhs, std::string* message) {
  CHECK(lhs != nullptr);
  if (!analyzed_) {
    if (!mkl_.DefineStructure(*lhs, message)) {
      return LinearSolverTerminationType::FATAL_ERROR;
    }
    if (!mkl_.Reorder(ordering_type_, message)) {
      return LinearSolverTerminationType::FATAL_ERROR;
    }
    analyzed_ = true;
  }

  if (!mkl_.Factorize(*lhs, true, message)) {
    return LinearSolverTerminationType::FAILURE;
  }
  return LinearSolverTerminationType::SUCCESS;
}

LinearSolverTerminationType MKLSparseCholesky::Solve(const double* lhs,
                                                     double* solution,
                                                     std::string* message) {
  CHECK(lhs != nullptr);
  if (!mkl_.Solve(lhs, solution, message)) {
    return LinearSolverTerminationType::FAILURE;
  }
  return LinearSolverTerminationType::SUCCESS;
}

}  // namespace ceres::internal

#endif
