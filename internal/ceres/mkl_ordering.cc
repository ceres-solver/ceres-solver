// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2026 Google Inc. All rights reserved.
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
// Author: sergiu.deitsch@gmail.com (Sergiu Deitsch)

#include "ceres/mkl_ordering.h"

#ifndef CERES_NO_MKL

#include <memory>
#include <string>

#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/event_logger.h"
#include "ceres/mkl_diagnostics.h"
#include "ceres/mkl_normal_matrix.h"
#include "ceres/mkl_pardiso.h"
#include "ceres/mkl_sparse_matrix.h"
#include "ceres/mkl_utils.h"
#include "ceres/types.h"
#include "mkl.h"

namespace ceres::internal {

namespace {

// Computes a PARDISO ordering of the symmetric sparsity pattern whose upper
// triangle pattern holds with sorted columns.
bool ComputeOrderingOfPattern(const sparse_matrix_t pattern,
                              const LinearSolverOrderingType ordering_type,
                              const int max_num_threads,
                              int* ordering,
                              std::string* message) {
  std::unique_ptr<CompressedRowSparseMatrix> matrix;
  return ExportMklCsr(pattern, &matrix, message) &&
         ComputePardisoOrdering(
             *matrix, ordering_type, max_num_threads, ordering, message);
}

}  // namespace

bool MklComputeOrdering(const MklCsrMatrix& matrix,
                        const LinearSolverOrderingType ordering_type,
                        const int max_num_threads,
                        int* ordering,
                        std::string* message) {
  EventLogger event_logger("MklComputeOrdering");
  MklSparseHandle normal_matrix;
  {
    const MklThreadScope thread_scope(max_num_threads);
    if (!ComputeAtAUsingMkl(matrix.get(), &normal_matrix, message)) {
      return false;
    }
  }
  event_logger.AddEvent("Form J'J");
  const bool success = ComputeOrderingOfPattern(
      normal_matrix.get(), ordering_type, max_num_threads, ordering, message);
  event_logger.AddEvent("PARDISO Ordering");
  return success;
}

bool MklComputeSchurOrdering(const MklCsrMatrix& e_matrix,
                             const MklCsrMatrix& f_matrix,
                             const LinearSolverOrderingType ordering_type,
                             const int max_num_threads,
                             int* ordering,
                             std::string* message) {
  EventLogger event_logger("MklComputeSchurOrdering");
  MklSparseHandle schur_pattern;
  {
    const MklThreadScope thread_scope(max_num_threads);
    sparse_matrix_t raw_etf_handle = nullptr;
    if (!CheckMklStatus(mkl_sparse_sp2m(SPARSE_OPERATION_TRANSPOSE,
                                        kGeneralMatrixDescriptor,
                                        e_matrix.get(),
                                        SPARSE_OPERATION_NON_TRANSPOSE,
                                        kGeneralMatrixDescriptor,
                                        f_matrix.get(),
                                        SPARSE_STAGE_FULL_MULT,
                                        &raw_etf_handle),
                        "sparse Schur product",
                        message)) {
      return false;
    }
    MklSparseHandle etf_handle(raw_etf_handle);
    // The result of mkl_sparse_sp2m is sorted first because mkl_sparse_syrk
    // can miss entries of an input whose columns are unsorted within a row.
    if (!CheckMklStatus(mkl_sparse_order(etf_handle.get()),
                        "sparse matrix ordering",
                        message)) {
      return false;
    }
    event_logger.AddEvent("Compute E'F");

    // The Schur complement couples two f_blocks if they share an e_block or a
    // residual block. Since all values are positive, the sum of both products
    // has the union of their sparsity patterns.
    MklSparseHandle etf_normal;
    MklSparseHandle f_normal;
    if (!ComputeAtAUsingMkl(etf_handle.get(), &etf_normal, message) ||
        !ComputeAtAUsingMkl(f_matrix.get(), &f_normal, message)) {
      return false;
    }
    etf_handle.reset();
    sparse_matrix_t raw_sum = nullptr;
    if (!CheckMklStatus(mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE,
                                         etf_normal.get(),
                                         1.0,
                                         f_normal.get(),
                                         &raw_sum),
                        "sparse matrix sum",
                        message)) {
      return false;
    }
    schur_pattern.reset(raw_sum);
    if (!CheckMklStatus(mkl_sparse_order(schur_pattern.get()),
                        "sparse matrix ordering",
                        message)) {
      return false;
    }
  }
  event_logger.AddEvent("Form Schur pattern");

  const bool success = ComputeOrderingOfPattern(
      schur_pattern.get(), ordering_type, max_num_threads, ordering, message);
  event_logger.AddEvent("PARDISO Ordering");
  return success;
}

}  // namespace ceres::internal

#endif  // CERES_NO_MKL
