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

#include "ceres/mkl_covariance.h"

#ifndef CERES_NO_MKL

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/context_impl.h"
#include "ceres/event_logger.h"
#include "ceres/internal/eigen.h"
#include "ceres/mkl_diagnostics.h"
#include "ceres/mkl_sparse_matrix.h"
#include "ceres/mkl_utils.h"
#include "ceres/parallel_for.h"
#include "mkl.h"

namespace ceres::internal {

namespace {

// Multiplier of the default column pivot threshold
// 20 (m + n) eps max ||J e_j|| used by SuiteSparseQR.
constexpr double kColumnPivotThresholdMultiplier = 20.0;

}  // namespace

bool ComputeCovarianceUsingMklSparseQR(const CRSMatrix& jacobian,
                                       const Covariance::Options& options,
                                       ContextImpl* context,
                                       CompressedRowSparseMatrix* covariance,
                                       std::string* message) {
  EventLogger event_logger("ComputeCovarianceUsingMklSparseQR");

  const int num_rows = jacobian.num_rows;
  const int num_cols = jacobian.num_cols;
  const int num_nonzeros = jacobian.values.size();
  if (num_rows < num_cols) {
    *message = absl::StrFormat(
        "MKL Sparse QR requires an overdetermined or square Jacobian, got %d "
        "rows and %d columns.",
        num_rows,
        num_cols);
    return false;
  }

  if (options.column_pivot_threshold > 0.0) {
    *message = absl::StrFormat(
        "MKL Sparse QR does not support a positive "
        "Covariance::Options::column_pivot_threshold, got %g, expected a "
        "value less than or equal to zero.",
        options.column_pivot_threshold);
    return false;
  }

  // oneMKL takes non-const arrays of MKL_INT, which MklCsrMatrix sorts in
  // place, so it gets its own copy of the Jacobian.
  MklCsrMatrix matrix;
  if (!matrix.Create(
          num_rows,
          num_cols,
          std::vector<MKL_INT>(jacobian.rows.begin(), jacobian.rows.end()),
          std::vector<MKL_INT>(jacobian.cols.begin(), jacobian.cols.end()),
          jacobian.values,
          message)) {
    return false;
  }

  const MklThreadScope thread_scope(options.num_threads);

  if (!CheckMklStatus(
          mkl_sparse_set_qr_hint(matrix.get(), SPARSE_QR_WITH_PIVOTS),
          "sparse QR hint",
          message)) {
    return false;
  }
  sparse_status_t status =
      mkl_sparse_qr_reorder(matrix.get(), kGeneralMatrixDescriptor);
  if (status == SPARSE_STATUS_SUCCESS) {
    status = mkl_sparse_d_qr_factorize(matrix.get(), matrix.mutable_values());
  }
  event_logger.AddEvent("QR Factorization");
  if (!CheckMklStatus(status, "sparse QR factorization", message)) {
    return false;
  }

  const int* rows = covariance->rows();
  const int* cols = covariance->cols();
  double* covariance_values = covariance->mutable_values();
  std::fill_n(covariance_values, covariance->num_nonzeros(), 0.0);
  std::vector<double> jacobian_column_norm(num_cols, 0.0);
  // The inverse column norm is needed for every column because the rank check
  // below inspects all of them.
  std::vector<double> inverse_column_norm(num_cols, 0.0);
  for (int index = 0; index < num_nonzeros; ++index) {
    const double value = jacobian.values[index];
    jacobian_column_norm[jacobian.cols[index]] =
        std::hypot(jacobian_column_norm[jacobian.cols[index]], value);
  }

  // MKL QR solves share workspace attached to the factorization handle, so
  // they stay serial. Solutions are gathered in blocks to amortize the
  // parallel region over the accumulation below.
  constexpr int kMaxSolutionsPerBlock = 64;
  constexpr std::size_t kSolutionBlockBytes = std::size_t{8} << 20;
  const int rows_per_block = std::max(
      1,
      std::min({kMaxSolutionsPerBlock,
                num_rows,
                static_cast<int>(kSolutionBlockBytes /
                                 (sizeof(double) * std::max(num_cols, 1)))}));

  Vector rhs = Vector::Zero(num_rows);
  // Each row holds the solution for one row of the Jacobian.
  Matrix solutions(rows_per_block, num_cols);

  const int num_threads = options.num_threads;
  context->EnsureMinimumThreads(num_threads - 1);

  for (int block_start = 0; block_start < num_rows;
       block_start += rows_per_block) {
    const int block_size = std::min(rows_per_block, num_rows - block_start);
    for (int offset = 0; offset < block_size; ++offset) {
      const int row = block_start + offset;
      rhs[row] = 1.0;
      const sparse_status_t solve_status =
          mkl_sparse_d_qr_solve(SPARSE_OPERATION_NON_TRANSPOSE,
                                matrix.get(),
                                nullptr,
                                SPARSE_LAYOUT_COLUMN_MAJOR,
                                1,
                                solutions.row(offset).data(),
                                num_cols,
                                rhs.data(),
                                num_rows);
      rhs[row] = 0.0;
      if (!CheckMklStatus(solve_status, "sparse QR solve", message)) {
        return false;
      }
    }

    // Accumulate only the requested covariance entries. Each row owns its
    // entries, so this is free of races.
    ParallelFor(
        context,
        0,
        num_cols,
        num_threads,
        [&covariance_values,
         &inverse_column_norm,
         &solutions,
         rows,
         cols,
         num_cols,
         block_size](int /*thread_id*/, int row) {
          for (int offset = 0; offset < block_size; ++offset) {
            const double* solution = solutions.row(offset).data();
            inverse_column_norm[row] =
                std::hypot(inverse_column_norm[row], solution[row]);
            for (int index = rows[row]; index < rows[row + 1]; ++index) {
              covariance_values[index] += solution[row] * solution[cols[index]];
            }
          }
        });
  }
  event_logger.AddEvent("Solve");

  // A pivoted QR factorization treats a column as zero once its norm after
  // orthogonalization drops below the column pivot threshold tau. For a full
  // rank J, a pivot r_kk bounds the norm of the corresponding row of
  // J+ = P R^-1 Q' from below by 1 / |r_kk|, and J+ does not depend on the
  // pivoting. Rejecting every J+ with a row norm of at least 1 / tau therefore
  // rejects every Jacobian for which a pivoted QR factorization has a pivot of
  // at most tau. Such a row also certifies
  // that the smallest singular value of J is at most tau.
  const double max_jacobian_column_norm =
      num_cols == 0 ? 0.0
                    : *std::max_element(jacobian_column_norm.begin(),
                                        jacobian_column_norm.end());
  const double column_pivot_threshold =
      kColumnPivotThresholdMultiplier *
      static_cast<double>(num_rows + num_cols) *
      std::numeric_limits<double>::epsilon() * max_jacobian_column_norm;
  for (int column = 0; column < num_cols; ++column) {
    // The negated comparison also rejects NaN.
    if (!(inverse_column_norm[column] * column_pivot_threshold < 1.0)) {
      *message = absl::StrFormat(
          "MKL Sparse QR detected numerical rank deficiency in column %d: "
          "row norm of J+ is %g, expected less than %g, the reciprocal of the "
          "column pivot threshold %g.",
          column,
          inverse_column_norm[column],
          1.0 / column_pivot_threshold,
          column_pivot_threshold);
      return false;
    }
  }

  event_logger.AddEvent("Rank Check");

  // The rank checks do not catch overflow, e.g., of a uniformly tiny Jacobian.
  for (int row = 0; row < covariance->num_rows(); ++row) {
    for (int index = rows[row]; index < rows[row + 1]; ++index) {
      if (!std::isfinite(covariance_values[index])) {
        *message = absl::StrFormat(
            "MKL Sparse QR produced a non-finite covariance value at row %d, "
            "column %d: %g, expected a finite value.",
            row,
            cols[index],
            covariance_values[index]);
        return false;
      }
    }
  }

  event_logger.AddEvent("Validation");
  return true;
}

}  // namespace ceres::internal

#endif  // CERES_NO_MKL
