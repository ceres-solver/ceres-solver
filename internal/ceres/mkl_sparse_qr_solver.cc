// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2026 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Author: Sergiu Deitsch

#include "ceres/mkl_sparse_qr_solver.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/str_format.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/event_logger.h"
#include "ceres/types.h"

#ifndef CERES_NO_MKL
#include "mkl.h"
#endif

namespace ceres::internal {

namespace {

#ifndef CERES_NO_MKL
std::string_view MKLStatusToString(const sparse_status_t status) {
  switch (status) {
    case SPARSE_STATUS_SUCCESS:
      return "success";
    case SPARSE_STATUS_NOT_INITIALIZED:
      return "not initialized";
    case SPARSE_STATUS_ALLOC_FAILED:
      return "allocation failed";
    case SPARSE_STATUS_INVALID_VALUE:
      return "invalid value";
    case SPARSE_STATUS_EXECUTION_FAILED:
      return "execution failed";
    case SPARSE_STATUS_INTERNAL_ERROR:
      return "internal error";
    case SPARSE_STATUS_NOT_SUPPORTED:
      return "not supported";
    default:
      return "unknown status";
  }
}

class MKLThreadScope {
 public:
  explicit MKLThreadScope(const int num_threads)
      : previous_num_threads_(mkl_set_num_threads_local(num_threads)) {}

  ~MKLThreadScope() { mkl_set_num_threads_local(previous_num_threads_); }

 private:
  const int previous_num_threads_;
};

bool CheckMKLStatus(const sparse_status_t status, std::string* message) {
  if (status == SPARSE_STATUS_SUCCESS) {
    return true;
  }

  *message = absl::StrFormat(
      "MKL Sparse QR returned status %d (%s), expected status %d (success).",
      static_cast<int>(status),
      MKLStatusToString(status),
      static_cast<int>(SPARSE_STATUS_SUCCESS));
  return false;
}
#endif

}  // namespace

MKLSparseQRSolver::MKLSparseQRSolver(const LinearSolver::Options& options)
    : options_(options) {}

LinearSolver::Summary MKLSparseQRSolver::SolveImpl(
    CompressedRowSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double* x) {
  EventLogger event_logger("MKLSparseQRSolver::Solve");
  LinearSolver::Summary summary;
  summary.num_iterations = 1;

#ifdef CERES_NO_MKL
  summary.termination_type = LinearSolverTerminationType::FATAL_ERROR;
  summary.message = "Ceres was compiled without MKL Sparse QR support.";
  return summary;
#else
  const MKLThreadScope thread_scope(options_.num_threads);
  const int num_rows = A->num_rows();
  const int num_cols = A->num_cols();
  const int num_nonzeros = A->num_nonzeros();
  const bool regularized = per_solve_options.D != nullptr;
  const int augmented_rows = num_rows + (regularized ? num_cols : 0);
  const int augmented_nonzeros = num_nonzeros + (regularized ? num_cols : 0);

  std::vector<MKL_INT> rows_start(augmented_rows);
  std::vector<MKL_INT> rows_end(augmented_rows);
  std::vector<MKL_INT> columns(augmented_nonzeros);
  std::vector<double> values(augmented_nonzeros);

  for (int row = 0; row < num_rows; ++row) {
    rows_start[row] = A->rows()[row];
    rows_end[row] = A->rows()[row + 1];
  }
  std::copy_n(A->cols(), num_nonzeros, columns.begin());
  std::copy_n(A->values(), num_nonzeros, values.begin());

  if (regularized) {
    for (int col = 0; col < num_cols; ++col) {
      const int row = num_rows + col;
      const int nonzero = num_nonzeros + col;
      rows_start[row] = nonzero;
      rows_end[row] = nonzero + 1;
      columns[nonzero] = col;
      values[nonzero] = per_solve_options.D[col];
    }
  }

  sparse_matrix_t matrix = nullptr;
  sparse_status_t status = mkl_sparse_d_create_csr(&matrix,
                                                   SPARSE_INDEX_BASE_ZERO,
                                                   augmented_rows,
                                                   num_cols,
                                                   rows_start.data(),
                                                   rows_end.data(),
                                                   columns.data(),
                                                   values.data());
  if (!CheckMKLStatus(status, &summary.message)) {
    summary.termination_type = LinearSolverTerminationType::FAILURE;
    return summary;
  }

  const auto destroy_matrix = [&matrix]() {
    if (matrix != nullptr) {
      mkl_sparse_destroy(matrix);
    }
  };

  matrix_descr descriptor{};
  descriptor.type = SPARSE_MATRIX_TYPE_GENERAL;

  status = mkl_sparse_qr_reorder(matrix, descriptor);
  event_logger.AddEvent("Symbolic Factorization");
  if (!CheckMKLStatus(status, &summary.message)) {
    destroy_matrix();
    summary.termination_type = LinearSolverTerminationType::FAILURE;
    return summary;
  }

  status = mkl_sparse_d_qr_factorize(matrix, values.data());
  event_logger.AddEvent("Numeric Factorization");
  if (!CheckMKLStatus(status, &summary.message)) {
    destroy_matrix();
    summary.termination_type = LinearSolverTerminationType::FAILURE;
    return summary;
  }

  std::vector<double> rhs(augmented_rows, 0.0);
  std::copy_n(b, num_rows, rhs.begin());
  status = mkl_sparse_d_qr_solve(SPARSE_OPERATION_NON_TRANSPOSE,
                                 matrix,
                                 values.data(),
                                 SPARSE_LAYOUT_COLUMN_MAJOR,
                                 1,
                                 x,
                                 num_cols,
                                 rhs.data(),
                                 augmented_rows);
  event_logger.AddEvent("Solve");
  destroy_matrix();

  if (!CheckMKLStatus(status, &summary.message)) {
    summary.termination_type = LinearSolverTerminationType::FAILURE;
    return summary;
  }

  summary.termination_type = LinearSolverTerminationType::SUCCESS;
  summary.message = "Success.";
  return summary;
#endif
}

}  // namespace ceres::internal
