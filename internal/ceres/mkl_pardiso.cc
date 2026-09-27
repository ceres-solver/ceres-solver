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

#include "ceres/mkl_pardiso.h"

#ifndef CERES_NO_MKL

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/strings/str_format.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/event_logger.h"
#include "ceres/mkl_pardiso_diagnostics.h"
#include "ceres/mkl_utils.h"
#include "ceres/types.h"
#include "mkl.h"

namespace ceres::internal {

// Overloads the PARDISO functions on their integer type. pardiso uses MKL_INT
// and pardiso_64 uses MKL_INT64 with either interface. Both therefore accept
// the same types with the ILP64 interface, which provides no 32-bit overloads.
struct Pardiso {
#ifndef MKL_ILP64
  static void Init(_MKL_DSS_HANDLE_t handle,
                   const MKL_INT* matrix_type,
                   MKL_INT* iparm) {
    pardisoinit(handle, matrix_type, iparm);
  }

  static void Call(_MKL_DSS_HANDLE_t handle,
                   const MKL_INT* max_factors,
                   const MKL_INT* factor_number,
                   const MKL_INT* matrix_type,
                   const MKL_INT* phase,
                   const MKL_INT* size,
                   const void* values,
                   const MKL_INT* rows,
                   const MKL_INT* columns,
                   MKL_INT* permutation,
                   const MKL_INT* rhs_count,
                   MKL_INT* iparm,
                   const MKL_INT* message_level,
                   void* rhs,
                   void* solution,
                   MKL_INT* error) {
    pardiso(handle,
            max_factors,
            factor_number,
            matrix_type,
            phase,
            size,
            values,
            rows,
            columns,
            permutation,
            rhs_count,
            iparm,
            message_level,
            rhs,
            solution,
            error);
  }
#endif  // MKL_ILP64

  // pardisoinit has no 64-bit variant. It only clears the handle and sets the
  // default parameters, which are widened afterwards.
  static void Init(_MKL_DSS_HANDLE_t handle,
                   const MKL_INT64* matrix_type,
                   MKL_INT64* iparm) {
    const MKL_INT narrow_matrix_type = static_cast<MKL_INT>(*matrix_type);
    std::array<MKL_INT, kPardisoParameterCount> narrow_iparm{};
    pardisoinit(handle, &narrow_matrix_type, narrow_iparm.data());
    std::copy(narrow_iparm.begin(), narrow_iparm.end(), iparm);
  }

  static void Call(_MKL_DSS_HANDLE_t handle,
                   const MKL_INT64* max_factors,
                   const MKL_INT64* factor_number,
                   const MKL_INT64* matrix_type,
                   const MKL_INT64* phase,
                   const MKL_INT64* size,
                   const void* values,
                   const MKL_INT64* rows,
                   const MKL_INT64* columns,
                   MKL_INT64* permutation,
                   const MKL_INT64* rhs_count,
                   MKL_INT64* iparm,
                   const MKL_INT64* message_level,
                   void* rhs,
                   void* solution,
                   MKL_INT64* error) {
    pardiso_64(handle,
               max_factors,
               factor_number,
               matrix_type,
               phase,
               size,
               values,
               rows,
               columns,
               permutation,
               rhs_count,
               iparm,
               message_level,
               rhs,
               solution,
               error);
  }
};

// PARDISO-compatible CSR matrix and the map back to the Ceres values.
struct PardisoMatrixState {
  std::vector<PardisoIndex> rows;
  std::vector<PardisoIndex> columns;
  // If every source row is sorted, the entries PARDISO keeps from a row are
  // contiguous and start at the source index stored here for each row.
  std::vector<int> source_offsets;
  // Otherwise, the source index of every PARDISO entry, or -1 for an inserted
  // diagonal entry.
  std::vector<int> source_indices;
  // Allocated by the first RefreshValues, so that computing an ordering does
  // not allocate values.
  std::vector<double> values;
  // Dimensions of the source matrix that the maps above refer to.
  int input_num_rows = 0;
  int input_num_nonzeros = 0;
};

class CERES_NO_EXPORT PardisoSolver final {
 public:
  PardisoSolver(OrderingType ordering_type,
                int max_num_threads,
                bool use_two_level_factorization)
      : ordering_type_(ordering_type),
        max_num_threads_(max_num_threads),
        use_two_level_factorization_(use_two_level_factorization) {}

  ~PardisoSolver() {
    if (matrix_) {
      Call(kPardisoRelease, nullptr, nullptr, nullptr, nullptr);
    }
  }

  // Convert Ceres upper triangular storage to PARDISO scalar CSR.
  bool DefineStructure(const CompressedRowSparseMatrix& matrix,
                       std::string* message) {
    CHECK_EQ(matrix.storage_type(),
             CompressedRowSparseMatrix::StorageType::UPPER_TRIANGULAR);
    if (matrix_) {
      // SparseCholesky requires the sparsity structure to stay fixed after
      // the first factorization. Only the dimensions are verified because
      // RefreshValues depends on them.
      if (matrix.num_rows() != matrix_->input_num_rows ||
          matrix.num_nonzeros() != matrix_->input_num_nonzeros) {
        *message = absl::StrFormat(
            "PARDISO matrix structure changed after symbolic analysis: got "
            "%d rows and %d nonzeros, expected %d rows and %d nonzeros.",
            matrix.num_rows(),
            matrix.num_nonzeros(),
            matrix_->input_num_rows,
            matrix_->input_num_nonzeros);
        return false;
      }
      return true;
    }

    const int num_rows = matrix.num_rows();
    const int* source_rows = matrix.rows();
    const int* source_columns = matrix.cols();
    bool rows_sorted = true;
    for (int row = 0; row < num_rows && rows_sorted; ++row) {
      rows_sorted = std::is_sorted(source_columns + source_rows[row],
                                   source_columns + source_rows[row + 1]);
    }

    // PARDISO requires sorted columns and an explicit diagonal, which is the
    // first entry of an upper triangular row. Block upper triangular matrices
    // store their diagonal blocks in full, so entries below the diagonal are
    // skipped.
    std::vector<PardisoIndex> rows(num_rows + 1);
    std::vector<PardisoIndex> columns;
    std::vector<int> source_offsets;
    std::vector<int> source_indices;
    columns.reserve(matrix.num_nonzeros() + num_rows);
    if (rows_sorted) {
      source_offsets.resize(num_rows);
      for (int row = 0; row < num_rows; ++row) {
        const int* row_end = source_columns + source_rows[row + 1];
        const int* kept =
            std::lower_bound(source_columns + source_rows[row], row_end, row);
        rows[row] = static_cast<PardisoIndex>(columns.size());
        source_offsets[row] = static_cast<int>(kept - source_columns);
        if (kept == row_end || *kept != row) {
          columns.push_back(row);
        }
        columns.insert(columns.end(), kept, row_end);
      }
    } else {
      source_indices.reserve(matrix.num_nonzeros() + num_rows);
      std::vector<std::pair<int, int>> entries;
      for (int row = 0; row < num_rows; ++row) {
        rows[row] = static_cast<PardisoIndex>(columns.size());
        entries.clear();
        for (int index = source_rows[row]; index < source_rows[row + 1];
             ++index) {
          if (source_columns[index] >= row) {
            entries.emplace_back(source_columns[index], index);
          }
        }
        std::sort(entries.begin(), entries.end());
        if (entries.empty() || entries.front().first != row) {
          columns.push_back(row);
          source_indices.push_back(-1);
        }
        for (const auto& [column, index] : entries) {
          columns.push_back(column);
          source_indices.push_back(index);
        }
      }
    }
    rows.back() = static_cast<PardisoIndex>(columns.size());

    PardisoMatrixState& state = matrix_.emplace();
    state.rows = std::move(rows);
    state.columns = std::move(columns);
    state.source_offsets = std::move(source_offsets);
    state.source_indices = std::move(source_indices);
    state.input_num_rows = matrix.num_rows();
    state.input_num_nonzeros = matrix.num_nonzeros();
    Initialize();
    if (ordering_type_ == OrderingType::NATURAL) {
      permutation_.resize(matrix.num_rows());
      std::iota(permutation_.begin(), permutation_.end(), 0);
    }
    return true;
  }

  // Refresh the PARDISO value array from the source matrix. Fill-in entries
  // introduced by DefineStructure stay zero.
  void RefreshValues(const CompressedRowSparseMatrix& matrix) {
    if (matrix_->values.empty()) {
      matrix_->values.resize(matrix_->columns.size());
    }
    if (!matrix_->source_offsets.empty()) {
      // The kept entries of a row end the PARDISO row, after a possibly
      // inserted diagonal entry.
      for (int row = 0; row < matrix_->input_num_rows; ++row) {
        const int source_begin = matrix_->source_offsets[row];
        const int count = matrix.rows()[row + 1] - source_begin;
        std::copy_n(matrix.values() + source_begin,
                    count,
                    matrix_->values.data() + matrix_->rows[row + 1] - count);
      }
      return;
    }
    for (int index = 0;
         index < static_cast<int>(matrix_->source_indices.size());
         ++index) {
      if (matrix_->source_indices[index] >= 0) {
        matrix_->values[index] =
            matrix.values()[matrix_->source_indices[index]];
      }
    }
  }

  LinearSolverTerminationType AnalyzeStructure(std::string* message) {
    if (analyzed_) {
      return LinearSolverTerminationType::SUCCESS;
    }

    const PardisoIndex status = Call(
        kPardisoAnalysis,
        matrix_->values.data(),
        ordering_type_ == OrderingType::NATURAL ? permutation_.data() : nullptr,
        nullptr,
        nullptr);
    if (!CheckPardisoStatus(status, "symbolic analysis", message)) {
      return PardisoErrorToTerminationType(status);
    }
    if (!CheckPardisoFactorNonzeros(iparm_[kPardisoFactorNonzerosParameter],
                                    message)) {
      return LinearSolverTerminationType::FATAL_ERROR;
    }
    analyzed_ = true;
    return LinearSolverTerminationType::SUCCESS;
  }

  LinearSolverTerminationType Factorize(std::string* message) {
    const PardisoIndex status = Call(
        kPardisoNumericalFactorization,
        matrix_->values.data(),
        ordering_type_ == OrderingType::NATURAL ? permutation_.data() : nullptr,
        nullptr,
        nullptr);
    CheckPardisoStatus(status, "factorization", message);
    return PardisoErrorToTerminationType(status);
  }

  LinearSolverTerminationType Solve(const double* rhs,
                                    double* solution,
                                    std::string* message) {
    const PardisoIndex status =
        Call(kPardisoSolve, nullptr, nullptr, rhs, solution);
    CheckPardisoStatus(status, "solve", message);
    return PardisoErrorToTerminationType(status);
  }

  bool ComputeOrdering(const CompressedRowSparseMatrix& matrix,
                       int* ordering,
                       std::string* message) {
    if (!DefineStructure(matrix, message)) {
      return false;
    }
    iparm_[kPardisoUserPermutationParameter] = kPardisoReturnPermutation;
    std::vector<PardisoIndex> permutation(matrix.num_cols());
    const PardisoIndex status =
        Call(kPardisoAnalysis, nullptr, permutation.data(), nullptr, nullptr);
    if (!CheckPardisoStatus(status, "ordering", message)) {
      return false;
    }
    std::vector<bool> seen(matrix.num_cols(), false);
    for (int index = 0; index < matrix.num_cols(); ++index) {
      if (permutation[index] < 0 || permutation[index] >= matrix.num_cols()) {
        *message = absl::StrFormat(
            "PARDISO ordering at index %d is %d, expected a value in the "
            "range [0, %d).",
            index,
            permutation[index],
            matrix.num_cols());
        return false;
      }
      if (seen[permutation[index]]) {
        *message = absl::StrFormat(
            "PARDISO ordering contains duplicate value %d at index %d, "
            "expected a permutation of [0, %d).",
            permutation[index],
            index,
            matrix.num_cols());
        return false;
      }
      seen[permutation[index]] = true;
      ordering[index] = static_cast<int>(permutation[index]);
    }
    return true;
  }

 private:
  void Initialize() {
    constexpr PardisoIndex matrix_type = kPardisoPositiveDefinite;
    // pardisoinit sets the defaults for the matrix type, which the settings
    // below override. mkl_pardiso_diagnostics.h describes the parameters.
    Pardiso::Init(pparam_, &matrix_type, iparm_.data());
    iparm_[kPardisoUseDefaultValuesParameter] = kPardisoUserParameters;
    iparm_[kPardisoIndexBaseParameter] = kPardisoZeroBasedIndexing;
    // pardisoinit enables two steps of iterative refinement. Ceres refines
    // solutions itself if max_num_refinement_iterations is positive.
    iparm_[kPardisoIterativeRefinementParameter] =
        kPardisoNoIterativeRefinement;
    iparm_[kPardisoParallelFactorizationParameter] =
        use_two_level_factorization_ ? kPardisoTwoLevelFactorization
                                     : kPardisoClassicFactorization;
    iparm_[kPardisoParallelSolveParameter] = kPardisoParallelSolve;
    if (ordering_type_ == OrderingType::NESDIS) {
      iparm_[kPardisoOrderingParameter] =
          kPardisoParallelNestedDissectionOrdering;
    } else if (ordering_type_ == OrderingType::AMD) {
      iparm_[kPardisoOrderingParameter] = kPardisoMinimumDegreeOrdering;
    } else if (ordering_type_ == OrderingType::NATURAL) {
      iparm_[kPardisoUserPermutationParameter] = kPardisoUseUserPermutation;
    }
    // The symbolic analysis fails if the reported factor nonzero count is
    // negative. pardisoinit also enables counting the operations, which can
    // slow down the symbolic analysis and is only for reporting.
    iparm_[kPardisoFactorNonzerosParameter] = kPardisoEnableReport;
    iparm_[kPardisoFactorOperationsParameter] =
        VLOG_IS_ON(3) ? kPardisoEnableReport : kPardisoDisabled;
    if (VLOG_IS_ON(3)) {
      previous_iparm_ = iparm_;
      has_iparm_snapshot_ = true;
      VLOG(3) << FormatPardisoIparmTable(
          "PARDISO iparm initialization", iparm_, nullptr);
    }
  }

  PardisoIndex Call(PardisoIndex phase,
                    const double* values,
                    PardisoIndex* permutation,
                    const double* rhs,
                    double* solution) {
    const MklThreadScope thread_scope(max_num_threads_);
    constexpr PardisoIndex max_factors = 1;
    constexpr PardisoIndex factor_number = 1;
    constexpr PardisoIndex rhs_count = 1;
    constexpr PardisoIndex message_level = 0;
    constexpr PardisoIndex matrix_type = kPardisoPositiveDefinite;
    PardisoIndex error = 0;
    // Every phase, including the release, follows DefineStructure.
    CHECK(matrix_.has_value());
    const PardisoIndex size =
        static_cast<PardisoIndex>(matrix_->rows.size()) - 1;
    // The matrix checker rejects invalid structures, e.g., unsorted columns,
    // which PARDISO can otherwise accept silently. Checking the symbolic
    // analysis costs little. Checking every numerical factorization and solve
    // measurably slows down large problems, so only debug builds do so.
#ifdef NDEBUG
    iparm_[kPardisoMatrixCheckerParameter] = phase == kPardisoAnalysis
                                                 ? kPardisoEnableMatrixChecker
                                                 : kPardisoDisabled;
#else
    iparm_[kPardisoMatrixCheckerParameter] = kPardisoEnableMatrixChecker;
#endif
    if (has_iparm_snapshot_) {
      previous_iparm_[kPardisoMatrixCheckerParameter] =
          iparm_[kPardisoMatrixCheckerParameter];
    }
    Pardiso::Call(pparam_,
                  &max_factors,
                  &factor_number,
                  &matrix_type,
                  &phase,
                  &size,
                  values,
                  matrix_->rows.data(),
                  matrix_->columns.data(),
                  permutation,
                  &rhs_count,
                  iparm_.data(),
                  &message_level,
                  const_cast<double*>(rhs),
                  solution,
                  &error);
    if (VLOG_IS_ON(3)) {
      if (phase != kPardisoRelease && has_iparm_snapshot_) {
        if (const std::string changes = FormatPardisoIparmTable(
                absl::StrFormat("PARDISO iparm changes after %s (%d)",
                                PardisoPhaseToString(phase),
                                phase),
                iparm_,
                &previous_iparm_);
            !changes.empty()) {
          VLOG(3) << changes;
        }
        previous_iparm_ = iparm_;
      }
      VLOG(3) << absl::StrFormat(
          "PARDISO %s (%d) returned %s (%d) for %d x %d matrix, requested "
          "threads %d, active MKL threads %d.",
          PardisoPhaseToString(phase),
          phase,
          PardisoErrorToString(error),
          error,
          size,
          size,
          max_num_threads_,
          mkl_get_max_threads());
    }
    return error;
  }

  const OrderingType ordering_type_;
  const int max_num_threads_;
  const bool use_two_level_factorization_;
  std::array<PardisoIndex, kPardisoParameterCount> iparm_{};
  void* pparam_[kPardisoParameterCount] = {};
  std::array<PardisoIndex, kPardisoParameterCount> previous_iparm_{};
  bool has_iparm_snapshot_ = false;
  bool analyzed_ = false;
  std::optional<PardisoMatrixState> matrix_;
  std::vector<PardisoIndex> permutation_;
};

bool ComputePardisoOrdering(const CompressedRowSparseMatrix& matrix,
                            const LinearSolverOrderingType ordering_type,
                            const int max_num_threads,
                            int* ordering,
                            std::string* message) {
  PardisoSolver solver(
      ordering_type == AMD ? OrderingType::AMD : OrderingType::NESDIS,
      max_num_threads,
      /*use_two_level_factorization=*/false);
  return solver.ComputeOrdering(matrix, ordering, message);
}

MklSparseCholesky::MklSparseCholesky(const OrderingType ordering_type,
                                     const int max_num_threads,
                                     const bool use_two_level_factorization)
    : solver_(std::make_unique<PardisoSolver>(
          ordering_type, max_num_threads, use_two_level_factorization)) {}

MklSparseCholesky::~MklSparseCholesky() = default;

std::unique_ptr<MklSparseCholesky> MklSparseCholesky::Create(
    const OrderingType ordering_type,
    const int max_num_threads,
    const bool use_two_level_factorization) {
  return std::unique_ptr<MklSparseCholesky>(new MklSparseCholesky(
      ordering_type, max_num_threads, use_two_level_factorization));
}

CompressedRowSparseMatrix::StorageType MklSparseCholesky::StorageType() const {
  return CompressedRowSparseMatrix::StorageType::UPPER_TRIANGULAR;
}

LinearSolverTerminationType MklSparseCholesky::Factorize(
    CompressedRowSparseMatrix* lhs, std::string* message) {
  EventLogger event_logger("MklSparseCholesky::Factorize");
  if (!solver_->DefineStructure(*lhs, message)) {
    return LinearSolverTerminationType::FATAL_ERROR;
  }
  event_logger.AddEvent("Define structure");
  solver_->RefreshValues(*lhs);
  if (const LinearSolverTerminationType analysis_termination_type =
          solver_->AnalyzeStructure(message);
      analysis_termination_type != LinearSolverTerminationType::SUCCESS) {
    return analysis_termination_type;
  }
  event_logger.AddEvent("Analyze structure");
  const LinearSolverTerminationType termination_type =
      solver_->Factorize(message);
  event_logger.AddEvent("Factorize");
  return termination_type;
}

LinearSolverTerminationType MklSparseCholesky::Solve(const double* rhs,
                                                     double* solution,
                                                     std::string* message) {
  return solver_->Solve(rhs, solution, message);
}

}  // namespace ceres::internal

#endif  // CERES_NO_MKL
