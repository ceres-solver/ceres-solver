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

#include "ceres/mkl_pardiso_diagnostics.h"

#ifndef CERES_NO_MKL

#include <array>
#include <limits>
#include <string>
#include <string_view>

#include "absl/strings/str_format.h"

namespace ceres::internal {

namespace {

constexpr PardisoIndex kPardisoInputInconsistent = -1;
constexpr PardisoIndex kPardisoNotEnoughMemory = -2;
constexpr PardisoIndex kPardisoReorderingProblem = -3;
constexpr PardisoIndex kPardisoNumericalFactorizationProblem = -4;
constexpr PardisoIndex kPardisoInternalProblem = -5;
constexpr PardisoIndex kPardisoNonsymmetricReorderingFailed = -6;
constexpr PardisoIndex kPardisoSingularDiagonalMatrix = -7;
constexpr PardisoIndex kPardisoIntegerOverflow = -8;
constexpr PardisoIndex kPardisoNotEnoughOutOfCoreMemory = -9;
constexpr PardisoIndex kPardisoOutOfCoreFileOpenError = -10;
constexpr PardisoIndex kPardisoOutOfCoreFileAccessError = -11;

constexpr std::string_view kPardisoIparmHeader = "iparm[]";
constexpr int kPardisoIparmColumnWidth =
    static_cast<int>(kPardisoIparmHeader.size());

std::string_view PardisoIparmValueDescription(const int index,
                                              const PardisoIndex value) {
  switch (index) {
    case kPardisoUseDefaultValuesParameter:
      return value == kPardisoDisabled ? "fill in defaults"
                                       : "use supplied values";
    case kPardisoOrderingParameter:
      switch (value) {
        case kPardisoMinimumDegreeOrdering:
          return "minimum degree";
        case kPardisoNestedDissectionOrdering:
          return "nested dissection of METIS";
        case kPardisoParallelNestedDissectionOrdering:
          return "parallel nested dissection";
        default:
          return "unknown ordering";
      }
    case kPardisoPreconditionedCgParameter:
      return value == kPardisoDisabled ? "factorization as required by phase"
                                       : "preconditioned CGS/CG";
    case kPardisoIterativeRefinementParameter:
      if (value == kPardisoNoIterativeRefinement) {
        return "no iterative refinement";
      }
      return value > 0 ? "maximum steps"
                       : "maximum steps in extended precision";
    case kPardisoUserPermutationParameter:
      switch (value) {
        case kPardisoIgnoreUserPermutation:
          return "ignore user permutation";
        case kPardisoUseUserPermutation:
          return "use supplied permutation";
        case kPardisoReturnPermutation:
          return "return computed permutation";
        default:
          return "unknown permutation mode";
      }
    case kPardisoParallelFactorizationParameter:
      switch (value) {
        case kPardisoClassicFactorization:
          return "classic factorization";
        case kPardisoTwoLevelFactorization:
          return "two-level factorization";
        default:
          return "unknown factorization mode";
      }
    case kPardisoParallelSolveParameter:
      switch (value) {
        case kPardisoDisabled:
          return "partitioning for one right-hand side";
        case kPardisoSequentialSolve:
          return "sequential solve";
        case kPardisoParallelSolve:
          return "parallel matrix partitioning";
        default:
          return "unknown solve mode";
      }
    case kPardisoMatrixCheckerParameter:
      return value == kPardisoDisabled ? "disabled" : "enabled";
    case kPardisoIndexBaseParameter:
      switch (value) {
        case kPardisoOneBasedIndexing:
          return "one-based";
        case kPardisoZeroBasedIndexing:
          return "zero-based";
        default:
          return "unknown index base";
      }
    case kPardisoFactorNonzerosParameter:
    case kPardisoFactorOperationsParameter:
      return value == kPardisoEnableReport ? "report enabled"
                                           : std::string_view{};
    default:
      return {};
  }
}

std::string FormatPardisoIparmValue(const int index, const PardisoIndex value) {
  const std::string_view description =
      PardisoIparmValueDescription(index, value);
  if (description.empty()) {
    return absl::StrFormat("%d", value);
  }
  return absl::StrFormat("%s (%d)", description, value);
}

// Describes how to avoid an overflow of the factor nonzero count that PARDISO
// reports using Integer, which is only possible by switching from 32-bit to
// 64-bit integers.
template <typename Integer>
struct FactorNonzerosOverflowRemedy;

template <>
struct FactorNonzerosOverflowRemedy<int> {
  static constexpr std::string_view kText =
      " Configure Ceres with MKL_INTERFACE_FULL=intel_ilp64 to use 64-bit "
      "integers.";
};

template <>
struct FactorNonzerosOverflowRemedy<MKL_INT64> {
  static constexpr std::string_view kText{};
};

}  // namespace

std::string_view PardisoErrorToString(const PardisoIndex error) {
  switch (error) {
    case PARDISO_NO_ERROR:
      return "success";
    case kPardisoInputInconsistent:
      return "input inconsistent";
    case kPardisoNotEnoughMemory:
      return "not enough memory";
    case kPardisoReorderingProblem:
      return "reordering problem";
    case kPardisoNumericalFactorizationProblem:
      return "zero pivot or numerical factorization problem";
    case kPardisoInternalProblem:
      return "internal problem";
    case kPardisoNonsymmetricReorderingFailed:
      return "reordering failed for a nonsymmetric matrix";
    case kPardisoSingularDiagonalMatrix:
      return "singular diagonal matrix";
    case kPardisoIntegerOverflow:
      return "32-bit integer overflow";
    case kPardisoNotEnoughOutOfCoreMemory:
      return "not enough memory for out-of-core execution";
    case kPardisoOutOfCoreFileOpenError:
      return "error opening out-of-core files";
    case kPardisoOutOfCoreFileAccessError:
      return "read or write error with out-of-core files";
    default:
      return "unknown";
  }
}

bool CheckPardisoStatus(const PardisoIndex actual,
                        const std::string_view operation,
                        std::string* message) {
  if (actual == PARDISO_NO_ERROR) {
    return true;
  }

  *message = absl::StrFormat("PARDISO %s returned %s (%d), expected %s (%d).",
                             operation,
                             PardisoErrorToString(actual),
                             actual,
                             PardisoErrorToString(PARDISO_NO_ERROR),
                             PARDISO_NO_ERROR);
  return false;
}

bool CheckPardisoFactorNonzeros(const PardisoIndex factor_nonzeros,
                                std::string* message) {
  if (factor_nonzeros >= 0) {
    return true;
  }

  *message = absl::StrFormat(
      "PARDISO symbolic analysis reports %d nonzeros in the Cholesky factor, "
      "expected a value in [0, %d].%s",
      factor_nonzeros,
      std::numeric_limits<PardisoIndex>::max(),
      FactorNonzerosOverflowRemedy<PardisoIndex>::kText);
  return false;
}

LinearSolverTerminationType PardisoErrorToTerminationType(
    const PardisoIndex error) {
  switch (error) {
    case PARDISO_NO_ERROR:
      return LinearSolverTerminationType::SUCCESS;
    case kPardisoNumericalFactorizationProblem:
    case kPardisoSingularDiagonalMatrix:
      return LinearSolverTerminationType::FAILURE;
    default:
      return LinearSolverTerminationType::FATAL_ERROR;
  }
}

std::string_view PardisoPhaseToString(const PardisoIndex phase) {
  switch (phase) {
    case kPardisoAnalysis:
      return "symbolic analysis";
    case kPardisoNumericalFactorization:
      return "numerical factorization";
    case kPardisoSolve:
      return "solve";
    case kPardisoRelease:
      return "release";
    default:
      return "unknown phase";
  }
}

std::string FormatPardisoIparmTable(
    const std::string_view title,
    const std::array<PardisoIndex, kPardisoParameterCount>& current,
    const std::array<PardisoIndex, kPardisoParameterCount>* previous) {
  std::string rows;
  for (const PardisoIparmEntry& entry : kPardisoIparmEntries) {
    const std::string current_value =
        FormatPardisoIparmValue(entry.index, current[entry.index]);
    if (previous == nullptr) {
      absl::StrAppendFormat(&rows,
                            "  %*d %-70s %s\n",
                            kPardisoIparmColumnWidth,
                            entry.index,
                            entry.name,
                            current_value);
    } else if ((*previous)[entry.index] != current[entry.index]) {
      absl::StrAppendFormat(
          &rows,
          "  %*d %-70s %-30s %s\n",
          kPardisoIparmColumnWidth,
          entry.index,
          entry.name,
          FormatPardisoIparmValue(entry.index, (*previous)[entry.index]),
          current_value);
    }
  }
  if (rows.empty()) {
    return {};
  }

  if (previous == nullptr) {
    return absl::StrFormat("\n%s\n  %*s %-70s %s\n%s",
                           title,
                           kPardisoIparmColumnWidth,
                           kPardisoIparmHeader,
                           "Parameter",
                           "Value",
                           rows);
  }
  return absl::StrFormat("\n%s\n  %*s %-70s %-30s %s\n%s",
                         title,
                         kPardisoIparmColumnWidth,
                         kPardisoIparmHeader,
                         "Parameter",
                         "Previous",
                         "Current",
                         rows);
}

}  // namespace ceres::internal

#endif  // CERES_NO_MKL
