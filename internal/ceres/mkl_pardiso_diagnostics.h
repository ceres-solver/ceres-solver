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

#ifndef CERES_INTERNAL_MKL_PARDISO_DIAGNOSTICS_H_
#define CERES_INTERNAL_MKL_PARDISO_DIAGNOSTICS_H_

#include "ceres/internal/config.h"

#ifndef CERES_NO_MKL

#include <array>
#include <string>
#include <string_view>

#include "ceres/internal/export.h"
#include "ceres/linear_solver.h"
#include "mkl.h"

namespace ceres::internal {

// Integer type of the PARDISO arguments. pardiso uses MKL_INT, which is the
// narrowest type the linked interface supports. 64-bit indices increase the
// memory PARDISO requires. MKL_INT64 selects pardiso_64, which accepts 64-bit
// indices with both the LP64 and the ILP64 interface.
using PardisoIndex = MKL_INT;

CERES_NO_EXPORT bool CheckPardisoStatus(PardisoIndex actual,
                                        std::string_view operation,
                                        std::string* message);

// Returns false and describes the problem in message if the number of
// nonzeros in the Cholesky factor that PARDISO reports after symbolic analysis
// is negative.
CERES_NO_EXPORT bool CheckPardisoFactorNonzeros(PardisoIndex factor_nonzeros,
                                                std::string* message);

CERES_NO_EXPORT std::string_view PardisoErrorToString(PardisoIndex error);
// Numerical breakdowns of the factorization map to FAILURE because the caller
// can recover from them, e.g., by increasing the regularization. Every other
// error maps to FATAL_ERROR.
CERES_NO_EXPORT LinearSolverTerminationType
PardisoErrorToTerminationType(PardisoIndex error);
CERES_NO_EXPORT std::string_view PardisoPhaseToString(PardisoIndex phase);

inline constexpr int kPardisoParameterCount = 64;

// PARDISO phases, matrix types, iparm indices and iparm values shared by the
// solver and its diagnostics. The iparm indices are zero-based as in C. See
// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/pardiso-iparm-parameter.html
// for the iparm parameters.
inline constexpr PardisoIndex kPardisoAnalysis = 11;
inline constexpr PardisoIndex kPardisoNumericalFactorization = 22;
inline constexpr PardisoIndex kPardisoSolve = 33;
inline constexpr PardisoIndex kPardisoRelease = -1;
inline constexpr PardisoIndex kPardisoPositiveDefinite = 2;
inline constexpr int kPardisoUseDefaultValuesParameter = 0;
inline constexpr int kPardisoOrderingParameter = 1;
inline constexpr int kPardisoPreconditionedCgParameter = 3;
inline constexpr int kPardisoUserPermutationParameter = 4;
inline constexpr int kPardisoRefinementStepsPerformedParameter = 6;
inline constexpr int kPardisoIterativeRefinementParameter = 7;
inline constexpr int kPardisoPerturbedPivotsParameter = 13;
inline constexpr int kPardisoPeakSymbolicMemoryParameter = 14;
inline constexpr int kPardisoPermanentSymbolicMemoryParameter = 15;
inline constexpr int kPardisoNumericalMemoryParameter = 16;
inline constexpr int kPardisoFactorNonzerosParameter = 17;
inline constexpr int kPardisoFactorOperationsParameter = 18;
inline constexpr int kPardisoCgDiagnosticsParameter = 19;
inline constexpr int kPardisoParallelFactorizationParameter = 23;
inline constexpr int kPardisoParallelSolveParameter = 24;
inline constexpr int kPardisoMatrixCheckerParameter = 26;
inline constexpr int kPardisoIndexBaseParameter = 34;
inline constexpr PardisoIndex kPardisoUserParameters = 1;
inline constexpr PardisoIndex kPardisoOneBasedIndexing = 0;
inline constexpr PardisoIndex kPardisoZeroBasedIndexing = 1;
inline constexpr PardisoIndex kPardisoEnableMatrixChecker = 1;
inline constexpr PardisoIndex kPardisoDisabled = 0;
inline constexpr PardisoIndex kPardisoMinimumDegreeOrdering = 0;
inline constexpr PardisoIndex kPardisoNestedDissectionOrdering = 2;
inline constexpr PardisoIndex kPardisoParallelNestedDissectionOrdering = 3;
inline constexpr PardisoIndex kPardisoIgnoreUserPermutation = 0;
inline constexpr PardisoIndex kPardisoUseUserPermutation = 1;
inline constexpr PardisoIndex kPardisoReturnPermutation = 2;
inline constexpr PardisoIndex kPardisoClassicFactorization = 0;
inline constexpr PardisoIndex kPardisoTwoLevelFactorization = 1;
inline constexpr PardisoIndex kPardisoSequentialSolve = 1;
inline constexpr PardisoIndex kPardisoParallelSolve = 2;
inline constexpr PardisoIndex kPardisoNoIterativeRefinement = 0;
inline constexpr PardisoIndex kPardisoEnableReport = -1;

struct PardisoIparmEntry {
  int index;
  std::string_view name;
};

// The iparm parameters that verbose logging reports.
inline constexpr std::array kPardisoIparmEntries{
    PardisoIparmEntry{kPardisoUseDefaultValuesParameter, "Use default values"},
    PardisoIparmEntry{kPardisoOrderingParameter, "Fill-in reducing ordering"},
    PardisoIparmEntry{kPardisoPreconditionedCgParameter,
                      "Preconditioned CGS/CG"},
    PardisoIparmEntry{kPardisoUserPermutationParameter, "User permutation"},
    PardisoIparmEntry{kPardisoRefinementStepsPerformedParameter,
                      "Iterative refinement steps performed"},
    PardisoIparmEntry{kPardisoIterativeRefinementParameter,
                      "Iterative refinement steps"},
    PardisoIparmEntry{kPardisoPerturbedPivotsParameter,
                      "Number of perturbed pivots"},
    PardisoIparmEntry{kPardisoPeakSymbolicMemoryParameter,
                      "Peak symbolic memory in KiB"},
    PardisoIparmEntry{kPardisoPermanentSymbolicMemoryParameter,
                      "Permanent symbolic memory in KiB"},
    PardisoIparmEntry{kPardisoNumericalMemoryParameter,
                      "Peak numerical factorization and solve memory in KiB"},
    PardisoIparmEntry{kPardisoFactorNonzerosParameter, "Factor nonzeros"},
    PardisoIparmEntry{kPardisoFactorOperationsParameter,
                      "Factorization operations in millions"},
    PardisoIparmEntry{kPardisoCgDiagnosticsParameter, "CG/CGS diagnostics"},
    PardisoIparmEntry{kPardisoParallelFactorizationParameter,
                      "Parallel factorization control"},
    PardisoIparmEntry{kPardisoParallelSolveParameter,
                      "Parallel forward/backward solve control"},
    PardisoIparmEntry{kPardisoMatrixCheckerParameter, "Matrix checker"},
    PardisoIparmEntry{kPardisoIndexBaseParameter,
                      "CSR row and column index base"}};

// Formats the parameters of kPardisoIparmEntries as a table. If previous is not
// null, only the parameters whose value differs from previous are listed, and
// the result is empty if none differs.
CERES_NO_EXPORT std::string FormatPardisoIparmTable(
    std::string_view title,
    const std::array<PardisoIndex, kPardisoParameterCount>& current,
    const std::array<PardisoIndex, kPardisoParameterCount>* previous);

}  // namespace ceres::internal

#endif  // CERES_NO_MKL

#endif  // CERES_INTERNAL_MKL_PARDISO_DIAGNOSTICS_H_
