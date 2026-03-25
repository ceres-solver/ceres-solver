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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/types.h"

#include <algorithm>
#include <cctype>
#include <string>

#include "absl/log/log.h"
#include "ceres/internal/config.h"

namespace ceres {

// clang-format off
#define CASESTR(x) case x: return #x
#define STRENUM(x) if (value == #x) { *type = x; return true; }
// clang-format on

static void UpperCase(std::string* input) {
  std::transform(input->begin(), input->end(), input->begin(), ::toupper);
}

const char* LinearSolverTypeToString(LinearSolverType type) {
  switch (type) {
    CASESTR(DENSE_NORMAL_CHOLESKY);
    CASESTR(DENSE_QR);
    CASESTR(SPARSE_NORMAL_CHOLESKY);
    CASESTR(DENSE_SCHUR);
    CASESTR(SPARSE_SCHUR);
    CASESTR(ITERATIVE_SCHUR);
    CASESTR(CGNR);
    default:
      return "UNKNOWN";
  }
}

bool StringToLinearSolverType(const std::string& value,
                              LinearSolverType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(DENSE_NORMAL_CHOLESKY);
  STRENUM(DENSE_QR);
  STRENUM(SPARSE_NORMAL_CHOLESKY);
  STRENUM(DENSE_SCHUR);
  STRENUM(SPARSE_SCHUR);
  STRENUM(ITERATIVE_SCHUR);
  STRENUM(CGNR);
#undef value
  return false;
}

const char* PreconditionerTypeToString(PreconditionerType type) {
  switch (type) {
    CASESTR(IDENTITY);
    CASESTR(JACOBI);
    CASESTR(SCHUR_JACOBI);
    CASESTR(SCHUR_POWER_SERIES_EXPANSION);
    CASESTR(CLUSTER_JACOBI);
    CASESTR(CLUSTER_TRIDIAGONAL);
    CASESTR(SUBSET);
    default:
      return "UNKNOWN";
  }
}

bool StringToPreconditionerType(const std::string& value,
                                PreconditionerType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(IDENTITY);
  STRENUM(JACOBI);
  STRENUM(SCHUR_JACOBI);
  STRENUM(SCHUR_POWER_SERIES_EXPANSION);
  STRENUM(CLUSTER_JACOBI);
  STRENUM(CLUSTER_TRIDIAGONAL);
  STRENUM(SUBSET);
#undef value
  return false;
}

const char* SparseLinearAlgebraLibraryTypeToString(
    SparseLinearAlgebraLibraryType type) {
  switch (type) {
    CASESTR(SUITE_SPARSE);
    CASESTR(EIGEN_SPARSE);
    CASESTR(ACCELERATE_SPARSE);
    CASESTR(CUDA_SPARSE);
    CASESTR(NO_SPARSE);
    default:
      return "UNKNOWN";
  }
}

bool StringToSparseLinearAlgebraLibraryType(
    const std::string& value, SparseLinearAlgebraLibraryType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(SUITE_SPARSE);
  STRENUM(EIGEN_SPARSE);
  STRENUM(ACCELERATE_SPARSE);
  STRENUM(CUDA_SPARSE);
  STRENUM(NO_SPARSE);
#undef value
  return false;
}

const char* LinearSolverOrderingTypeToString(LinearSolverOrderingType type) {
  switch (type) {
    CASESTR(AMD);
    CASESTR(NESDIS);
    default:
      return "UNKNOWN";
  }
}

bool StringToLinearSolverOrderingType(const std::string& value,
                                      LinearSolverOrderingType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(AMD);
  STRENUM(NESDIS);
#undef value
  return false;
}

const char* DenseLinearAlgebraLibraryTypeToString(
    DenseLinearAlgebraLibraryType type) {
  switch (type) {
    CASESTR(EIGEN);
    CASESTR(LAPACK);
    CASESTR(CUDA);
    default:
      return "UNKNOWN";
  }
}

bool StringToDenseLinearAlgebraLibraryType(
    const std::string& value, DenseLinearAlgebraLibraryType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(EIGEN);
  STRENUM(LAPACK);
  STRENUM(CUDA);
#undef value
  return false;
}

const char* TrustRegionStrategyTypeToString(TrustRegionStrategyType type) {
  switch (type) {
    CASESTR(LEVENBERG_MARQUARDT);
    CASESTR(DOGLEG);
    default:
      return "UNKNOWN";
  }
}

bool StringToTrustRegionStrategyType(const std::string& value,
                                     TrustRegionStrategyType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(LEVENBERG_MARQUARDT);
  STRENUM(DOGLEG);
#undef value
  return false;
}

const char* DoglegTypeToString(DoglegType type) {
  switch (type) {
    CASESTR(TRADITIONAL_DOGLEG);
    CASESTR(SUBSPACE_DOGLEG);
    default:
      return "UNKNOWN";
  }
}

bool StringToDoglegType(const std::string& value, DoglegType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(TRADITIONAL_DOGLEG);
  STRENUM(SUBSPACE_DOGLEG);
#undef value
  return false;
}

const char* MinimizerTypeToString(MinimizerType type) {
  switch (type) {
    CASESTR(TRUST_REGION);
    CASESTR(LINE_SEARCH);
    default:
      return "UNKNOWN";
  }
}

bool StringToMinimizerType(const std::string& value, MinimizerType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(TRUST_REGION);
  STRENUM(LINE_SEARCH);
#undef value
  return false;
}

const char* LineSearchDirectionTypeToString(LineSearchDirectionType type) {
  switch (type) {
    CASESTR(STEEPEST_DESCENT);
    CASESTR(NONLINEAR_CONJUGATE_GRADIENT);
    CASESTR(LBFGS);
    CASESTR(BFGS);
    default:
      return "UNKNOWN";
  }
}

bool StringToLineSearchDirectionType(const std::string& value,
                                     LineSearchDirectionType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(STEEPEST_DESCENT);
  STRENUM(NONLINEAR_CONJUGATE_GRADIENT);
  STRENUM(LBFGS);
  STRENUM(BFGS);
#undef value
  return false;
}

const char* LineSearchTypeToString(LineSearchType type) {
  switch (type) {
    CASESTR(ARMIJO);
    CASESTR(WOLFE);
    default:
      return "UNKNOWN";
  }
}

bool StringToLineSearchType(const std::string& value, LineSearchType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(ARMIJO);
  STRENUM(WOLFE);
#undef value
  return false;
}

const char* LineSearchInterpolationTypeToString(
    LineSearchInterpolationType type) {
  switch (type) {
    CASESTR(BISECTION);
    CASESTR(QUADRATIC);
    CASESTR(CUBIC);
    default:
      return "UNKNOWN";
  }
}

bool StringToLineSearchInterpolationType(const std::string& value,
                                         LineSearchInterpolationType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(BISECTION);
  STRENUM(QUADRATIC);
  STRENUM(CUBIC);
#undef value
  return false;
}

const char* NonlinearConjugateGradientTypeToString(
    NonlinearConjugateGradientType type) {
  switch (type) {
    CASESTR(FLETCHER_REEVES);
    CASESTR(POLAK_RIBIERE);
    CASESTR(HESTENES_STIEFEL);
    default:
      return "UNKNOWN";
  }
}

bool StringToNonlinearConjugateGradientType(
    const std::string& value, NonlinearConjugateGradientType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(FLETCHER_REEVES);
  STRENUM(POLAK_RIBIERE);
  STRENUM(HESTENES_STIEFEL);
#undef value
  return false;
}

const char* CovarianceAlgorithmTypeToString(CovarianceAlgorithmType type) {
  switch (type) {
    CASESTR(DENSE_SVD);
    CASESTR(SPARSE_QR);
    default:
      return "UNKNOWN";
  }
}

bool StringToCovarianceAlgorithmType(const std::string& value,
                                     CovarianceAlgorithmType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(DENSE_SVD);
  STRENUM(SPARSE_QR);
#undef value
  return false;
}

const char* NumericDiffMethodTypeToString(NumericDiffMethodType type) {
  switch (type) {
    CASESTR(CENTRAL);
    CASESTR(FORWARD);
    CASESTR(RIDDERS);
    default:
      return "UNKNOWN";
  }
}

bool StringToNumericDiffMethodType(const std::string& value,
                                   NumericDiffMethodType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(CENTRAL);
  STRENUM(FORWARD);
  STRENUM(RIDDERS);
#undef value
  return false;
}

const char* VisibilityClusteringTypeToString(VisibilityClusteringType type) {
  switch (type) {
    CASESTR(CANONICAL_VIEWS);
    CASESTR(SINGLE_LINKAGE);
    default:
      return "UNKNOWN";
  }
}

bool StringToVisibilityClusteringType(const std::string& value,
                                      VisibilityClusteringType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(CANONICAL_VIEWS);
  STRENUM(SINGLE_LINKAGE);
#undef value
  return false;
}

const char* TerminationTypeToString(TerminationType type) {
  switch (type) {
    CASESTR(CONVERGENCE);
    CASESTR(NO_CONVERGENCE);
    CASESTR(FAILURE);
    CASESTR(USER_SUCCESS);
    CASESTR(USER_FAILURE);
    default:
      return "UNKNOWN";
  }
}

const char* LoggingTypeToString(LoggingType type) {
  switch (type) {
    CASESTR(SILENT);
    CASESTR(PER_MINIMIZER_ITERATION);
    default:
      return "UNKNOWN";
  }
}

bool StringToLoggingType(const std::string& value, LoggingType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(SILENT);
  STRENUM(PER_MINIMIZER_ITERATION);
#undef value
  return false;
}

const char* DumpFormatTypeToString(DumpFormatType type) {
  switch (type) {
    CASESTR(CONSOLE);
    CASESTR(TEXTFILE);
    default:
      return "UNKNOWN";
  }
}

bool StringToDumpFormatType(const std::string& value, DumpFormatType* type) {
  std::string value_upper = value;
  UpperCase(&value_upper);
#define value value_upper
  STRENUM(CONSOLE);
  STRENUM(TEXTFILE);
#undef value
  return false;
}

#undef CASESTR
#undef STRENUM

bool IsSchurType(LinearSolverType type) {
  // clang-format off
  return ((type == SPARSE_SCHUR) ||
          (type == DENSE_SCHUR)  ||
          (type == ITERATIVE_SCHUR));
  // clang-format on
}

bool IsSparseLinearAlgebraLibraryTypeAvailable(
    SparseLinearAlgebraLibraryType type) {
  if (type == SUITE_SPARSE) {
#ifdef CERES_NO_SUITESPARSE
    return false;
#else
    return true;
#endif
  }

  if (type == ACCELERATE_SPARSE) {
#ifdef CERES_NO_ACCELERATE_SPARSE
    return false;
#else
    return true;
#endif
  }

  if (type == EIGEN_SPARSE) {
#ifdef CERES_USE_EIGEN_SPARSE
    return true;
#else
    return false;
#endif
  }

  if (type == CUDA_SPARSE) {
#ifdef CERES_NO_CUDA
    return false;
#else
    return true;
#endif
  }

  if (type == NO_SPARSE) {
    return true;
  }

  LOG(WARNING) << "Unknown sparse linear algebra library " << type;
  return false;
}

bool IsDenseLinearAlgebraLibraryTypeAvailable(
    DenseLinearAlgebraLibraryType type) {
  if (type == EIGEN) {
    return true;
  }

  if (type == LAPACK) {
#ifdef CERES_NO_LAPACK
    return false;
#else
    return true;
#endif
  }

  if (type == CUDA) {
#ifdef CERES_NO_CUDA
    return false;
#else
    return true;
#endif
  }

  LOG(WARNING) << "Unknown dense linear algebra library " << type;
  return false;
}

}  // namespace ceres
