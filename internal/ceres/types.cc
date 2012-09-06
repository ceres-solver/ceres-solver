// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
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

#include <algorithm>
#include <string>
#include "ceres/types.h"
#include "glog/logging.h"

namespace ceres {

#define CASESTR(x) case x: return #x

#define STRENUM(x) if (value == #x) { *type = x; return true;}

void UpperCase(string* input) {
  std::transform(input->begin(), input->end(), input->begin(), ::toupper);
}

const char* LinearSolverTypeToString(LinearSolverType solver_type) {
  switch (solver_type) {
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

bool StringToLinearSolverType(string value, LinearSolverType* type) {
  UpperCase(&value);
  STRENUM(DENSE_NORMAL_CHOLESKY);
  STRENUM(DENSE_QR);
  STRENUM(SPARSE_NORMAL_CHOLESKY);
  STRENUM(DENSE_SCHUR);
  STRENUM(SPARSE_SCHUR);
  STRENUM(ITERATIVE_SCHUR);
  STRENUM(CGNR);
  return false;
}

const char* PreconditionerTypeToString(
    PreconditionerType preconditioner_type) {
  switch (preconditioner_type) {
    CASESTR(IDENTITY);
    CASESTR(JACOBI);
    CASESTR(SCHUR_JACOBI);
    CASESTR(CLUSTER_JACOBI);
    CASESTR(CLUSTER_TRIDIAGONAL);
    default:
      return "UNKNOWN";
  }
}

bool StringToPreconditionerType(string value, PreconditionerType* type) {
  UpperCase(&value);
  STRENUM(IDENTITY);
  STRENUM(JACOBI);
  STRENUM(SCHUR_JACOBI);
  STRENUM(CLUSTER_JACOBI);
  STRENUM(CLUSTER_TRIDIAGONAL);
  return false;
}

const char* SparseLinearAlgebraLibraryTypeToString(
    SparseLinearAlgebraLibraryType sparse_linear_algebra_library_type) {
  switch (sparse_linear_algebra_library_type) {
    CASESTR(SUITE_SPARSE);
    CASESTR(CX_SPARSE);
    default:
      return "UNKNOWN";
  }
}


bool StringToSparseLinearAlgebraLibraryType(
    string value,
    SparseLinearAlgebraLibraryType* type) {
  UpperCase(&value);
  STRENUM(SUITE_SPARSE);
  STRENUM(CX_SPARSE);
  return false;
}

const char* OrderingTypeToString(OrderingType ordering_type) {
  switch (ordering_type) {
    CASESTR(NATURAL);
    CASESTR(USER);
    CASESTR(SCHUR);
    default:
      return "UNKNOWN";
  }
}

bool StringToOrderingType(string value, OrderingType* type) {
  UpperCase(&value);
  STRENUM(NATURAL);
  STRENUM(USER);
  STRENUM(SCHUR);
  return false;
}

const char* TrustRegionStrategyTypeToString(
    TrustRegionStrategyType trust_region_strategy_type) {
  switch (trust_region_strategy_type) {
    CASESTR(LEVENBERG_MARQUARDT);
    CASESTR(DOGLEG);
    default:
      return "UNKNOWN";
  }
}

bool StringToTrustRegionStrategyType(string value,
                                     TrustRegionStrategyType* type) {
  UpperCase(&value);
  STRENUM(LEVENBERG_MARQUARDT);
  STRENUM(DOGLEG);
  return false;
}

const char* DoglegTypeToString(DoglegType dogleg_type) {
  switch (dogleg_type) {
    CASESTR(TRADITIONAL_DOGLEG);
    CASESTR(SUBSPACE_DOGLEG);
    default:
      return "UNKNOWN";
  }
}

bool StringToDoglegType(string value, DoglegType* type) {
  UpperCase(&value);
  STRENUM(TRADITIONAL_DOGLEG);
  STRENUM(SUBSPACE_DOGLEG);
  return false;
}

const char* SolverTerminationTypeToString(
    SolverTerminationType termination_type) {
  switch (termination_type) {
    CASESTR(NO_CONVERGENCE);
    CASESTR(FUNCTION_TOLERANCE);
    CASESTR(GRADIENT_TOLERANCE);
    CASESTR(PARAMETER_TOLERANCE);
    CASESTR(NUMERICAL_FAILURE);
    CASESTR(USER_ABORT);
    CASESTR(USER_SUCCESS);
    CASESTR(DID_NOT_RUN);
    default:
      return "UNKNOWN";
  }
}

const char* LinearSolverTerminationTypeToString(
    LinearSolverTerminationType termination_type) {
  switch (termination_type) {
    CASESTR(TOLERANCE);
    CASESTR(MAX_ITERATIONS);
    CASESTR(STAGNATION);
    CASESTR(FAILURE);
    default:
      return "UNKNOWN";
  }
}

#undef CASESTR
#undef STRENUM

bool IsSchurType(LinearSolverType type) {
  return ((type == SPARSE_SCHUR) ||
          (type == DENSE_SCHUR)  ||
          (type == ITERATIVE_SCHUR));
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

  if (type == CX_SPARSE) {
#ifdef CERES_NO_CXSPARSE
    return false;
#else
    return true;
#endif
  }

  LOG(WARNING) << "Unknown sparse linear algebra library " << type;
  return false;
}

}  // namespace ceres
