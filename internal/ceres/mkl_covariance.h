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

#ifndef CERES_INTERNAL_MKL_COVARIANCE_H_
#define CERES_INTERNAL_MKL_COVARIANCE_H_

#include "ceres/internal/config.h"

#ifndef CERES_NO_MKL

#include <string>

#include "ceres/covariance.h"
#include "ceres/crs_matrix.h"
#include "ceres/internal/export.h"

namespace ceres::internal {

class CompressedRowSparseMatrix;
class ContextImpl;

// Computes the entries of covariance that its sparsity pattern selects from
// [J'J]^-1 using oneMKL Sparse QR.
//
// oneMKL Sparse QR exposes neither R nor transposed solves, so the result is
// accumulated as J+ (J+)' from one least squares solve per Jacobian row. The
// solutions are gathered in blocks so the accumulation can be spread over
// threads. Rank deficiency is reported through the return value because the
// backend offers no rank query of its own.
//
// This declaration deliberately mentions no MKL type so that callers do not
// have to include the oneMKL headers.
CERES_NO_EXPORT bool ComputeCovarianceUsingMklSparseQR(
    const CRSMatrix& jacobian,
    const Covariance::Options& options,
    ContextImpl* context,
    CompressedRowSparseMatrix* covariance,
    std::string* message);

}  // namespace ceres::internal

#endif  // CERES_NO_MKL

#endif  // CERES_INTERNAL_MKL_COVARIANCE_H_
