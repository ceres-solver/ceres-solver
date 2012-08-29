// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
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
// Author: strandmark@google.com (Petter Strandmark)
//
// A simple C++ interface to CXSparse, enabling solving linear system with
// caching of the symbolic Cholesky factorization.

#ifndef CERES_NO_CXSPARSE
#include "ceres/cxsparse.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

CXSparse::CXSparse() : scratch_size_(0), scratch_(NULL) {
}

CXSparse::~CXSparse() {
  if (scratch_size_ > 0) {
    cs_free(scratch_);
  }
}

bool CXSparse::SolveCholesky(cs_di* A, css* factor, double* b) {
  // Make sure we have enough scratch space available.
  if (scratch_size_ < A->n) {
    if (scratch_size_ > 0) {
      cs_free(scratch_);
    }
    scratch_ = reinterpret_cast<CS_ENTRY*>(cs_malloc(A->n, sizeof(CS_ENTRY)));
  }

  // Solve using Cholesky factorization
  csn* N = cs_chol(A, factor);
  if (N == NULL) {
    LOG(WARNING) << "Cholesky factorization failed.";
    return false;
  }

  // When the Cholesky factorization succeeded, these methods are guaranteed to
  // succeed as well. In the comments below, "x" refers to the scratch space.
  // Set x = P * b.
  cs_ipvec(factor->pinv, b, scratch_, A->n);
  // Set x = L \ x.
  cs_lsolve(N->L, scratch_);
  // Set x = L' \ x.
  cs_ltsolve(N->L, scratch_);
  // Set b = P' * b.
  cs_pvec(factor->pinv, scratch_, b, A->n);

  // Free Cholesky factorization.
  cs_nfree(N);
  return true;
}

css* CXSparse::AnalyzeCholesky(CS_INT order, cs_di* A) {
  return cs_schol(order, A);
}

void CXSparse::Free(css* factor) {
  cs_sfree(factor);
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_NO_CXSPARSE
