// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2013 Google Inc. All rights reserved.
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

#ifndef CERES_INTERNAL_COVARIANCE_IMPL_H_
#define CERES_INTERNAL_COVARIANCE_IMPL_H_

#include <utility>
#include <map>
#include <vector>

#include "ceres/covariance.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/suitesparse.h"
#include "ceres/problem_impl.h"

namespace ceres {

namespace internal {

class CompressedRowSparseMatrix;

class CovarianceImpl {
 public:
  explicit CovarianceImpl(const Covariance::Options& options);
  ~CovarianceImpl();

  bool Compute(const vector<pair<double*, double*> >& covariance_blocks,
                       ProblemImpl* problem);

  bool GetCovarianceBlock(double* parameter_block1,
                          double* parameter_block2,
                          double* covariance_block);

  bool ComputeCovarianceSparsity(
      const vector<pair<double*, double*> >& covariance_blocks,
      ProblemImpl* problem);

  bool ComputeCovarianceValues();

  const CompressedRowSparseMatrix* covariance_matrix() const {
    return covariance_matrix_.get();
  }

 private:
  //const Covariance::Options& options_;
  ProblemImpl* problem_;
  Problem::EvaluateOptions evaluate_options_;
  map<double*, int> parameter_block_to_row_index_;
  scoped_ptr<CompressedRowSparseMatrix> covariance_matrix_;
  SuiteSparse ss_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_COVARIANCE_IMPL_H_
