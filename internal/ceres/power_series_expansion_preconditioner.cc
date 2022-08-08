// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2022 Google Inc. All rights reserved.
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
// Author: markshachkov@gmail.com (Mark Shachkov)

#include "ceres/power_series_expansion_preconditioner.h"

namespace ceres::internal {

PowerSeriesExpansionPreconditioner::PowerSeriesExpansionPreconditioner(
    const ImplicitSchurComplement* s, Preconditioner::Options options)
    : s_(s),
      options_(std::move(options)),
      b_init(num_rows()),
      b_temp(num_rows()),
      b_temp_previous(num_rows()) {}

PowerSeriesExpansionPreconditioner::~PowerSeriesExpansionPreconditioner() =
    default;

bool PowerSeriesExpansionPreconditioner::Update(const LinearOperator& A,
                                                const double* D) {
  return true;
}

void PowerSeriesExpansionPreconditioner::RightMultiply(const double* x,
                                                       double* y) const {
  VectorRef yref(y, num_rows());
  b_init.setZero();
  s_->block_diagonal_FtF_inverse()->RightMultiply(x, b_init.data());
  b_temp_previous = b_init;
  yref = b_init;

  const double norm_threshold = options_.e_tolerance * b_init.norm();

  for (int i = 1;; i++) {
    b_temp.setZero();
    s_->RightMultiply_Z(b_temp_previous.data(), b_temp.data());
    yref += b_temp;

    if (i >= options_.min_num_iterations &&
        (i >= options_.max_num_iterations || b_temp.norm() < norm_threshold))
      break;

    std::swap(b_temp_previous, b_temp);
  }
}

int PowerSeriesExpansionPreconditioner::num_rows() const {
  return s_->num_rows();
}

}  // namespace ceres::internal
