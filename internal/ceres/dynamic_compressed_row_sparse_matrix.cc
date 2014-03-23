// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2014 Google Inc. All rights reserved.
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
// Author: richie.stebbing@gmail.com (Richard Stebbing)

#include "ceres/dynamic_compressed_row_sparse_matrix.h"

namespace ceres {
namespace internal {

void DynamicCompressedRowSparseMatrix::Finalize(int num_additional) {
  // `num_additional` is provided as an argument so that additional
  // storage can be reserved when it is known by the finalizer.

  // Count the number of non-zeros and resize `cols_` and `values_`.
  int num_jacobian_nonzeros = 0;
  for (auto& cols : dynamic_cols_) {
    num_jacobian_nonzeros += (int)cols.size();
  }
  cols_.resize(num_jacobian_nonzeros + num_additional);
  values_.resize(num_jacobian_nonzeros + num_additional);

  // Flatten `dynamic_cols_` into `cols_` and `dynamic_values_`
  // into `values_`.
  int l = 0;
  for (int i = 0; i < num_rows_; ++i) {
    rows_[i] = l;
    std::copy(dynamic_cols_[i].begin(), dynamic_cols_[i].end(),
              &cols_[0] + l);
    std::copy(dynamic_values_[i].begin(), dynamic_values_[i].end(),
              &values_[0] + l);
    l += (int)dynamic_cols_[i].size();
  }
  rows_[num_rows_] = l;

  CHECK_EQ(l, num_jacobian_nonzeros);
}

}  // namespace internal
}  // namespace ceres
