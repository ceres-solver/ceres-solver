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

#include "ceres/weighted_cost_function.h"

#include <algorithm>
#include <numeric>

#include "ceres/cost_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/sized_cost_function.h"
#include "glog/logging.h"

namespace ceres {

WeightedCostFunction::WeightedCostFunction(
    const double* weight_matrix,
    const int num_rows,
    const int num_cols,
    CostFunction* wrapped_cost_function) {
  CHECK_EQ(num_cols, wrapped_cost_function->num_residuals())
      << "The weight matrix must have the same number "
      << "of columns as the number of residuals returned "
      << "by the wrapped_cost_function.";

  wrapped_cost_function_.reset(wrapped_cost_function);
  const vector<int16>& parameter_block_sizes =
      wrapped_cost_function_->parameter_block_sizes();

  set_num_residuals(num_rows);
  *mutable_parameter_block_sizes() = parameter_block_sizes;

  const int num_elements = num_rows * num_cols;
  weight_matrix_.reset(new double[num_elements]);
  std::copy(weight_matrix, weight_matrix + num_elements, weight_matrix_.get());

  residuals_.reset(new double[num_cols]);
  const int num_parameters = std::accumulate(parameter_block_sizes.begin(),
                                             parameter_block_sizes.end(),
                                             0);
  jacobian_values_.reset(new double[num_parameters * num_cols]);
  jacobians_.reset(new double*[parameter_block_sizes.size()]);
}

bool WeightedCostFunction::Evaluate(double const* const* parameters,
                                    double* residuals,
                                    double** jacobians) const {
  const int num_rows = num_residuals();
  const int num_cols =  wrapped_cost_function_->num_residuals();
  ConstMatrixRef w(weight_matrix_.get(), num_rows, num_cols);
  const vector<int16>& parameter_block_sizes = this->parameter_block_sizes();

  if (jacobians != NULL) {
    int cursor = 0;
    for (int i = 0; i < parameter_block_sizes.size(); ++i) {
      if (jacobians[i] == NULL) {
        jacobians_[i] = NULL;
      } else {
        jacobians_[i] = jacobian_values_.get() + cursor;
        cursor += parameter_block_sizes[i] * num_cols;
      }
    }
  }

  if (!wrapped_cost_function_->Evaluate(parameters,
                                        residuals_.get(),
                                        jacobians_.get())) {
    return false;
  }

  VectorRef(residuals, num_rows) =
      w * ConstVectorRef(residuals_.get(), num_cols);
  if (jacobians != NULL) {
    for (int i = 0; i < parameter_block_sizes.size(); ++i) {
      if (jacobians[i] == NULL) {
        continue;
      }

      MatrixRef(jacobians[i], num_rows, parameter_block_sizes[i]) =
          w * ConstMatrixRef(jacobians_[i], num_cols, parameter_block_sizes[i]);
    }
  }

  return true;
}

}  // namespace ceres
