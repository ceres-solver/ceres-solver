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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_INTERNAL_AUGMENTED_LAGRANGIAN_COST_FUNCTION_H_
#define CERES_INTERNAL_AUGMENTED_LAGRANGIAN_COST_FUNCTION_H_

#include "ceres/cost_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"

namespace ceres {
namespace experimental {

class AugmentedLagrangianCostFunction : public CostFunction {
 public:
  AugmentedLagrangianCostFunction(CostFunction* constraint)
      : constraint_(constraint) {
    set_num_residuals(constraint_->num_residuals());
    lambda_.resize(num_residuals());
    lambda_.setZero();
    mu_ = 1.0;
    *mutable_parameter_block_sizes() = constraint->parameter_block_sizes();
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    if (!constraint_->Evaluate(parameters, residuals, jacobians)) {
      return false;
    }

    const double scale = 1.0 / sqrt(2 * mu_);
    VectorRef residuals_ref(residuals, num_residuals());
    residuals_ref = residuals_ref  * scale + lambda_ * sqrt(mu_ / 2.0);
    if (jacobians == NULL) {
      return true;
    }

    const vector<int32>& block_sizes = parameter_block_sizes();
    const int num_parameter_blocks = block_sizes.size();
    for (int i = 0; i < num_parameter_blocks; ++i) {
      if (jacobians[i] != NULL) {
        MatrixRef(jacobians[i], num_residuals(), block_sizes[i]) *= scale;
      }
    }

    return true;
  }

  bool UpdateLambda(const vector<double*>& parameter_blocks) {
    Vector scratch(num_residuals());
    if (!constraint_->Evaluate(&parameter_blocks[0], scratch.data(), NULL)) {
      return false;
    }

    lambda_ += scratch / mu_;
    return true;
  }

  Vector* mutable_lambda() { return &lambda_; }
  const Vector& lambda() const { return lambda_; }
  double mu() const { return mu_; }
  void set_mu(double mu) { mu_ = mu; }
  CostFunction* constraint() const { return constraint_.get(); }

 private:
  ::ceres::internal::scoped_ptr<CostFunction> constraint_;
  Vector lambda_;
  double mu_;
};

}  // namespace experimental
}  // namespace ceres

#endif  // CERES_INTERNAL_AUGMENTED_LAGRANGIAN_COST_FUNCTION_H_
