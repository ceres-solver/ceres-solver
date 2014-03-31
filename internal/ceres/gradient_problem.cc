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

#include "ceres/gradient_problem.h"
#include "glog/logging.h"

namespace ceres {

GradientProblem::GradientProblem(FirstOrderFunction* function)
    : function_(function),
      parameterization_(NULL),
      scratch_(new double[function_->NumParameters()]) {
}

GradientProblem::GradientProblem(FirstOrderFunction* function,
                                 LocalParameterization* parameterization)
      : function_(function),
        parameterization_(parameterization),
        scratch_(new double[function_->NumParameters()]) {
  CHECK_EQ(function_->NumParameters(), parameterization_->GlobalSize());
}

int GradientProblem::NumParameters() const {
  return function_->NumParameters();
}

int GradientProblem::NumLocalParameters() const {
    return (parameterization_.get() == NULL)
        ? NumParameters()
        : parameterization_->LocalSize();
}


bool GradientProblem::Evaluate(const double* parameters,
                               double* cost,
                               double* gradient) const {
  if (gradient == NULL || parameterization_.get() == NULL) {
    return function_->Evaluate(parameters, cost, gradient);
  }

  return (function_->Evaluate(parameters, cost, scratch_.get()) &&
          parameterization_->MultiplyByJacobian(parameters,
                                                1,
                                                scratch_.get(),
                                                gradient));
}

bool GradientProblem::Plus(const double* x,
                           const double* delta,
                           double* x_plus_delta) const {
  if (parameterization_.get() != NULL) {
    return parameterization_->Plus(x, delta, x_plus_delta);
  }

  for (int i = 0; i < NumParameters(); ++i) {
    x_plus_delta[i] = x[i] + delta[i];
  }
  return true;
}

}  // namespace ceres
