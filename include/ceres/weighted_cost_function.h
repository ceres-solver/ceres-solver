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

#ifndef CERES_PUBLIC_WEIGHTED_COST_FUNCTION_H_
#define CERES_PUBLIC_WEIGHTED_COST_FUNCTION_H_

#include "ceres/cost_function.h"
#include "ceres/internal/scoped_ptr.h"

namespace ceres {

// A cost function whose residuals are obtained by multiplying the
// residuals of another cost function (wrapped_cost_function) by a
// weight matrix, i.e,
//
//   weighted_residuals = weight_matrix * residuals
//
// The most common usage of this class is to add covariance weighting
// to a cost function, e.g to get a cost function with cost
//
//   cost = 1/2 r(x)'S^{-1}r(x)
//
// Let FooCostFunction be a CostFunction that implements the
// computation of r(x) and let
//
//   weight_matrix = S^{-1/2}
//
// then
//
//   CostFunction* cost_function = new FooCostFunction(...);
//   WeightedCostFunction weighted_cost_function(weight_matrix,
//                                               num_rows,
//                                               num_cols,
//                                               cost_function);
//
class WeightedCostFunction: public CostFunction {
 public:
  // The weight_matrix is a row-major matrix.
  //
  // num_rows is the number of residuals returned by
  // WeightedCostFunction.
  //
  // num_cols must be equal to the number of residuals returned by
  // wrapped_cost_function.
  //
  // WeightedCostFunction takes ownership of the
  // wrapped_cost_function.
  WeightedCostFunction(const double* weight_matrix,
                       const int num_rows,
                       const int num_cols,
                       CostFunction* wrapped_cost_function);

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const;

 private:
  internal::scoped_ptr<CostFunction> wrapped_cost_function_;
  internal::scoped_array<double> weight_matrix_;
  internal::scoped_array<double> residuals_;
  internal::scoped_array<double> jacobian_values_;
  internal::scoped_array<double*> jacobians_;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_WEIGHTED_COST_FUNCTION_H_
