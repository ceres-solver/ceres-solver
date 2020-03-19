// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2020 Google Inc. All rights reserved.
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
// Author: darius.rueckert@fau.de (Darius Rueckert)
//
//
#ifndef CERES_INTERNAL_AUTODIFF_BENCHMARKS_LINEAR_COST_FUNCTIONS_H_
#define CERES_INTERNAL_AUTODIFF_BENCHMARKS_LINEAR_COST_FUNCTIONS_H_

#include "ceres/codegen/codegen_cost_function.h"
#include "ceres/rotation.h"

namespace ceres {

struct Linear1CostFunction : public ceres::CodegenCostFunction<1, 1> {
  template <typename T>
  bool operator()(const T* const x, T* residuals) const {
    residuals[0] = x[0] + T(10);
    return true;
  }
#ifdef WITH_CODE_GENERATION
#include "benchmarks/linear1costfunction.h"
#else
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    return false;
  }
#endif
};

struct Linear10CostFunction : public ceres::CodegenCostFunction<10, 10> {
  template <typename T>
  bool operator()(const T* const x, T* residuals) const {
    for (int i = 0; i < 10; ++i) {
      residuals[i] = x[i] + T(i);
    }
    return true;
  }
#ifdef WITH_CODE_GENERATION
#include "benchmarks/linear10costfunction.h"
#else
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    return false;
  }
#endif
};
}  // namespace ceres

#endif  // CERES_INTERNAL_AUTODIFF_BENCHMARKS_LINEAR_COST_FUNCTIONS_H_
