// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2019 Google Inc. All rights reserved.
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
// Author: darius.rueckert@fau.de (Darius Rueckert)
//
#ifndef CERES_PUBLIC_CODEGEN_COST_FUNCTION_H_
#define CERES_PUBLIC_CODEGEN_COST_FUNCTION_H_

#include "ceres/codegen/macros.h"
#include "ceres/sized_cost_function.h"

namespace ceres {

// This is the interface for automatically generating cost functor derivative
// code. The template parameters are identical to those of SizedCostFunction<>.
// The behaviour of this class changes between code generation and problem
// solving.
//
// During code generation (when CERES_CODEGEN is defined), this class doesn't do
// anything. The templated operator() from your derived cost functor is used.
//
// After code generation (when CERES_CODEGEN is not defined), this class is a
// SizedCostFunction. The required Evaluate() function was generated and must be
// included into your cost functor.
//
// Usage Example:
//
//   #include "ceres/codegen/cost_function.h"
//
//   struct HelloWorldCostFunction : public ceres::CodegenCostFunction<1, 1> {
//     // A default constructor is required, because the code generator has to
//     // create an object of the cost function.
//     HelloWorldCostFunction() = default;
//     template <typename T>
//     bool operator()(const T* x, T* residual) const {
//       residual[0] = x[0] * x[0];
//       return true;
//     }
//   #include "examples/helloworldcostfunction.h"
//   };

template <int kNumResiduals_, int... Ns>
class CodegenCostFunction
#ifdef CERES_CODEGEN
// During code generation we can't derive from SizedCostFunction.
// The variadic evaluation with Jets would try to call the empty Evaluate()
// instead of the templated functor.
#else
    : public SizedCostFunction<kNumResiduals_, Ns...>
#endif
{
 public:
  static constexpr int kNumResiduals = kNumResiduals_;
  static_assert(kNumResiduals > 0,
                "Cost functions must have at least one residual block.");
  static_assert(kNumResiduals != DYNAMIC,
                "Code generation for dynamic residuals is not yet supported.");
  static_assert(internal::StaticParameterDims<Ns...>::kIsValid,
                "Invalid parameter block dimension detected. Each parameter "
                "block dimension must be bigger than zero.");
  using ParameterDims = internal::StaticParameterDims<Ns...>;
};

}  // namespace ceres
#endif  // CERES_PUBLIC_CODEGEN_COST_FUNCTION_H_
