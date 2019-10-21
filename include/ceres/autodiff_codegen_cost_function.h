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
// To use the Ceres AutoDiffCodeGen functionality, your cost functors must
//    (1) derive from AutoDiffCodeGenCostFunction
//    (2) have a default constructor
//    (3) provide a templated operator()
//
// Example:
//
//   struct SquareFunctor : public AutoDiffCodeGenCostFunction<1, 1> {
//     template <typename T>
//     bool operator()(const T* x, T* residual) const {
//       residual[0] = x[0] * x[0];
//       return true;
//     }
//   };
//
// This is the only public C++ interface of the AutoDiffCodeGen system. All
// other methods are located in ceres::internal. The generation itself is
// initiated by the build system (CMake).
//
// Let's assume, SquareFunctor is in the file square_functor.h. To generate the
// derivative code, the following line has to be added to your CMakeLists.txt:
//
// ceres_autodiff_codegen(
//      square_functor.h,      # Input file
//      SquareFunctor,         # Name of the cost functor
//      square_functor_gen.h,  # Output file
//      gen::SquareFunctor     # Output CMake target
// )
//
// Now make sure to add the new CMake target to your executable/library. This
// makes sure that your target can find the generated file and it is
// automatically regenerated if the input changes.
//
// add_dependencies(MyExecutable gen::SquareFunctor)

#ifndef CERES_PUBLIC_AUTODIFF_CODEGEN_COST_FUNCTION_H_
#define CERES_PUBLIC_AUTODIFF_CODEGEN_COST_FUNCTION_H_

#include "internal/parameter_dims.h"

namespace ceres {

template <int kNumResiduals, int... Ns>
class AutoDiffCodeGenCostFunction {
 public:
  static_assert(kNumResiduals > 0,
                "Cost functions must have at least one residual block. A "
                "dynamic number of residuals is not supported yet.");
  static_assert(internal::StaticParameterDims<Ns...>::kIsValid,
                "Invalid parameter block dimension detected. Each parameter "
                "block dimension must be bigger than zero.");

  using ParameterDims = internal::StaticParameterDims<Ns...>;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_AUTODIFF_CODEGEN_COST_FUNCTION_H_
