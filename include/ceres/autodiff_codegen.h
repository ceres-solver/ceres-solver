
// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
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
// AutoDiffCodeGen is able to generate C++ code of the _Evaluate_ function from
// a given cost functor. This includes refinded code for the residuals as well
// as code for all partial derivatives. Usage example:
//
//
// struct CostFunctorSimple {
//   template <typename T>
//   bool operator()(const T* const x, T* residual) const {
//     residual[0] = x[0] * x[0];
//     return true;
//   }
// };
// ....
//  ceres::AutoDiffCodeGen<CostFunctorSimple, 1, 1> codeGen(
//      new CostFunctorSimple());
//  codeGen.Generate();
//
//
// Console output:
// bool Evaluate(double const* const* parameters, double* residuals, double**
// jacobians)
// {
//   // This code is generated with ceres::AutoDiffCodeGen
//   // See ceres/autodiff_codegen.h for more informations.
//   const double v_4 = parameters[0][0];
//   const double v_8 = v_4 * v_4;
//   const double v_11 = v_4 + v_4;
//   residuals[0] = v_8;
//   jacobians[0][0] = v_11;
//   return true;
// }
//
//
// Overview
//
// autodiff_codegen.h
//    - Creates Jet objects and calls the cost functor (the expression tree is
//      created here)
//    - Adds input-output expressions
//    - Starts code gen and prints the result
// expressions.h
//    - Contains Expression types and additional utilities for
//      creation/modification
//    - Includes a kind of micro-compiler, which traverses the expression tree
//      for code generation and then
//      applies a few simple optimizations.
// expression_jet.h
//    - Similar to ceres::Jet but generates an expression tree instead of
//      evaluating the operations.
//
//
// The basic idea is to create a custom type (here: ExpressionJet) which prints
// all the operations on the console instead of evaluating them. As an example,
// we can implement an operator* which prints itself:
//
// void operator*(ExpressionJet a, ExpressionJet b){
//  std::cout << a.name() << "*" << b.name() << std::endl;
// }
//
// If we now call a templated cost-functor with our ExpressionJet all
// multiplications will be printed to the console. In the same way we can
// generate the derivative code by just printing it in the operator overload.
//
// Unfortunately a direct output to the console has a few disadvantages. First,
// we lose all the meta information and it is hard to add external constraints
// such as the input and output variables. Second, long math expressions will
// create temporary objects, so the .name() function in the pseudo code
// above is not trivial. Third, AutoDiff in general generates lots of trivial
// expressions such as
// a = 0;
// b = a + a;
// , which make the generated code hardly readable.
//
// To solve these issues, we do not directly output the code to the console, but
// rather generate intermediate code in static single assignment (SSA) form.
// SSA-Form: https://en.wikipedia.org/wiki/Static_single_assignment_form
//
// In this intermediate representation, we can add input/output expressions and
// code apply optimization. The optimizations currently include:
//
// - trivial assignment propagation
//      a = b
//      c = a;    ->    c = b;
//
// - zero-one evaluation
//      a = b * 0;    ->    a = 0;
//      a = b + 0;    ->    a = b;
//      ...
//
// - dead code elimination
//      Analyze which expressions contribute to the output variables. Remove
//      everything else

#ifndef CERES_PUBLIC_AutoDiffCodeGen_H_
#define CERES_PUBLIC_AutoDiffCodeGen_H_

#include "ceres/internal/autodiff.h"
#include "ceres/types.h"
#include "expression_jet.h"

#include "ceres/autodiff_cost_function.h"
#include "ceres/sized_cost_function.h"

namespace ceres {
template <typename CostFunctor, int kNumResiduals, int... Ns>
struct AutoDiffCodeGen : public SizedCostFunction<kNumResiduals, Ns...> {
  explicit AutoDiffCodeGen(CostFunctor* functor) : functor_(functor) {}

 public:
  bool Generate() {
    using T = double;
    using ParameterDims =
        typename SizedCostFunction<kNumResiduals, Ns...>::ParameterDims;
    typedef ExpressionJet<T, ParameterDims::kNumParameters> JetT;

    auto num_outputs = SizedCostFunction<kNumResiduals, Ns...>::num_residuals();
    internal::FixedArray<JetT, (256 * 7) / sizeof(JetT)> x(
        ParameterDims::kNumParameters + num_outputs);

    using Parameters = typename ParameterDims::Parameters;

    // These are the positions of the respective jets in the fixed array x.
    std::array<JetT*, ParameterDims::kNumParameterBlocks> unpacked_parameters =
        ParameterDims::GetUnpackedParameters(x.data());
    JetT* output = x.data() + ParameterDims::kNumParameters;

    int totalParamId = 0;
    for (int i = 0; i < ParameterDims::kNumParameterBlocks; ++i) {
      for (int j = 0; j < ParameterDims::GetDim(i); ++j) {
        JetT& J = x[totalParamId];
        J.a = JetT::factory.ParameterExpr(
            i,
            "parameters[" + std::to_string(i) + "][" + std::to_string(j) + "]");
        J.v = JetT::factory
                  .template ConstantExprArray<ParameterDims::kNumParameters>(
                      totalParamId);
        totalParamId++;
      }
    }

    if (!internal::VariadicEvaluate<ParameterDims>(
            *functor_, unpacked_parameters.data(), output)) {
      return false;
    }

    // the (non-dervied) function
    CodeFunction f;
    f.removedUnusedParameters = false;
    f.factory = JetT::factory;

    for (int i = 0; i < num_outputs; ++i) {
      auto& J = output[i];
      auto res = f.factory.OutputAssignExpr(
          J.a, "residuals[" + std::to_string(i) + "]");
      f.targets.push_back(res);
    }

    totalParamId = 0;
    for (int i = 0; i < ParameterDims::kNumParameterBlocks; ++i) {
      for (int j = 0; j < ParameterDims::GetDim(i); ++j) {
        for (int r = 0; r < num_outputs; ++r) {
          auto& J = output[r];
          // all partial derivatives
          auto res = f.factory.OutputAssignExpr(
              J.v[totalParamId],
              "jacobians[" + std::to_string(i) + "][" +
                  std::to_string(r * ParameterDims::GetDim(i) + j) + "]");
          f.targets.push_back(res);
        }
        totalParamId++;
      }
    }
    f.generate();
    f.print();
    return true;
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    return false;
  }

  std::unique_ptr<CostFunctor> functor_;
};

}  // namespace ceres
#endif
