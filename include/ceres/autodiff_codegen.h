
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
//  =============================================================================
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
// rather generate intermediate code in static single assignment (SSA) form. In
// SSA-form each variable is assigned exactly once. Here an example:
//
// const double v_63 = cos(v_24);
// const double v_71 = v_48 / v_62;
// const double v_73 = v_71 * v_63;
// const double v_76 = -(v_73);
// const double v_78 = v_28 / v_62;
// const double v_79 = v_76 / v_62;
//
// The single assignment property can be easily verified, because all variables
// are 'const'. An additional property of our implementation here is that the
// right hand side is exactly one expression. An expression is for example a
// binary operator or a function call. In this form, applying code optimizations
// is very efficient.
//
// More on SSA-form: https://en.wikipedia.org/wiki/Static_single_assignment_form
//
//  =============================================================================
//
// The optimizations currently implemented in expressions.h are:
// - trivial assignment propagation
// - zero-one evaluation
// - dead code elimination
// - common expression elimination
// - constant folding
//
//
//   Trivial Assignment Propagation
//      a = b     ->    NOP
//      c = a;    ->    c = b;
// Trivial assignments can be removed, because the right hand side cannot change
// and therefore a==b over the complete life time.
//
//
//   Zero-One Evaluation
//      a = b * 0;    ->    a = 0;
//      a = b + 0;    ->    a = b;
//      ...
// Arithmetic expressions including a 0 or 1 are simplified if possible. This
// optimization is especially targeted at AutoDiff, because many 0-1 operations
// are generated during initialization. Reminder: The 'partial derivative part'
// is initialized with [0,0,1,0,0,0,...].
//
//
//     Dead Code Elimination
//       ...                   ->     ...
//       v_4 = v_0 * v_2;      ->     NOP
//       v_5 = v_0 + v_1;      ->     v_5 = v_0 + v_1;
//       residual[0] = v_5;    ->     residual[0] = v_5;
// Code that does not contribute to the output variables is removed. This
// removal is done by a bottom-up tree traversal. We start at the output
// expression and traverse the tree by visiting the parameters. All expressions
// that have never been visited are dead code.
//
//
//    Redundant Expression Elimination
//      v_5 = f(v_1,v_2);     ->    v_5 = f(v_1,v_2);
//      ...                   ->    ...
//      v_15 = f(v_1,v_2);    ->    v_15 = v_5;
// Per definition of the SSA-form we know that variables will not change over
// their lifetime. Therefore, we have to evaluate identical expressions only
// once (the second appearance will be replaced by a trivial assignment to the
// first). An important note is, that this optimization requires the expressions
// to be side-effect free. You can disable this optimization in the settings.
//
//   Constant Folding
//      v_0 = 3.1415;      ->   v_0 = 3.1415;
//      v_1 = 1;           ->   v_1 = 1;
//      v_2 = v_0 + v_1;   ->   v_2 = 4.1415
//      v_3 = sin(v_0);    ->   v_3 = 0;
// If an expression is side-effect free and all parameters are constant, this
// expression can be evaluated at compile time. The implementation is straight
// forward: Check if all parameters are constant. If yes: Evaluate expression
// and replace by constant assignemnt.
//
//  =============================================================================
//
// There is one fundamental problem using AutoDiff for code generation. Code is
// only generated for expressions which are actually evaluated. Let's look at
// the following simple example:
//
//  ...
//  double a;
//  if(condition)
//    a = 5;
//  else
//    a = 7;
//  ...
//
//  Since we generated the expression tree during run-time, only one branch is
//  "seen". The generated code will be either
//      double a = 5;
//  or
//      double a = 7;
// depending on the value of condition during evaluation.
//
// For our case that means, static branching (conditions not depending on
// parameters) will work, but dynamic branching will fail. A common case of
// static branching are fixed size for loops.
//
// double sum = 0;
// for(int i = 0; i < 3; ++i)
//    sum += vector(i);
//
// These will work, generating the following unrolled code:
//
// double sum_1 = 0;
// double sum_2 = sum_1 + vector(0);
// double sum_3 = sum_2 + vector(1);
// double sum_4 = sum_3 + vector(1);
//
//  =============================================================================
//
//  How to make dynamic branching work with AutoDiffCodeGen
//
// As seen above, dynamic branching with if/else will not work. The trick to
// overcome this problem is to use the Phi-function. The Phi-function itself is
// equivalent to the ternary '?' operator.
//
// PHI(bool c, double a, double b) { return c ? a : b; }
//
// In compiler theory, this Phi-function is used to generate SSA code for
// branches. In the example above, the variable 'a' is assigned at two
// locations, which is not valid in SSA. Using PHI we can now define a valid SSA
// for the snipped:
//
// double a_1 = 5;
// double a_2 = 7;
// double a = PHI(condition,a_1,a_2);
//
// This code is now valid SSA and can be derived with AutoDiffCodeGen, because
// every expression can be 'seen'. In some cases this might be less  efficient
// than the original version, especially for long jmps over code that is rarely
// evaluated. If performance is the major concern it might be beneficial to
// manually add the branches back into the generated code.
//
// Conclusion: Convert branches to phi-functions before generating the
// Derivatives.
//
//  =============================================================================
//
//  Local Variables in Cost Functors.
//
// Let's consider the following two cost functors:
//
//  class CostConstant {
//   public:
//    template <typename T>
//    bool operator()(const T* const x, T* cost) const {
//      T ex = T(3);
//      cost[0] = ex * x[0];
//      return true;
//    }
//  }
//
//  class CostLocal {
//   public:
//    template <typename T>
//    bool operator()(const T* const x, T* cost) const {
//      T ex = T(localVariable);
//      cost[0] = ex * x[0];
//      return true;
//    }
//   double localVariable = 3;
//  }
//
// So what's the difference? The first functor defines a compile-time constant
// 'ex' which is initialized to 3. The second functor uses a local variable
// which is initialized to 3, but might change during runtime.
//
// If we now use AutoDiffCodeGen on both versions, the exact same code is
// generated:
//
//  bool Evaluate_ResidualAndJacobian(double const* const* parameters, double*
//  residuals, double** jacobians)
//  {
//    // This code is generated with ceres::AutoDiffCodeGen
//    // See ceres/autodiff_codegen.h for more informations.
//    const double v_0 = parameters[0][0];
//    const double v_2 = 3.000000;
//    const double v_6 = v_2 * v_0;
//    residuals[0] = v_6;
//    jacobians[0][0] = v_2;
//    return true;
//  }
//
// As you can see, the constant 3 is actually integrated into the program, which
// is only desired for the first functor. For the second functor we would like
// to have the following change:
//
// const double v_2 = 3.000000;     ->     const double v_2 = localVariable;
//
// The easiest way to achieve this transformation automatically is to use the
// CERES_EXTERNAL_CONSTANT macro. The modified CostLocal should now look like:
//
//  class CostLocal {
//   public:
//    template <typename T>
//    bool operator()(const T* const x, T* cost) const {
//      T ex = CERES_EXTERNAL_CONSTANT(localVariable);
//      cost[0] = ex * x[0];
//      return true;
//    }
//   double localVariable = 3;
//  }
//
// Using this macro, the localVariable is kept and integrated into the final
// code. The implementation idea behind this macro is the following:
// If we are in code generation mode (T is of type ExpressionJet), then a
// special assignment expression is generated which includes the name of the
// local variable on the right hand side.
//
#ifndef CERES_PUBLIC_AutoDiffCodeGen_H_
#define CERES_PUBLIC_AutoDiffCodeGen_H_

#include "ceres/internal/autodiff.h"
#include "ceres/internal/expression_operators.h"
#include "ceres/types.h"

namespace ceres {
template <typename CostFunctor, int kNumResiduals, int... Ns>
struct AutoDiffCodeGen {
  explicit AutoDiffCodeGen(CostFunctor* functor) : functor_(functor) {}

  using ParameterDims = internal::StaticParameterDims<Ns...>;

 public:
  bool Generate(
      const CodeGenerationSettings& settings = CodeGenerationSettings()) {
    using T = Expression;
    //     using JetT = ExpressionJet<T, ParameterDims::kNumParameters>;
    using JetT = Jet<Expression, ParameterDims::kNumParameters>;

    CodeFactory factory;
    codeFactory = &factory;

    auto num_outputs = kNumResiduals;
    internal::FixedArray<JetT, (256 * 7) / sizeof(JetT)> x(
        ParameterDims::kNumParameters + num_outputs);

    using Parameters = typename ParameterDims::Parameters;

    // These are the positions of the respective jets in the fixed array x.
    std::array<JetT*, ParameterDims::kNumParameterBlocks> unpacked_parameters =
        ParameterDims::GetUnpackedParameters(x.data());
    JetT* output = x.data() + ParameterDims::kNumParameters;

    factory.tmpExpressions.clear();

    int totalParamId = 0;
    for (int i = 0; i < ParameterDims::kNumParameterBlocks; ++i) {
      for (int j = 0; j < ParameterDims::GetDim(i); ++j) {
        JetT& J = x[totalParamId];
        J.a = factory.ParameterExpr(
            i,
            "parameters[" + std::to_string(i) + "][" + std::to_string(j) + "]");
        for (int k = 0; k < ParameterDims::kNumParameters; ++k) {
          if (k == totalParamId)
            J.v(k) = factory.ConstantExpr(1);
          else
            J.v(k) = factory.ConstantExpr(0);
        }
        totalParamId++;
      }
    }

    if (!internal::VariadicEvaluate<ParameterDims>(
            *functor_, unpacked_parameters.data(), output)) {
      return false;
    }

    factory.check();

    CodeGenerator f(settings);
    f.name =
        "bool Evaluate_Residual(double const* const* parameters, double* "
        "residuals)";
    f.factory = factory;
    for (int i = 0; i < num_outputs; ++i) {
      auto& J = output[i];
      auto res = f.factory.OutputAssignExpr(
          J.a, "residuals[" + std::to_string(i) + "]");
      f.targets.push_back(res);
    }

    {
      // generate a function for the residual only
      auto cpy = f;
      cpy.generate();
      cpy.print();
    }

    f.name =
        "bool Evaluate_ResidualAndJacobian(double const* const* parameters, "
        "double* "
        "residuals, double** jacobians)";

    totalParamId = 0;
    for (int i = 0; i < ParameterDims::kNumParameterBlocks; ++i) {
      for (int j = 0; j < ParameterDims::GetDim(i); ++j) {
        for (int r = 0; r < num_outputs; ++r) {
          auto& J = output[r];
          // all partial derivatives
          auto res = f.factory.OutputAssignExpr(
              (J.v[totalParamId]),
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

  std::unique_ptr<CostFunctor> functor_;
};

}  // namespace ceres
#endif
