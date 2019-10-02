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
// See ceres/autodiff_codegen.h for a complete overview.
//
// This file contains the basic expression type, which is used during code
// creation. Only assignment expressions of the following form are supported:
//
// result = [constant|binary_expr|functioncall]
//
// Examples:
// v_78 = v_28 / v_62;
// v_97 = exp(v_20);
// v_89 = 3.000000;
//
//
#ifndef CERES_PUBLIC_EXPRESSION_H_
#define CERES_PUBLIC_EXPRESSION_H_

#include <cmath>
#include <iosfwd>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace ceres {
namespace internal {

/**
 * @brief A type-safe reference to 'Expression'.
 *
 * This class represents a scalar value that creates new expressions during
 * evaluation. ExpressionRef can be used as template parameter for cost functors
 * and Jets.
 *
 * ExpressionRef should be passed by value.
 */
struct ExpressionRef {
  using ExpressionId = int;
  static constexpr ExpressionId kInvalidExpressionId = -1;

  ExpressionId id = kInvalidExpressionId;

  ExpressionRef() = default;

  // Create a constant expression directly from a double value.
  // v_0 = 123;
  ExpressionRef(double constant);

  // Make sure this conversion is explicit so we don't convert it to bool by
  // accident.
  explicit operator ExpressionId() { return id; }

  // Compound operators (defined in expression_arithmetic.h)
  inline ExpressionRef& operator+=(ExpressionRef y);
  inline ExpressionRef& operator-=(ExpressionRef y);
  inline ExpressionRef& operator*=(ExpressionRef y);
  inline ExpressionRef& operator/=(ExpressionRef y);
};

/**
 * @brief A reference to a comparison expressions.
 *
 * This additonal type is required, so that we can detect invalid conditions
 * during compile time.
 */
struct ComparisonExpressionRef {
  ExpressionRef id;
  explicit ComparisonExpressionRef(ExpressionRef id) : id(id) {}
};

enum class ExpressionType {
  // v_0 = 3.1415;
  COMPILE_TIME_CONSTANT,

  // External constant. For example a local member of the cost-functor.
  // v_0 = _observed_point_x;
  EXTERNAL_CONSTANT,

  // Input parameter
  // v_0 = parameters[1][5];
  PARAMETER,

  // Output Variable Assignemnt
  // residual[0] = v_51;
  OUTPUT_ASSIGN,

  // Trivial Assignment
  // v_1 = v_0;
  ASSIGN,

  // Binary Arithmetic Operations
  // v_2 = v_0 + v_1
  PLUS,
  MINUS,
  MULTIPLICATION,
  DIVISION,

  // Unary Arithmetic Operation
  // v_1 = -(v_0);
  UNARY_MINUS,

  // Binary Comparision. (<,>,&&,...)
  // This is the only expressions which returns a 'bool'.
  // const bool v_2 = v_0 < v_1
  BINARY_COMPARE,

  // General Function Call.
  // v_5 = f(v_0,v_1,...)
  FUNCTION_CALL,

  // The ternary ?-operator. Separated from the general function call for easier
  // access.
  // v_3 = ternary(v_0,v_1,v_2);
  TERNARY,

  // No Operation. A placeholder for 'empty' expressions which will be removed
  // later.
  NOP
};

class Expression {
 public:
  bool IsSimpleArithmetic();

  // Checks "other" is identical to "this" so that one of the epxressions can be
  // replaced by a trivial assignemnt. Used during common subexpression
  // elimination.
  bool IsReplaceableBy(const Expression& other);

  // If this expression is the compile time constant '0' or '1'.
  bool IsConstantZero();
  bool IsConstantOne();

  // Replace this expression by 'other'.
  // The current id will be not replaced. That means other experssions
  // referencing this one stay valid.
  void Replace(const Expression& other);

  // if this expression has 'other' as a parameter
  bool DependsOn(ExpressionRef other);

  // Converts this expression into a NOP
  void MakeNop();

  // The return type as a string.
  // Usually "const double" except for comparison, which is "const bool".
  std::string ResultTypeString() const;

  // Returns the target name.
  //  v_0 = v_1 + v_2;
  // -> return "v_0"
  std::string LhsName() const;

 public:
  // These functions create the corresponding expression, add them to an
  // internal vector and return a reference to them.
  static ExpressionRef MakeConstant(double v);
  static ExpressionRef MakeExternalConstant(const std::string& name);
  static ExpressionRef MakeParameter(const std::string& name);
  static ExpressionRef MakeAssign(ExpressionRef v);
  static ExpressionRef MakeUnaryMinus(ExpressionRef v);
  static ExpressionRef MakeOutput(ExpressionRef v, const std::string& name);
  static ExpressionRef MakeFunction(const std::string& name,
                                    std::vector<ExpressionRef> params_);
  static ExpressionRef MakeTernary(ComparisonExpressionRef c,
                                   ExpressionRef a,
                                   ExpressionRef b);
  static ExpressionRef MakeBinaryCompare(const std::string& name,
                                         ExpressionRef l,
                                         ExpressionRef r);
  static ExpressionRef MakeBinaryArithmetic(ExpressionType type_,
                                            ExpressionRef l,
                                            ExpressionRef r);

  static Expression& Data(ExpressionRef e) { return expression_data[e.id]; }

 private:
  // During execution, the expressions add themself into this vector. This
  // allows us to see what code was executed and optimize it later.
  static std::vector<Expression> expression_data;
  static Expression& MakeExpr(ExpressionType type_);

  // Private constructor to ensure all expressions are created using the static
  // functions above.
  Expression(ExpressionType type_, ExpressionRef id_);

  ExpressionRef id_;

  // Depending on the type this name is one of the following:
  //  (type == FUNCTION_CALL) -> the function name
  //  (type == PARAMETER)     -> the parameter name
  //  (type == OUTPUT_ASSIGN) -> the output variable name
  //  (type == BINARY_COMPARE)-> the comparison symbol "<","&&",...
  std::string name_;

  ExpressionType type_ = ExpressionType::NOP;

  // Expressions have different number of parameters. For example a binary "+"
  // has 2 parameters and a function call so "sin" has 1 parameter. Here, a
  // reference to these paratmers is stored. Note: The order matters!
  std::vector<ExpressionRef> params_;

  // Only valid if type == COMPILE_TIME_CONSTANT
  double constant_value_ = 0;
};

}  // namespace internal
}  // namespace ceres
#endif
