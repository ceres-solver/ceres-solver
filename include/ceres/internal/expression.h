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

struct ComparisonExpression {
  ExpressionRef id;
  explicit ComparisonExpression(ExpressionRef id) : id(id) {}
};

enum class ExpressionType {
  // Compile time constant.
  // v_0 = 3.1415;
  CONSTANT,
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
  MULT,
  DIV,
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
  // The PHI function. Separated from the general function call for easier
  // access.
  // v_3 = PHI(v_0,v_1,v_2);
  PHI_FUNCTION,
  // No Operation. A placeholder for 'empty' expressions which will be removed
  // later.
  NOP,

  INVALID,
};

class Expression {
 public:
  bool isSimpleArithmetic();

  // Checks "other" is identical to "this" so that one of the epxressions can be
  // replaced by a trivial assignemnt. Used during common subexpression
  // elimination.
  bool isReplaceableBy(const Expression& other);

  // replace this expression by 'other'
  // copies everything except the id
  // -> all other expressions referencing this one stay intact.
  void replace(const Expression& other);

  // if this expression has 'other' as a parameter
  bool dependsOn(ExpressionRef other);

  // If this expression is the compile time constant '0' or '1'.
  bool isConstantZero();
  bool isConstantOne();

  // Converts this expression into a NOP
  void makeNOP();

  // The return type as a string.
  // Usually "const double" except for comparison, which is "const bool".
  std::string resultTypeString() const;

  // Returns the target name.
  //  v_0 = v_1 + v_2;
  // -> return "v_0"
  std::string lhs_name() const;

 public:
  // These functions create the corresponding expression, add them to an
  // internal vector and return a reference to them.
  static ExpressionRef ConstantExpr(double v);
  static ExpressionRef ExternalConstantExpr(const std::string& name);
  static ExpressionRef ParameterExpr(const std::string& lhs_name);
  static ExpressionRef AssignExpr(ExpressionRef v);
  static ExpressionRef UnaryMinusExpr(ExpressionRef v);
  static ExpressionRef OutputAssignExpr(ExpressionRef v,
                                        const std::string& lhs_name);
  static ExpressionRef FunctionExpr(const std::string& lhs_name,
                                    std::vector<ExpressionRef> params_);
  static ExpressionRef Phi(ComparisonExpression c,
                           ExpressionRef a,
                           ExpressionRef b);
  static ExpressionRef BinaryCompare(const std::string& lhs_name,
                                     ExpressionRef l,
                                     ExpressionRef r);
  static ExpressionRef BinaryExpr(ExpressionType type_,
                                  ExpressionRef l,
                                  ExpressionRef r);

  static Expression& data(ExpressionRef e) { return expression_data[e.id]; }

 private:
  static std::vector<Expression> expression_data;
  static Expression& Expr(ExpressionType type_);

  // Private constructor to ensure all expressions are created using the static
  // functions above.
  Expression(ExpressionType type_, ExpressionRef id_);

  ExpressionRef id_;

  // Depending on the type this name is one of the following:
  //  (type == FUNCTION_CALL) -> the function name
  //  (type == PARAMETER)     -> the parameter name
  //  (type == OUTPUT_ASSIGN) -> the output variable name
  //  (type == BINARY_COMPARE)-> the comparison symbol "<","&&",...
  std::string function_name_;

  ExpressionType type_ = ExpressionType::INVALID;

  std::vector<ExpressionRef> params_;

  // Only valid if type == CONSTANT
  double constant_value_ = 0;
};

}  // namespace ceres

#endif
