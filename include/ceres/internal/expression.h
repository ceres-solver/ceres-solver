
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

#include "Eigen/Core"
#include "ceres/internal/port.h"

#include <cmath>
#include <iosfwd>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace ceres {

enum class ExpressionType {
  // Compile time constant.
  // v_0 = 3.1415;
  CONSTANT,
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
  // A Phi Block is an if-else condition generated from PHI functions during
  // optimization.
  PHI_BLOCK,
  // No Operation. A placeholder for 'empty' expressions which will be removed
  // later.
  NOP,

  INVALID,
};

using ExpressionId = int;
constexpr ExpressionId kInvalidExpressionId = -1;

class CodeFactory;
struct Expression {
  ExpressionId id = kInvalidExpressionId;

  Expression() {}

  // Create a constant expression directly from a double value.
  // v_0 = 123;
  Expression(double constant);

  // Make sure this conversion is explicit so we don't convert it to bool by
  // accident.
  explicit operator ExpressionId() { return id; }

  // The expression adds itself to the codefactory during creation.
  static CodeFactory* codeFactory;
};

struct ComparisonExpression {
  Expression id;
  explicit ComparisonExpression(Expression id) : id(id) {}
};

struct PhiBlock {
  int N;
  Expression condition;
  // expressions which are only used for one of the branches
  std::vector<Expression> localExpressionsTrue, localExpressionsFalse;
  std::vector<Expression> targets;
  std::vector<std::pair<Expression, Expression>> parameters;

  int codeStart = -1;
  int codeEnd = -1;

  bool leftReferenced(Expression expr) const;
  bool rightReferenced(Expression expr) const;
  Expression start() { return targets.front(); }
};

class ExpressionData {
 private:
  // Only the code factory is allowed to create new expressions
  friend class CodeFactory;
  ExpressionData(ExpressionType type, int N, Expression id, double value = 0)
      : type(type), N(N), id(id), value(value) {}

 public:
  // If the function is evaluatable at compile time and it can be optimized for
  // constant 0 or 1 parameters.
  bool zeroOnePropagateable();

  // Checks "other" is identical to "this" so that one of the epxressions can be
  // replaced by a trivial assignemnt. Used during common subexpression
  // elimination.
  bool isRedundant(const ExpressionData& other);

  // if this expression has 'other' as a parameter
  bool references(Expression other);

  // replace this expression by 'other'
  // copies everything except the id
  // -> all other expressions referencing this one stay intact.
  void replace(const ExpressionData& other);

  bool compileTimeConstant();

  bool shouldPrint() { return type != ExpressionType::NOP && !local; }

  // If this expression is the compile time constant '0' or '1'.
  bool isZeroOne();

  // Converts this expression into a NOP
  void makeNOP();

  bool isPhi() const {
    return type == ExpressionType::PHI_BLOCK ||
           type == ExpressionType::PHI_FUNCTION;
  }

  // The return type as a string.
  // Usually "const double" except for comparison, which is "const bool".
  std::string resultType() const;

  // Returns the target name.
  //  v_0 = v_1 + v_2;
  // -> return "v_0"
  std::string name() const;

  // Generates the actual c++ code for this expression.
  // This is exactly one line except for PhiBlock expressions.
  std::string generate(const std::vector<ExpressionData>& code) const;

  // Debug Check.
  void check() const;

  Expression id;
  int targetId = -1;
  bool local = false;

  // only valid if type==FUNCTION_CALL
  std::string functionName;
  ExpressionType type = ExpressionType::INVALID;

  double value;  // only for constant with N == 0
  int N;         // number of params
  std::vector<Expression> params;
  bool externalConstant = false;

  // only valid if type == PHI_BLOCK
  PhiBlock phiBlock;

  mutable int tmp;
};

class CodeFactory {
 public:
  std::vector<ExpressionData> expression_data;

  ExpressionData& operator()(Expression i) { return expression_data[i.id]; }

  const ExpressionData& operator()(Expression i) const {
    return expression_data[i.id];
  }

  // These functions create the corresponding expression, add them to an
  // internal vector and return a reference to them.
  Expression ConstantExpr(double v);
  Expression ParameterExpr(int v, const std::string& name);
  Expression AssignExpr(Expression v);
  Expression UnaryMinusExpr(Expression v);
  Expression OutputAssignExpr(Expression v, const std::string& name);
  Expression FunctionExpr(const std::string& name,
                          std::vector<Expression> params);
  Expression Phi(Expression c, Expression a, Expression b);
  Expression PhiBlockExpr(const PhiBlock& block);
  Expression BinaryCompare(const std::string& name, Expression l, Expression r);
  Expression BinaryExpr(ExpressionType type, Expression l, Expression r);

  void check();

 private:
  Expression Expr(ExpressionType type, int N, double value = 0);
};

}  // namespace ceres

#endif
