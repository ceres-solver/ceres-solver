
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

enum ExpressionType {
  CONSTANT,
  PARAMETER,
  ASSIGN,
  BINARY_PLUS,
  BINARY_MINUS,
  BINARY_MULT,
  BINARY_DIV,
  BINARY_COMPARE,
  UNARY_MINUS,
  FUNCTION_CALL,
  RETURN,
  OUTPUT_ASSIGN,
  NOP,
  INVALID,
};

using ExpressionId = int;
constexpr ExpressionId invalidExpressionId = -1;

struct Expression {
  ExpressionId id = invalidExpressionId;

  Expression() {}
  // create a constant expression
  Expression(double constant);
  explicit operator ExpressionId() { return id; }
};

struct ComparisonExpression {
  Expression id;
  explicit ComparisonExpression(Expression _id) : id(_id) {}
};

struct ExpressionData {
 private:
  // Only the code factory is allowed to create new expressions
  friend class CodeFactory;
  ExpressionData(ExpressionType type, int N, Expression id, double value = 0)
      : type(type), N(N), id(id), value(value) {}

 public:
  // If the function is evaluatable at compile time
  // + - * /
  bool zeroOnePropagateable();

  bool isRedundant(const ExpressionData& other);

  // replace this expression by 'other'
  // copies everything except the id
  // -> all other expressions referencing this one stay intact.
  void replace(const ExpressionData& other);

  std::string resultType() const;

  bool compileTimeConstant();

  bool isZeroOne();

  std::string name() const;

  std::string paramName() const;
  void check() const;
  std::string generate(const std::vector<ExpressionData>& code) const;

  Expression id;
  int targetId = -1;
  std::string typeString = "const double";

  // only valid if type==FUNCTION_CALL
  std::string functionName;
  ExpressionType type = INVALID;

  double value;  // only for constant with N == 0
  int N;         // number of params
  std::vector<Expression> params;
  bool externalConstant = false;

  mutable int tmp;
};

struct CodeFactory {
  std::vector<ExpressionData> tmpExpressions;

  ExpressionData& operator()(Expression i) { return tmpExpressions[i.id]; }

  const ExpressionData& operator()(Expression i) const {
    return tmpExpressions[i.id];
  }

  Expression ConstantExpr(double v);

  Expression ParameterExpr(int v, const std::string& name);

  Expression AssignExpr(Expression v);

  Expression UnaryMinusExpr(Expression v);

  Expression ReturnExpr(Expression v);
  Expression OutputAssignExpr(Expression v, const std::string& name);

  Expression FunctionExpr(const std::string& name,
                          std::vector<Expression> params);

  Expression BinaryCompare(const std::string& name, Expression l, Expression r);

  Expression BinaryExpr(ExpressionType type, Expression l, Expression r);

  void check();

 private:
  Expression Expr(ExpressionType type, int N, double value = 0);
};

// Let's use a global variable here so we can access it from the overloaded
// operators
CERES_EXPORT extern CodeFactory* codeFactory;

struct CodeGenerationSettings {
  bool addInlineKeyword = true;

  // this string we be added to the function names.
  // usefull if you want to generate multiple functors into the same file.
  std::string function_prefix = "";

  // If this is false, none of the optimizations below will be executed
  bool optimize = true;

  bool optimize_unusedParameters = false;
  bool optimize_deadCode = true;
  bool optimize_zeroones = true;
  bool optimize_trivialAssignments = true;
  bool optimize_constants = true;
  bool optimize_commonExpressionis = true;
};

struct CodeGenerator {
  // local copy here
  CodeFactory factory;

  // name of the function
  std::string name = "foo";
  std::string returnValue = "bool";

  std::vector<Expression> targets;

  CodeGenerator(const CodeGenerationSettings& settings) : settings(settings) {}

  void generate();
  void print(std::ostream& strm);

 private:
  CodeGenerationSettings settings;
  // final code indexing into the expression array of factory
  std::vector<Expression> code;

  // traverse through the expression tree from the target node.
  // all expressions found will be added to the output.
  std::vector<Expression> traverseAndCollect(Expression target);
  void DeadCodeRemoval();

  bool ZeroOnePropagation();

  bool TrivialAssignmentElimination();

  bool ConstantFolding();

  bool CommonSubexpressionElimination();

  void optimize();
};

}  // namespace ceres

#endif
