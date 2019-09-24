
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
#ifndef CERES_PUBLIC_EXPRESSIONS_H_
#define CERES_PUBLIC_EXPRESSIONS_H_

#include "Eigen/Core"

#include <cmath>
#include <iosfwd>
#include <iostream>
#include <string>
#include <vector>

namespace ceres {
enum Type {
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
};

using ExpressionId = int;
constexpr ExpressionId invalidExpressionId = -1;

// Use a wrapper class to make sure the expression id is not converted to bool.
class CodeFactory;
struct ComparisonExpressionId {
  ExpressionId id;
  CodeFactory* factory;
  explicit ComparisonExpressionId(ExpressionId _id) : id(_id) {}
};

struct ExpressionBase {
  ExpressionId id;
  int targetId = -1;
  std::string typeString = "const double";

 private:
  friend class CodeFactory;

  // only valid if type==FUNCTION_CALL
  std::string functionName;

  ExpressionBase(Type type, int N, int id, double value = 0)
      : type(type), N(N), id(id), value(value) {}

 public:
  // If the function is evaluatable at compile time
  // + - * /
  bool zeroOnePropagateable() {
    switch (type) {
      case BINARY_PLUS:
      case BINARY_MULT:
      case BINARY_DIV:
      case BINARY_MINUS:
      case UNARY_MINUS:
        return true;
      default:
        return false;
    }
  }

  // replace this expression by 'other'
  // copies everything except the id
  // -> all other expressions referencing this one stay intact.
  void replace(const ExpressionBase& other) {
    auto currentId = id;
    auto currentTarget = targetId;
    (*this) = other;
    id = currentId;
    targetId = currentTarget;
  }

  std::string resultType() const {
    if (type == BINARY_COMPARE) return "const bool";

    return typeString;
  }

  bool isZeroOne() { return type == CONSTANT && (value == 0 || value == 1); }

  std::string name() const { return "v_" + std::to_string(id); }

  std::string paramName() const {
    assert(type == PARAMETER);
    //        return "x_" + std::to_string(int(value));
    return functionName;
  }
  void check() const { assert(N == params.size()); }
  std::string generate(const std::vector<ExpressionBase>& code) const {
    check();

    std::string res;

    res = name() + " = ";

    res = resultType() + " " + res;

    std::string binaryOperator;

    switch (type) {
      case ASSIGN:
        res += code[params[0]].name() + ";";
        return res;
      case CONSTANT:
        res += std::to_string(value) + ";";
        return res;
      case PARAMETER:
        res += paramName() + ";";
        return res;
      case RETURN:
        return "return " + code[params[0]].name() + ";";
      case OUTPUT_ASSIGN:
        return functionName + " = " + code[params[0]].name() + ";";

      case BINARY_PLUS:
        binaryOperator = "+";
        break;
      case BINARY_MULT:
        binaryOperator = "*";
        break;
      case BINARY_MINUS:
        binaryOperator = "-";
        break;
      case BINARY_DIV:
        binaryOperator = "/";
        break;

      case BINARY_COMPARE:
        binaryOperator = functionName;
        break;
      case UNARY_MINUS:
        res += "-";
        break;
      case FUNCTION_CALL:
        res += functionName;
        break;
      case NOP:
        return "NOP;";
      default:
        res += "UNKNOWM ";
    }

    if (!binaryOperator.empty()) {
      assert(params.size() == 2);
      res += code[params[0]].name() + " " + binaryOperator + " " +
             code[params[1]].name() + ";";
    } else {
      res += "(";

      for (int i = 0; i < N; ++i) {
        auto& ex = code[params[i]];
        res += ex.name();
        if (i < N - 1) res += ",";
      }
      res += ");";
    }

    return res;
  }

  Type type;

  double value;  // only for constant with N == 0
  int N;         // number of params
  std::vector<ExpressionId> params;

  mutable int tmp;
};

struct CodeFactory {
  std::vector<ExpressionBase> tmpExpressions;

  ExpressionBase& operator()(ExpressionId i) { return tmpExpressions[i]; }
  const ExpressionBase& operator()(ExpressionId i) const {
    return tmpExpressions[i];
  }

  ExpressionId Expr(Type type, int N, double value = 0) {
    ExpressionBase expr(type, N, tmpExpressions.size(), value);
    tmpExpressions.push_back(expr);
    return expr.id;
  }

  ExpressionId ConstantExpr(double v) { return Expr(CONSTANT, 0, v); }

  ExpressionId ParameterExpr(int v, const std::string& name) {
    auto expr = Expr(PARAMETER, 0, v);
    ((*this)(expr)).functionName = name;
    return expr;
  }

  ExpressionId AssignExpr(ExpressionId v) {
    auto expr = Expr(ASSIGN, 1);
    ((*this)(expr)).params.push_back(v);
    return expr;
  }

  ExpressionId UnaryMinusExpr(ExpressionId v) {
    auto expr = Expr(UNARY_MINUS, 1);
    ((*this)(expr)).params.push_back(v);
    return expr;
  }

  ExpressionId ReturnExpr(ExpressionId v) {
    auto expr = Expr(RETURN, 1);
    ((*this)(expr)).params.push_back(v);
    return expr;
  }
  ExpressionId OutputAssignExpr(ExpressionId v, const std::string& name) {
    auto expr = Expr(OUTPUT_ASSIGN, 1);
    ((*this)(expr)).params.push_back(v);
    ((*this)(expr)).functionName = name;
    return expr;
  }

  template <unsigned long N>
  std::array<ExpressionId, N> ConstantExprArray(int v) {
    std::array<ExpressionId, N> res;
    for (int i = 0; i < N; ++i) {
      if (i == v) {
        res[i] = ConstantExpr(1);
      } else {
        res[i] = ConstantExpr(0);
      }
    }
    return res;
  }

  ExpressionId FunctionExpr(const std::string& name,
                            std::vector<ExpressionId> params) {
    auto expr = Expr(FUNCTION_CALL, params.size());
    ((*this)(expr)).params = params;
    ((*this)(expr)).functionName = name;
    return expr;
  }

  ExpressionId BinaryCompare(const std::string& name,
                             ExpressionId l,
                             ExpressionId r) {
    auto expr = Expr(BINARY_COMPARE, 2);
    ((*this)(expr)).params.push_back(l);
    ((*this)(expr)).params.push_back(r);
    ((*this)(expr)).functionName = name;
    return expr;
  }

  ExpressionId BinaryExpr(Type type, ExpressionId l, ExpressionId r) {
    auto expr = Expr(type, 2);
    ((*this)(expr)).params.push_back(l);
    ((*this)(expr)).params.push_back(r);
    return expr;
  }

  template <unsigned long N>
  std::array<ExpressionId, N> BinaryExpr(Type type,
                                         const std::array<ExpressionId, N>& l,
                                         const std::array<ExpressionId, N>& r) {
    std::array<ExpressionId, N> res;
    for (int i = 0; i < N; ++i) {
      res[i] = BinaryExpr(type, l[i], r[i]);
    }
    return res;
  }

  template <unsigned long N>
  std::array<ExpressionId, N> BinaryExpr(Type type,
                                         ExpressionId l,
                                         const std::array<ExpressionId, N>& r) {
    std::array<ExpressionId, N> res;
    for (int i = 0; i < N; ++i) {
      res[i] = BinaryExpr(type, l, r[i]);
    }
    return res;
  }

  template <unsigned long N>
  std::array<ExpressionId, N> BinaryExpr(Type type,
                                         const std::array<ExpressionId, N>& l,
                                         ExpressionId r) {
    std::array<ExpressionId, N> res;
    for (int i = 0; i < N; ++i) {
      res[i] = BinaryExpr(type, l[i], r);
    }
    return res;
  }
};

struct CodeFunction {
  // local copy here
  CodeFactory factory;

  // final code indexing into the expression array of factory
  std::vector<ExpressionId> code;

  // name of the function
  std::string name;

  std::vector<ExpressionId> targets;

  bool removedUnusedParameters = true;

  // traverse through the expression tree from the target node.
  // all expressions found will be added to the output.
  // this is basically a dead code elimination
  std::vector<ExpressionId> traverseAndCollect(ExpressionId target) {
    std::vector<ExpressionId> code;

    std::vector<ExpressionId> stack;
    stack.push_back(target);

    while (!stack.empty()) {
      auto& top = factory(stack.back());
      stack.pop_back();
      if (top.tmp == 1) continue;
      code.push_back(top.id);
      top.tmp = 1;

      // add all dependencies
      for (auto& expr : top.params) {
        stack.push_back(expr);
      }
    }
    std::sort(code.begin(), code.end());

    return code;
  }

  void traverseAndCollect() {
    code.clear();

    for (auto& expr : factory.tmpExpressions) {
      expr.tmp = 0;
    }

    if (!removedUnusedParameters) {
      // add all parameter expressions
      for (auto& expr : factory.tmpExpressions) {
        if (expr.type == PARAMETER) {
          expr.tmp = 1;
          code.push_back(expr.id);
        }
      }
    }

    for (int i = 0; i < targets.size(); ++i) {
      auto target = targets[i];
      auto codenew = traverseAndCollect(target);
      code.insert(code.end(), codenew.begin(), codenew.end());
    }

    std::sort(code.begin(), code.end());
  }

  void generate() {
    for (int i = 0; i < targets.size(); ++i) {
      auto target = targets[i];
      factory(target).targetId = i;
    }

    traverseAndCollect();

    zeroOneAssignemntPropagation();

    // traverse again to remove dead code
    traverseAndCollect();

    for (int i = 0; i < targets.size(); ++i) {
      auto target = targets[i];
      factory(target).targetId = i;
    }
  }

  // an optimization to propagate 0 and 1 into the arithmetic expressions.
  // Additionally also the assignments are propagated forwards.
  // Current operations:
  //
  // a = 0 * b   ->    a = 0
  // a = 0 + b   ->    a = b
  //
  // a = b
  // c = d + a;  ->    c = d + b
  void zeroOneAssignemntPropagation() {
    //        std::cout << "Running zeroOneAssignemntPropagation" << std::endl;
    // iterate over the code multiple times until nothing changes anymore

    auto findZeroOneParam = [this](ExpressionBase& expr) -> ExpressionId {
      for (int i = 0; i < expr.params.size(); ++i) {
        auto& pexpr = factory(expr.params[i]);
        if (pexpr.isZeroOne()) {
          return i;
        }
      }
      return -1;
    };

    bool changed = true;
    while (changed) {
      changed = false;

      // find 0-1 operations, evaluate and replace by assignments
      for (auto id : code) {
        auto& expr = factory(id);
        if (!expr.zeroOnePropagateable()) continue;

        auto zo = findZeroOneParam(expr);
        if (zo != -1) {
          //                    std::cout << "found zero one param " <<
          //                    expr.generate(factory.tmpExpressions)
          //                    << std::endl;

          // either zero or one
          bool isZero = false;
          auto& zoexpr = factory(expr.params[zo]);
          isZero = zoexpr.value == 0;

          auto otherBinaryId = (zo == 0) ? 1 : 0;

          if (expr.type == BINARY_MULT && isZero) {
            // a = 0 * b   ->   a = 0
            auto id = factory.ConstantExpr(0);
            expr.replace(factory(id));
            changed = true;
          } else if (expr.type == BINARY_MULT && !isZero) {
            // a = 1 * b   ->    a = b
            auto id = factory.AssignExpr(expr.params[otherBinaryId]);
            expr.replace(factory(id));
            changed = true;
          } else if (expr.type == BINARY_PLUS && isZero) {
            // a = 0 + b   ->    a = b
            auto id = factory.AssignExpr(expr.params[otherBinaryId]);
            expr.replace(factory(id));
            changed = true;
          } else if (expr.type == BINARY_DIV && isZero) {
            if (zo == 0) {
              // a = 0 / b   ->    a = 0
              auto id = factory.ConstantExpr(0);
              expr.replace(factory(id));
              changed = true;
            } else {
              // a = b / 0   ->    error
              std::cout << "Warning division by zero detected! Line: "
                        << expr.generate(factory.tmpExpressions) << std::endl;
            }
          } else if (expr.type == BINARY_MINUS && isZero) {
            if (zo == 0) {
              // a = 0 - b   ->    a = -b
              auto id = factory.UnaryMinusExpr(expr.params[otherBinaryId]);
              expr.replace(factory(id));
              changed = true;
            } else {
              // a = b - 0   ->    a = b
              auto id = factory.AssignExpr(expr.params[otherBinaryId]);
              expr.replace(factory(id));
              changed = true;
            }
          } else if (expr.type == UNARY_MINUS && isZero) {
            // a = -0   ->   a = 0
            auto id = factory.ConstantExpr(0);
            expr.replace(factory(id));
            changed = true;
          }
        }
      }

      // eliminate assignments
      // a = b
      // c = a + d
      //
      // ->   c = b + d    (the variable a is removed)
      for (auto id : code) {
        auto& expr = factory(id);
        if (expr.type != ASSIGN) continue;

        auto target = id;
        auto src = expr.params[0];

        // go over code and find expressions wit 'target' as a paramter and
        // replace with src
        for (auto id2 : code) {
          auto& expr2 = factory(id2);

          for (auto& p : expr2.params) {
            if (p == target) {
              p = src;
            }
          }
        }

        if (expr.targetId != -1) {
          // we are going to eliminate expr
          // ->
          targets[expr.targetId] = src;
          factory(src).targetId = expr.targetId;
          expr.targetId = -1;
        }

        // elimintate variable
        expr.type = NOP;
        expr.params.clear();
        expr.N = 0;
      }
    }
  }

  void print() {
    std::cout << "bool Evaluate(double const* const* parameters, double* "
                 "residuals, double** jacobians)"
              << std::endl;
    std::string indent = "  ";

    std::cout << "{" << std::endl;
    std::cout << indent
              << "// This code is generated with ceres::AutoDiffCodeGen"
              << std::endl;
    std::cout << indent
              << "// See ceres/autodiff_codegen.h for more informations."
              << std::endl;
    for (auto f : code) {
      std::cout << indent << factory(f).generate(factory.tmpExpressions)
                << std::endl;
    }
    std::cout << indent << "return true;" << std::endl;
    std::cout << "}" << std::endl;
  }
};

}  // namespace ceres

#endif
