
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
#include "ceres/internal/expression.h"

namespace ceres {
CodeFactory* codeFactory = nullptr;

bool ExpressionData::zeroOnePropagateable() {
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

bool ExpressionData::isRedundant(const ExpressionData& other) {
  if (type == NOP) return false;

  // check everything except the id
  if (!(std::make_tuple(type, externalConstant, functionName, value, N) ==
        std::make_tuple(other.type,
                        other.externalConstant,
                        other.functionName,
                        other.value,
                        other.N)))
    return false;

  // check if params are equal too
  for (int i = 0; i < N; ++i) {
    if (params[i].id != other.params[i].id) return false;
  }
  return true;
}

void ExpressionData::replace(const ExpressionData& other) {
  auto currentId = id;
  auto currentTarget = targetId;
  (*this) = other;
  id = currentId;
  targetId = currentTarget;
}

std::string ExpressionData::resultType() const {
  if (type == BINARY_COMPARE) return "const bool";

  return typeString;
}

bool ExpressionData::compileTimeConstant() {
  return type == CONSTANT && !externalConstant;
}

bool ExpressionData::isZeroOne() {
  return compileTimeConstant() && (value == 0 || value == 1);
}

std::string ExpressionData::name() const {
  return "v_" + std::to_string(id.id);
}

std::string ExpressionData::paramName() const {
  assert(type == PARAMETER);
  //        return "x_" + std::to_string(int(value));
  return functionName;
}

void ExpressionData::check() const {
  assert(N == params.size());
  assert(type != INVALID);
}

template <typename T>
inline std::string double_to_string_precise(const T a_value, const int n = 6) {
  std::ostringstream out;
  out.precision(n);
  out << std::scientific << a_value;
  return out.str();
}

std::string ExpressionData::generate(
    const std::vector<ExpressionData>& code) const {
  check();

  std::string res;

  res = name() + " = ";

  res = resultType() + " " + res;

  std::string binaryOperator;

  switch (type) {
    case ASSIGN:
      res += code[params[0].id].name() + ";";
      return res;
    case CONSTANT:
      if (externalConstant) {
        res += functionName + ";";
      } else {
        res += double_to_string_precise(value, 20) + ";";
      }
      return res;
    case PARAMETER:
      res += paramName() + ";";
      return res;
    case RETURN:
      return "return " + code[params[0].id].name() + ";";
    case OUTPUT_ASSIGN:
      return functionName + " = " + code[params[0].id].name() + ";";

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
    res += code[params[0].id].name() + " " + binaryOperator + " " +
           code[params[1].id].name() + ";";
  } else {
    res += "(";

    for (int i = 0; i < N; ++i) {
      auto& ex = code[params[i].id];
      res += ex.name();
      if (i < N - 1) res += ",";
    }
    res += ");";
  }

  return res;
}

Expression CodeFactory::ConstantExpr(double v) { return Expr(CONSTANT, 0, v); }

Expression CodeFactory::ParameterExpr(int v, const std::string& name) {
  auto expr = Expr(PARAMETER, 0, v);
  ((*this)(expr)).functionName = name;
  return expr;
}

Expression CodeFactory::AssignExpr(Expression v) {
  auto expr = Expr(ASSIGN, 1);
  ((*this)(expr)).params.push_back(v);
  return expr;
}

Expression CodeFactory::UnaryMinusExpr(Expression v) {
  auto expr = Expr(UNARY_MINUS, 1);
  ((*this)(expr)).params.push_back(v);
  return expr;
}

Expression CodeFactory::ReturnExpr(Expression v) {
  auto expr = Expr(RETURN, 1);
  ((*this)(expr)).params.push_back(v);
  return expr;
}

Expression CodeFactory::OutputAssignExpr(Expression v,
                                         const std::string& name) {
  auto expr = Expr(OUTPUT_ASSIGN, 1);
  ((*this)(expr)).params.push_back(v);
  ((*this)(expr)).functionName = name;
  return expr;
}

Expression CodeFactory::FunctionExpr(const std::string& name,
                                     std::vector<Expression> params) {
  auto expr = Expr(FUNCTION_CALL, params.size());
  ((*this)(expr)).params = params;
  ((*this)(expr)).functionName = name;
  return expr;
}

Expression CodeFactory::BinaryCompare(const std::string& name,
                                      Expression l,
                                      Expression r) {
  auto expr = Expr(BINARY_COMPARE, 2);
  ((*this)(expr)).params.push_back(l);
  ((*this)(expr)).params.push_back(r);
  ((*this)(expr)).functionName = name;
  return expr;
}

Expression CodeFactory::BinaryExpr(ExpressionType type,
                                   Expression l,
                                   Expression r) {
  auto expr = Expr(type, 2);
  ((*this)(expr)).params.push_back(l);
  ((*this)(expr)).params.push_back(r);
  return expr;
}

void CodeFactory::check() {
  for (int i = 0; i < tmpExpressions.size(); ++i) {
    auto& expr = tmpExpressions[i];
    expr.check();
    assert(i == expr.id.id);
  }
}

Expression CodeFactory::Expr(ExpressionType type, int N, double value) {
  auto id = tmpExpressions.size();
  Expression e;
  e.id = id;
  ExpressionData expr(type, N, e, value);
  tmpExpressions.push_back(expr);
  return e;
}

std::vector<Expression> CodeGenerator::traverseAndCollect(Expression target) {
  std::vector<Expression> code;

  std::vector<Expression> stack;
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
  std::sort(code.begin(), code.end(), [](Expression a, Expression b) {
    return a.id < b.id;
  });

  return code;
}

void CodeGenerator::DeadCodeRemoval() {
  code.clear();

  for (auto& expr : factory.tmpExpressions) {
    expr.tmp = 0;
  }

  if (!settings.optimize_unusedParameters) {
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

  //    std::sort(code.begin(), code.end());
  std::sort(code.begin(), code.end(), [](Expression a, Expression b) {
    return a.id < b.id;
  });
}

void CodeGenerator::generate() {
  for (int i = 0; i < targets.size(); ++i) {
    auto target = targets[i];
    factory(target).targetId = i;
  }

  optimize();

  for (int i = 0; i < targets.size(); ++i) {
    auto target = targets[i];
    factory(target).targetId = i;
  }
}

bool CodeGenerator::ZeroOnePropagation() {
  auto findZeroOneParam = [this](ExpressionData& expr) -> int {
    for (int i = 0; i < expr.params.size(); ++i) {
      auto& pexpr = factory(expr.params[i]);
      if (pexpr.isZeroOne()) {
        return i;
      }
    }
    return -1;
  };

  bool changed = false;

  // find 0-1 operations, evaluate and replace by assignments
  for (auto initial_id : code) {
    //      auto& expr = factory(initial_id);
    if (!factory(initial_id).zeroOnePropagateable()) continue;

    auto zo = findZeroOneParam(factory(initial_id));
    if (zo != -1) {
      // either zero or one
      bool isZero = false;
      auto& zoexpr = factory(factory(initial_id).params[zo]);
      isZero = zoexpr.value == 0;
      auto otherBinaryId = (zo == 0) ? 1 : 0;

      if (factory(initial_id).type == BINARY_MULT && isZero) {
        // a = 0 * b   ->   a = 0
        auto id = factory.ConstantExpr(0);
        factory(initial_id).replace(factory(id));
        changed = true;
      } else if (factory(initial_id).type == BINARY_MULT && !isZero) {
        // a = 1 * b   ->    a = b
        auto id = factory.AssignExpr(factory(initial_id).params[otherBinaryId]);
        factory(initial_id).replace(factory(id));
        changed = true;
      } else if (factory(initial_id).type == BINARY_PLUS && isZero) {
        // a = 0 + b   ->    a = b
        auto id = factory.AssignExpr(factory(initial_id).params[otherBinaryId]);
        factory(initial_id).replace(factory(id));
        changed = true;
      } else if (factory(initial_id).type == BINARY_DIV && isZero) {
        if (zo == 0) {
          // a = 0 / b   ->    a = 0
          auto id = factory.ConstantExpr(0);
          factory(initial_id).replace(factory(id));
          changed = true;
        } else {
          // a = b / 0   ->    error
          std::cout << "Warning division by zero detected! Line: "
                    << factory(initial_id).generate(factory.tmpExpressions)
                    << std::endl;
        }
      } else if (factory(initial_id).type == BINARY_MINUS && isZero) {
        if (zo == 0) {
          // a = 0 - b   ->    a = -b
          auto id =
              factory.UnaryMinusExpr(factory(initial_id).params[otherBinaryId]);
          factory(initial_id).replace(factory(id));
          changed = true;
        } else {
          // a = b - 0   ->    a = b
          auto id =
              factory.AssignExpr(factory(initial_id).params[otherBinaryId]);
          factory(initial_id).replace(factory(id));
          changed = true;
        }
      } else if (factory(initial_id).type == UNARY_MINUS && isZero) {
        // a = -0   ->   a = 0
        auto id = factory.ConstantExpr(0);
        factory(initial_id).replace(factory(id));
        changed = true;
      }
    }
  }
  return changed;
}

bool CodeGenerator::TrivialAssignmentElimination() {
  // eliminate assignments
  // a = b
  // c = a + d
  //
  // ->   c = b + d    (the variable a is removed)
  bool changed = false;
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
        if (p.id == target.id) {
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
    changed = true;
  }
  return changed;
}

bool CodeGenerator::ConstantFolding() {
  // returns true if all the paramters of an expressions are from type
  // CONSTANT
  auto allParametersConstants = [this](ExpressionData& expr) -> bool {
    for (int i = 0; i < expr.params.size(); ++i) {
      auto& pexpr = factory(expr.params[i]);
      if (!pexpr.compileTimeConstant()) return false;
    }
    return true;
  };

  std::vector<double> params;

  bool changed = false;
  // constant folding
  for (auto id : code) {
    if (allParametersConstants(factory(id))) {
      //        std::cout << "found full constant expressioins: "
      //                  << expr.generate(factory.tmpExpressions) <<
      //                  std::endl;

      // let's extract all paramters and put them into an array for easier
      // access
      params.clear();
      for (auto& p : factory(id).params) {
        params.push_back(factory(p).value);
      }

      switch (factory(id).type) {
        case BINARY_MINUS:
          factory(id).replace(
              factory(factory.ConstantExpr(params[0] - params[1])));
          changed = true;
          break;
        case BINARY_PLUS:
          factory(id).replace(
              factory(factory.ConstantExpr(params[0] + params[1])));
          changed = true;
          break;
        case BINARY_MULT:
          factory(id).replace(
              factory(factory.ConstantExpr(params[0] * params[1])));
          changed = true;
          break;
        case BINARY_DIV:
          factory(id).replace(
              factory(factory.ConstantExpr(params[0] / params[1])));
          changed = true;
          break;
        case UNARY_MINUS:
          factory(id).replace(factory(factory.ConstantExpr(-params[0])));
          changed = true;
          break;
        case FUNCTION_CALL: {
          // compile time evaluate some functions
          if (factory(id).functionName == "sin") {
            factory(id).replace(factory(factory.ConstantExpr(sin(params[0]))));
            changed = true;
          } else if (factory(id).functionName == "cos") {
            factory(id).replace(factory(factory.ConstantExpr(cos(params[0]))));
            changed = true;
          }
          // TODO: add more functions here
          break;
        }
        default:
          break;
      }
    }
  }
  return changed;
}

bool CodeGenerator::CommonSubexpressionElimination() {
  bool changed = false;
  for (int i = 0; i < code.size(); ++i) {
    auto id = code[i];

    // find an identical expression after the current expr.
    for (int j = i + 1; j < code.size(); ++j) {
      auto& other = factory(code[j]);
      if (factory(id).isRedundant(other)) {
        // replace with an assignment to first expression
        auto aexpr = factory.AssignExpr(id);
        factory(code[j]).replace(factory(aexpr));
        changed = true;
      }
    }
  }
  return changed;
}

void CodeGenerator::optimize() {
  DeadCodeRemoval();

  if (!settings.optimize) return;

  bool changed = true;
  // Optimize until convergence
  while (changed) {
    changed = false;
    if (settings.optimize_deadCode) DeadCodeRemoval();
    if (settings.optimize_zeroones) changed |= ZeroOnePropagation();
    if (settings.optimize_trivialAssignments)
      changed |= TrivialAssignmentElimination();
    if (settings.optimize_constants) changed |= ConstantFolding();
    if (settings.optimize_commonExpressionis)
      changed |= CommonSubexpressionElimination();
  }
}

void CodeGenerator::print(std::ostream& strm) {
  std::string indent = "  ";

  std::string finalName = returnValue + " " + settings.function_prefix + name;

  if (settings.addInlineKeyword) {
    finalName = "inline " + finalName;
  }

  strm << finalName << std::endl;
  strm << "{" << std::endl;
  strm << indent << "// This code is generated with ceres::AutoDiffCodeGen"
       << std::endl;
  strm << indent << "// See ceres/autodiff_codegen.h for more informations."
       << std::endl;
  for (auto f : code) {
    strm << indent << factory(f).generate(factory.tmpExpressions) << std::endl;
  }
  strm << indent << "return true;" << std::endl;
  strm << "}" << std::endl;
}

}  // namespace ceres
