
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

#include "ceres/internal/expression.h"

namespace ceres {
CodeFactory* Expression::codeFactory = nullptr;

bool PhiBlock::leftReferenced(Expression expr) const {
  // check if it's one of the parameters
  for (auto& p : parameters) {
    if (p.first.id == expr.id) return true;
  }
  // check if a local expression references it
  for (auto& p : localExpressionsTrue) {
    std::cout << p.id << " = ";
    for (auto& p2 : (*Expression::codeFactory)(p).params) {
      std::cout << p2.id << " ";
      if (expr.id == p2.id) return true;
    }
    std::cout << std::endl;
  }
  return false;
}

bool PhiBlock::rightReferenced(Expression expr) const {
  // check if it's one of the parameters
  for (auto p : parameters) {
    if (p.second.id == expr.id) return true;
  }
  // check if a local expression references it
  for (auto p : localExpressionsFalse) {
    for (auto p2 : (*Expression::codeFactory)(p).params) {
      if (expr.id == p2.id) return true;
    }
  }
  return false;
}

bool ExpressionData::zeroOnePropagateable() {
  switch (type) {
    case ExpressionType::PLUS:
    case ExpressionType::MULT:
    case ExpressionType::DIV:
    case ExpressionType::MINUS:
    case ExpressionType::UNARY_MINUS:
      return true;
    default:
      return false;
  }
}

bool ExpressionData::isRedundant(const ExpressionData& other) {
  if (type == ExpressionType::NOP) return false;

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

bool ExpressionData::references(Expression other) {
  for (auto p : params) {
    if (p.id == other.id) return true;
  }
  return false;
}

void ExpressionData::replace(const ExpressionData& other) {
  auto currentId = id;
  auto currentTarget = targetId;
  (*this) = other;
  id = currentId;
  targetId = currentTarget;
}

std::string ExpressionData::resultType() const {
  if (type == ExpressionType::BINARY_COMPARE) return "const bool";
  return "const double";
}

bool ExpressionData::compileTimeConstant() {
  return type == ExpressionType::CONSTANT && !externalConstant;
}

bool ExpressionData::isZeroOne() {
  return compileTimeConstant() && (value == 0 || value == 1);
}

void ExpressionData::makeNOP() {
  type = ExpressionType::NOP;
  params.clear();
}

std::string ExpressionData::name() const {
  return "v_" + std::to_string(id.id);
}

void ExpressionData::check() const {
  assert(N == params.size());
  assert(type != ExpressionType::INVALID);
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
    case ExpressionType::ASSIGN:
      res += code[params[0].id].name() + ";";
      return res;
    case ExpressionType::CONSTANT:
      if (externalConstant) {
        res += functionName + ";";
      } else {
        res += double_to_string_precise(value, 20) + ";";
      }
      return res;
    case ExpressionType::PARAMETER:
      res += functionName + ";";
      return res;
    case ExpressionType::OUTPUT_ASSIGN:
      return functionName + " = " + code[params[0].id].name() + ";";

    case ExpressionType::PLUS:
      binaryOperator = "+";
      break;
    case ExpressionType::MULT:
      binaryOperator = "*";
      break;
    case ExpressionType::MINUS:
      binaryOperator = "-";
      break;
    case ExpressionType::DIV:
      binaryOperator = "/";
      break;

    case ExpressionType::BINARY_COMPARE:
      binaryOperator = functionName;
      break;
    case ExpressionType::UNARY_MINUS:
      res += "-";
      break;
    case ExpressionType::FUNCTION_CALL:
      res += functionName;
      break;
    case ExpressionType::PHI_FUNCTION:
      res += "ceres::PHI";
      break;
    case ExpressionType::PHI_BLOCK: {
      // This generates the following code
      //
      //  double target_0, target_1,....;
      //  if(condition)
      //  {
      //      <local_true_0>;
      //      <local_true_1>;
      //      ...
      //      target_0 = parameters_0_true
      //      target_1 = parameters_1_true
      //      ...
      //  }else{
      //      <local_false_0>;
      //      <local_false_1>;
      //      ...
      //      target_0 = parameters_0_false
      //      target_1 = parameters_1_false
      //      ...
      //  }
      //
      std::string ident = "  ";
      std::string declaration = "double ";
      for (int i = 0; i < phiBlock.N; ++i) {
        declaration += code[phiBlock.targets[i].id].name();
        if (i < phiBlock.N - 1) {
          declaration += ", ";
        }
      }
      declaration += ";\n";

      std::string conditionStart =
          ident + "if(" + code[phiBlock.condition.id].name() + ")\n";
      conditionStart += ident + "{ \n";

      std::string localTrue;
      for (auto l : phiBlock.localExpressionsTrue) {
        localTrue += ident + ident + code[l.id].generate(code) + "\n";
        assert(!code[l.id].isPhi());
      }

      // target assignemnts for "true" case
      std::string targetTrue;
      for (int i = 0; i < phiBlock.N; ++i) {
        targetTrue += ident + ident + code[phiBlock.targets[i].id].name() +
                      " = " + code[phiBlock.parameters[i].first.id].name() +
                      ";\n";
      }

      std::string conditionElse = ident + "}else{\n";

      std::string localFalse;
      for (auto l : phiBlock.localExpressionsFalse) {
        localFalse += ident + ident + code[l.id].generate(code) + "\n";
        assert(!code[l.id].isPhi());
      }

      std::string targetFalse;
      for (int i = 0; i < phiBlock.N; ++i) {
        targetFalse += ident + ident + code[phiBlock.targets[i].id].name() +
                       " = " + code[phiBlock.parameters[i].second.id].name() +
                       ";\n";
      }

      std::string conditionEnd = ident + "}";

      return declaration + conditionStart + localTrue + targetTrue +
             conditionElse + localFalse + targetFalse + conditionEnd;
    }
    case ExpressionType::NOP:
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

Expression CodeFactory::ConstantExpr(double v) {
  return Expr(ExpressionType::CONSTANT, 0, v);
}

Expression CodeFactory::ParameterExpr(int v, const std::string& name) {
  auto expr = Expr(ExpressionType::PARAMETER, 0, v);
  ((*this)(expr)).functionName = name;
  return expr;
}

Expression CodeFactory::AssignExpr(Expression v) {
  auto expr = Expr(ExpressionType::ASSIGN, 1);
  ((*this)(expr)).params.push_back(v);
  return expr;
}

Expression CodeFactory::UnaryMinusExpr(Expression v) {
  auto expr = Expr(ExpressionType::UNARY_MINUS, 1);
  ((*this)(expr)).params.push_back(v);
  return expr;
}

Expression CodeFactory::OutputAssignExpr(Expression v,
                                         const std::string& name) {
  auto expr = Expr(ExpressionType::OUTPUT_ASSIGN, 1);
  ((*this)(expr)).params.push_back(v);
  ((*this)(expr)).functionName = name;
  return expr;
}

Expression CodeFactory::FunctionExpr(const std::string& name,
                                     std::vector<Expression> params) {
  auto expr = Expr(ExpressionType::FUNCTION_CALL, params.size());
  ((*this)(expr)).params = params;
  ((*this)(expr)).functionName = name;
  return expr;
}

Expression CodeFactory::Phi(Expression c, Expression a, Expression b) {
  auto expr = Expr(ExpressionType::PHI_FUNCTION, 3);
  ((*this)(expr)).params.push_back(c);
  ((*this)(expr)).params.push_back(a);
  ((*this)(expr)).params.push_back(b);
  return expr;
}

Expression CodeFactory::PhiBlockExpr(const PhiBlock& block) {
  auto expr = Expr(ExpressionType::PHI_BLOCK, 0);
  ((*this)(expr)).phiBlock = block;
  return expr;
}

Expression CodeFactory::BinaryCompare(const std::string& name,
                                      Expression l,
                                      Expression r) {
  auto expr = Expr(ExpressionType::BINARY_COMPARE, 2);
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
  for (int i = 0; i < expression_data.size(); ++i) {
    auto& expr = expression_data[i];
    expr.check();
    assert(i == expr.id.id);
  }
}

Expression CodeFactory::Expr(ExpressionType type, int N, double value) {
  auto id = expression_data.size();
  Expression e;
  e.id = id;
  ExpressionData expr(type, N, e, value);
  expression_data.push_back(expr);
  return e;
}

}  // namespace ceres
