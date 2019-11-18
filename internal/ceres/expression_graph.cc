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

#include "ceres/internal/expression_graph.h"
#include <iostream>
#include "glog/logging.h"
namespace ceres {
namespace internal {

static ExpressionGraph* expression_pool = nullptr;

void StartRecordingExpressions() {
  CHECK(expression_pool == nullptr)
      << "Expression recording must be stopped before calling "
         "StartRecordingExpressions again.";
  expression_pool = new ExpressionGraph;
}

ExpressionGraph StopRecordingExpressions() {
  CHECK(expression_pool)
      << "Expression recording hasn't started yet or you tried "
         "to stop it twice.";
  ExpressionGraph result = std::move(*expression_pool);
  delete expression_pool;
  expression_pool = nullptr;
  return result;
}

ExpressionGraph* GetCurrentExpressionGraph() { return expression_pool; }

Expression& ExpressionGraph::CreateArithmeticExpression(ExpressionType type,
                                                        ExpressionId lhs_id) {
  if (lhs_id == kInvalidExpressionId) {
    // We are creating a new temporary variable.
    // -> The new lhs_id is the index into the graph
    lhs_id = static_cast<ExpressionId>(expressions_.size());
  } else {
    // The left hand side already exists.
  }

  Expression expr(type, lhs_id);
  expressions_.push_back(expr);
  return expressions_.back();
}

Expression& ExpressionGraph::CreateControlExpression(ExpressionType type) {
  Expression expr(type, kInvalidExpressionId);
  expressions_.push_back(expr);
  return expressions_.back();
}

bool ExpressionGraph::DependsOn(ExpressionId A, ExpressionId B) const {
  // Depth first search on the expression graph
  // Equivalent Recursive Implementation:
  //   if (A.DirectlyDependsOn(B)) return true;
  //   for (auto p : A.params_) {
  //     if (pool[p.id].DependsOn(B, pool)) return true;
  //   }
  std::vector<ExpressionId> stack = ExpressionForId(A).arguments_;
  while (!stack.empty()) {
    auto top = stack.back();
    stack.pop_back();
    if (top == B) {
      return true;
    }
    auto& expr = ExpressionForId(top);
    stack.insert(stack.end(), expr.arguments_.begin(), expr.arguments_.end());
  }
  return false;
}

bool ExpressionGraph::IsValid() const {
  // 1. All dependencies are backwards
  for (ExpressionId id = 0; id < Size(); ++id) {
    const auto& expr = ExpressionForId(id);

    // The left hand side must be backwards.
    // id == lhs_id is allowed.
    if (expr.lhs_id() > id) {
      LOG(ERROR) << "ExpressionGraph::IsValid failed. Expression " << id
                 << " has a forward pointing lhs_id (" << expr.lhs_id() << ")";
      return false;
    }

    // All arguments must point backwards.
    // id == arg is not allowed.
    for (auto arg : expr.arguments()) {
      if (arg >= id) {
        LOG(ERROR) << "ExpressionGraph::IsValid failed. Expression " << id
                   << " has a forward pointing argument (" << arg << ")";
        return false;
      }
    }
  }

  // 2. All lhs_ids point to declared variables.
  for (ExpressionId id = 0; id < Size(); ++id) {
    const auto lhs = ExpressionForId(id).lhs_id();

    if (lhs == kInvalidExpressionId) {
      continue;
    }

    const auto& lhs_expr = ExpressionForId(lhs);

    if (lhs != lhs_expr.lhs_id()) {
      LOG(ERROR) << "ExpressionGraph::IsValid failed. Expression " << id
                 << " writes to an undeclared variable (" << lhs << ")";
      return false;
    }
  }

  // 3. Matching If/Else/Endif expressions
  std::vector<ExpressionType> stack;
  for (ExpressionId id = 0; id < Size(); ++id) {
    const auto& expr = ExpressionForId(id);

    if (expr.type() == ExpressionType::IF) {
      stack.push_back(ExpressionType::IF);
    } else if (expr.type() == ExpressionType::ELSE) {
      // The top element must be an IF
      if (!stack.empty() && stack.back() == ExpressionType::IF) {
        stack.push_back(ExpressionType::ELSE);
      } else {
        LOG(ERROR) << "ExpressionGraph::IsValid failed. ELSE must be preceeded "
                      "by IF. ExpressionId: "
                   << id;
        return false;
      }
    } else if (expr.type() == ExpressionType::ENDIF) {
      // The top element must be either IF or ELSE.
      if (!stack.empty() && stack.back() == ExpressionType::IF) {
        // consume this if
        stack.pop_back();
      } else if (!stack.empty() && stack.back() == ExpressionType::ELSE) {
        // consume else and then consume if
        stack.pop_back();
        if (!stack.empty() && stack.back() == ExpressionType::IF) {
          stack.pop_back();
        } else {
          LOG(ERROR)
              << "ExpressionGraph::IsValid failed. ELSE must be preceeded "
                 "by IF. ExpressionId: "
              << id;
          return false;
        }
      } else {
        LOG(ERROR)
            << "ExpressionGraph::IsValid failed. ENDIF must be preceeded "
               "by IF or ELSE. ExpressionId: "
            << id;
        return false;
      }
    }
  }
  // All IF/ELSE have to be consumed.
  if (!stack.empty()) {
    LOG(ERROR) << "ExpressionGraph::IsValid failed. Missing ENDIF";
    return false;
  }

  return true;
}

bool ExpressionGraph::IsEquivalentTo(const ExpressionGraph& other) const {
  if (Size() != other.Size()) {
    return false;
  }
  for (ExpressionId id = 0; id < Size(); ++id) {
    if (!ExpressionForId(id).IsEquivalentTo(other.ExpressionForId(id))) {
      return false;
    }
  }
  return true;
}

bool ExpressionGraph::IsSemanticallyEquivalentTo(
    const ExpressionGraph& other) const {
  if (!IsValid() || !other.IsValid()) {
    LOG(ERROR) << "ExpressionGraph::IsSemanticallyEquivalentTo failed. Both "
                  "graphs must be valid ";
    return false;
  }

  // Collect all output expressions
  std::vector<ExpressionId> output1, output2;
  for (ExpressionId id = 0; id < Size(); ++id) {
    const auto& expr = ExpressionForId(id);
    if (expr.type() == ExpressionType::OUTPUT_ASSIGNMENT) {
      output1.push_back(id);
    }
  }
  for (ExpressionId id = 0; id < other.Size(); ++id) {
    const auto& expr = other.ExpressionForId(id);
    if (expr.type() == ExpressionType::OUTPUT_ASSIGNMENT) {
      output2.push_back(id);
    }
  }
  if (output1.size() != output2.size()) {
    LOG(ERROR) << "ExpressionGraph::IsSemanticallyEquivalentTo failed. Both "
                  "graphs must have the same number of output expressions "
               << output1.size() << " != " << output2.size();
    return false;
  }
  // Now test if they assign to the same variable in the same order. The order
  // must be identical, because we assume possible side-effect of this
  // expression.
  for (size_t i = 0; i < output1.size(); ++i) {
    const auto& expr1 = ExpressionForId(output1[i]);
    const auto& expr2 = other.ExpressionForId(output2[i]);
    if (expr1.name() != expr2.name()) {
      LOG(ERROR) << "ExpressionGraph::IsSemanticallyEquivalentTo failed. The "
                    "output expression are either in incorrect order or they "
                    "assign to different variables "
                 << expr1.name() << " != " << expr2.name();
      return false;
    }
  }

  // Traverse both graphs simultaneously and test if all encountered expressions
  // are semantically equivalent.
  // The vectors output1 and output2 are now used as stacks.
  while (!output1.empty() && !output2.empty()) {
    // Take top expression from the stack.
    ExpressionId id1 = output1.back();
    ExpressionId id2 = output2.back();
    output1.pop_back();
    output2.pop_back();

    if (id1 == kInvalidExpressionId && id2 == kInvalidExpressionId) {
      // Both nodes beeing invalid is ok.
      continue;
    } else if (id1 == kInvalidExpressionId || id2 == kInvalidExpressionId) {
      // only one being invalid is an error.
      return false;
    }

    const auto& expr1 = ExpressionForId(id1);
    const auto& expr2 = other.ExpressionForId(id2);

    if (!expr1.IsSemanticallyEquivalentTo(expr2)) {
      return false;
    }

    // All pervious variables these expressions depend on
    std::vector<ExpressionId> past_dependencies1 = expr1.arguments();
    std::vector<ExpressionId> past_dependencies2 = expr2.arguments();

    // Push lhs and all previous writes to if it is a backwards reference
    if (expr1.HasValidLhs() && id1 != expr1.lhs_id()) {
      past_dependencies1.push_back(expr1.lhs_id());
    }
    if (expr2.HasValidLhs() && id2 != expr2.lhs_id()) {
      past_dependencies2.push_back(expr2.lhs_id());
    }

    // Add dependencies
    output1.insert(
        output1.end(), past_dependencies1.begin(), past_dependencies1.end());
    output2.insert(
        output2.end(), past_dependencies2.begin(), past_dependencies2.end());

    // Add assignments to the dependencies that are between the declaration and
    // the current expression
    for (auto dep_id : past_dependencies1) {
      for (auto id = dep_id + 1; id < id1; ++id) {
        if (ExpressionForId(id).lhs_id() == dep_id) {
          output1.push_back(id);
          //          std::cout << "1:  " << id << " dep " << dep_id <<
          //          std::endl;
        }
      }
    }
    for (auto dep_id : past_dependencies2) {
      for (auto id = dep_id + 1; id < id2; ++id) {
        if (other.ExpressionForId(id).lhs_id() == dep_id) {
          output2.push_back(id);
          //          std::cout << "2:  " << id << " dep " << dep_id <<
          //          std::endl;
        }
      }
    }

    // Push the parent
    output1.push_back(GetParentControlExpression(id1));
    output2.push_back(other.GetParentControlExpression(id2));
    //    std::cout << "parent of " << id1 << " is " << output1.back()
    //              << " parent of " << id2 << " is " << output2.back() <<
    //              std::endl;
  }

  return true;
}  // namespace internal

ExpressionId ExpressionGraph::GetParentControlExpression(
    ExpressionId id) const {
  // Traverse upwards and find first matching IF/ELSE
  for (ExpressionId i = id - 1; i >= 0; --i) {
    const auto& expr = ExpressionForId(i);

    if (expr.type() == ExpressionType::ENDIF) {
      // Found a nested block
      // -> Jump over it
      i = FindMatchingIf(i);
    } else if (expr.type() == ExpressionType::IF ||
               expr.type() == ExpressionType::ELSE) {
      // Found the parent!
      return i;
    }
  }

  return kInvalidExpressionId;
}

ExpressionId ExpressionGraph::FindMatchingIf(ExpressionId id) const {
  auto initial_type = ExpressionForId(id).type();
  CHECK(initial_type == ExpressionType::ELSE ||
        initial_type == ExpressionType::ENDIF)
      << "FindMatchingIf is only valid on ELSE or ENDIF expressions.";

  // Traverse upwards and find first matching IF/ELSE
  for (ExpressionId i = id - 1; i >= 0; --i) {
    const auto& expr = ExpressionForId(i);
    if (expr.type() == ExpressionType::IF) {
      // Found it! We are done!
      return i;
    } else if (expr.type() == ExpressionType::ELSE) {
      if (initial_type == ExpressionType::ELSE) {
        // We found two ELSE in a row. This is actually an error and should not
        // happen.
        CHECK(false) << "IF-ELSE-ENDIF missmatch detected.";
        return kInvalidExpressionId;
      }
      // We found an else but we are looking for the matching IF
      // -> Just look for the matching if to this else
      initial_type = ExpressionType::ELSE;
      continue;
    } else if (expr.type() == ExpressionType::ENDIF) {
      // We found another endif. That means there is another if/else block
      // nested into this one.
      // -> Find the corresponding if recursively and continue iterating from
      // there
      auto matching_if = FindMatchingIf(i);
      if (matching_if == kInvalidExpressionId) {
        return kInvalidExpressionId;
      }
      i = matching_if;
      continue;
    }
  }
  CHECK(false) << "IF-ELSE-ENDIF missmatch detected.";
  return kInvalidExpressionId;
}

void ExpressionGraph::InsertExpression(
    ExpressionId location,
    ExpressionType type,
    ExpressionId lhs_id,
    const std::vector<ExpressionId>& arguments,
    const std::string& name,
    double value) {
  ExpressionId last_expression_id = Size() - 1;

  // Increase size by adding a dummy expression.
  expressions_.push_back(Expression(ExpressionType::NOP, kInvalidExpressionId));

  // Move everything after id back and update references
  for (ExpressionId id = last_expression_id; id >= location; --id) {
    expressions_[id + 1] = expressions_[id];

    if (expressions_[id + 1].lhs_id_ >= location) {
      expressions_[id + 1].lhs_id_++;
    }

    for (auto& arg : expressions_[id + 1].arguments_) {
      if (arg >= location) {
        arg++;
      }
    }
  }

  // Insert new expression at the correct place
  Expression expr(type, lhs_id);
  expr.arguments_ = arguments;
  expr.name_ = name;
  expr.value_ = value;
  expressions_[location] = expr;
}
}  // namespace internal
}  // namespace ceres
