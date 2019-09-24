
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

#include "ceres/internal/code_generator.h"

namespace ceres {

std::vector<Expression> CodeGenerator::traverseAndCollect(Expression target) {
  std::vector<Expression> code;

  std::vector<Expression> stack;
  // start at the root node of the tree
  stack.push_back(target);

  // depth first search
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
  return code;
}

void CodeGenerator::DeadCodeRemoval() {
  code.clear();

  for (auto& expr : factory.expression_data) {
    expr.tmp = 0;
  }

  if (!settings.removeUnusedParameters) {
    // add all parameter expressions
    for (auto& expr : factory.expression_data) {
      if (expr.type == ExpressionType::PARAMETER) {
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

      if (factory(initial_id).type == ExpressionType::MULT && isZero) {
        // a = 0 * b   ->   a = 0
        auto id = factory.ConstantExpr(0);
        factory(initial_id).replace(factory(id));
        changed = true;
      } else if (factory(initial_id).type == ExpressionType::MULT && !isZero) {
        // a = 1 * b   ->    a = b
        auto id = factory.AssignExpr(factory(initial_id).params[otherBinaryId]);
        factory(initial_id).replace(factory(id));
        changed = true;
      } else if (factory(initial_id).type == ExpressionType::PLUS && isZero) {
        // a = 0 + b   ->    a = b
        auto id = factory.AssignExpr(factory(initial_id).params[otherBinaryId]);
        factory(initial_id).replace(factory(id));
        changed = true;
      } else if (factory(initial_id).type == ExpressionType::DIV && isZero) {
        if (zo == 0) {
          // a = 0 / b   ->    a = 0
          auto id = factory.ConstantExpr(0);
          factory(initial_id).replace(factory(id));
          changed = true;
        } else {
          // a = b / 0   ->    error
          std::cout << "Warning division by zero detected! Line: "
                    << factory(initial_id).generate(factory.expression_data)
                    << std::endl;
        }
      } else if (factory(initial_id).type == ExpressionType::MINUS && isZero) {
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
      } else if (factory(initial_id).type == ExpressionType::UNARY_MINUS &&
                 isZero) {
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
    if (expr.type != ExpressionType::ASSIGN) continue;

    auto target = id;
    auto src = expr.params[0];

    // go over code and find expressions with 'target' as a paramter and
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
    expr.type = ExpressionType::NOP;
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
        case ExpressionType::MINUS:
          factory(id).replace(
              factory(factory.ConstantExpr(params[0] - params[1])));
          changed = true;
          break;
        case ExpressionType::PLUS:
          factory(id).replace(
              factory(factory.ConstantExpr(params[0] + params[1])));
          changed = true;
          break;
        case ExpressionType::MULT:
          factory(id).replace(
              factory(factory.ConstantExpr(params[0] * params[1])));
          changed = true;
          break;
        case ExpressionType::DIV:
          factory(id).replace(
              factory(factory.ConstantExpr(params[0] / params[1])));
          changed = true;
          break;
        case ExpressionType::UNARY_MINUS:
          factory(id).replace(factory(factory.ConstantExpr(-params[0])));
          changed = true;
          break;
        case ExpressionType::FUNCTION_CALL: {
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

void CodeGenerator::PhiBlockMerging() {
  std::vector<PhiBlock> phiBlocks;

  int state = 0;

  PhiBlock currentBlock;
  for (int i = 0; i < code.size(); ++i) {
    auto id = code[i];
    if (factory(id).type == ExpressionType::PHI_FUNCTION) {
      if (state == 0) {
        // new block - none is active
        currentBlock = PhiBlock();
        currentBlock.codeStart = i;
        currentBlock.condition = factory(id).params[0];
        state = 1;
      } else if (state == 1) {
        if (currentBlock.condition.id != factory(id).params[0].id) {
          // a neigbour block but with a different condition
          // -> start new one
          currentBlock.codeEnd = i;
          phiBlocks.push_back(currentBlock);
          currentBlock = PhiBlock();
          currentBlock.codeStart = i;
          currentBlock.condition = factory(id).params[0];
        }
      }
      currentBlock.targets.push_back(factory(id).id);
      currentBlock.parameters.emplace_back(factory(id).params[1],
                                           factory(id).params[2]);
    } else {
      if (state == 1) {
        // we are currently collecting phi functions but we encountered a
        // different expression
        // -> finish block
        currentBlock.codeEnd = i;
        phiBlocks.push_back(currentBlock);
        state = 0;
      }
    }
  }

  for (auto& pb : phiBlocks) {
    pb.N = pb.parameters.size();
  }

#if 0
  // print blocks
  for (auto pb : phiBlocks) {
    std::cout << "block" << std::endl;
    std::cout << "condition " << pb.condition.id << std::endl;
    std::cout << "params" << std::endl;
    for (auto d : pb.parameters)
      std::cout << d.first.id << " " << d.second.id << std::endl;
    std::cout << "targetes" << std::endl;
    for (auto d : pb.targets) std::cout << d.id << std::endl;
  }
#endif

  // add expressions to local
  // can be disabled for debugging purposes
#if 1
  for (auto& pb : phiBlocks) {
    auto codeAfter = pb.codeEnd;
    // start one instruction before the block
    // go backwards to previous
    for (int i = pb.codeStart - 1; i >= 0; --i) {
      auto eid = code[i];

      // don't construct nested phi blocks
      if (factory(eid).isPhi()) continue;

      bool foundReference = false;
      // let's check if this expression referenced by any instruction before the
      // phi block. this also makes it invalid because we go upwards
      for (auto j = i + 1; j < pb.codeStart; ++j) {
        auto& expr = factory(code[j]);
        if (!expr.shouldPrint()) continue;
        if (expr.references(eid)) {
          foundReference = true;
          break;
        }
      }

      // let's check if this expression referenced by any instruction ofter the
      // phi block
      for (auto j = codeAfter; j < code.size(); ++j) {
        auto& expr = factory(code[j]);
        if (expr.references(eid)) {
          foundReference = true;
          break;
        }
      }

      if (foundReference) continue;

      // So 'eid' is only reference by the phi block
      // Now do one of the following
      // 1. eid is referenced by true AND false branch -> do nothing
      // 2. eid is referenced by true branch -> move to true branch
      // 3. eid is referenced by false branch -> move to false branch

      bool lr = false;
      for (auto e : pb.parameters) {
        if (e.first.id == eid.id) lr = true;
      }
      for (auto& p : pb.localExpressionsTrue) {
        for (auto& p2 : factory(p).params) {
          if (eid.id == p2.id) lr = true;
        }
      }

      bool rr = false;
      for (auto e : pb.parameters) {
        if (e.second.id == eid.id) rr = true;
      }

      for (auto& p : pb.localExpressionsFalse) {
        for (auto& p2 : factory(p).params) {
          if (eid.id == p2.id) rr = true;
        }
      }

      if (lr && rr) {
        // referenced in both branches -> do nothing
      } else if (lr) {
        factory(eid).local = true;
        pb.localExpressionsTrue.push_back(eid);
      } else if (rr) {
        factory(eid).local = true;
        pb.localExpressionsFalse.push_back(eid);
      } else {
        // not refernced at all -> do nothing
      }
    }

    std::reverse(pb.localExpressionsTrue.begin(),
                 pb.localExpressionsTrue.end());
    std::reverse(pb.localExpressionsFalse.begin(),
                 pb.localExpressionsFalse.end());
  }
#endif
  for (auto pb : phiBlocks) {
    // replace first expression by phi block expr
    assert(pb.parameters.size() > 0);

    auto blockExpr = factory.PhiBlockExpr(pb);
    factory(pb.targets[0]).replace(factory(blockExpr));

    // make the remaining expressions to nops
    for (int i = 1; i < pb.N; ++i) {
      factory(pb.targets[i]).type = ExpressionType::NOP;  // makeNOP();
    }
  }
}

std::vector<std::vector<Expression> >
CodeGenerator::computeInverseDependencies() {
  std::vector<std::vector<Expression> > data(factory.expression_data.size());

  for (int i = 0; i < code.size(); ++i) {
    auto id = code[i];
    if (factory(id).type == ExpressionType::NOP) continue;
    for (auto dep : factory(id).params) {
      data[dep.id].push_back(id);
    }
  }
  return data;
}

void CodeGenerator::optimize() {
  DeadCodeRemoval();

  if (!settings.optimize) return;

  bool changed = true;
  // Optimize until convergence
  while (changed) {
    changed = false;
    if (settings.removeDeadCode) DeadCodeRemoval();
    if (settings.propagateZeroOnes) changed |= ZeroOnePropagation();
    if (settings.removeTrivialAssignments)
      changed |= TrivialAssignmentElimination();
    if (settings.foldConstants) changed |= ConstantFolding();
    if (settings.eliimiateCommonSubeEpressions)
      changed |= CommonSubexpressionElimination();
  }

  if (settings.mergePhiBlocks) PhiBlockMerging();
}

void CodeGenerator::print(std::ostream& strm) {
  generate();

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
    if (factory(f).shouldPrint())
      strm << indent << factory(f).generate(factory.expression_data)
           << std::endl;
  }
  strm << indent << "return true;" << std::endl;
  strm << "}" << std::endl;
}

}  // namespace ceres
