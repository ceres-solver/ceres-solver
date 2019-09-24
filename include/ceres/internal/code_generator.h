
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
#ifndef CERES_PUBLIC_CODE_GENERATOR_H_
#define CERES_PUBLIC_CODE_GENERATOR_H_

#include "Eigen/Core"
#include "ceres/internal/port.h"
#include "expression.h"

#include <cmath>
#include <iosfwd>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace ceres {

struct CodeGenerationOptions {
  bool addInlineKeyword = true;

  // this string we be added to the function names.
  // usefull if you want to generate multiple functors into the same file.
  std::string function_prefix = "";

  // If this is false, none of the optimizations below will be executed
  bool optimize = true;

  bool removeUnusedParameters = false;
  bool removeDeadCode = true;
  bool propagateZeroOnes = true;
  bool removeTrivialAssignments = true;
  bool foldConstants = true;
  bool eliimiateCommonSubeEpressions = true;
  bool mergePhiBlocks = true;
};

struct CodeGenerator {
  // Local copy here so we can modify the code.
  CodeFactory factory;

  // name of the function
  std::string name = "foo";
  std::string returnValue = "bool";

  std::vector<Expression> targets;

  CodeGenerator(const CodeGenerationOptions& settings) : settings(settings) {}

  void print(std::ostream& strm);

 private:
  CodeGenerationOptions settings;
  // final code indexing into the expression array of factory
  std::vector<Expression> code;

  // traverse through the expression tree from the target node.
  // all expressions found will be added to the output.
  std::vector<Expression> traverseAndCollect(Expression target);

  void generate();

  void DeadCodeRemoval();

  bool ZeroOnePropagation();

  bool TrivialAssignmentElimination();

  bool ConstantFolding();

  bool CommonSubexpressionElimination();

  void PhiBlockMerging();

  // returns all expressions which depend on 'expr'
  std::vector<std::vector<Expression>> computeInverseDependencies();

  void optimize();
};

}  // namespace ceres

#endif
