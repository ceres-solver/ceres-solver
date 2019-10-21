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
#ifndef CERES_PUBLIC_CODE_GENERATOR_H_
#define CERES_PUBLIC_CODE_GENERATOR_H_

#include "expression_graph.h"

namespace ceres {
namespace internal {

// This class is used to convert an expression graph into a string. The typical
// pipeline is:
//
// 1. Record ExpressionGraph
// 2. Optimize ExpressionGraph
// 3. Generate code (this class here)
class CodeGenerator {
 public:
  // Initialize the code generator. This creates a copy of 'graph'.
  // TODO: Not sure if we need a copy.
  CodeGenerator(const ExpressionGraph& graph);

  void Print(std::ostream& strm);

 private:
  // The following format is used for converting expressions to strings.
  // <LhsType> <LhsName> <Operator> <Rhs>
  //
  // double    v_1       =          sin(v_0)
  // bool      v_2       =          v_1 < v_0
  //           v_2       =          v_0
  //                     if         (v_7)
  std::string ToString(ExpressionId id);

  std::string LhsTypeString(const Expression& expr);
  std::string LhsNameString(const Expression& expr);
  std::string OperatorString(const Expression& expr);
  std::string RhsString(const Expression& expr);

  // Returns the lhs name and checks if it's valid.
  std::string LhsNameStringCheck(ExpressionId id);
  std::string ValueToString(double value);

  ExpressionGraph graph_;
};

}  // namespace internal
}  // namespace ceres
#endif
