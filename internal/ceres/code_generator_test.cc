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

#include "ceres/internal/code_generator.h"
#include "ceres/internal/expression_graph.h"
#include "ceres/internal/expression_ref.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

template <typename T>
void Square(const T* x, T* residual) {
  residual[0] = x[0] * x[0];
}

// Important:
//
// For know this "test" only contains a demo that helps me implementing and
// debugging. This will be replaced by an actual test in later stages of the
// patch.
TEST(CodeGenerator, Demo) {
  using T = ExpressionRef;

  StartRecordingExpressions();

  T x = MakeParameter("x[0]");
  T residual;
  Square<T>(&x, &residual);
  MakeOutput(residual, "residual[0]");

  auto graph = StopRecordingExpressions();

  CodeGenerator::Options options;
  options.header = "void SquareFunctor(const double* x, double* residual)";
  options.add_root_block = true;
  CodeGenerator gen(graph, options);

  std::cout << std::endl;
  gen.Print(std::cout);
  std::cout << std::endl;
}

TEST(CodeGenerator, DemoJet) {
  using T = Jet<ExpressionRef, 1>;

  StartRecordingExpressions();

  T x = T(sqrt(2.0));  // T(MakeParameter("x[0]"), 0);
  T residual;
  Square<T>(&x, &residual);

  MakeOutput(residual.a, "residual[0]");
  MakeOutput(residual.v[0], "jacobian[0][0]");

  auto graph = StopRecordingExpressions();

  CodeGenerator::Options options;
  options.header =
      "void SquareFunctorJet(const double* x, double* residual, double** "
      "jacobian)";
  options.add_root_block = true;
  CodeGenerator gen(graph, options);

  std::cout << std::endl;
  gen.Print(std::cout);
  std::cout << std::endl;
}

}  // namespace internal
}  // namespace ceres
