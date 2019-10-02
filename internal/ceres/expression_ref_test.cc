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

#include "ceres/internal/expression_ref.h"
#include "ceres/internal/expression_graph.h"

#include "gtest/gtest.h"

namespace ceres {
namespace internal {

template <typename T>
void test_plus(T* x, T* y, T* result) {
  result[0] = x[0] + y[0];
}

// This function is only for demonstration.
// After a simple code-gen is done this will be generated to a new file from the
// functor above.
void test_plus_generated(double* x, double* y, double* result) {
  const double v_0 = x[0];
  const double v_1 = y[0];
  const double v_2 = v_0 + v_1;
  result[0] = v_2;
}

TEST(ExpressionRef, Operators) {
  double x = 5, y = 8;
  double result_ref = 0, result_gen = 0;

  test_plus(&x, &y, &result_ref);
  test_plus_generated(&x, &y, &result_gen);
  EXPECT_EQ(result_ref, result_gen);
}

}  // namespace internal
}  // namespace ceres
