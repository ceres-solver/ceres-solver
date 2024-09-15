// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
// http://ceres-solver.org/
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
// Author: strandmark@google.com (Petter Strandmark)

#include "ceres/gradient_problem.h"

#include <memory>

#include "ceres/manifold.h"
#include "gtest/gtest.h"

namespace ceres::internal {

class QuadraticTestFunction : public ceres::FirstOrderFunction {
 public:
  explicit QuadraticTestFunction(bool* flag_to_set_on_destruction = nullptr)
      : flag_to_set_on_destruction_(flag_to_set_on_destruction) {}

  ~QuadraticTestFunction() override {
    if (flag_to_set_on_destruction_) {
      *flag_to_set_on_destruction_ = true;
    }
  }

  bool Evaluate(const double* parameters,
                double* cost,
                double* gradient) const final {
    const double x = parameters[0];
    cost[0] = x * x;
    if (gradient != nullptr) {
      gradient[0] = 2.0 * x;
    }
    return true;
  }

  int NumParameters() const final { return 1; }

 private:
  bool* flag_to_set_on_destruction_;
};

TEST(GradientProblem, TakesOwnershipOfFirstOrderFunction) {
  bool is_destructed = false;
  {
    ceres::GradientProblem problem(
        std::make_unique<QuadraticTestFunction>(&is_destructed));
  }
  EXPECT_TRUE(is_destructed);
}

TEST(GradientProblem, EvaluationWithManifoldAndNoGradient) {
  ceres::GradientProblem problem(std::make_unique<QuadraticTestFunction>(),
                                 std::make_unique<EuclideanManifold<1>>());
  double x = 7.0;
  double cost = 0;
  problem.Evaluate(&x, &cost, nullptr);
  EXPECT_EQ(x * x, cost);
}

TEST(GradientProblem, EvaluationWithoutManifoldAndWithGradient) {
  ceres::GradientProblem problem(std::make_unique<QuadraticTestFunction>());
  double x = 7.0;
  double cost = 0;
  double gradient = 0;
  problem.Evaluate(&x, &cost, &gradient);
  EXPECT_EQ(2.0 * x, gradient);
}

TEST(GradientProblem, EvaluationWithManifoldAndWithGradient) {
  ceres::GradientProblem problem(std::make_unique<QuadraticTestFunction>(),
                                 std::make_unique<EuclideanManifold<1>>());
  double x = 7.0;
  double cost = 0;
  double gradient = 0;
  problem.Evaluate(&x, &cost, &gradient);
  EXPECT_EQ(2.0 * x, gradient);
}

}  // namespace ceres::internal
