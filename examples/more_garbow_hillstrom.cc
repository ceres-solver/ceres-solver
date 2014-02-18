// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2014 Google Inc. All rights reserved.
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
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// Bounds constrained test problems from the paper
//
// Testing Unconstrained Optimization Software
// Jorge J. More, Burton S. Garbow and Kenneth E. Hillstrom
// ACM Transactions on Mathematical Software, 7(1), pp. 17-41, 1981
//
// A subset of these problems were augmented with bounds and used for
// testing bounds constrained optimization algorithms by
//
// A Trust Region Approach to Linearly Constrained Optimization
// David M. Gay
// Numerical Analysis (Griffiths, D.F., ed.), pp. 72-105
// Lecture Notes in Mathematics 1066, Springer Verlag, 1984.
//
// The latter paper is behind a paywall. We obtained the bounds on the
// variables and the function values at the global minimums from
//
// http://www.mat.univie.ac.at/~neum/glopt/bounds.html
//
// A problem is considered solved if of the log relative error of its
// objective function is at least 5.


#include <cmath>
#include <iostream>
#include "ceres/ceres.h"

namespace ceres {
namespace examples {

#define BEGIN_BOUNDS_TEST(name, num_parameters, num_residuals)          \
  struct name {                                                         \
    static const int kNumParameters = num_parameters;                   \
    static const double initial_x[kNumParameters];                      \
    static const double lower_bounds[kNumParameters];                   \
    static const double upper_bounds[kNumParameters];                   \
    static const double optimal_cost;                                   \
    static CostFunction* Create() {                                     \
      return new AutoDiffCostFunction<name,                             \
                                      num_residuals,                    \
                                      num_parameters>(new name);        \
    }                                                                   \
    template <typename T>                                               \
    bool operator()(const T* const x, T* residual) const {

#define END_BOUNDS_TEST return true; } };

BEGIN_BOUNDS_TEST(TestProblem3, 2, 2)
  const T x1 = x[0];
  const T x2 = x[1];
  residual[0] = T(10000.0) * x1 * x2 - T(1.0);
  residual[1] = exp(-x1) + exp(-x2) - T(1.0001);
END_BOUNDS_TEST;

const double TestProblem3::initial_x[] = {0.0, 1.0};
const double TestProblem3::lower_bounds[] = {0.0, 1.0};
const double TestProblem3::upper_bounds[] = {1.0, 9.0};
const double TestProblem3::optimal_cost = 0.15125900e-9;


BEGIN_BOUNDS_TEST(TestProblem4, 2, 3)
  const T x1 = x[0];
  const T x2 = x[1];
  residual[0] = x1  - T(1000000.0);
  residual[1] = x2 - T(0.000002);
  residual[2] = x1 * x2 - T(2.0);
END_BOUNDS_TEST;

const double TestProblem4::initial_x[] = {1.0, 1.0};
const double TestProblem4::lower_bounds[] = {0.0, 0.00003};
const double TestProblem4::upper_bounds[] = {1000000.0, 100.0};
const double TestProblem4::optimal_cost = 0.78400000e3;

BEGIN_BOUNDS_TEST(TestProblem5, 2, 3)
    const T x1 = x[0];
    const T x2 = x[1];
    residual[0] = T(1.5) - x1 * (T(1.0) - x2);
    residual[1] = T(2.25) - x1 * (T(1.0) - x2 * x2);
    residual[2] = T(2.625) - x1 * (T(1.0) - x2 * x2 * x2);
END_BOUNDS_TEST;

const double TestProblem5::initial_x[] = {1.0, 1.0};
const double TestProblem5::lower_bounds[] = {0.6, 0.5};
const double TestProblem5::upper_bounds[] = {10.0, 100.0};
const double TestProblem5::optimal_cost = 0.0;

BEGIN_BOUNDS_TEST(TestProblem7, 3, 3)
  const T x1 = x[0];
  const T x2 = x[1];
  const T x3 = x[2];
  const T theta = T(0.5 / M_PI)  * atan(x2 / x1) + (x1 > 0.0 ? T(0.0) : T(0.5));

  residual[0] = T(10.0) * (x3 - T(10.0) * theta);
  residual[1] = T(10.0) * (sqrt(x1 * x1 + x2 * x2) - T(1.0));
  residual[2] = x3;
END_BOUNDS_TEST;

const double TestProblem7::initial_x[] = {-1.0, 0.0, 0.0};
const double TestProblem7::lower_bounds[] = {-100.0, -1.0, -1.0};
const double TestProblem7::upper_bounds[] = {0.8, 1.0, 1.0};
const double TestProblem7::optimal_cost = 0.99042212;

BEGIN_BOUNDS_TEST(TestProblem9, 3, 15)
  const T x1 = x[0];
  const T x2 = x[1];
  const T x3 = x[2];

  double y[] = {0.0009, 0.0044, 0.0175, 0.0540, 0.1295, 0.2420, 0.3521,
                0.3989,
                0.3521, 0.2420, 0.1295, 0.0540, 0.0175, 0.0044, 0.0009};
  for (int i = 0; i < 15; ++i) {
    const T t_i = T((8.0 - i - 1.0) / 2.0);
    const T y_i = T(y[i]);
    residual[i] = x1 * exp( -x2 * (t_i - x3) * (t_i - x3) / T(2.0)) - y_i;
  }
END_BOUNDS_TEST;

const double TestProblem9::initial_x[] = {0.4, 1.0, 0.0};
const double TestProblem9::lower_bounds[] = {0.398, 1.0 ,-0.5};
const double TestProblem9::upper_bounds[] = {4.2, 2.0, 0.1};
const double TestProblem9::optimal_cost = 0.11279300e-7;

#undef BEGIN_BOUNDS_TEST
#undef END_BOUNDS_TEST

template<typename TestProblem> string Solve() {
  double x[TestProblem::kNumParameters];
  std::copy(TestProblem::initial_x,
            TestProblem::initial_x + TestProblem::kNumParameters,
            x);

  Problem problem;
  problem.AddResidualBlock(TestProblem::Create(), NULL, x);
  for (int i = 0; i < TestProblem::kNumParameters; ++i) {
    problem.SetParameterLowerBound(x, i, TestProblem::lower_bounds[i]);
    problem.SetParameterUpperBound(x, i, TestProblem::upper_bounds[i]);
  }

  Solver::Options options;
  options.parameter_tolerance = 1e-18;
  options.function_tolerance = 1e-18;
  options.gradient_tolerance = 1e-18;
  options.max_num_iterations = 1000;
  options.linear_solver_type = DENSE_QR;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  const double kMinLogRelativeError = 5.0;
  const double log_relative_error = -std::log10(
      std::abs(2.0 * summary.final_cost - TestProblem::optimal_cost) /
      (TestProblem::optimal_cost > 0.0 ? TestProblem::optimal_cost : 1.0));

  return (log_relative_error >= kMinLogRelativeError
          ? "Success\n"
          : "Failure\n");
}

}  // namespace examples
}  // namespace ceres

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  using ceres::examples::Solve;

  std::cout << "Test problem 3 : " << Solve<ceres::examples::TestProblem3>();
  std::cout << "Test problem 4 : " << Solve<ceres::examples::TestProblem4>();
  std::cout << "Test problem 5 : " << Solve<ceres::examples::TestProblem5>();
  std::cout << "Test problem 7 : " << Solve<ceres::examples::TestProblem7>();
  std::cout << "Test problem 9 : " << Solve<ceres::examples::TestProblem9>();

  return 0;
}
