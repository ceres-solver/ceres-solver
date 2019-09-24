// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2018 Google Inc. All rights reserved.
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
// Authors: sameeragarwal@google.com (Sameer Agarwal)

#include <memory>

#include "benchmark/benchmark.h"
#include "ceres/autodiff_codegen.h"
#include "ceres/ceres.h"
#include "ceres/jet.h"

//#define GEN_CODE

namespace ceres {

// From the NIST problem collection.
struct Rat43CostFunctor {
  Rat43CostFunctor(const double x, const double y) : x_(x), y_(y) {}

  template <typename T>
  bool operator()(const T* parameters, T* residuals) const {
    const T& b1 = parameters[0];
    const T& b2 = parameters[1];
    const T& b3 = parameters[2];
    const T& b4 = parameters[3];
#ifdef GEN_CODE
    T x = CERES_EXTERNAL_CONSTANT(x_);
    T y = CERES_EXTERNAL_CONSTANT(y_);
#else
    T x(x_);
    T y(y_);
#endif
    residuals[0] = b1 * pow(T(1.0) + exp(b2 - b3 * x), T(-1.0) / b4) - y;
    return true;
  }

  EIGEN_DONT_INLINE bool Evaluate_Residual(double const* const* parameters,
                                           double* residuals) {
    // This code is generated with ceres::AutoDiffCodeGen
    // See ceres/autodiff_codegen.h for more informations.
    const double v_0 = parameters[0][0];
    const double v_5 = parameters[0][1];
    const double v_10 = parameters[0][2];
    const double v_15 = parameters[0][3];
    const double v_20 = x_;
    const double v_25 = y_;
    const double v_30 = 1.000000;
    const double v_40 = v_10 * v_20;
    const double v_58 = v_5 - v_40;
    const double v_68 = exp(v_58);
    const double v_78 = v_30 + v_68;
    const double v_83 = -1.000000;
    const double v_94 = v_30 / v_15;
    const double v_95 = v_83 * v_94;
    const double v_113 = pow(v_78, v_95);
    const double v_137 = v_0 * v_113;
    const double v_155 = v_137 - v_25;
    residuals[0] = v_155;
    return true;
  }
  EIGEN_DONT_INLINE bool Evaluate_ResidualAndJacobian(
      double const* const* parameters, double* residuals, double** jacobians) {
    // This code is generated with ceres::AutoDiffCodeGen
    // See ceres/autodiff_codegen.h for more informations.
    const double v_0 = parameters[0][0];
    const double v_1 = 1.000000;
    const double v_5 = parameters[0][1];
    const double v_10 = parameters[0][2];
    const double v_15 = parameters[0][3];
    const double v_20 = x_;
    const double v_25 = y_;
    const double v_40 = v_10 * v_20;
    const double v_58 = v_5 - v_40;
    const double v_61 = -(v_20);
    const double v_68 = exp(v_58);
    const double v_71 = v_68 * v_61;
    const double v_78 = v_1 + v_68;
    const double v_83 = -1.000000;
    const double v_94 = v_1 / v_15;
    const double v_95 = v_83 * v_94;
    const double v_103 = -(v_95);
    const double v_107 = v_103 * v_94;
    const double v_113 = pow(v_78, v_95);
    const double v_115 = v_95 + v_83;
    const double v_116 = pow(v_78, v_115);
    const double v_117 = v_95 * v_116;
    const double v_118 = log(v_78);
    const double v_119 = v_113 * v_118;
    const double v_121 = v_117 * v_68;
    const double v_122 = v_117 * v_71;
    const double v_127 = v_119 * v_107;
    const double v_137 = v_0 * v_113;
    const double v_139 = v_0 * v_121;
    const double v_140 = v_0 * v_122;
    const double v_141 = v_0 * v_127;
    const double v_155 = v_137 - v_25;
    residuals[0] = v_155;
    jacobians[0][0] = v_113;
    jacobians[0][1] = v_139;
    jacobians[0][2] = v_140;
    jacobians[0][3] = v_141;
    return true;
  }

  // from http://ceres-solver.org/analytical_derivatives.html
  EIGEN_DONT_INLINE bool Evaluate_Analytic(double const* const* parameters,
                                           double* residuals,
                                           double** jacobians) const {
    const double b1 = parameters[0][0];
    const double b2 = parameters[0][1];
    const double b3 = parameters[0][2];
    const double b4 = parameters[0][3];

    const double t1 = exp(b2 - b3 * x_);
    const double t2 = 1 + t1;
    const double t3 = pow(t2, -1.0 / b4);
    residuals[0] = b1 * t3 - y_;

    if (!jacobians) return true;
    double* jacobian = jacobians[0];
    if (!jacobian) return true;

    const double t4 = pow(t2, -1.0 / b4 - 1);
    jacobian[0] = t3;
    jacobian[1] = -b1 * t1 * t4 / b4;
    jacobian[2] = -x_ * jacobian[1];
    jacobian[3] = b1 * log(t2) * t3 / (b4 * b4);
    return true;
  }

 private:
  const double x_;
  const double y_;
};

// Simple implementation of autodiff using Jets directly instead of
// going through the machinery of AutoDiffCostFunction, which does
// the same thing, but much more generically.
class Rat43Automatic : public ceres::SizedCostFunction<1, 4> {
 public:
  Rat43Automatic(const Rat43CostFunctor* functor) : functor_(functor) {}
  virtual ~Rat43Automatic() {}
  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const final {
    if (!jacobians) {
      return (*functor_)(parameters[0], residuals);
    }

    typedef ceres::Jet<double, 4> JetT;
    JetT jets[4];
    for (int i = 0; i < 4; ++i) {
      jets[i].a = parameters[0][i];
      jets[i].v.setZero();
      jets[i].v[i] = 1.0;
    }

    JetT result;
    (*functor_)(jets, &result);

    residuals[0] = result.a;
    for (int i = 0; i < 4; ++i) {
      jacobians[0][i] = result.v[i];
    }
    return true;
  }

 private:
  std::unique_ptr<const Rat43CostFunctor> functor_;
};

volatile double globalSum = 0;
static void BM_Rat43AutoDiff(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4.};
  double* parameters[] = {parameter_block1};

  double jacobian1[] = {0.0, 0.0, 0.0, 0.0};
  double residuals;
  double* jacobians[] = {jacobian1};
  const double x = 0.2;
  const double y = 0.3;
  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<Rat43CostFunctor, 1, 4>(
          new Rat43CostFunctor(x, y)));

  // use autodiff
  while (state.KeepRunning()) {
    cost_function->Evaluate(
        parameters, &residuals, state.range(0) ? jacobians : nullptr);
  }
}

static void BM_Rat43AutoDiffCodeGen(benchmark::State& state) {
#ifdef GEN_CODE
  // Use these lines to generate the code which is included above
  ceres::AutoDiffCodeGen<Rat43CostFunctor, 1, 4> codeGen(
      new Rat43CostFunctor(0, 0));
  codeGen.Generate();
  std::terminate();
#endif

  double parameter_block1[] = {1., 2., 3., 4.};
  double* parameters[] = {parameter_block1};

  double jacobian1[] = {0.0, 0.0, 0.0, 0.0};
  double residuals;
  double* jacobians[] = {jacobian1};
  const double x = 0.2;
  const double y = 0.3;
  Rat43CostFunctor rawFunctor(x, y);

  // use generated code
  while (state.KeepRunning()) {
    if (state.range(0)) {
      rawFunctor.Evaluate_ResidualAndJacobian(
          parameters, &residuals, jacobians);
    } else {
      rawFunctor.Evaluate_Residual(parameters, &residuals);
    }
  }
}

static void BM_Rat43AnalyticOptimized(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4.};
  double* parameters[] = {parameter_block1};

  double jacobian1[] = {0.0, 0.0, 0.0, 0.0};
  double residuals;
  double* jacobians[] = {jacobian1};
  const double x = 0.2;
  const double y = 0.3;
  Rat43CostFunctor rawFunctor(x, y);

  // use generated code
  while (state.KeepRunning()) {
    rawFunctor.Evaluate_Analytic(
        parameters, &residuals, state.range(0) ? jacobians : nullptr);
  }
}

BENCHMARK(BM_Rat43AutoDiff)->Arg(0)->Arg(1);
BENCHMARK(BM_Rat43AnalyticOptimized)->Arg(0)->Arg(1);
BENCHMARK(BM_Rat43AutoDiffCodeGen)->Arg(0)->Arg(1);

}  // namespace ceres

BENCHMARK_MAIN();
