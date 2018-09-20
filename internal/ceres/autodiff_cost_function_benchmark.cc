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
#include "ceres/ceres.h"
#include "ceres/jet.h"

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
    residuals[0] = b1 * pow(1.0 + exp(b2 - b3 * x_), -1.0 / b4) - y_;
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
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
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

  while (state.KeepRunning()) {
    cost_function->Evaluate(
        parameters, &residuals, state.range(0) ? jacobians : nullptr);
  }
}
BENCHMARK(BM_Rat43AutoDiff)->Arg(0)->Arg(1);

}  // namespace ceres

BENCHMARK_MAIN();
