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
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// This example fits the curve f(x;m,c) = e^(m * x + c) to data, minimizing the
// sum squared loss.
// #include "FitTest.h"
#include <ceres/ceres.h>
#include <glog/logging.h>

template <typename T>
class SphericalCovariance {
 public:
  SphericalCovariance(T range, T sill, T nugget)
      : _nugget(nugget), _range(range), _sill(sill) {
    _psill = _sill - _nugget;
  }

 public:
  T compute(double d) const {
    if (d <= _range) {
      return _psill * ((3.0 * d) / (2.0 * _range) -
                       (d * d * d) / (2.0 * _range * _range * _range)) +
             _nugget;
    } else {
      return _psill + _nugget;
    }
  }

 private:
  T _sill;
  T _range;
  T _nugget;
  T _psill;
};

struct ModelResidual {
  ModelResidual(double lag, double gamma) : _lag(lag), _gamma(gamma) {}

  template <typename T>
  bool operator()(const T* const range,
                  const T* const nugget,
                  const T* const sill,
                  T* residual) const {
    residual[0] =
        _gamma -
        SphericalCovariance(range[0], sill[0], nugget[0]).compute(_lag);

    // residual[0] = y_ - exp(m[0] * x_ + c[0]);
    return true;
  }

 private:
  // Observations for a sample.
  const double _lag;
  const double _gamma;
};

void ceres_test() {
  using namespace ceres;
  ceres::Problem problem;

  std::vector<double> lag = {
      175.65234, 390.07074, 617.2337,  846.20544, 1079.8468, 1312.8428,
      1545.525,  1777.5623, 2009.1091, 2239.874,  2472.7234, 2709.9663,
      2941.5889, 3174.672,  3406.3713, 3639.4817, 3873.5212, 4107.823,
      4341.223,  4571.714,  4805.605,  5041.677,  5271.1084, 5503.067,
      5737.291,  5970.111,  6202.1523, 6437.138,  6670.4116, 6892.423};
  std::vector<double> gamma = {
      20.815123, 21.075365, 19.551971, 20.397446, 22.835442, 24.360342,
      28.06185,  27.736881, 28.389353, 27.340208, 30.65644,  31.15702,
      29.316727, 28.009079, 26.655233, 24.42093,  20.318785, 23.195473,
      22.581009, 23.794619, 22.329227, 22.553814, 21.483736, 20.519196,
      21.407993, 18.173586, 21.122591, 25.138948, 24.998138, 34.060234};

  double nugget = 20.601930259621913;
  double sill = 28.062121873701631;
  double range = 3639.4816871947096;

  // nugget = 18.362237;
  // sill = 25.0602215;
  // range = 1901.215665;

  double maxRange = 6892.4227592761508;
  double maxNugget = 34.060233524360626;
  double maxSill = 34.060233524360626;

  for (size_t i = 0; i < lag.size(); ++i) {
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<ModelResidual, 1, 1, 1, 1>(
            new ModelResidual{lag[i], gamma[i]});

    // SoftLOneLoss CauchyLoss HuberLoss
    ceres::LossFunction* loss = nullptr;

    problem.AddResidualBlock(cost_function, loss, &range, &nugget, &sill);
  }

  problem.SetParameterLowerBound(&nugget, 0, 0.0);
  // problem.SetParameterUpperBound(&nugget, 0, maxNugget);

  problem.SetParameterLowerBound(&range, 0, 0.0);
  // problem.SetParameterUpperBound(&range, 0, maxRange);

  problem.SetParameterLowerBound(&sill, 0, 0.0);
  // problem.SetParameterUpperBound(&sill, 0, maxSill);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  options.function_tolerance = 0.0;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << std::endl;
  std::cout << "Fitted parameters:" << std::endl;
  std::cout << "Nugget = " << nugget << std::endl;
  std::cout << "Range = " << range << std::endl;
  std::cout << "Sill = " << sill << std::endl;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  ceres_test();
}
