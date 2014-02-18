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

#include "ceres/ceres.h"

namespace ceres {
namespace examples {

struct MGH3Functor {
  template <typename T> bool operator()(const T* const x, T* residual) const {
    T x1 = x[0];
    T x2 = x[1];
    residual[0] = T(10000.0) * x1 * x2 - T(1.0);
    residual[1] = exp(-x1) + exp(-x2) - T(1.0001);
    return true;
  }

  static CostFunction* Create() {
    return new AutoDiffCostFunction<MGH3Functor, 2, 2>(new MGH3Functor);
  }
};

void MGH3() {
  Problem problem;
  double x[2] = {0.0, 1.0};
  double lower_bounds[] = {0.0, 1.0};
  double upper_bounds[] = {1.0, 9.0};
  problem.AddResidualBlock(MGH3Functor::Create(), NULL, x);
  problem.SetParameterLowerBound(x, 0, lower_bounds[0]);
  problem.SetParameterLowerBound(x, 1, lower_bounds[1]);
  problem.SetParameterUpperBound(x, 0, upper_bounds[0]);
  problem.SetParameterUpperBound(x, 1, upper_bounds[1]);

  Solver::Options options;
  options.max_num_iterations = 1000;
  options.linear_solver_type = DENSE_QR;
  Solver::Summary summary;

  LOG(INFO) << "x : " << x[0] << " " << x[1];
  Solve(options, &problem, &summary);
  LOG(INFO) << "x : " << x[0] << " " << x[1];
  LOG(INFO) << summary.BriefReport();
};

struct MGH4Functor {
  template <typename T> bool operator()(const T* const x, T* residual) const {
    T x1 = x[0];
    T x2 = x[1];
    residual[0] = x1  - T(1000000.0);
    residual[1] = x2 - T(0.000002);
    residual[2] = x1 * x2 - T(2.0);
    return true;
  }

  static CostFunction* Create() {
    return new AutoDiffCostFunction<MGH4Functor, 3, 2>(new MGH4Functor);
  }
};

void MGH4() {
  Problem problem;
  double x[2] = {1.0, 1.0};
  double lower_bounds[] = {0.0, 0.00003};
  double upper_bounds[] = {1000000.0, 100.0};

  problem.AddResidualBlock(MGH4Functor::Create(), NULL, x);
  problem.SetParameterLowerBound(x, 0, lower_bounds[0]);
  problem.SetParameterLowerBound(x, 1, lower_bounds[1]);
  problem.SetParameterUpperBound(x, 0, upper_bounds[0]);
  problem.SetParameterUpperBound(x, 1, upper_bounds[1]);

  Solver::Options options;
  options.max_num_iterations = 1000;
  options.linear_solver_type = DENSE_QR;
  Solver::Summary summary;

  LOG(INFO) << "x : " << x[0] << " " << x[1];
  Solve(options, &problem, &summary);
  LOG(INFO) << "x : " << x[0] << " " << x[1];
  LOG(INFO) << summary.BriefReport();
};

struct MGH5Functor {
  template <typename T> bool operator()(const T* const x, T* residual) const {
    T x1 = x[0];
    T x2 = x[1];
    residual[0] = T(1.5) - x1 * (T(1.0) - x2);
    residual[1] = T(2.25) - x1 * (T(1.0) - x2 * x2);
    residual[2] = T(2.625) - x1 * (T(1.0) - x2 * x2 * x2);
    return true;
  }

  static CostFunction* Create() {
    return new AutoDiffCostFunction<MGH5Functor, 3, 2>(new MGH5Functor);
  }
};

void MGH5() {
  Problem problem;
  double x[2] = {1.0, 1.0};
  double lower_bounds[] = {0.6, 0.5};
  double upper_bounds[] = {10.0, 100.0};

  problem.AddResidualBlock(MGH5Functor::Create(), NULL, x);
  problem.SetParameterLowerBound(x, 0, lower_bounds[0]);
  problem.SetParameterLowerBound(x, 1, lower_bounds[1]);
  problem.SetParameterUpperBound(x, 0, upper_bounds[0]);
  problem.SetParameterUpperBound(x, 1, upper_bounds[1]);

  Solver::Options options;
  options.max_num_iterations = 1000;
  options.linear_solver_type = DENSE_QR;
  Solver::Summary summary;

  LOG(INFO) << "x : " << x[0] << " " << x[1];
  Solve(options, &problem, &summary);
  LOG(INFO) << "x : " << x[0] << " " << x[1];
  LOG(INFO) << summary.BriefReport();
};

struct MGH7Functor {
  template <typename T> bool operator()(const T* const x, T* residual) const {
    const T x1 = x[0];
    const T x2 = x[1];
    const T x3 = x[2];
    const T theta = T(0.5 / M_PI)  * atan(x2 / x1) + (x1 > 0.0 ? T(0.0) : T(0.5));

    residual[0] = T(10.0) * (x3 - T(10.0) * theta);
    residual[1] = T(10.0) * (sqrt(x1 * x1 + x2 * x2) - T(1.0));
    residual[2] = x3;
    return true;
  }

  static CostFunction* Create() {
    return new AutoDiffCostFunction<MGH7Functor, 3, 3>(new MGH7Functor);
  }
};

void MGH7() {
  Problem problem;
  double x[] = {-1.0, 0.0, 0.0};
  double lower_bounds[] = {-100.0, -1.0, -1.0};
  double upper_bounds[] = {0.8, 1.0, 1.0};

  problem.AddResidualBlock(MGH7Functor::Create(), NULL, x);
  problem.SetParameterLowerBound(x, 0, lower_bounds[0]);
  problem.SetParameterLowerBound(x, 1, lower_bounds[1]);
  problem.SetParameterLowerBound(x, 2, lower_bounds[2]);

  problem.SetParameterUpperBound(x, 0, upper_bounds[0]);
  problem.SetParameterUpperBound(x, 1, upper_bounds[1]);
  problem.SetParameterUpperBound(x, 2, upper_bounds[2]);

  Solver::Options options;
   options.max_num_iterations = 1000;
  options.linear_solver_type = DENSE_QR;
  Solver::Summary summary;

  LOG(INFO) << "x : " << x[0] << " " << x[1] << " " << x[2];
  Solve(options, &problem, &summary);
  LOG(INFO) << "x : " << x[0] << " " << x[1] << " " << x[2];
  LOG(INFO) << summary.BriefReport();
};

struct MGH9Functor {
  template <typename T> bool operator()(const T* const x, T* residual) const {
    const T x1 = x[0];
    const T x2 = x[1];
    const T x3 = x[2];

    double y[] = {0.0009, 0.0044, 0.0175, 0.0540, 0.1295, 0.2420, 0.3521,
                  0.3989,
                  0.3521, 0.2420, 0.1295, 0.0540, 0.0175, 0.0044, 0.0009};

    for (int i = 0; i < 15; ++i) {
      T t_i = T((8.0 - i - 1.0) / 2.0);
      T y_i = T(y[i]);
      residual[i] = x1 * exp( -x2 * (t_i - x3) * (t_i - x3) / T(2.0)) - y_i;
    }

    return true;
  }

  static CostFunction* Create() {
    return new AutoDiffCostFunction<MGH9Functor, 15, 3>(new MGH9Functor);
  }
};

void MGH9() {
  Problem problem;
  double x[] = {0.4, 1.0, 0.0};
  double lower_bounds[] = {0.398, 1.0 ,-0.5};
  double upper_bounds[] = {4.2, 2.0, 0.1};

  problem.AddResidualBlock(MGH9Functor::Create(), NULL, x);
  problem.SetParameterLowerBound(x, 0, lower_bounds[0]);
  problem.SetParameterLowerBound(x, 1, lower_bounds[1]);
  problem.SetParameterLowerBound(x, 2, lower_bounds[2]);

  problem.SetParameterUpperBound(x, 0, upper_bounds[0]);
  problem.SetParameterUpperBound(x, 1, upper_bounds[1]);
  problem.SetParameterUpperBound(x, 2, upper_bounds[2]);
  Solver::Options options;
  options.max_num_iterations = 1000;
  options.linear_solver_type = DENSE_QR;
  Solver::Summary summary;

  LOG(INFO) << "x : " << x[0] << " " << x[1] << " " << x[2];
  Solve(options, &problem, &summary);
  LOG(INFO) << "x : " << x[0] << " " << x[1] << " " << x[2];
  LOG(INFO) << summary.BriefReport();
};


}  // namespace examples
}  // namespace ceres

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  //  ceres::examples::MGH3();
  //ceres::examples::MGH4();
  //ceres::examples::MGH5();
  //ceres::examples::MGH7();
  ceres::examples::MGH9();
  return 0;
}
