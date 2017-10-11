// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
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
// Author: keir@google.com (Keir Mierle)
//
// A simple example of using the Ceres minimizer.
//
// Minimize 0.5 (10 - x)^2 using jacobian matrix computed using
// automatic differentiation.

#include "ceres/ceres.h"
#include "ceres/tiny_solver.h"
#include "glog/logging.h"
#include <Eigen/Dense>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::TinySolver;


// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
struct CostFunctor {
  template <typename T> bool operator()(const T* const x, T* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};

template <int kNumResiduals, int kNumParameters>
class CostFunctionAdapter {
 public:
  typedef double Scalar;
  enum { NUM_PARAMETERS = kNumParameters, NUM_RESIDUALS = kNumResiduals };

  CostFunctionAdapter(const CostFunction& cost_function)
      : cost_function_(cost_function) {
    CHECK_EQ(cost_function_.parameter_block_sizes().size(), 1);
    if (NUM_PARAMETERS != Eigen::Dynamic) {
      CHECK_EQ(cost_function_.num_residuals(), NUM_RESIDUALS);
    }
    if (NUM_RESIDUALS != Eigen::Dynamic) {
      CHECK_EQ(cost_function_.parameter_block_sizes()[0], NUM_PARAMETERS);
    }

    if (NUM_PARAMETERS == Eigen::Dynamic || NUM_RESIDUALS == Eigen::Dynamic) {
      row_major_jacobian_.resize(cost_function_.num_residuals(),
                                 cost_function_.parameter_block_sizes()[0]);
    }
  }

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    if (!jacobian) {
      return cost_function_.Evaluate(&parameters, residuals, NULL);
    }

    double* j[1];
    j[0] = row_major_jacobian_.data();
    if (!cost_function_.Evaluate(&parameters, residuals, j)) {
      return false;
    }

    Eigen::Map<Eigen::Matrix<double, NUM_RESIDUALS, NUM_PARAMETERS>>
        col_major_jacobian(jacobian,
                           cost_function_.num_residuals(),
                           cost_function_.parameter_block_sizes()[0]);
    col_major_jacobian = row_major_jacobian_;
    return true;
  }

  int NumResiduals() const { return cost_function_.num_residuals(); }
  int NumParameters() const {
    return cost_function_.parameter_block_sizes()[0];
  }

  const CostFunction& cost_function_;
  mutable Eigen::Matrix<double, NUM_RESIDUALS, NUM_PARAMETERS, Eigen::RowMajor>
      row_major_jacobian_;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  Eigen::Matrix<double, 1, 1> x;
  x(0) = 0.5;
  const Eigen::Matrix<double, 1, 1> initial_x = x;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  CostFunction* cost_function =
      new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);

  CostFunctionAdapter<1, 1> cfa(*cost_function);

  TinySolver<CostFunctionAdapter<1, 1>> solver;
  solver.Solve(cfa, &x);

  std::cout << "x : " << initial_x
            << " -> " << x << "\n";
  return 0;
}
