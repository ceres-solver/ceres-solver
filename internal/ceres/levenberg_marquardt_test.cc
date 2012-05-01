// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
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
// Author: keir@google.com (Keir Mierle)
//
// This tests the Levenberg-Marquardt loop using a direct Evaluator
// implementation, rather than having a test that goes through all the Program
// and Problem machinery.

#include <cmath>
#include "ceres/dense_qr_solver.h"
#include "ceres/dense_sparse_matrix.h"
#include "ceres/evaluator.h"
#include "ceres/levenberg_marquardt.h"
#include "ceres/linear_solver.h"
#include "ceres/minimizer.h"
#include "ceres/internal/port.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// Templated Evaluator for Powell's function. The template parameters
// indicate which of the four variables/columns of the jacobian are
// active. This is equivalent to constructing a problem and using the
// SubsetLocalParameterization. This allows us to test the support for
// the Evaluator::Plus operation besides checking for the basic
// performance of the LevenbergMarquardt algorithm.
template <bool col1, bool col2, bool col3, bool col4>
class PowellEvaluator2 : public Evaluator {
 public:
  PowellEvaluator2()
      : num_active_cols_(
          (col1 ? 1 : 0) +
          (col2 ? 1 : 0) +
          (col3 ? 1 : 0) +
          (col4 ? 1 : 0)) {
    VLOG(1) << "Columns: "
            << col1 << " "
            << col2 << " "
            << col3 << " "
            << col4;
  }

  virtual ~PowellEvaluator2() {}

  // Implementation of Evaluator interface.
  virtual SparseMatrix* CreateJacobian() const {
    CHECK(col1 || col2 || col3 || col4);
    DenseSparseMatrix* dense_jacobian =
        new DenseSparseMatrix(NumResiduals(), NumEffectiveParameters());
    dense_jacobian->SetZero();
    return dense_jacobian;
  }

  virtual bool Evaluate(const double* state,
                        double* cost,
                        double* residuals,
                        SparseMatrix* jacobian) {
    double x1 = state[0];
    double x2 = state[1];
    double x3 = state[2];
    double x4 = state[3];

    VLOG(1) << "State: "
            << "x1=" << x1 << ", "
            << "x2=" << x2 << ", "
            << "x3=" << x3 << ", "
            << "x4=" << x4 << ".";

    double f1 = x1 + 10.0 * x2;
    double f2 = sqrt(5.0) * (x3 - x4);
    double f3 = pow(x2 - 2.0 * x3, 2.0);
    double f4 = sqrt(10.0) * pow(x1 - x4, 2.0);

    VLOG(1) << "Function: "
            << "f1=" << f1 << ", "
            << "f2=" << f2 << ", "
            << "f3=" << f3 << ", "
            << "f4=" << f4 << ".";

    *cost = (f1*f1 + f2*f2 + f3*f3 + f4*f4) / 2.0;

    VLOG(1) << "Cost: " << *cost;

    if (residuals != NULL) {
      residuals[0] = f1;
      residuals[1] = f2;
      residuals[2] = f3;
      residuals[3] = f4;
    }

    if (jacobian != NULL) {
      DenseSparseMatrix* dense_jacobian;
      dense_jacobian = down_cast<DenseSparseMatrix*>(jacobian);
      dense_jacobian->SetZero();

      AlignedMatrixRef jacobian_matrix = dense_jacobian->mutable_matrix();
      CHECK_EQ(jacobian_matrix.cols(), num_active_cols_);

      int column_index = 0;
      if (col1) {
        jacobian_matrix.col(column_index++) <<
            1.0,
            0.0,
            0.0,
            sqrt(10) * 2.0 * (x1 - x4) * (1.0 - x4);
      }
      if (col2) {
        jacobian_matrix.col(column_index++) <<
            10.0,
            0.0,
            2.0*(x2 - 2.0*x3)*(1.0 - 2.0*x3),
            0.0;
      }

      if (col3) {
        jacobian_matrix.col(column_index++) <<
            0.0,
            sqrt(5.0),
            2.0*(x2 - 2.0*x3)*(x2 - 2.0),
            0.0;
      }

      if (col4) {
        jacobian_matrix.col(column_index++) <<
            0.0,
            -sqrt(5.0),
            0.0,
            sqrt(10) * 2.0 * (x1 - x4) * (x1 - 1.0);
      }
      VLOG(1) << "\n" << jacobian_matrix;
    }
    return true;
  }

  virtual bool Plus(const double* state,
                    const double* delta,
                    double* state_plus_delta) const {
    int delta_index = 0;
    state_plus_delta[0] = (col1  ? state[0] + delta[delta_index++] : state[0]);
    state_plus_delta[1] = (col2  ? state[1] + delta[delta_index++] : state[1]);
    state_plus_delta[2] = (col3  ? state[2] + delta[delta_index++] : state[2]);
    state_plus_delta[3] = (col4  ? state[3] + delta[delta_index++] : state[3]);
    return true;
  }

  virtual int NumEffectiveParameters() const { return num_active_cols_; }
  virtual int NumParameters()          const { return 4; }
  virtual int NumResiduals()           const { return 4; }

 private:
  const int num_active_cols_;
};

// Templated function to hold a subset of the columns fixed and check
// if the solver converges to the optimal values or not.
template<bool col1, bool col2, bool col3, bool col4>
void IsSolveSuccessful() {
  LevenbergMarquardt lm;
  Solver::Options solver_options;
  Minimizer::Options minimizer_options(solver_options);
  minimizer_options.gradient_tolerance = 1e-26;
  minimizer_options.function_tolerance = 1e-26;
  minimizer_options.parameter_tolerance = 1e-26;
  LinearSolver::Options linear_solver_options;
  DenseQRSolver linear_solver(linear_solver_options);

  double initial_parameters[4] = { 3, -1, 0, 1.0 };
  double final_parameters[4] = { -1.0, -1.0, -1.0, -1.0 };

  // If the column is inactive, then set its value to the optimal
  // value.
  initial_parameters[0] = (col1 ? initial_parameters[0] : 0.0);
  initial_parameters[1] = (col2 ? initial_parameters[1] : 0.0);
  initial_parameters[2] = (col3 ? initial_parameters[2] : 0.0);
  initial_parameters[3] = (col4 ? initial_parameters[3] : 0.0);

  PowellEvaluator2<col1, col2, col3, col4> powell_evaluator;

  Solver::Summary summary;
  lm.Minimize(minimizer_options,
              &powell_evaluator,
              &linear_solver,
              initial_parameters,
              final_parameters,
              &summary);

  // The minimum is at x1 = x2 = x3 = x4 = 0.
  EXPECT_NEAR(0.0, final_parameters[0], 0.001);
  EXPECT_NEAR(0.0, final_parameters[1], 0.001);
  EXPECT_NEAR(0.0, final_parameters[2], 0.001);
  EXPECT_NEAR(0.0, final_parameters[3], 0.001);
};

TEST(LevenbergMarquardt, PowellsSingularFunction) {
  // This case is excluded because this has a local minimum and does
  // not find the optimum. This should not affect the correctness of
  // this test since we are testing all the other 14 combinations of
  // column activations.

  // IsSolveSuccessful<true, true, false, true>();

  IsSolveSuccessful<true,  true,  true,  true>();
  IsSolveSuccessful<true,  true,  true,  false>();
  IsSolveSuccessful<true,  false, true,  true>();
  IsSolveSuccessful<false, true,  true,  true>();
  IsSolveSuccessful<true,  true,  false, false>();
  IsSolveSuccessful<true,  false, true,  false>();
  IsSolveSuccessful<false, true,  true,  false>();
  IsSolveSuccessful<true,  false, false, true>();
  IsSolveSuccessful<false, true,  false, true>();
  IsSolveSuccessful<false, false, true,  true>();
  IsSolveSuccessful<true,  false, false, false>();
  IsSolveSuccessful<false, true,  false, false>();
  IsSolveSuccessful<false, false, true,  false>();
  IsSolveSuccessful<false, false, false, true>();
}


}  // namespace internal
}  // namespace ceres
