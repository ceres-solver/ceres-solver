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
// Author: mierle@gmail.com (Keir Mierle)
//
// A stripped down implementation of Levenberg-Marquardt, intended for solving
// small dense problems with low-latency. The implementation takes care to do
// as much allocation up front as possible, so that the memory needed to solve
// can get allocated up front and reused for many solves.
//
// Implementation taken almost straight from:
//
// [1] K. Madsen, H. Nielsen, O. Tingleoff. Methods for Non-linear Least
//     Squares Problems.
//     http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf

#ifndef CERES_PUBLIC_SMALL_SOLVER_H_
#define CERES_PUBLIC_SMALL_SOLVER_H_

#include <cmath>

#include "ceres/types.h"
#include "Eigen/Core"
#include "Eigen/LU"
#include "glog/logging.h"

namespace ceres {

template<typename Function,
         typename Solver = Eigen::PartialPivLU<
           Eigen::Matrix<typename Function::Scalar,
                         Function::NUM_PARAMETERS,
                         Function::NUM_PARAMETERS> > >
class SmallSolver {
 public:
  enum {
    NUM_RESIDUALS = Function::NUM_RESIDUALS,
    NUM_PARAMETERS = Function::NUM_PARAMETERS
  };
  typedef typename Function::Scalar Scalar;
  typedef typename Eigen::Matrix<Scalar, NUM_PARAMETERS, 1> Parameters;

  // TODO(keir): Some of these knobs can be derived from each other and
  // removed, instead of requiring the user to set them.
  enum Status {
    RUNNING,
    GRADIENT_TOO_SMALL,            // eps > max(J'*f(x))
    RELATIVE_STEP_SIZE_TOO_SMALL,  // eps > ||dx|| / ||x||
    ERROR_TOO_SMALL,               // eps > ||f(x)||
    HIT_MAX_ITERATIONS,
  };

  struct SolverParameters {
    SolverParameters()
       : gradient_threshold(1e-16),
         relative_step_threshold(1e-16),
         error_threshold(1e-16),
         initial_scale_factor(1e-3),
         max_iterations(100) {}
    Scalar gradient_threshold;       // eps > max(J'*f(x))
    Scalar relative_step_threshold;  // eps > ||dx|| / ||x||
    Scalar error_threshold;          // eps > ||f(x)||
    Scalar initial_scale_factor;     // Initial u for solving normal equations.
    int    max_iterations;           // Maximum number of solver iterations.
  };

  struct Results {
    Scalar error_magnitude;     // ||f(x)||
    Scalar gradient_magnitude;  // ||J'f(x)||
    int    iterations;
    Status status;
  };

  Status Update(const Function& function, const Parameters &x) {
    // TODO(keir): Handle false return from the cost function.
    function(&x(0), &error_(0), &jacobian_(0, 0));
    error_ = -error_;

    // This explicitly computes the normal equations, which is numerically
    // unstable. Nevertheless, it is often good enough and is fast.
    jtj_ = jacobian_.transpose() * jacobian_;
    g_ = jacobian_.transpose() * error_;
    if (g_.array().abs().maxCoeff() < params.gradient_threshold) {
      return GRADIENT_TOO_SMALL;
    } else if (error_.norm() < params.error_threshold) {
      return ERROR_TOO_SMALL;
    }
    return RUNNING;
  }

  Results solve(const Function& function, Parameters* x_and_min) {
    Parameters& x = *x_and_min;
    results.status = Update(function, x);

    Scalar u = Scalar(params.initial_scale_factor * jtj_.diagonal().maxCoeff());
    Scalar v = 2;

    int i;
    for (i = 0; results.status == RUNNING && i < params.max_iterations; ++i) {
      VLOG(3) << "iteration: " << i;
      VLOG(3) << "||f(x)||: " << error_.norm();
      VLOG(3) << "max(g): " << g_.array().abs().maxCoeff();
      VLOG(3) << "u: " << u;
      VLOG(3) << "v: " << v;

      jtj_augmented_ = jtj_;
      jtj_augmented_.diagonal().array() += u;

      // TODO: Preallocate
      Solver solver(jtj_augmented_);
      dx_ = solver.solve(g_);
      bool solved = (jtj_augmented_ * dx_).isApprox(g_);
      if (!solved) {
        LOG(ERROR) << "Failed to solve";
      }
      if (solved && dx_.norm() < params.relative_step_threshold * x.norm()) {
        results.status = RELATIVE_STEP_SIZE_TOO_SMALL;
        break;
      }
      if (solved) {
        x_new_ = x + dx_;
        // Rho is the ratio of the actual reduction in error to the reduction
        // in error that would be obtained if the problem was linear. See [1]
        // for details.
        // TODO: Error handling on user eval.
        function(&x_new_[0], &f_x_new_[0], NULL);
        Scalar rho((error_.squaredNorm() - f_x_new_.squaredNorm())
                   / dx_.dot(u * dx_ + g_));
        if (rho > 0) {
          // Accept the Gauss-Newton step because the linear model fits well.
          x = x_new_;
          results.status = Update(function, x);
          Scalar tmp = Scalar(2 * rho - 1);
          u = u*std::max(1 / 3., 1 - tmp * tmp * tmp);
          v = 2;
          continue;
        }
      }
      // Reject the update because either the normal equations failed to solve
      // or the local linear model was not good (rho < 0). Instead, increase u
      // to move closer to gradient descent.
      u *= v;
      v *= 2;
    }
    if (results.status == RUNNING) {
      results.status = HIT_MAX_ITERATIONS;
    }
    results.error_magnitude = error_.norm();
    results.gradient_magnitude = g_.norm();
    results.iterations = i;
    return results;
  }

  SolverParameters params;
  Results results;

 private:
  Parameters dx_, x_new_, g_;
  Eigen::Matrix<Scalar, NUM_RESIDUALS, 1> error_, f_x_new_;
  Eigen::Matrix<Scalar, NUM_RESIDUALS, NUM_PARAMETERS> jacobian_;
  Eigen::Matrix<Scalar, NUM_PARAMETERS, NUM_PARAMETERS> jtj_, jtj_augmented_;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_SMALL_SOLVER_H_
