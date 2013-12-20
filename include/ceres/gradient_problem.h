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

#ifndef CERES_PUBLIC_GRADIENT_PROBLEM_H_
#define CERES_PUBLIC_GRADIENT_PROBLEM_H_

namespace ceres {

// Instances of GradientProblem represent general non-linear
// optimization problems that must be solved using just the value of
// the objective function and its gradient.
//
// Unlike the Problem class, which can only be used to model
// non-linear least squares problems, instances of GradientProblem are
// not restricted in the form of the objective function.
//
// Example usage:
//
// To minimize the Rosenbrock function
//
// f(x,y) = (1-x)^2 + 100(y - x^2)^2;
//
// We will need to implement the following class
//
// class Rosenbrock : public ceres::GradientProblem {
//  public:
//   virtual ~Rosenbrock() {}
//
//   virtual bool Evaluate(const double* parameters,
//                         double* cost,
//                         double* gradient) const {
//     const double x = parameters[0];
//     const double y = parameters[1];

//     cost[0] = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
//     if (gradient != NULL) {
//       gradient[0] = -2.0 * (1.0 - x) - 200.0 * (y - x * x) * 2.0 * x;
//       gradient[1] = 200.0 * (y - x * x);
//     }
//     return true;
//   };
//
//   virtual int NumParameters() const { return 2; };
// };

class GradientProblem {
 public:
  virtual ~GradientProblem();

  // Evaluate the cost and (optionally) the gradient of the objective
  // function.
  //
  // cost is guaranteed never to be NULL.
  //
  // gradient is an array of size NumTangentSpaceParameters(). if not
  // NULL, then it is the user's responsibility to evaluate and
  // populate the gradient array.
  //
  // The return value indicates whether the computation of the
  // objective function and/or gradient was successful or not.
  virtual bool Evaluate(const double* parameters,
                        double* cost,
                        double* gradient) const = 0;

  // Degrees of freedom of the optimization problem.
  virtual int NumParameters() const = 0;

  // The following two methods determine properties of the tangent
  // space (space of derivatives) of the problem. If all the
  // parameters of the problem live in a Euclidean space then there is
  // nothing to do.
  //
  // If however, the tangent space has a dimension less than the
  // dimension of the parameter vector, then it is the user's
  // responsibility to define this method and implement the Plus and
  // Evaluate methods appropriately.
  virtual int NumTangentSpaceParameters() const;

  // x_plus_delta = Plus(x, delta);
  //
  // Unless overridden by the user, the default implementation assumes
  // that the parameter space is Euclidean and the usual addition
  // operation on vectors is to be used, i.e.:
  //
  //   x_plus_delta = x + delta;
  //
  // See local_parameterization.h for more details on the Plus
  // operation.
  virtual bool Plus(const double* x,
                    const double* delta,
                    double* x_plus_delta) const;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_GRADIENT_PROBLEM_H_
