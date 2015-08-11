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
// Author: standmark@google.com (Petter Strandmark)

#ifndef CERES_PUBLIC_COMPOSITE_FIRST_ORDER_FUNCTION_H_
#define CERES_PUBLIC_COMPOSITE_FIRST_ORDER_FUNCTION_H_

#include <vector>
#include "ceres/gradient_problem.h"

namespace ceres {

// A first-order function representing a sum of terms, each of which is
// computed from a CostFunction.
//
// This class represents a function
//
//    N 
//   SUM  f_i(x)
//   i=1 
//
// where each f_i typically only depends on a subset of all N variables. Each
// f_i is added as a CostFunction.
class CERES_EXPORT CompositeFirstOrderFunction : public FirstOrderFunction {
 public:
  CompositeFirstOrderFunction();
  virtual ~CompositeFirstOrderFunction();

  virtual bool Evaluate(const double* const parameters, double* cost,
                        double* gradient) const;
  virtual int NumParameters() const;

  // Adds a term to the function. If the CostFunction exposes more than one
  // residual, their sum is added to the function.
  //
  // The Problem object takes ownership of the cost_function passed to it and
  // the cost_function remains live for the life if the
  // CompositeFirstOrderFunction.
  void AddTerm(CostFunction* cost_function, double* x0);
  void AddTerm(CostFunction* cost_function, double* x0, double* x1);
  void AddTerm(CostFunction* cost_function, double* x0, double* x1, double* x2);
  void AddTerm(CostFunction* cost_function, const std::vector<double*> xs);

  // Writes the parameters used to create the function to one vector suitable
  // for use with GradientProblemSolver.
  void InitialSolution(double* all_parameters);

  // Fills the parameters passed to AddTerm calls from a vector with all
  // parameters packed together. The all_parameters argument must have length
  // NumParameters().
  void ParseSolution(const double* const all_parameters) const;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_COMPOSITE_FIRST_ORDER_FUNCTION_H_
