// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2013 Google Inc. All rights reserved.
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
// Author: mierle@gmail.com (Keir Mierle)
//
// An incomplete C API for Ceres.
//
// TODO(keir): Figure out why logging does not seem to work.

#include <vector>
#include <iostream>  // XXX remove me
#include "ceres/c_api.h"
#include "ceres/cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "ceres/types.h"  // for std
#include "glog/logging.h"

using ceres::Problem;

void ceres_init(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
}

ceres_problem_t* ceres_create_problem() {
  return reinterpret_cast<ceres_problem_t*>(new Problem);
}

void ceres_free_problem(ceres_problem_t* problem) {
  delete reinterpret_cast<Problem*>(problem);
}

class CallbackCostFunction : public ceres::CostFunction {
 public:
  CallbackCostFunction(ceres_cost_function_t cost_function,
                       void* user_data,
                       int num_residuals,
                       int num_parameter_blocks,
                       int* parameter_block_sizes)
      : cost_function_(cost_function),
        user_data_(user_data) {
    set_num_residuals(num_residuals);
    for (int i = 0; i < num_parameter_blocks; ++i) {
      mutable_parameter_block_sizes()->push_back(parameter_block_sizes[i]);
    }
  }
  virtual ~CallbackCostFunction() {}

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    return (*cost_function_)(user_data_,
                             const_cast<double**>(parameters),
                             residuals,
                             jacobians);
  }

 private:
  ceres_cost_function_t cost_function_;
  void* user_data_;
};

ceres_residual_block_id_t* ceres_problem_add_residual_block(
    ceres_problem_t* problem,
    ceres_cost_function_t cost_function,
    ceres_loss_function_t loss_function,
    void* user_data,
    int num_residuals,
    int num_parameter_blocks,
    int* parameter_block_sizes,
    double** parameters) {
  ceres::CostFunction* callback_cost_function =
      new CallbackCostFunction(cost_function,
                               user_data,
                               num_residuals,
                               num_parameter_blocks,
                               parameter_block_sizes);
  
  Problem* ceres_problem = reinterpret_cast<Problem*>(problem);

  std::vector<double*> parameter_blocks(parameters,
                                        parameters + num_parameter_blocks);
  return reinterpret_cast<ceres_residual_block_id_t*>(
      ceres_problem->AddResidualBlock(callback_cost_function,
                                      NULL, /* Ignore loss for now */
                                      parameter_blocks));
}

void ceres_solve(ceres_problem_t* c_problem) {
  Problem* problem = reinterpret_cast<Problem*>(c_problem);
  
  // TODO(keir): Obviously, this way of setting options won't scale or last.
  // Instead, figure out a way to specify some of the options without
  // duplicating everything.
  ceres::Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);
  std::cout << summary.FullReport() << "\n";
}
