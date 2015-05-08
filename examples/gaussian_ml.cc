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

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;

namespace ceres {

// A first-order function representing a sum of terms, each of which is
// computed from a CostFunction.
class CompositeFirstOrderFunction : public FirstOrderFunction {
 public:
  CompositeFirstOrderFunction();
  virtual ~CompositeFirstOrderFunction();

  virtual bool Evaluate(const double* const parameters, double* cost,
                        double* gradient) const;
  virtual int NumParameters() const;

  // Adds a term to the function. the CostFunction must expose exactly one
  // residual, which is added to the function.
  void AddTerm(CostFunction* cost_function, double* x0, double* x1);

  // Writes the parameters used to create the function to one vector suitable
  // for use with GradientProblemSolver.
  void InitialSolution(double* all_parameters);

  // Sets the parameters used to create the function from a vector with all
  // parameters packed together.
  void ParseSolution(const double* const all_parameters) const;

 protected:
  void AddParameterBlock(double* values, int size);

 private:
  int num_parameters_;

  struct Term {
    Term() : cost_function_(NULL) {}
    CostFunction* cost_function_;
    std::vector<int> offsets_;
  };

  struct Block {
    Block() : global_offset_(0), size(0) {}
    int global_offset_;
    int size;
  };

  std::vector<Term> terms_;
  std::map<double*, Block> parameter_blocks_;
  std::set<CostFunction*> cost_functions_to_delete_;
};

CompositeFirstOrderFunction::CompositeFirstOrderFunction()
    : num_parameters_(0) {}

CompositeFirstOrderFunction::~CompositeFirstOrderFunction() {
  for (std::set<CostFunction*>::iterator cost_function_itr =
           cost_functions_to_delete_.begin();
       cost_function_itr != cost_functions_to_delete_.end();
       ++cost_function_itr) {
    delete *cost_function_itr;
  }
}

void CompositeFirstOrderFunction::AddTerm(CostFunction* cost_function,
                                          double* x0, double* x1) {
  CHECK(cost_function->num_residuals() == 1);
  AddParameterBlock(x0, cost_function->parameter_block_sizes()[0]);
  AddParameterBlock(x1, cost_function->parameter_block_sizes()[1]);

  Term term;
  term.cost_function_ = cost_function;
  term.offsets_.push_back(parameter_blocks_[x0].global_offset_);
  term.offsets_.push_back(parameter_blocks_[x1].global_offset_);
  terms_.push_back(term);
  cost_functions_to_delete_.insert(cost_function);
}

void CompositeFirstOrderFunction::AddParameterBlock(double* values, int size) {
  Block& block = parameter_blocks_[values];
  if (block.size != 0) {
    CHECK(block.size == size);
    return;
  }
  block.size = size;
  block.global_offset_ = num_parameters_;
  num_parameters_ += block.size;
}

bool CompositeFirstOrderFunction::Evaluate(const double* const parameters,
                                           double* cost,
                                           double* gradient) const {
  *cost = 0;
  for (int t = 0; t < terms_.size(); ++t) {
    const Term& term = terms_[t];
    std::vector<const double*> parameters_for_evaluation;
    std::vector<double*> gradient_for_evaluation;
    for (int i = 0; i < term.offsets_.size(); ++i) {
      parameters_for_evaluation.push_back(parameters + term.offsets_[i]);
      gradient_for_evaluation.push_back(gradient + term.offsets_[i]);
    }

    double term_cost = 0;
    bool success;
    if (gradient != NULL) {
      success = term.cost_function_->Evaluate(parameters_for_evaluation.data(),
                                              &term_cost,
                                              gradient_for_evaluation.data());
    } else {
      success = term.cost_function_->Evaluate(parameters_for_evaluation.data(),
                                              &term_cost, NULL);
    }
    if (!success) {
      return false;
    }
    *cost += term_cost;
  }
  return true;
}

void CompositeFirstOrderFunction::InitialSolution(double* parameters) {
  for (std::map<double*, Block>::const_iterator itr = parameter_blocks_.begin();
       itr != parameter_blocks_.end(); ++itr) {
    for (int i = 0; i < itr->second.size; ++i) {
      parameters[itr->second.global_offset_ + i] = itr->first[i];
    }
  }
}

void CompositeFirstOrderFunction::ParseSolution(
    const double* const parameters) const {
  for (std::map<double*, Block>::const_iterator itr = parameter_blocks_.begin();
       itr != parameter_blocks_.end(); ++itr) {
    for (int i = 0; i < itr->second.size; ++i) {
      itr->first[i] = parameters[itr->second.global_offset_ + i];
    }
  }
}

int CompositeFirstOrderFunction::NumParameters() const {
  return num_parameters_;
}
}

//------------------------------------------------------------------------------

// This is the maximum likelihood term corresponing to one sample from a
// Gaussian distribution.
struct GaussianMLFunction {
  GaussianMLFunction(double sample) : sample_(sample) {}

  template <typename T>
  bool operator()(const T* const mu, const T* const sigma, T* residual) const {
    T diff = (*mu - T(sample_)) / *sigma;
    residual[0] = T(0.5) * diff * diff + log(*sigma);
    return true;
  }

 private:
  const double sample_;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  ceres::GradientProblemSolver::Options options;
  options.minimizer_progress_to_stdout = true;

  // Some "Gaussian" data.
  std::vector<double> data;
  data.push_back(9.9);
  data.push_back(10.0);
  data.push_back(10.2);
  data.push_back(10.3);
  data.push_back(9.8);
  data.push_back(9.6);
  data.push_back(10.1);

  // Build the ML function.
  ceres::CompositeFirstOrderFunction* ml_sum =
      new ceres::CompositeFirstOrderFunction();
  double mu = 0.0;  // Initial guess.
  double sigma = 1.0;
  for (int i = 0; i < data.size(); ++i) {
    ml_sum->AddTerm(new AutoDiffCostFunction<GaussianMLFunction, 1, 1, 1>(
                        new GaussianMLFunction(data.at(i))),
                    &mu, &sigma);
  }

  ceres::GradientProblemSolver::Summary summary;
  ceres::GradientProblem problem(ml_sum);

  double cost = 0;
  double parameters[2];
  ml_sum->InitialSolution(parameters);
  ml_sum->Evaluate(parameters, &cost, NULL);
  ceres::Solve(options, problem, parameters, &summary);
  ml_sum->ParseSolution(parameters);

  std::cout << summary.FullReport() << "\n";
  std::cout << "mu    = " << mu << "\n";
  std::cout << "sigma = " << sigma << "\n";
  return 0;
}
