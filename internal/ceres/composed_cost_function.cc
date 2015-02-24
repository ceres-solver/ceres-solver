// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
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
// Author: richie.stebbing@gmail.com (Richard Stebbing)

#include <numeric>
#include "ceres/composed_cost_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/fixed_array.h"
#include "glog/logging.h"

namespace ceres {

// An object that represents an input (or internal) function to the composed
// cost function. It stores either a cost function or a pointer to a raw
// parameter block. `parameter_block_indices[i]` is the index of the `i`th
// parameter, with respect to all parameters input to the composed cost
// function and is set in `AddInputCostFunction` and `AddInputParameterBlock`.
struct ComposedCostFunction::ComposedCostFunctionInput {
  ComposedCostFunctionInput(CostFunction* g,
                            Ownership owns_g=TAKE_OWNERSHIP)
    : g(CHECK_NOTNULL(g)),
      owns_g(owns_g),
      size(g->num_residuals())
    {}

  ComposedCostFunctionInput(int size)
      : g(NULL),
        owns_g(DO_NOT_TAKE_OWNERSHIP),
        size(size)
    {}

  ~ComposedCostFunctionInput() {
    if (owns_g == DO_NOT_TAKE_OWNERSHIP) {
      g.release();
    }
  }

  internal::scoped_ptr<CostFunction> g;
  Ownership owns_g;
  int size;

  std::vector<int> parameter_block_indices;
};

ComposedCostFunction::ComposedCostFunction(
  CostFunction* f,
  Ownership ownership)
    : f_(CHECK_NOTNULL(f)),
      owns_f_(ownership),
      is_finalised_(false) {
  set_num_residuals(f->num_residuals());

  // `parameter_block_sizes` are set in `AddInternalParameterBlock`.
}

ComposedCostFunction::~ComposedCostFunction() {
  if (owns_f_ == DO_NOT_TAKE_OWNERSHIP) {
    f_.release();
  }

  for (int i = 0; i < inputs_.size(); ++i) {
    delete inputs_[i];
  }
}

void ComposedCostFunction::AddInputCostFunction(
    CostFunction* g,
    const std::vector<double*>& parameter_blocks,
    Ownership ownership) {
  const std::vector<int>& parameter_block_sizes =
    CHECK_NOTNULL(g)->parameter_block_sizes();

  internal::scoped_ptr<ComposedCostFunctionInput> input(
      new ComposedCostFunctionInput(g, ownership));

  // For each parameter block which is an input for `g`, determine its index
  // with respect to all input parameter blocks seen so far.
  for (int i = 0; i < parameter_blocks.size(); ++i) {
    input->parameter_block_indices.push_back(
      AddInternalParameterBlock(parameter_blocks[i],
                                parameter_block_sizes[i]));
  }

  inputs_.push_back(input.release());
}

void ComposedCostFunction::AddInputParameterBlock(double* values, int size) {
  internal::scoped_ptr<ComposedCostFunctionInput> input(
    new ComposedCostFunctionInput(size));

  input->parameter_block_indices.push_back(
    AddInternalParameterBlock(CHECK_NOTNULL(values), size));

  inputs_.push_back(input.release());
}

void ComposedCostFunction::Finalize() {
  CHECK(!is_finalised_);

  // Ensure that all inputs to `f_` have been added.
  CHECK_EQ(f_->parameter_block_sizes().size(), inputs_.size());

  // Ensure that the number of residuals for each input function --- or the
  // size of each raw input block --- is the same as the dimension of the input
  // parameter blocks for `f_`.
  for (int i = 0; i < inputs_.size(); ++i) {
    const ComposedCostFunctionInput& input = *inputs_[i];
    if (input.g != NULL) {
      CHECK_EQ(f_->parameter_block_sizes()[i], input.g->num_residuals());
    } else {
      CHECK_EQ(f_->parameter_block_sizes()[i],
               parameter_block_sizes()[input.parameter_block_indices[0]]);
    }
  }

  // `parameter_to_jacobian_blocks_[k]` stores the information about which
  // internal (input) Jacobian blocks use parameter `k`. Precisely, it is a
  // vector of pairs `(i, j)`, with each pair corresponding to the `j`th
  // parameter of the `i`th input to the composed cost function.
  parameter_to_jacobian_blocks_.resize(parameter_blocks().size());
  for (int i = 0; i < inputs_.size(); ++i) {
    const std::vector<int>& block_indices =
      inputs_[i]->parameter_block_indices;
    for (int j = 0; j < block_indices.size(); ++j) {
      parameter_to_jacobian_blocks_[block_indices[j]].push_back(
        std::make_pair(i, j));
    }
  }

  is_finalised_ = true;
}

bool ComposedCostFunction::Evaluate(const double* const* parameters,
                                    double* residuals,
                                    double** jacobians) const {
  CHECK(is_finalised_);

  // Allocate the memory necessary to evaluate all of the internal (input)
  // residuals.
  const std::vector<int>& int_residual_sizes = f_->parameter_block_sizes();
  const int int_residuals_size = std::accumulate(int_residual_sizes.begin(),
                                                 int_residual_sizes.end(),
                                                 0);

  internal::FixedArray<double> int_residuals_data(int_residuals_size);
  internal::FixedArray<double*> int_residuals(inputs_.size());
  for (int i = 0, cursor = 0; i < inputs_.size(); ++i) {
    int_residuals[i] = &int_residuals_data[cursor];
    cursor += int_residual_sizes[i];
  }

  // If Jacobians are not required then evaluate each internal function and
  // return the residuals immediately.
  if (jacobians == NULL) {
    for (int i = 0; i < inputs_.size(); ++i) {
      if (!InternalEvaluate(i, parameters, int_residuals[i], NULL)) {
        return false;
      }
    }

    return f_->Evaluate(int_residuals.get(), residuals, NULL);
  }

  // Otherwise, determine the number of internal Jacobian blocks ...
  int num_int_jacobians = 0;
  for (int i = 0; i < inputs_.size(); ++i) {
    num_int_jacobians += inputs_[i]->parameter_block_indices.size();
  }

  // ... and setup `int_jacobians`, where `int_jacobians[i][j]` is a
  // pointer to the Jacobian for the `j`th parameter of the `i`th internal
  // function. These pointers are initialised to NULL, but are set once it is
  // determined which ones need to be evaluated.
  internal::FixedArray<double*> int_jacobians_flat(num_int_jacobians);
  std::fill(int_jacobians_flat.begin(), int_jacobians_flat.end(),
            static_cast<double*>(NULL));
  internal::FixedArray<double**> int_jacobians(inputs_.size());
  for (int i = 0, cursor = 0; i < inputs_.size(); ++i) {
    int_jacobians[i] = &int_jacobians_flat[cursor];
    cursor += inputs_[i]->parameter_block_indices.size();
  }

  // Determine which Jacobians for the composed cost function need to be
  // evaluated, and the total size necessary to allocate for the internal
  // Jacobians.
  internal::FixedArray<unsigned char> f_requires_jacobian(inputs_.size());
  std::fill(f_requires_jacobian.begin(), f_requires_jacobian.end(), 0);

  int int_jacobians_data_size = 0;
  for (int i = 0; i < parameter_blocks_.size(); ++i) {
    if (jacobians[i] == NULL) {
      continue;
    }

    const std::vector<std::pair<int, int> >& jacobian_blocks =
      parameter_to_jacobian_blocks_[i];
    for (int j = 0; j < jacobian_blocks.size(); ++j) {
      const std::pair<int, int>& t = jacobian_blocks[j];
      int_jacobians_data_size += inputs_[t.first]->size *
                                 parameter_block_sizes()[i];
      f_requires_jacobian[t.first] = 1;
    }
  }
  internal::FixedArray<double> int_jacobians_data(int_jacobians_data_size);

  // Finally, set each `int_jacobians[i][j]`.
  for (int i = 0, cursor = 0; i < parameter_blocks_.size(); ++i) {
    if (jacobians[i] == NULL) {
      continue;
    }

    const std::vector<std::pair<int, int> >& jacobian_blocks =
      parameter_to_jacobian_blocks_[i];
    for (int j = 0; j < jacobian_blocks.size(); ++j) {
      const std::pair<int, int>& t = jacobian_blocks[j];
      int_jacobians[t.first][t.second] = &int_jacobians_data[cursor];
      cursor += inputs_[t.first]->size * parameter_block_sizes()[i];
    }
  }

  // Determine the total size necessary to allocate for the Jacobians of the
  // composed cost function ...
  int f_jacobians_data_size = 0;
  for (int i = 0; i < inputs_.size(); ++i) {
    if (!f_requires_jacobian[i]) {
      continue;
    }
    f_jacobians_data_size += f_->num_residuals() * int_residual_sizes[i];
  }

  // ... and allocate and setup `f_jacobians`.
  internal::FixedArray<double> f_jacobians_data(f_jacobians_data_size);

  internal::FixedArray<double*> f_jacobians(inputs_.size());
  std::fill(f_jacobians.begin(), f_jacobians.end(),
            static_cast<double*>(NULL));
  for (int i = 0, cursor = 0; i < inputs_.size(); ++i) {
    if (!f_requires_jacobian[i]) {
      continue;
    }
    f_jacobians[i] = &f_jacobians_data[cursor];
    cursor += f_->num_residuals() * int_residual_sizes[i];
  }

  // Evaluate the residuals and Jacobians for the input functions.
  for (int i = 0; i < inputs_.size(); ++i) {
    if (!InternalEvaluate(i, parameters, int_residuals[i], int_jacobians[i])) {
      return false;
    }
  }

  // Evaluate the residuals and Jacobians for the composed function.
  if (!f_->Evaluate(int_residuals.get(), residuals, f_jacobians.get())) {
    return false;
  }

  // Apply the chain rule to set the complete Jacobians.
  for (int i = 0; i < parameter_blocks_.size(); ++i) {
    if (jacobians[i] == NULL) {
      continue;
    }

    MatrixRef J(jacobians[i], f_->num_residuals(), parameter_block_sizes()[i]);
    J.setZero();

    const std::vector<std::pair<int, int> >& jacobian_blocks =
      parameter_to_jacobian_blocks_[i];
    for (int j = 0; j < jacobian_blocks.size(); ++j) {
      const std::pair<int, int>& t = jacobian_blocks[j];
      J += MatrixRef(f_jacobians[t.first],
                     f_->num_residuals(),
                     int_residual_sizes[t.first]) *
           MatrixRef(int_jacobians[t.first][t.second],
                     int_residual_sizes[t.first],
                     parameter_block_sizes()[i]);
    }
  }

  return true;
}

int ComposedCostFunction::AddInternalParameterBlock(
    double* values, int size) {
  // Append `values` to `parameter_blocks_` if necessary and return its index.
  std::vector<double*>::iterator it = std::find(parameter_blocks_.begin(),
                                                parameter_blocks_.end(),
                                                values);
  int j = -1;
  if (it != parameter_blocks_.end()) {
    j = std::distance(parameter_blocks_.begin(), it);
  } else {
    parameter_blocks_.push_back(values);
    mutable_parameter_block_sizes()->push_back(size);
    j = parameter_blocks_.size() - 1;
  }

  return j;
}

bool ComposedCostFunction::InternalEvaluate(int input_index,
                                            const double* const* parameters,
                                            double* residuals,
                                            double** jacobians) const {
  const ComposedCostFunctionInput& input = *inputs_[input_index];
  const std::vector<int>& block_indices = input.parameter_block_indices;

  if (input.g != NULL) {
    internal::FixedArray<const double*> g_parameters(block_indices.size());
    for (int i = 0; i < block_indices.size(); ++i) {
      g_parameters[i] = parameters[block_indices[i]];
    }

    return input.g->Evaluate(g_parameters.get(), residuals, jacobians);
  } else {
    int i = block_indices[0];
    const double* parameter = parameters[i];
    const int block_size = parameter_block_sizes()[i];
    std::copy(parameter, parameter + block_size, residuals);
    if (jacobians != NULL && jacobians[0] != NULL) {
      MatrixRef(jacobians[0], block_size, block_size).setIdentity();
    }

    return true;
  }
}

}  // namespace ceres
