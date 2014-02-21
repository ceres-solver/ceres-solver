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

#include "ceres/constrained_problem.h"
#include "ceres/problem.h"
#include "ceres/cost_function.h"
#include "ceres/loss_function.h"
#include "ceres/augmented_lagrangian_cost_function.h"

namespace ceres {
namespace experimental {

ConstrainedProblem::~ConstrainedProblem() {
}

ResidualBlockId ConstrainedProblem::AddResidualBlock(
    CostFunction* cost_function,
    LossFunction* loss_function,
    const vector<double*>& parameter_blocks) {
  return problem_.AddResidualBlock(cost_function,
                                   loss_function,
                                   parameter_blocks);
}

ResidualBlockId ConstrainedProblem::AddResidualBlock(
    CostFunction* cost_function,
    LossFunction* loss_function,
    double* x0) {
  return problem_.AddResidualBlock(cost_function,
                                         loss_function,
                                         x0);
}

ResidualBlockId ConstrainedProblem::AddResidualBlock(
    CostFunction* cost_function,
    LossFunction* loss_function,
    double* x0, double* x1) {
  return problem_.AddResidualBlock(cost_function,
                                         loss_function,
                                         x0, x1);
}

ResidualBlockId ConstrainedProblem::AddResidualBlock(
    CostFunction* cost_function,
    LossFunction* loss_function,
    double* x0, double* x1, double* x2) {
  return problem_.AddResidualBlock(cost_function,
                                   loss_function,
                                   x0, x1, x2);
}

ResidualBlockId ConstrainedProblem::AddResidualBlock(
    CostFunction* cost_function,
    LossFunction* loss_function,
    double* x0, double* x1, double* x2, double* x3) {
  return problem_.AddResidualBlock(cost_function,
                                   loss_function,
                                   x0, x1, x2, x3);
}

ResidualBlockId ConstrainedProblem::AddResidualBlock(
    CostFunction* cost_function,
    LossFunction* loss_function,
    double* x0, double* x1, double* x2, double* x3, double* x4) {
  return problem_.AddResidualBlock(cost_function,
                                   loss_function,
                                   x0, x1, x2, x3, x4);
}

ResidualBlockId ConstrainedProblem::AddResidualBlock(
    CostFunction* cost_function,
    LossFunction* loss_function,
    double* x0, double* x1, double* x2, double* x3, double* x4, double* x5) {
  return problem_.AddResidualBlock(cost_function,
                                   loss_function,
                                   x0, x1, x2, x3, x4, x5);
}

ResidualBlockId ConstrainedProblem::AddResidualBlock(
    CostFunction* cost_function,
    LossFunction* loss_function,
    double* x0, double* x1, double* x2, double* x3, double* x4, double* x5,
    double* x6) {
  return problem_.AddResidualBlock(cost_function,
                                   loss_function,
                                   x0, x1, x2, x3, x4, x5, x6);
}

ResidualBlockId ConstrainedProblem::AddResidualBlock(
    CostFunction* cost_function,
    LossFunction* loss_function,
    double* x0, double* x1, double* x2, double* x3, double* x4, double* x5,
    double* x6, double* x7) {
  return problem_.AddResidualBlock(cost_function,
                                   loss_function,
                                   x0, x1, x2, x3, x4, x5, x6, x7);
}

ResidualBlockId ConstrainedProblem::AddResidualBlock(
    CostFunction* cost_function,
    LossFunction* loss_function,
    double* x0, double* x1, double* x2, double* x3, double* x4, double* x5,
    double* x6, double* x7, double* x8) {
  return problem_.AddResidualBlock(cost_function,
                                   loss_function,
                                   x0, x1, x2, x3, x4, x5, x6, x7, x8);
}

ResidualBlockId ConstrainedProblem::AddResidualBlock(
    CostFunction* cost_function,
    LossFunction* loss_function,
    double* x0, double* x1, double* x2, double* x3, double* x4, double* x5,
    double* x6, double* x7, double* x8, double* x9) {
  return problem_.AddResidualBlock(
      cost_function,
      loss_function,
      x0, x1, x2, x3, x4, x5, x6, x7, x8, x9);
}


ConstraintBlockId ConstrainedProblem::AddConstraintBlock(
    CostFunction* cost_function,
    ConstraintType constraint_type,
    const vector<double*>& parameter_blocks) {
  AugmentedLagrangianCostFunction* al_cost_function =
      new AugmentedLagrangianCostFunction(cost_function);
  const ConstraintBlockId constraint_block_id =
      problem_.AddResidualBlock(al_cost_function,
                                NULL,
                                parameter_blocks);
  constraints_[constraint_block_id] = al_cost_function;
  return constraint_block_id;
}

ConstraintBlockId ConstrainedProblem::AddConstraintBlock(
    CostFunction* cost_function,
    ConstraintType constraint_type,
    double* x0) {
  vector<double*> parameter_blocks;
  parameter_blocks.push_back(x0);
  return AddConstraintBlock(cost_function, constraint_type, parameter_blocks);
}

ConstraintBlockId ConstrainedProblem::AddConstraintBlock(
    CostFunction* cost_function,
    ConstraintType constraint_type,
    double* x0, double* x1) {
  vector<double*> parameter_blocks;
  parameter_blocks.push_back(x0);
  parameter_blocks.push_back(x1);
  return AddConstraintBlock(cost_function, constraint_type, parameter_blocks);
}

ConstraintBlockId ConstrainedProblem::AddConstraintBlock(
    CostFunction* cost_function,
    ConstraintType constraint_type,
    double* x0, double* x1, double* x2) {
  vector<double*> parameter_blocks;
  parameter_blocks.push_back(x0);
  parameter_blocks.push_back(x1);
  parameter_blocks.push_back(x2);
  return AddConstraintBlock(cost_function, constraint_type, parameter_blocks);
}

ConstraintBlockId ConstrainedProblem::AddConstraintBlock(
    CostFunction* cost_function,
    ConstraintType constraint_type,
    double* x0, double* x1, double* x2,
    double* x3) {
  vector<double*> parameter_blocks;
  parameter_blocks.push_back(x0);
  parameter_blocks.push_back(x1);
  parameter_blocks.push_back(x2);
  parameter_blocks.push_back(x3);
  return AddConstraintBlock(cost_function, constraint_type, parameter_blocks);
}

ConstraintBlockId ConstrainedProblem::AddConstraintBlock(
    CostFunction* cost_function,
    ConstraintType constraint_type,
    double* x0, double* x1, double* x2,
    double* x3, double* x4) {
  vector<double*> parameter_blocks;
  parameter_blocks.push_back(x0);
  parameter_blocks.push_back(x1);
  parameter_blocks.push_back(x2);
  parameter_blocks.push_back(x3);
  parameter_blocks.push_back(x4);
  return AddConstraintBlock(cost_function, constraint_type, parameter_blocks);
}

ConstraintBlockId ConstrainedProblem::AddConstraintBlock(
    CostFunction* cost_function,
    ConstraintType constraint_type,
    double* x0, double* x1, double* x2,
    double* x3, double* x4, double* x5) {
  vector<double*> parameter_blocks;
  parameter_blocks.push_back(x0);
  parameter_blocks.push_back(x1);
  parameter_blocks.push_back(x2);
  parameter_blocks.push_back(x3);
  parameter_blocks.push_back(x4);
  return AddConstraintBlock(cost_function, constraint_type, parameter_blocks);
}

ConstraintBlockId ConstrainedProblem::AddConstraintBlock(
    CostFunction* cost_function,
    ConstraintType constraint_type,
    double* x0, double* x1, double* x2,
    double* x3, double* x4, double* x5,
    double* x6) {
  vector<double*> parameter_blocks;
  parameter_blocks.push_back(x0);
  parameter_blocks.push_back(x1);
  parameter_blocks.push_back(x2);
  parameter_blocks.push_back(x3);
  parameter_blocks.push_back(x4);
  parameter_blocks.push_back(x5);
  parameter_blocks.push_back(x6);
  return AddConstraintBlock(cost_function, constraint_type, parameter_blocks);
}

ConstraintBlockId ConstrainedProblem::AddConstraintBlock(
    CostFunction* cost_function,
    ConstraintType constraint_type,
    double* x0, double* x1, double* x2,
    double* x3, double* x4, double* x5,
    double* x6, double* x7) {
  vector<double*> parameter_blocks;
  parameter_blocks.push_back(x0);
  parameter_blocks.push_back(x1);
  parameter_blocks.push_back(x2);
  parameter_blocks.push_back(x3);
  parameter_blocks.push_back(x4);
  parameter_blocks.push_back(x5);
  parameter_blocks.push_back(x6);
  parameter_blocks.push_back(x7);
  return AddConstraintBlock(cost_function, constraint_type, parameter_blocks);
}

ConstraintBlockId ConstrainedProblem::AddConstraintBlock(
    CostFunction* cost_function,
    ConstraintType constraint_type,
    double* x0, double* x1, double* x2,
    double* x3, double* x4, double* x5,
    double* x6, double* x7, double* x8) {
  vector<double*> parameter_blocks;
  parameter_blocks.push_back(x0);
  parameter_blocks.push_back(x1);
  parameter_blocks.push_back(x2);
  parameter_blocks.push_back(x3);
  parameter_blocks.push_back(x4);
  parameter_blocks.push_back(x5);
  parameter_blocks.push_back(x6);
  parameter_blocks.push_back(x7);
  parameter_blocks.push_back(x8);
  return AddConstraintBlock(cost_function, constraint_type, parameter_blocks);
}

ConstraintBlockId ConstrainedProblem::AddConstraintBlock(
    CostFunction* cost_function,
    ConstraintType constraint_type,
    double* x0, double* x1, double* x2,
    double* x3, double* x4, double* x5,
    double* x6, double* x7, double* x8,
    double* x9) {
  vector<double*> parameter_blocks;
  parameter_blocks.push_back(x0);
  parameter_blocks.push_back(x1);
  parameter_blocks.push_back(x2);
  parameter_blocks.push_back(x3);
  parameter_blocks.push_back(x4);
  parameter_blocks.push_back(x5);
  parameter_blocks.push_back(x6);
  parameter_blocks.push_back(x7);
  parameter_blocks.push_back(x8);
  parameter_blocks.push_back(x9);
  return AddConstraintBlock(cost_function, constraint_type, parameter_blocks);
}


void ConstrainedProblem::AddParameterBlock(double* values, int size) {
  problem_.AddParameterBlock(values, size);
}

void ConstrainedProblem::AddParameterBlock(
    double* values,
    int size,
    LocalParameterization* local_parameterization) {
  problem_.AddParameterBlock(values, size, local_parameterization);
};

void ConstrainedProblem::SetParameterBlockConstant(double* values) {
  problem_.SetParameterBlockConstant(values);
}

void ConstrainedProblem::SetParameterBlockVariable(double* values) {
  problem_.SetParameterBlockVariable(values);
}

void ConstrainedProblem::SetParameterization(
    double* values,
    LocalParameterization* local_parameterization) {
  problem_.SetParameterization(values, local_parameterization);
}

const LocalParameterization* ConstrainedProblem::GetParameterization(
    double* values) const {
  return problem_.GetParameterization(values);
}

void ConstrainedProblem::SetParameterLowerBound(double* values,
                                                int index,
                                                double lower_bound) {
  problem_.SetParameterLowerBound(values, index, lower_bound);
}

void ConstrainedProblem::SetParameterUpperBound(double* values,
                                                int index,
                                                double upper_bound) {
  problem_.SetParameterUpperBound(values, index, upper_bound);
}

// TODO(sameeragarwal): The following four methods are over estimates
// since they will include the slack variables and the augmented
// lagrangian terms.
int ConstrainedProblem::NumParameterBlocks() const {
  return problem_.NumParameterBlocks();
}

int ConstrainedProblem::NumParameters() const {
  return problem_.NumParameters();
}

int ConstrainedProblem::NumResidualBlocks() const {
  return problem_.NumResidualBlocks();
}

int ConstrainedProblem::NumResiduals() const {
  return problem_.NumResiduals();
}

int ConstrainedProblem::ParameterBlockSize(const double* values) const {
  return problem_.ParameterBlockSize(values);
}

int ConstrainedProblem::ParameterBlockLocalSize(const double* values) const {
  return problem_.ParameterBlockLocalSize(values);
}

}  // namespace ceres
}  // namespace experimental
