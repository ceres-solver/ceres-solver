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
// Author: richie.stebbing@gmail.com (Richard Stebbing)

#include "ceres/dynamic_compressed_row_jacobian_writer.h"

#include "ceres/casts.h"
#include "ceres/dynamic_compressed_row_sparse_matrix.h"
#include "ceres/parameter_block.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"

namespace ceres {
namespace internal {

SparseMatrix* DynamicCompressedRowJacobianWriter::CreateJacobian() const {
  const vector<ResidualBlock*>& residual_blocks =
      program_->residual_blocks();

  // Initialize `jacobian` with zero number of `max_num_nonzeros`.
  const int num_residuals = program_->NumResiduals();
  const int num_effective_parameters = program_->NumEffectiveParameters();

  DynamicCompressedRowSparseMatrix* jacobian =
      new DynamicCompressedRowSparseMatrix(num_residuals,
                                           num_effective_parameters,
                                           0);

  // Populate the row and column block vectors for use by block
  // oriented ordering algorithms. This is useful when
  // Solver::Options::use_block_amd = true.
  const vector<ParameterBlock*>& parameter_blocks =
      program_->parameter_blocks();
  vector<int>& col_blocks = *(jacobian->mutable_col_blocks());
  col_blocks.resize(parameter_blocks.size());
  for (int i = 0; i <  parameter_blocks.size(); ++i) {
    col_blocks[i] = parameter_blocks[i]->LocalSize();
  }

  vector<int>& row_blocks = *(jacobian->mutable_row_blocks());
  row_blocks.resize(residual_blocks.size());
  for (int i = 0; i <  residual_blocks.size(); ++i) {
    row_blocks[i] = residual_blocks[i]->NumResiduals();
  }

  return jacobian;
}

void DynamicCompressedRowJacobianWriter::Write(int residual_id,
                                               int residual_offset,
                                               double **jacobians,
                                               SparseMatrix* base_jacobian) {
  DynamicCompressedRowSparseMatrix* jacobian =
    down_cast<DynamicCompressedRowSparseMatrix*>(base_jacobian);

  // Get the `residual_block` of interest.
  const ResidualBlock* residual_block =
      program_->residual_blocks()[residual_id];
  const int num_parameter_blocks = residual_block->NumParameterBlocks();
  const int num_residuals = residual_block->NumResiduals();

  // It is necessary to determine the order of the jacobian blocks before
  // inserting them into the DynamicCompressedRowSparseMatrix.
  vector<pair<int, int> > evaluated_jacobian_blocks;
  for (int j = 0; j < num_parameter_blocks; ++j) {
    const ParameterBlock* parameter_block =
        residual_block->parameter_blocks()[j];
    if (!parameter_block->IsConstant()) {
      evaluated_jacobian_blocks.push_back(
          make_pair(parameter_block->index(), j));
    }
  }
  sort(evaluated_jacobian_blocks.begin(), evaluated_jacobian_blocks.end());

  // `residual_offset` is the residual row in the global jacobian.
  // Empty the jacobian rows.
  jacobian->ClearRows(residual_offset, num_residuals);

  // Iterate over each parameter block.
  for (int i = 0; i < evaluated_jacobian_blocks.size(); ++i) {
    const ParameterBlock* parameter_block =
        program_->parameter_blocks()[evaluated_jacobian_blocks[i].first];
    const int argument = evaluated_jacobian_blocks[i].second;
    const int parameter_block_size = parameter_block->LocalSize();

    // For each parameter block only insert its non-zero entries.
    for (int r = 0; r < num_residuals; ++r) {
      for (int c = 0; c < parameter_block_size; ++c) {
        const double& v = jacobians[argument][r * parameter_block_size + c];
        // Only insert non-zero entries.
        if (v != 0.0) {
          jacobian->InsertEntry(
            residual_offset + r, parameter_block->delta_offset() + c, v);
        }
      }
    }
  }

}

}  // namespace internal
}  // namespace ceres
