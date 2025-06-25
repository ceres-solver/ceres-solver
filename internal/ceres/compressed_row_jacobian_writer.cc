// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
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

#include "ceres/compressed_row_jacobian_writer.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "ceres/casts.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/parameter_block.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"
#include "ceres/scratch_evaluate_preparer.h"

namespace ceres::internal {
void CompressedRowJacobianWriter::PopulateJacobianRowAndColumnBlockVectors(
    const Program* program, CompressedRowSparseMatrix* jacobian) {
  const auto& parameter_blocks = program->parameter_blocks();
  auto* col_blocks = jacobian->mutable_col_blocks();
  col_blocks->resize(parameter_blocks.size());
  int col_pos = 0;
  std::transform(parameter_blocks.begin(),
                 parameter_blocks.end(),
                 col_blocks->begin(),
                 [&col_pos](const ParameterBlock* pb) {
                   CompressedRowSparseMatrix::Block block = {pb->TangentSize(),
                                                             col_pos};
                   col_pos += pb->TangentSize();
                   return block;
                 });

  const auto& residual_blocks = program->residual_blocks();
  auto* row_blocks = jacobian->mutable_row_blocks();
  row_blocks->resize(residual_blocks.size());
  int row_pos = 0;
  std::transform(residual_blocks.begin(),
                 residual_blocks.end(),
                 row_blocks->begin(),
                 [&row_pos](const ResidualBlock* rb) {
                   CompressedRowSparseMatrix::Block block = {rb->NumResiduals(),
                                                             row_pos};
                   row_pos += rb->NumResiduals();
                   return block;
                 });
}

void CompressedRowJacobianWriter::GetOrderedParameterBlocks(
    const Program* program,
    int residual_id,
    std::vector<std::pair<int, int>>* evaluated_jacobian_blocks) {
  auto residual_block = program->residual_blocks()[residual_id];
  const int num_parameter_blocks = residual_block->NumParameterBlocks();

  for (int j = 0; j < num_parameter_blocks; ++j) {
    auto parameter_block = residual_block->parameter_blocks()[j];
    if (!parameter_block->IsConstant()) {
      evaluated_jacobian_blocks->push_back(
          std::make_pair(parameter_block->index(), j));
    }
  }
  std::sort(evaluated_jacobian_blocks->begin(),
            evaluated_jacobian_blocks->end());
}

std::unique_ptr<SparseMatrix> CompressedRowJacobianWriter::CreateJacobian()
    const {
  const auto& residual_blocks = program_->residual_blocks();

  const int total_num_residuals = program_->NumResiduals();
  const int total_num_effective_parameters = program_->NumEffectiveParameters();

  // Count the number of jacobian nonzeros.
  //
  // We used an unsigned int here, so that we can compare it INT_MAX without
  // triggering overflow behaviour.
  unsigned int num_jacobian_nonzeros = total_num_effective_parameters;
  for (auto* residual_block : residual_blocks) {
    const int num_residuals = residual_block->NumResiduals();
    const int num_parameter_blocks = residual_block->NumParameterBlocks();
    for (int j = 0; j < num_parameter_blocks; ++j) {
      auto parameter_block = residual_block->parameter_blocks()[j];
      if (!parameter_block->IsConstant()) {
        num_jacobian_nonzeros += num_residuals * parameter_block->TangentSize();
        if (num_jacobian_nonzeros > std::numeric_limits<int>::max()) {
          LOG(ERROR) << "Unable to create Jacobian matrix: Too many entries in "
                        "the Jacobian matrix. num_jacobian_nonzeros = "
                     << num_jacobian_nonzeros;
          return nullptr;
        }
      }
    }
  }

  // Allocate storage for the jacobian with some extra space at the end.
  // Allocate more space than needed to store the jacobian so that when the LM
  // algorithm adds the diagonal, no reallocation is necessary. This reduces
  // peak memory usage significantly.
  auto jacobian = std::make_unique<CompressedRowSparseMatrix>(
      total_num_residuals,
      total_num_effective_parameters,
      static_cast<int>(num_jacobian_nonzeros));

  // At this stage, the CompressedRowSparseMatrix is an invalid state. But
  // this seems to be the only way to construct it without doing a memory
  // copy.
  int* rows = jacobian->mutable_rows();
  int* cols = jacobian->mutable_cols();

  int row_pos = 0;
  rows[0] = 0;
  for (auto* residual_block : residual_blocks) {
    const int num_parameter_blocks = residual_block->NumParameterBlocks();

    // Count the number of derivatives for a row of this residual block and
    // build a list of active parameter block indices.
    int num_derivatives = 0;
    absl::InlinedVector<int, 4> parameter_indices;
    for (int j = 0; j < num_parameter_blocks; ++j) {
      auto parameter_block = residual_block->parameter_blocks()[j];
      if (!parameter_block->IsConstant()) {
        parameter_indices.push_back(parameter_block->index());
        num_derivatives += parameter_block->TangentSize();
      }
    }

    // Sort the parameters by their position in the state vector.
    std::sort(parameter_indices.begin(), parameter_indices.end());
    if (adjacent_find(parameter_indices.begin(), parameter_indices.end()) !=
        parameter_indices.end()) {
      std::string parameter_block_description;
      for (int j = 0; j < num_parameter_blocks; ++j) {
        auto parameter_block = residual_block->parameter_blocks()[j];
        parameter_block_description += parameter_block->ToString() + "\n";
      }
      LOG(FATAL) << "Ceres internal error: "
                 << "Duplicate parameter blocks detected in a cost function. "
                 << "This should never happen. Please report this to "
                 << "the Ceres developers.\n"
                 << "Residual Block: " << residual_block->ToString() << "\n"
                 << "Parameter Blocks: " << parameter_block_description;
    }

    // Update the row indices.
    const int num_residuals = residual_block->NumResiduals();
    for (int j = 0; j < num_residuals; ++j) {
      rows[row_pos + j + 1] = rows[row_pos + j] + num_derivatives;
    }

    // Iterate over parameter blocks in the order which they occur in the
    // parameter vector. This code mirrors that in Write(), where jacobian
    // values are updated.
    int col_pos = 0;
    for (int parameter_index : parameter_indices) {
      auto parameter_block = program_->parameter_blocks()[parameter_index];
      const int parameter_block_size = parameter_block->TangentSize();

      for (int r = 0; r < num_residuals; ++r) {
        // This is the position in the values array of the jacobian where this
        // row of the jacobian block should go.
        const int column_block_begin = rows[row_pos + r] + col_pos;
        for (int c = 0; c < parameter_block_size; ++c) {
          cols[column_block_begin + c] = parameter_block->delta_offset() + c;
        }
      }
      col_pos += parameter_block_size;
    }
    row_pos += num_residuals;
  }
  CHECK_EQ(num_jacobian_nonzeros - total_num_effective_parameters,
           rows[total_num_residuals]);

  PopulateJacobianRowAndColumnBlockVectors(program_, jacobian.get());

  return jacobian;
}

void CompressedRowJacobianWriter::Write(int residual_id,
                                        int residual_offset,
                                        double** jacobians,
                                        SparseMatrix* base_jacobian) {
  auto* jacobian = down_cast<CompressedRowSparseMatrix*>(base_jacobian);

  double* jacobian_values = jacobian->mutable_values();
  const int* jacobian_rows = jacobian->rows();

  auto residual_block = program_->residual_blocks()[residual_id];
  const int num_residuals = residual_block->NumResiduals();

  std::vector<std::pair<int, int>> evaluated_jacobian_blocks;
  GetOrderedParameterBlocks(program_, residual_id, &evaluated_jacobian_blocks);

  // Where in the current row does the jacobian for a parameter block begin.
  int col_pos = 0;

  // Iterate over the jacobian blocks in increasing order of their
  // positions in the reduced parameter vector.
  for (auto& evaluated_jacobian_block : evaluated_jacobian_blocks) {
    auto parameter_block =
        program_->parameter_blocks()[evaluated_jacobian_block.first];
    const int argument = evaluated_jacobian_block.second;
    const int parameter_block_size = parameter_block->TangentSize();

    // Copy one row of the jacobian block at a time.
    for (int r = 0; r < num_residuals; ++r) {
      // Position of the r^th row of the current jacobian block.
      const double* block_row_begin =
          jacobians[argument] + r * parameter_block_size;

      // Position in the values array of the jacobian where this
      // row of the jacobian block should go.
      double* column_block_begin =
          jacobian_values + jacobian_rows[residual_offset + r] + col_pos;

      std::copy(block_row_begin,
                block_row_begin + parameter_block_size,
                column_block_begin);
    }
    col_pos += parameter_block_size;
  }
}

}  // namespace ceres::internal
