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
// Author: sameeragarwal@google.com (Sameer Agarwal)

// TODO(sameeragarwal): Currently this is a single file compiling
// everything. The class is small enough that it is not a problem IMO,
// but this can be easily done the same way the schur eliminator is
// being templated and specialized, controlled by the same macro.

#include "ceres/internal/eigen.h"
#include "ceres/partitioned_matrix_view.h"
#include "ceres/partitioned_matrix_view_impl.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

PartitionedMatrixViewBase*
PartitionedMatrixViewBase::Create(const LinearSolver::Options& options,
                                  const BlockSparseMatrix& matrix_) {
  const int num_col_blocks_e = options.elimination_groups[0];

  if ((options.row_block_size == 2) &&
      (options.e_block_size == 2) &&
      (options.f_block_size == 2)) {
    return new PartitionedMatrixView<2, 2, 2>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == 2) &&
      (options.e_block_size == 2) &&
      (options.f_block_size == 3)) {
    return new PartitionedMatrixView<2, 2, 3>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == 2) &&
      (options.e_block_size == 2) &&
      (options.f_block_size == 4)) {
    return new PartitionedMatrixView<2, 2, 4>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == 2) &&
      (options.e_block_size == 2) &&
      (options.f_block_size == Eigen::Dynamic)) {
    return new PartitionedMatrixView<2, 2, Eigen::Dynamic>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == 2) &&
      (options.e_block_size == 3) &&
      (options.f_block_size == 3)) {
    return new PartitionedMatrixView<2, 3, 3>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == 2) &&
      (options.e_block_size == 3) &&
      (options.f_block_size == 4)) {
    return new PartitionedMatrixView<2, 3, 4>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == 2) &&
      (options.e_block_size == 3) &&
      (options.f_block_size == 9)) {
    return new PartitionedMatrixView<2, 3, 9>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == 2) &&
      (options.e_block_size == 3) &&
      (options.f_block_size == Eigen::Dynamic)) {
    return new PartitionedMatrixView<2, 3, Eigen::Dynamic>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == 2) &&
      (options.e_block_size == 4) &&
      (options.f_block_size == 3)) {
    return new PartitionedMatrixView<2, 4, 3>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == 2) &&
      (options.e_block_size == 4) &&
      (options.f_block_size == 4)) {
    return new PartitionedMatrixView<2, 4, 4>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == 2) &&
      (options.e_block_size == 4) &&
      (options.f_block_size == Eigen::Dynamic)) {
    return new PartitionedMatrixView<2, 4, Eigen::Dynamic>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == 2) &&
      (options.e_block_size == Eigen::Dynamic) &&
      (options.f_block_size == Eigen::Dynamic)) {
    return new PartitionedMatrixView<2, Eigen::Dynamic, Eigen::Dynamic>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == 4) &&
      (options.e_block_size == 4) &&
      (options.f_block_size == 2)) {
    return new PartitionedMatrixView<4, 4, 2>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == 4) &&
      (options.e_block_size == 4) &&
      (options.f_block_size == 3)) {
    return new PartitionedMatrixView<4, 4, 3>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == 4) &&
      (options.e_block_size == 4) &&
      (options.f_block_size == 4)) {
    return new PartitionedMatrixView<4, 4, 4>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == 4) &&
      (options.e_block_size == 4) &&
      (options.f_block_size == Eigen::Dynamic)) {
    return new PartitionedMatrixView<4, 4, Eigen::Dynamic>(matrix_, num_col_blocks_e);
  }
  if ((options.row_block_size == Eigen::Dynamic) &&
      (options.e_block_size == Eigen::Dynamic) &&
      (options.f_block_size == Eigen::Dynamic)) {
    return new PartitionedMatrixView<Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic>(matrix_, num_col_blocks_e);
  }

   VLOG(1) << "Template specializations not found for <"
          << options.row_block_size << ","
          << options.e_block_size << ","
          << options.f_block_size << ">";
   return  new PartitionedMatrixView<Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic>(matrix_, num_col_blocks_e);
}

}  // namespace internal
}  // namespace ceres
