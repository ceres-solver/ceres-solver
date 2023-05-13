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
// Authors: dmitriy.korchemkin@gmail.com (Dmitriy Korchemkin)

#include "ceres/cuda_partitioned_block_sparse_crs_view.h"

#ifndef CERES_NO_CUDA

#include "ceres/cuda_block_structure.h"
#include "ceres/cuda_kernels.h"

namespace ceres::internal {

CudaPartitionedBlockSparseCRSView::CudaPartitionedBlockSparseCRSView(
    const BlockSparseMatrix& bsm,
    const int num_col_blocks_e,
    ContextImpl* context)
    : permutation_(context, bsm.num_nonzeros()),
      streamed_buffer_(context, kMaxTemporaryArraySize) {
  const auto& bs = *bsm.block_structure();
  CudaBlockSparseStructure block_structure(bs, context);
  // Determine number of non-zeros in left submatrix
  // Row-blocks are at least 1 row high, thus we can use a temporary array of
  // num_rows for ComputeNonZerosInColumnBlockSubMatrix; and later reuse it for
  // FillCRSStructurePartitioned
  const int num_rows = bsm.num_rows();
  CudaBuffer<int> temp_rows(context, num_rows);
  const int num_nonzeros_e =
      ComputeNonZerosInColumnBlockSubMatrix(block_structure.num_row_blocks(),
                                            num_col_blocks_e,
                                            block_structure.row_block_offsets(),
                                            block_structure.cells(),
                                            block_structure.row_blocks(),
                                            block_structure.col_blocks(),
                                            temp_rows.data(),
                                            context->DefaultStream());
  const int num_nonzeros_f = bsm.num_nonzeros() - num_nonzeros_e;

  const int num_cols_e = num_col_blocks_e < bs.cols.size()
                             ? bs.cols[num_col_blocks_e].position
                             : bsm.num_cols();
  const int num_cols_f = bsm.num_cols() - num_cols_e;
  crs_matrix_e_ = std::make_unique<CudaSparseMatrix>(
      num_rows, num_cols_e, num_nonzeros_e, context);
  crs_matrix_f_ = std::make_unique<CudaSparseMatrix>(
      num_rows, num_cols_f, num_nonzeros_f, context);

  FillCRSStructurePartitioned(block_structure.num_row_blocks(),
                              num_col_blocks_e,
                              num_cols_e,
                              num_rows,
                              block_structure.row_block_offsets(),
                              block_structure.cells(),
                              block_structure.row_blocks(),
                              block_structure.col_blocks(),
                              crs_matrix_e_->mutable_rows(),
                              crs_matrix_e_->mutable_cols(),
                              crs_matrix_f_->mutable_rows(),
                              crs_matrix_f_->mutable_cols(),
                              temp_rows.data(),
                              permutation_.data(),
                              context->DefaultStream());
  UpdateValues(bsm);
}
void CudaPartitionedBlockSparseCRSView::UpdateValues(
    const BlockSparseMatrix& bsm) {
  streamed_buffer_.CopyToGpu(
      bsm.values(),
      bsm.num_nonzeros(),
      [permutation = permutation_.data(),
       values_e = crs_matrix_e_->mutable_values(),
       values_f = crs_matrix_f_->mutable_values()](
          const double* values, int num_values, int offset, auto stream) {
        PermuteValuesPartitioned(offset,
                                 num_values,
                                 permutation,
                                 values,
                                 values_e,
                                 values_f,
                                 stream);
      });
}

}  // namespace ceres::internal
#endif  // CERES_NO_CUDA
