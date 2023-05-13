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

#include "ceres/cuda_block_sparse_crs_view.h"

#ifndef CERES_NO_CUDA

#include "ceres/cuda_block_structure.h"
#include "ceres/cuda_kernels.h"

namespace ceres::internal {

CudaBlockSparseCRSView::CudaBlockSparseCRSView(const BlockSparseMatrix& bsm,
                                               ContextImpl* context)
    : permutation_(context, bsm.num_nonzeros()),
      streamed_buffer_(context, kMaxTemporaryArraySize) {
  CudaBlockSparseStructure block_structure(*bsm.block_structure(), context);
  crs_matrix_ = std::make_unique<CudaSparseMatrix>(
      bsm.num_rows(), bsm.num_cols(), bsm.num_nonzeros(), context);
  FillCRSStructure(block_structure.num_row_blocks(),
                   bsm.num_rows(),
                   block_structure.row_block_offsets(),
                   block_structure.cells(),
                   block_structure.row_blocks(),
                   block_structure.col_blocks(),
                   crs_matrix_->mutable_rows(),
                   crs_matrix_->mutable_cols(),
                   permutation_.data(),
                   context->DefaultStream());
  UpdateValues(bsm);
}
void CudaBlockSparseCRSView::UpdateValues(const BlockSparseMatrix& bsm) {
  streamed_buffer_.CopyToGpu(
      bsm.values(),
      bsm.num_nonzeros(),
      [permutation = permutation_.data(),
       values_to = crs_matrix_->mutable_values()](
          const double* values, int num_values, int offset, auto stream) {
        PermuteValues(
            num_values, permutation + offset, values, values_to, stream);
      });
}

}  // namespace ceres::internal
#endif  // CERES_NO_CUDA
