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
//

#ifndef CERES_INTERNAL_CUDA_BLOCK_SPARSE_CRS_VIEW_H_
#define CERES_INTERNAL_CUDA_BLOCK_SPARSE_CRS_VIEW_H_

#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA

#include <memory>

#include "ceres/block_sparse_matrix.h"
#include "ceres/cuda_buffer.h"
#include "ceres/cuda_sparse_matrix.h"
#include "ceres/cuda_streamed_buffer.h"

namespace ceres::internal {
// We use cuSPARSE library for SpMV operations. However, it does not support
// block-sparse format with varying size of the blocks. Thus, we perform the
// following operations in order to compute products of block-sparse matrices
// and dense vectors on gpu:
//  - Once per block-sparse structure update:
//    - Compute CRS structure from block-sparse structure
//    - Compute permutation from block-sparse values to CRS values
//  - Once per block-sparse values update:
//    - Update values in CRS matrix with values of block-sparse matrix
//
// Since there are no constraints on positions of cells in value array of
// block-sparse matrix, a permutation from block-sparse values to CRS
// values is stored explicitly.
//
// Example: given matrix with the following block-structure
//  [ 1 2 | | 6 7 ]
//  [ 3 4 | | 8 9 ]
//  [-----+-+-----]
//  [     |5|     ]
// with values stored as values_block_sparse = [1, 2, 3, 4, 5, 6, 7, 8, 9],
// permutation from block-sparse to CRS is p = [0, 1, 4, 5, 8, 2, 3, 6, 7];
// and i-th block-sparse value has index p[i] in CRS values array
//
// This allows to avoid storing both CRS and block-sparse values in GPU memory.
// Instead, block-sparse values are transferred to gpu memory as a disjoint set
// of small continuous segments with simultaneous permutation of the values into
// correct order
class CERES_NO_EXPORT CudaBlockSparseCRSView {
 public:
  // Initializes internal CRS matrix and permutation from block-sparse to CRS
  // values. The following objects are stored in gpu memory for the whole
  // lifetime of the object
  //  - crs_matrix_: CRS matrix
  //  - permutation_: permutation from block-sparse to CRS value order
  //  (num_nonzeros integer values)
  //  - streamed_buffer_: helper for value updating
  // The following objects are created temporarily during construction:
  //  - CudaBlockSparseStructure: block-sparse structure of block-sparse matrix
  //  - num_rows integer values: row to row-block map
  //  If copy_values flag is set to false, only structure of block-sparse matrix
  //  bsm is captured, and values are left uninitialized
  CudaBlockSparseCRSView(const BlockSparseMatrix& bsm, ContextImpl* context);

  const CudaSparseMatrix* crs_matrix() const { return crs_matrix_.get(); }
  CudaSparseMatrix* mutable_crs_matrix() { return crs_matrix_.get(); }

  // Update values of crs_matrix_ using values of block-sparse matrix.
  // Assumes that bsm has the same block-sparse structure as matrix that was
  // used for construction.
  void UpdateValues(const BlockSparseMatrix& bsm);

 private:
  // Value permutation kernel performs a single element-wise operation per
  // thread, thus performing permutation in blocks of 8 megabytes of
  // block-sparse  values seems reasonable
  static constexpr int kMaxTemporaryArraySize = 1 * 1024 * 1024;
  std::unique_ptr<CudaSparseMatrix> crs_matrix_;
  // Permutation from block-sparse to CRS value order.
  // permutation_[i] = index of i-th block-sparse value in CRS values
  CudaBuffer<int> permutation_;
  CudaStreamedBuffer<double> streamed_buffer_;
};

}  // namespace ceres::internal

#endif  // CERES_NO_CUDA
#endif  // CERES_INTERNAL_CUDA_BLOCK_SPARSE_CRS_VIEW_H_
