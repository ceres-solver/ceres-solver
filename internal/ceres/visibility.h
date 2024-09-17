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
// Author: kushalav@google.com (Avanish Kushal)
//         sameeragarwal@google.com (Sameer Agarwal)
//
// Functions to manipulate visibility information from the block
// structure of sparse matrices.

#ifndef CERES_INTERNAL_VISIBILITY_H_
#define CERES_INTERNAL_VISIBILITY_H_

#include <memory>
#include <vector>

#include "absl/container/btree_set.h"
#include "ceres/graph.h"
#include "ceres/internal/disable_warnings.h"
#include "ceres/internal/export.h"

namespace ceres::internal {

struct CompressedRowBlockStructure;

// Given a compressed row block structure, computes the set of
// e_blocks "visible" to each f_block. If an e_block co-occurs with an
// f_block in a residual block, it is visible to the f_block. The
// first num_eliminate_blocks columns blocks are e_blocks and the rest
// f_blocks.
//
// In a structure from motion problem, e_blocks correspond to 3D
// points and f_blocks correspond to cameras.
CERES_NO_EXPORT void ComputeVisibility(
    const CompressedRowBlockStructure& block_structure,
    int num_eliminate_blocks,
    std::vector<absl::btree_set<int>>* visibility);

// Given f_block visibility as computed by the ComputeVisibility
// function above, construct and return a graph whose vertices are
// f_blocks and an edge connects two vertices if they have at least one
// e_block in common. The weight of this edge is normalized dot
// product between the visibility vectors of the two
// vertices/f_blocks.
//
// This graph reflects the sparsity structure of reduced camera
// matrix/Schur complement matrix obtained by eliminating the e_blocks
// from the normal equations.
//
// Caller acquires ownership of the returned WeightedGraph pointer
// (heap-allocated).
CERES_NO_EXPORT std::unique_ptr<WeightedGraph<int>> CreateSchurComplementGraph(
    const std::vector<absl::btree_set<int>>& visibility);

}  // namespace ceres::internal

#include "ceres/internal/reenable_warnings.h"

#endif  // CERES_INTERNAL_VISIBILITY_H_
