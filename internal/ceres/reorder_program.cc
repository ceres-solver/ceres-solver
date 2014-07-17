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

#include "ceres/reorder_program.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "ceres/cxsparse.h"
#include "ceres/internal/port.h"
#include "ceres/ordered_groups.h"
#include "ceres/parameter_block.h"
#include "ceres/parameter_block_ordering.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"
#include "ceres/solver.h"
#include "ceres/suitesparse.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/types.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {
namespace {

// Find the minimum index of any parameter block to the given residual.
// Parameter blocks that have indices greater than num_eliminate_blocks are
// considered to have an index equal to num_eliminate_blocks.
static int MinParameterBlock(const ResidualBlock* residual_block,
                             int num_eliminate_blocks) {
  int min_parameter_block_position = num_eliminate_blocks;
  for (int i = 0; i < residual_block->NumParameterBlocks(); ++i) {
    ParameterBlock* parameter_block = residual_block->parameter_blocks()[i];
    if (!parameter_block->IsConstant()) {
      CHECK_NE(parameter_block->index(), -1)
          << "Did you forget to call Program::SetParameterOffsetsAndIndex()? "
          << "This is a Ceres bug; please contact the developers!";
      min_parameter_block_position = std::min(parameter_block->index(),
                                              min_parameter_block_position);
    }
  }
  return min_parameter_block_position;
}

void OrderingForSparseNormalCholeskyUsingSuiteSparse(
    const TripletSparseMatrix& tsm_block_jacobian_transpose,
    const vector<ParameterBlock*>& parameter_blocks,
    const ParameterBlockOrdering& parameter_block_ordering,
    int* ordering) {
#ifdef CERES_NO_SUITESPARSE
  LOG(FATAL) << "Congratulations, you found a Ceres bug! "
             << "Please report this error to the developers.";
#else
  SuiteSparse ss;
  cholmod_sparse* block_jacobian_transpose =
      ss.CreateSparseMatrix(
          const_cast<TripletSparseMatrix*>(&tsm_block_jacobian_transpose));

  // No CAMD or the user did not supply a useful ordering, then just
  // use regular AMD.
  if (parameter_block_ordering.NumGroups() <= 1 ||
      !SuiteSparse::IsConstrainedApproximateMinimumDegreeOrderingAvailable()) {
    ss.ApproximateMinimumDegreeOrdering(block_jacobian_transpose, &ordering[0]);
  } else {
    vector<int> constraints;
    for (int i = 0; i < parameter_blocks.size(); ++i) {
      constraints.push_back(
          parameter_block_ordering.GroupId(
              parameter_blocks[i]->mutable_user_state()));
    }
    ss.ConstrainedApproximateMinimumDegreeOrdering(block_jacobian_transpose,
                                                   &constraints[0],
                                                   ordering);
  }

  ss.Free(block_jacobian_transpose);
#endif  // CERES_NO_SUITESPARSE
}

void OrderingForSparseNormalCholeskyUsingCXSparse(
    const TripletSparseMatrix& tsm_block_jacobian_transpose,
    int* ordering) {
#ifdef CERES_NO_CXSPARSE
  LOG(FATAL) << "Congratulations, you found a Ceres bug! "
             << "Please report this error to the developers.";
#else  // CERES_NO_CXSPARSE
  // CXSparse works with J'J instead of J'. So compute the block
  // sparsity for J'J and compute an approximate minimum degree
  // ordering.
  CXSparse cxsparse;
  cs_di* block_jacobian_transpose;
  block_jacobian_transpose =
      cxsparse.CreateSparseMatrix(
            const_cast<TripletSparseMatrix*>(&tsm_block_jacobian_transpose));
  cs_di* block_jacobian = cxsparse.TransposeMatrix(block_jacobian_transpose);
  cs_di* block_hessian =
      cxsparse.MatrixMatrixMultiply(block_jacobian_transpose, block_jacobian);
  cxsparse.Free(block_jacobian);
  cxsparse.Free(block_jacobian_transpose);

  cxsparse.ApproximateMinimumDegreeOrdering(block_hessian, ordering);
  cxsparse.Free(block_hessian);
#endif  // CERES_NO_CXSPARSE
}

}  // namespace

bool ApplyOrdering(const ProblemImpl::ParameterMap& parameter_map,
                   const ParameterBlockOrdering& ordering,
                   Program* program,
                   string* error) {
  const int num_parameter_blocks =  program->NumParameterBlocks();
  if (ordering.NumElements() != num_parameter_blocks) {
    *error = StringPrintf("User specified ordering does not have the same "
                          "number of parameters as the problem. The problem"
                          "has %d blocks while the ordering has %d blocks.",
                          num_parameter_blocks,
                          ordering.NumElements());
    return false;
  }

  vector<ParameterBlock*>* parameter_blocks =
      program->mutable_parameter_blocks();
  parameter_blocks->clear();

  const map<int, set<double*> >& groups =
      ordering.group_to_elements();

  for (map<int, set<double*> >::const_iterator group_it = groups.begin();
       group_it != groups.end();
       ++group_it) {
    const set<double*>& group = group_it->second;
    for (set<double*>::const_iterator parameter_block_ptr_it = group.begin();
         parameter_block_ptr_it != group.end();
         ++parameter_block_ptr_it) {
      ProblemImpl::ParameterMap::const_iterator parameter_block_it =
          parameter_map.find(*parameter_block_ptr_it);
      if (parameter_block_it == parameter_map.end()) {
        *error = StringPrintf("User specified ordering contains a pointer "
                              "to a double that is not a parameter block in "
                              "the problem. The invalid double is in group: %d",
                              group_it->first);
        return false;
      }
      parameter_blocks->push_back(parameter_block_it->second);
    }
  }
  return true;
}

bool LexicographicallyOrderResidualBlocks(const int num_eliminate_blocks,
                                          Program* program,
                                          string* error) {
  CHECK_GE(num_eliminate_blocks, 1)
      << "Congratulations, you found a Ceres bug! Please report this error "
      << "to the developers.";

  // Create a histogram of the number of residuals for each E block. There is an
  // extra bucket at the end to catch all non-eliminated F blocks.
  vector<int> residual_blocks_per_e_block(num_eliminate_blocks + 1);
  vector<ResidualBlock*>* residual_blocks = program->mutable_residual_blocks();
  vector<int> min_position_per_residual(residual_blocks->size());
  for (int i = 0; i < residual_blocks->size(); ++i) {
    ResidualBlock* residual_block = (*residual_blocks)[i];
    int position = MinParameterBlock(residual_block, num_eliminate_blocks);
    min_position_per_residual[i] = position;
    DCHECK_LE(position, num_eliminate_blocks);
    residual_blocks_per_e_block[position]++;
  }

  // Run a cumulative sum on the histogram, to obtain offsets to the start of
  // each histogram bucket (where each bucket is for the residuals for that
  // E-block).
  vector<int> offsets(num_eliminate_blocks + 1);
  std::partial_sum(residual_blocks_per_e_block.begin(),
                   residual_blocks_per_e_block.end(),
                   offsets.begin());
  CHECK_EQ(offsets.back(), residual_blocks->size())
      << "Congratulations, you found a Ceres bug! Please report this error "
      << "to the developers.";

  CHECK(find(residual_blocks_per_e_block.begin(),
             residual_blocks_per_e_block.end() - 1, 0) !=
        residual_blocks_per_e_block.end())
      << "Congratulations, you found a Ceres bug! Please report this error "
      << "to the developers.";

  // Fill in each bucket with the residual blocks for its corresponding E block.
  // Each bucket is individually filled from the back of the bucket to the front
  // of the bucket. The filling order among the buckets is dictated by the
  // residual blocks. This loop uses the offsets as counters; subtracting one
  // from each offset as a residual block is placed in the bucket. When the
  // filling is finished, the offset pointerts should have shifted down one
  // entry (this is verified below).
  vector<ResidualBlock*> reordered_residual_blocks(
      (*residual_blocks).size(), static_cast<ResidualBlock*>(NULL));
  for (int i = 0; i < residual_blocks->size(); ++i) {
    int bucket = min_position_per_residual[i];

    // Decrement the cursor, which should now point at the next empty position.
    offsets[bucket]--;

    // Sanity.
    CHECK(reordered_residual_blocks[offsets[bucket]] == NULL)
        << "Congratulations, you found a Ceres bug! Please report this error "
        << "to the developers.";

    reordered_residual_blocks[offsets[bucket]] = (*residual_blocks)[i];
  }

  // Sanity check #1: The difference in bucket offsets should match the
  // histogram sizes.
  for (int i = 0; i < num_eliminate_blocks; ++i) {
    CHECK_EQ(residual_blocks_per_e_block[i], offsets[i + 1] - offsets[i])
        << "Congratulations, you found a Ceres bug! Please report this error "
        << "to the developers.";
  }
  // Sanity check #2: No NULL's left behind.
  for (int i = 0; i < reordered_residual_blocks.size(); ++i) {
    CHECK(reordered_residual_blocks[i] != NULL)
        << "Congratulations, you found a Ceres bug! Please report this error "
        << "to the developers.";
  }

  // Now that the residuals are collected by E block, swap them in place.
  swap(*program->mutable_residual_blocks(), reordered_residual_blocks);
  return true;
}

void MaybeReorderSchurComplementColumnsUsingSuiteSparse(
    const ParameterBlockOrdering& parameter_block_ordering,
    Program* program) {
  // Pre-order the columns corresponding to the schur complement if
  // possible.
#ifndef CERES_NO_SUITESPARSE
  SuiteSparse ss;
  if (!SuiteSparse::IsConstrainedApproximateMinimumDegreeOrderingAvailable()) {
    return;
  }

  vector<int> constraints;
  vector<ParameterBlock*>& parameter_blocks =
      *(program->mutable_parameter_blocks());

  for (int i = 0; i < parameter_blocks.size(); ++i) {
    constraints.push_back(
        parameter_block_ordering.GroupId(
            parameter_blocks[i]->mutable_user_state()));
  }

  // Renumber the entries of constraints to be contiguous integers
  // as camd requires that the group ids be in the range [0,
  // parameter_blocks.size() - 1].
  MapValuesToContiguousRange(constraints.size(), &constraints[0]);

  // Set the offsets and index for CreateJacobianSparsityTranspose.
  program->SetParameterOffsetsAndIndex();
  // Compute a block sparse presentation of J'.
  scoped_ptr<TripletSparseMatrix> tsm_block_jacobian_transpose(
      program->CreateJacobianBlockSparsityTranspose());


  cholmod_sparse* block_jacobian_transpose =
      ss.CreateSparseMatrix(tsm_block_jacobian_transpose.get());

  vector<int> ordering(parameter_blocks.size(), 0);
  ss.ConstrainedApproximateMinimumDegreeOrdering(block_jacobian_transpose,
                                                 &constraints[0],
                                                 &ordering[0]);
  ss.Free(block_jacobian_transpose);

  const vector<ParameterBlock*> parameter_blocks_copy(parameter_blocks);
  for (int i = 0; i < program->NumParameterBlocks(); ++i) {
    parameter_blocks[i] = parameter_blocks_copy[ordering[i]];
  }
#endif
}

bool ReorderProgramForSchurTypeLinearSolver(
    const LinearSolverType linear_solver_type,
    const SparseLinearAlgebraLibraryType sparse_linear_algebra_library_type,
    const ProblemImpl::ParameterMap& parameter_map,
    ParameterBlockOrdering* parameter_block_ordering,
    Program* program,
    string* error) {
  if (parameter_block_ordering->NumGroups() == 1) {
    // If the user supplied an parameter_block_ordering with just one
    // group, it is equivalent to the user supplying NULL as an
    // parameter_block_ordering. Ceres is completely free to choose the
    // parameter block ordering as it sees fit. For Schur type solvers,
    // this means that the user wishes for Ceres to identify the
    // e_blocks, which we do by computing a maximal independent set.
    vector<ParameterBlock*> schur_ordering;
    const int num_eliminate_blocks =
        ComputeStableSchurOrdering(*program, &schur_ordering);

    CHECK_EQ(schur_ordering.size(), program->NumParameterBlocks())
        << "Congratulations, you found a Ceres bug! Please report this error "
        << "to the developers.";

    // Update the parameter_block_ordering object.
    for (int i = 0; i < schur_ordering.size(); ++i) {
      double* parameter_block = schur_ordering[i]->mutable_user_state();
      const int group_id = (i < num_eliminate_blocks) ? 0 : 1;
      parameter_block_ordering->AddElementToGroup(parameter_block, group_id);
    }

    // We could call ApplyOrdering but this is cheaper and
    // simpler.
    swap(*program->mutable_parameter_blocks(), schur_ordering);
  } else {
    // The user provided an ordering with more than one elimination
    // group. Trust the user and apply the ordering.
    if (!ApplyOrdering(parameter_map,
                       *parameter_block_ordering,
                       program,
                       error)) {
      return false;
    }
  }

  if (linear_solver_type == SPARSE_SCHUR &&
      sparse_linear_algebra_library_type == SUITE_SPARSE) {
    MaybeReorderSchurComplementColumnsUsingSuiteSparse(
        *parameter_block_ordering,
        program);
  }

  program->SetParameterOffsetsAndIndex();
  // Schur type solvers also require that their residual blocks be
  // lexicographically ordered.
  const int num_eliminate_blocks =
      parameter_block_ordering->group_to_elements().begin()->second.size();
  if (!LexicographicallyOrderResidualBlocks(num_eliminate_blocks,
                                            program,
                                            error)) {
    return false;
  }

  program->SetParameterOffsetsAndIndex();
  return true;
}

bool ReorderProgramForSparseNormalCholesky(
    const SparseLinearAlgebraLibraryType sparse_linear_algebra_library_type,
    const ParameterBlockOrdering& parameter_block_ordering,
    Program* program,
    string* error) {

  if (sparse_linear_algebra_library_type != SUITE_SPARSE &&
      sparse_linear_algebra_library_type != CX_SPARSE &&
      sparse_linear_algebra_library_type != EIGEN_SPARSE) {
    *error = "Unknown sparse linear algebra library.";
    return false;
  }

  // For Eigen, there is nothing to do. This is because Eigen in its
  // current stable version does not expose a method for doing
  // symbolic analysis on pre-ordered matrices, so a block
  // pre-ordering is a bit pointless.
  //
  // The dev version as recently as July 20, 2014 has support for
  // pre-ordering. Once this becomes more widespread, or we add
  // support for detecting Eigen versions, we can add support for this
  // along the lines of CXSparse.
  if (sparse_linear_algebra_library_type == EIGEN_SPARSE) {
    program->SetParameterOffsetsAndIndex();
    return true;
  }

  // Set the offsets and index for CreateJacobianSparsityTranspose.
  program->SetParameterOffsetsAndIndex();
  // Compute a block sparse presentation of J'.
  scoped_ptr<TripletSparseMatrix> tsm_block_jacobian_transpose(
      program->CreateJacobianBlockSparsityTranspose());

  vector<int> ordering(program->NumParameterBlocks(), 0);
  vector<ParameterBlock*>& parameter_blocks =
      *(program->mutable_parameter_blocks());

  if (sparse_linear_algebra_library_type == SUITE_SPARSE) {
    OrderingForSparseNormalCholeskyUsingSuiteSparse(
        *tsm_block_jacobian_transpose,
        parameter_blocks,
        parameter_block_ordering,
        &ordering[0]);
  } else if (sparse_linear_algebra_library_type == CX_SPARSE){
    OrderingForSparseNormalCholeskyUsingCXSparse(
        *tsm_block_jacobian_transpose,
        &ordering[0]);
  }

  // Apply ordering.
  const vector<ParameterBlock*> parameter_blocks_copy(parameter_blocks);
  for (int i = 0; i < program->NumParameterBlocks(); ++i) {
    parameter_blocks[i] = parameter_blocks_copy[ordering[i]];
  }

  program->SetParameterOffsetsAndIndex();
  return true;
}

}  // namespace internal
}  // namespace ceres
