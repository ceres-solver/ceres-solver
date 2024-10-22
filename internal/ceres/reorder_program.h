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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_INTERNAL_REORDER_PROGRAM_H_
#define CERES_INTERNAL_REORDER_PROGRAM_H_

#include <string>

#include "ceres/internal/disable_warnings.h"
#include "ceres/internal/export.h"
#include "ceres/linear_solver.h"
#include "ceres/parameter_block_ordering.h"
#include "ceres/problem_impl.h"
#include "ceres/types.h"

namespace ceres::internal {

class Program;

// Reorder the parameter blocks in program using the ordering
CERES_NO_EXPORT bool ApplyOrdering(
    const ProblemImpl::ParameterMap& parameter_map,
    const ParameterBlockOrdering& ordering,
    Program* program,
    std::string* error);

// Reorder the residuals for program, if necessary, so that the residuals
// involving each E block occur together. This is a necessary condition for the
// Schur eliminator, which works on these "row blocks" in the jacobian.
CERES_NO_EXPORT bool LexicographicallyOrderResidualBlocks(
    int size_of_first_elimination_group, Program* program, std::string* error);

// Schur type solvers require that all parameter blocks eliminated
// by the Schur eliminator occur before others and the residuals be
// sorted in lexicographic order of their parameter blocks.
//
// If the parameter_block_ordering only contains one elimination
// group then a maximal independent set is computed and used as the
// first elimination group, otherwise the user's ordering is used.
//
// If the linear solver type is SPARSE_SCHUR and support for
// constrained fill-reducing ordering is available in the sparse
// linear algebra library (SuiteSparse version >= 4.2.0) then
// columns of the schur complement matrix are ordered to reduce the
// fill-in the Cholesky factorization.
//
// Upon return, ordering contains the parameter block ordering that
// was used to order the program.
CERES_NO_EXPORT bool ReorderProgramForSchurTypeLinearSolver(
    LinearSolverType linear_solver_type,
    SparseLinearAlgebraLibraryType sparse_linear_algebra_library_type,
    LinearSolverOrderingType linear_solver_ordering_type,
    const ProblemImpl::ParameterMap& parameter_map,
    const int max_num_threads,
    ParameterBlockOrdering* parameter_block_ordering,
    Program* program,
    std::string* error);

// Sparse cholesky factorization routines when doing the sparse
// cholesky factorization of the Jacobian matrix, reorders its
// columns to reduce the fill-in. Compute this permutation and
// re-order the parameter blocks.
//
// When using SuiteSparse, if the parameter_block_ordering contains
// more than one elimination group and support for constrained
// fill-reducing ordering is available in the sparse linear algebra
// library (SuiteSparse version >= 4.2.0) then the fill reducing
// ordering will take it into account, otherwise it will be ignored.
CERES_NO_EXPORT bool ReorderProgramForSparseCholesky(
    SparseLinearAlgebraLibraryType sparse_linear_algebra_library_type,
    LinearSolverOrderingType linear_solver_ordering_type,
    const ParameterBlockOrdering& parameter_block_ordering,
    const int start_row_block,
    const int max_num_threads,
    Program* program,
    std::string* error);

// Reorder the residual blocks in the program so that all the residual
// blocks in bottom_residual_blocks are at the bottom. The return
// value is the number of residual blocks in the program in "top" part
// of the Program, i.e., the ones not included in
// bottom_residual_blocks.
//
// This number can be different from program->NumResidualBlocks() -
// bottom_residual_blocks.size() because we allow
// bottom_residual_blocks to contain residual blocks not present in
// the Program.
CERES_NO_EXPORT int ReorderResidualBlocksByPartition(
    const std::unordered_set<ResidualBlockId>& bottom_residual_blocks,
    Program* program);

// The return value of this function indicates whether the columns of
// the Jacobian can be reordered using a fill reducing ordering.
CERES_NO_EXPORT bool AreJacobianColumnsOrdered(
    LinearSolverType linear_solver_type,
    PreconditionerType preconditioner_type,
    SparseLinearAlgebraLibraryType sparse_linear_algebra_library_type,
    LinearSolverOrderingType linear_solver_ordering_type);

}  // namespace ceres::internal

#include "ceres/internal/reenable_warnings.h"

#endif  // CERES_INTERNAL_REORDER_PROGRAM_H_
