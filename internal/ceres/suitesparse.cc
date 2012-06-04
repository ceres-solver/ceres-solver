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

#ifndef CERES_NO_SUITESPARSE


#include "ceres/suitesparse.h"

#include <numeric>
#include "cholmod.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/triplet_sparse_matrix.h"
namespace ceres {
namespace internal {

cholmod_sparse* SuiteSparse::CreateSparseMatrix(TripletSparseMatrix* A) {
  cholmod_triplet triplet;

  triplet.nrow = A->num_rows();
  triplet.ncol = A->num_cols();
  triplet.nzmax = A->max_num_nonzeros();
  triplet.nnz = A->num_nonzeros();
  triplet.i = reinterpret_cast<void*>(A->mutable_rows());
  triplet.j = reinterpret_cast<void*>(A->mutable_cols());
  triplet.x = reinterpret_cast<void*>(A->mutable_values());
  triplet.stype = 0;  // Matrix is not symmetric.
  triplet.itype = CHOLMOD_INT;
  triplet.xtype = CHOLMOD_REAL;
  triplet.dtype = CHOLMOD_DOUBLE;

  return cholmod_triplet_to_sparse(&triplet, triplet.nnz, &cc_);
}


cholmod_sparse* SuiteSparse::CreateSparseMatrixTranspose(
    TripletSparseMatrix* A) {
  cholmod_triplet triplet;

  triplet.ncol = A->num_rows();  // swap row and columns
  triplet.nrow = A->num_cols();
  triplet.nzmax = A->max_num_nonzeros();
  triplet.nnz = A->num_nonzeros();

  // swap rows and columns
  triplet.j = reinterpret_cast<void*>(A->mutable_rows());
  triplet.i = reinterpret_cast<void*>(A->mutable_cols());
  triplet.x = reinterpret_cast<void*>(A->mutable_values());
  triplet.stype = 0;  // Matrix is not symmetric.
  triplet.itype = CHOLMOD_INT;
  triplet.xtype = CHOLMOD_REAL;
  triplet.dtype = CHOLMOD_DOUBLE;

  return cholmod_triplet_to_sparse(&triplet, triplet.nnz, &cc_);
}

cholmod_sparse* SuiteSparse::CreateSparseMatrixTransposeView(
    CompressedRowSparseMatrix* A) {
  cholmod_sparse* m = new cholmod_sparse_struct;
  m->nrow = A->num_cols();
  m->ncol = A->num_rows();
  m->nzmax = A->num_nonzeros();

  m->p = reinterpret_cast<void*>(A->mutable_rows());
  m->i = reinterpret_cast<void*>(A->mutable_cols());
  m->x = reinterpret_cast<void*>(A->mutable_values());

  m->stype = 0;  // Matrix is not symmetric.
  m->itype = CHOLMOD_INT;
  m->xtype = CHOLMOD_REAL;
  m->dtype = CHOLMOD_DOUBLE;
  m->sorted = 1;
  m->packed = 1;

  return m;
}

cholmod_dense* SuiteSparse::CreateDenseVector(const double* x,
                                              int in_size,
                                              int out_size) {
    CHECK_LE(in_size, out_size);
    cholmod_dense* v = cholmod_zeros(out_size, 1, CHOLMOD_REAL, &cc_);
    if (x != NULL) {
      memcpy(v->x, x, in_size*sizeof(*x));
    }
    return v;
}

cholmod_factor* SuiteSparse::AnalyzeCholesky(cholmod_sparse* A) {
  cc_.nmethods = 1 ;
  cc_.method[0].ordering = CHOLMOD_AMD;
  cc_.supernodal = CHOLMOD_AUTO;
  cholmod_factor* factor = cholmod_analyze(A, &cc_);
  CHECK_EQ(cc_.status, CHOLMOD_OK)
      << "Cholmod symbolic analysis failed " << cc_.status;
  CHECK_NOTNULL(factor);
  return factor;
}


cholmod_factor* SuiteSparse::AnalyzeCholeskyWithUserOrdering(cholmod_sparse* A,
                                                             int* ordering) {
  CHECK_NOTNULL(ordering);
  cc_.nmethods = 1 ;
  cc_.method[0].ordering = CHOLMOD_GIVEN;
  cholmod_factor* factor  =
      cholmod_analyze_p(A, ordering, NULL, 0, &cc_);
  CHECK_EQ(cc_.status, CHOLMOD_OK)
      << "Cholmod symbolic analysis failed " << cc_.status;
  CHECK_NOTNULL(factor);
  return factor;
}

bool SuiteSparse::Cholesky(cholmod_sparse* A, cholmod_factor* L) {
  CHECK_NOTNULL(A);
  CHECK_NOTNULL(L);

  cc_.quick_return_if_not_posdef = 1;
  int status = cholmod_factorize(A, L, &cc_);
  switch (cc_.status) {
    case CHOLMOD_NOT_INSTALLED:
      LOG(WARNING) << "Cholmod failure: method not installed.";
      return false;
    case CHOLMOD_OUT_OF_MEMORY:
      LOG(WARNING) << "Cholmod failure: out of memory.";
      return false;
    case CHOLMOD_TOO_LARGE:
      LOG(WARNING) << "Cholmod failure: integer overflow occured.";
      return false;
    case CHOLMOD_INVALID:
      LOG(WARNING) << "Cholmod failure: invalid input.";
      return false;
    case CHOLMOD_NOT_POSDEF:
      // TODO(sameeragarwal): These two warnings require more
      // sophisticated handling going forward. For now we will be
      // strict and treat them as failures.
      LOG(WARNING) << "Cholmod warning: matrix not positive definite.";
      return false;
    case CHOLMOD_DSMALL:
      LOG(WARNING) << "Cholmod warning: D for LDL' or diag(L) or "
                   << "LL' has tiny absolute value.";
      return false;
    case CHOLMOD_OK:
      if (status != 0) {
        return true;
      }
      LOG(WARNING) << "Cholmod failure: cholmod_factorize returned zero "
                   << "but cholmod_common::status is CHOLMOD_OK."
                   << "Please report this to ceres-solver@googlegroups.com.";
      return false;
    default:
      LOG(WARNING) << "Unknown cholmod return code. "
                   << "Please report this to ceres-solver@googlegroups.com.";
      return false;
  }
  return false;
}

cholmod_dense* SuiteSparse::Solve(cholmod_factor* L,
                                  cholmod_dense* b) {
  if (cc_.status != CHOLMOD_OK) {
    LOG(WARNING) << "CHOLMOD status NOT OK";
    return NULL;
  }

  return cholmod_solve(CHOLMOD_A, L, b, &cc_);
}

cholmod_dense* SuiteSparse::SolveCholesky(cholmod_sparse* A,
                                          cholmod_factor* L,
                                          cholmod_dense* b) {
  CHECK_NOTNULL(A);
  CHECK_NOTNULL(L);
  CHECK_NOTNULL(b);

  if (Cholesky(A, L)) {
    return Solve(L, b);
  }

  return NULL;
}

bool SuiteSparse::BlockAMDOrdering(const vector<int>& blocks,
                                   const set<pair<int, int> >& block_pairs,
                                   vector<int>* ordering) {
  VLOG(2) << "num_blocks " << blocks.size();
  VLOG(2) << "block pairs: " << block_pairs.size();

  const int num_blocks = blocks.size();
  TripletSparseMatrix block_sparsity(num_blocks,
                                     num_blocks,
                                     block_pairs.size());
  int* rows = block_sparsity.mutable_rows();
  int* cols = block_sparsity.mutable_cols();
  double* values = block_sparsity.mutable_values();
  int i = 0;
  for (set<pair<int, int> >::const_iterator it = block_pairs.begin();
       it != block_pairs.end();
       ++it, ++i) {
    rows[i] = it->first;
    cols[i] = it->second;
    values[i] = 1.0;
  }
  block_sparsity.set_num_nonzeros(block_pairs.size());

  cholmod_sparse* ccs_block_sparsity = CreateSparseMatrix(&block_sparsity);
  // The matrix is symmetric, and the upper triangular part of the
  // matrix contains the values.
  ccs_block_sparsity->stype = 1;

  vector<int> block_permutation(num_blocks);
  const int status = cholmod_amd(ccs_block_sparsity,
                                 NULL,
                                 0,
                                 &block_permutation[0],
                                 &cc_);
  Free(ccs_block_sparsity);
  if (!status) {
    return false;
  }

  // Lift the permutation to the full scalar permutation
  vector<int> block_positions(num_blocks);
  std::partial_sum(blocks.begin(), blocks.end(), block_positions.begin());
  ordering->resize(block_positions.back());
  int cursor = 0;
  for (int i = 0; i < num_blocks; ++i) {
    const int block_id = block_permutation[i];
    const int block_size = blocks[block_id];
    int block_position = block_positions[block_id] - blocks[0];
    for ( int j = 0; j < block_size; ++j) {
      (*ordering)[cursor++] = block_position++;
    }
  }
  return true;
}


}  // namespace internal
}  // namespace ceres

#endif  // CERES_NO_SUITESPARSE
