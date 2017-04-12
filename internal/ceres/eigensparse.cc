// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2017 Google Inc. All rights reserved.
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
//
// A simple C++ interface to the Eigen's Sparse Cholesky routines.

#include "ceres/eigensparse.h"

#include <sstream>
#include "Eigen/SparseCholesky"
#include "Eigen/SparseCore"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/linear_solver.h"

// TODO(sameeragarwal): Need ifdefs and versioning support.
namespace ceres {
namespace internal {

template <typename Solver>
class EigenSparseCholeskyTemplate : public EigenSparseCholesky {
 public:
  EigenSparseCholeskyTemplate() : analyzed_(false) {}
  virtual ~EigenSparseCholeskyTemplate() {}

  virtual LinearSolverTerminationType Factorize(
      const Eigen::SparseMatrix<double>& lhs, std::string* message) {
    if (!analyzed_) {
      solver_.analyzePattern(lhs);
      analyzed_ = true;

      if (VLOG_IS_ON(2)) {
        std::stringstream ss;
        solver_.dumpMemory(ss);
        VLOG(2) << "Symbolic Analysis\n" << ss.str();
      }

      if (solver_.info() != Eigen::Success) {
        *message = "Eigen failure. Unable to find symbolic factorization.";
        return LINEAR_SOLVER_FATAL_ERROR;
      }
    }

    solver_.factorize(lhs);
    if (solver_.info() != Eigen::Success) {
      *message = "Eigen failure. Unable to find numeric factorization.";
      return LINEAR_SOLVER_FAILURE;
    }
    return LINEAR_SOLVER_SUCCESS;
  }

  // rhs and solution can point to the same part of memory.
  virtual LinearSolverTerminationType Solve(const double* rhs,
                                            double* solution,
                                            std::string* message) {
    CHECK(analyzed_)
        << "Solve called without a call to Factorize first.";

    VectorRef(solution, solver_.cols()) =
        solver_.solve(ConstVectorRef(rhs, solver_.cols()));
    if (solver_.info() != Eigen::Success) {
      *message = "Eigen failure. Unable to do triangular solve.";
      return LINEAR_SOLVER_FAILURE;
    }
    return LINEAR_SOLVER_SUCCESS;
  }

  virtual LinearSolverTerminationType Factorize(CompressedRowSparseMatrix* lhs,
                                                std::string* message) {
    // TODO(sameeragarwal): Pay attention to the storage type
    CHECK_EQ(lhs->storage_type(), CompressedRowSparseMatrix::LOWER_TRIANGULAR);
    Eigen::MappedSparseMatrix<double, Eigen::ColMajor> eigen_lhs(
        lhs->num_rows(),
        lhs->num_rows(),
        lhs->num_nonzeros(),
        lhs->mutable_rows(),
        lhs->mutable_cols(),
        lhs->mutable_values());
    return Factorize(eigen_lhs, message);
  }

 private:
  bool analyzed_;
  Solver solver_;
};

// TODO(sameeragarwal) : Add the Eigen Versioning logic back in again.
EigenSparseCholesky* EigenSparseCholesky::Create(
    const OrderingType ordering_type) {
  typedef Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>,
                                Eigen::Upper,
                                Eigen::AMDOrdering<int> >
      WithAMDOrdering;
  typedef Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>,
                                Eigen::Upper,
                                Eigen::NaturalOrdering<int> >
      WithNaturalOrdering;
  if (ordering_type == AMD) {
    return new EigenSparseCholeskyTemplate<WithAMDOrdering>();
  } else {
    return new EigenSparseCholeskyTemplate<WithNaturalOrdering>();
  }
}

EigenSparseCholesky::~EigenSparseCholesky() {}

}  // namespace internal
}  // namespace ceres
