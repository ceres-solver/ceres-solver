// This include must come before any #ifndef check on Ceres compile options.
#include "ceres/internal/port.h"

#ifndef CERES_NO_APPLE_ACCELERATE_SOLVERS

#include "ceres/apple_accelerate.h"

#include <string>
#include <vector>

#include "ceres/compressed_col_sparse_matrix_utils.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/triplet_sparse_matrix.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

std::unique_ptr<SparseCholesky>
AppleAccelerateCholesky::Create(OrderingType ordering_type) {
  return std::unique_ptr<SparseCholesky>(
      new AppleAccelerateCholesky(ordering_type));
}

AppleAccelerateCholesky::AppleAccelerateCholesky(
    const OrderingType ordering_type)
    : ordering_type_(ordering_type) {}

AppleAccelerateCholesky::~AppleAccelerateCholesky() {
}

CompressedRowSparseMatrix::StorageType
AppleAccelerateCholesky::StorageType() const {
  return CompressedRowSparseMatrix::LOWER_TRIANGULAR;
}

LinearSolverTerminationType
AppleAccelerateCholesky::Factorize(CompressedRowSparseMatrix* lhs,
                                   std::string* message) {
  (void)lhs;
  (void)message;
  return LINEAR_SOLVER_SUCCESS;
}

LinearSolverTerminationType
AppleAccelerateCholesky::Solve(const double* rhs,
                               double* solution,
                               std::string* message) {
  (void)rhs;
  (void)solution;
  (void)message;
  return LINEAR_SOLVER_SUCCESS;
}

std::unique_ptr<SparseCholesky>
FloatAppleAccelerateCholesky::Create(OrderingType ordering_type) {
  return std::unique_ptr<SparseCholesky>(
      new FloatAppleAccelerateCholesky(ordering_type));
}

FloatAppleAccelerateCholesky::FloatAppleAccelerateCholesky(
    const OrderingType ordering_type)
    : ordering_type_(ordering_type) {}

FloatAppleAccelerateCholesky::~FloatAppleAccelerateCholesky() {
}

CompressedRowSparseMatrix::StorageType
FloatAppleAccelerateCholesky::StorageType() const {
  return CompressedRowSparseMatrix::LOWER_TRIANGULAR;
}

LinearSolverTerminationType
FloatAppleAccelerateCholesky::Factorize(CompressedRowSparseMatrix* lhs,
                                        std::string* message) {
  (void)lhs;
  (void)message;
  return LINEAR_SOLVER_SUCCESS;
}

LinearSolverTerminationType
FloatAppleAccelerateCholesky::Solve(const double* rhs,
                                    double* solution,
                                    std::string* message) {
  (void)rhs;
  (void)solution;
  (void)message;
  return LINEAR_SOLVER_SUCCESS;
}

}
}

#endif  // CERES_NO_APPLE_ACCELERATE_SOLVERS
