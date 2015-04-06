#include "ceres/eigen_dense_cholesky.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/port.h"
#include "Eigen/Dense"

namespace ceres {
namespace internal {

using Eigen::Upper;

Eigen::ComputationInfo
InvertUpperTriangularUsingCholesky(const int size,
                                   const double* values,
                                   double* inverse_values) {
  ConstMatrixRef m(values, size, size);
  MatrixRef inverse(inverse_values, size, size);

  // On ARM we have experienced significant numerical problems with
  // Eigen's LLT implementation. Defining
  // CERES_USE_LDLT_FOR_EIGEN_CHOLESKY switches to using the slightly
  // more expensive but much more numerically well behaved LDLT
  // factorization algorithm.

#ifdef CERES_USE_LDLT_FOR_EIGEN_CHOLESKY
  Eigen::LDLT<Matrix, Upper> cholesky = m.selfadjointView<Upper>().ldlt();
#else
  Eigen::LLT<Matrix, Upper> cholesky = m.selfadjointView<Upper>().llt();
#endif

  inverse = cholesky.solve(Matrix::Identity(size, size));
  return cholesky.info();
}

Eigen::ComputationInfo
SolveUpperTriangularUsingCholesky(int size,
                                  const double* lhs_values,
                                  const double* rhs_values,
                                  double* solution_values) {
  ConstMatrixRef lhs(lhs_values, size, size);

  // On ARM we have experienced significant numerical problems with
  // Eigen's LLT implementation. Defining
  // CERES_USE_LDLT_FOR_EIGEN_CHOLESKY switches to using the slightly
  // more expensive but much more numerically well behaved LDLT
  // factorization algorithm.

#ifdef CERES_USE_LDLT_FOR_EIGEN_CHOLESKY
  Eigen::LDLT<Matrix, Upper> cholesky = lhs.selfadjointView<Upper>().ldlt();
#else
  Eigen::LLT<Matrix, Upper> cholesky = lhs.selfadjointView<Upper>().llt();
#endif

  if (cholesky.info() == Eigen::Success) {
    VectorRef solution(solution_values, size);
    if (solution_values == rhs_values) {
      cholesky.solveInPlace(solution);
    } else {
      ConstVectorRef rhs(rhs_values, size);
      solution = cholesky.solve(rhs);
    }
  }

  return cholesky.info();
}

}  // namespace internal
}  // namespace ceres
