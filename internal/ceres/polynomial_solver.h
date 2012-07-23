#ifndef CERES_INTERNAL_POLYNOMIAL_SOLVER_H_
#define CERES_INTERNAL_POLYNOMIAL_SOLVER_H_

#include "ceres/internal/eigen.h"

namespace ceres {
namespace internal {

// Use the companion matrix eigenvalues to determine the roots of the polynomial
// sum_{i=0}^N p_i x^{N-i}.
// Returns 0 on success.
int FindPolynomialRoots(const Vector& polynomial, Vector* real, Vector* imag = NULL);

// Evaluate the polynomial at x using the Horner scheme.
inline double EvaluatePolynomial(const Vector& polynomial, double x) {
  double v = 0.0;
  for (int i=0; i<polynomial.size(); ++i) {
    v = v * x + polynomial(i);
  }
  return v;
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_POLYNOMIAL_SOLVER_H_
