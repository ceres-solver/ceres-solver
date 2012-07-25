// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
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

#include "ceres/polynomial_solver.h"

#include <limits>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include "gtest/gtest.h"

namespace ceres {
namespace internal {
namespace {
// Return a value suitable for comparison of an error
// abs(x - reference), assuming that up to units many
// machine precision errors are allowed.
double Epsilon(double reference, double units) {
  return units * std::numeric_limits<double>::epsilon() * std::abs(reference);
}

// The same as above, where units is 64.
double Epsilon(double reference) {
  return Epsilon(reference, 64.0);
}

// Return the constant polynomial p(x) = 1.23.
Vector ConstantPolynomial(double value) {
  Vector poly(1);
  poly(0) = value;
  return poly;
}

// Return the polynomial p(x) = poly(x) * (x - root).
Vector AddRealRoot(const Vector& poly, double root) {
  Vector poly2(poly.size() + 1);
  poly2.setZero();
  poly2.head(poly.size()) += poly;
  poly2.tail(poly.size()) -= root * poly;
  return poly2;
}

// Return the polynomial 
// p(x) = poly(x) * (x - real - imag*i) * (x - real + imag*i).
Vector AddComplexRootPair(const Vector& poly, double real, double imag) {
  Vector poly2(poly.size() + 2);
  poly2.setZero();
  // Multiply poly by x^2 - 2real + abs(real,imag)^2
  poly2.head(poly.size()) += poly;
  poly2.segment(1, poly.size()) -= 2 * real * poly;
  poly2.tail(poly.size()) += (real*real + imag*imag) * poly;
  return poly2;
}

// Sort the entries in a vector.
// Needed because the roots are not returned in sorted order.
Vector SortVector(const Vector& in) {
  Vector out(in);
  std::sort(out.data(), out.data() + out.size());
  return out;
}

// Run a test with the polynomial defined by the N real roots in roots_real.
// If use_real is false, NULL is passed as the real argument to FindPolynomialRoots.
// If use_imaginary is false, NULL is passed as the imaginary argument to FindPolynomialRoots.
template<int N>
void RunPolynomialTestRealRoots(const double (&roots_real)[N], bool use_real, bool use_imaginary) {
  Vector real;
  Vector imaginary;
  Vector poly = ConstantPolynomial(1.23);
  for (int i = 0; i < N; ++i) {
    poly = AddRealRoot(poly, real_roots[i]);
  }
  bool success = FindPolynomialRoots(poly, use_real ? &real : NULL, use_imaginary ? &imaginary : NULL);

  EXPECT_EQ(success, true);
  if (use_real) {
    EXPECT_EQ(real.size(), N);
    real = SortVector(real);
    for (int i = 0; i < N; ++i) {
      EXPECT_LT(std::abs(real(i) - real_roots[i]), Epsilon(real_roots[i] == 0 ? 1.0 : real_roots[i]));
    }
  }
  if (use_imaginary) {
    EXPECT_EQ(imaginary.size(), N);
    for (int i = 0; i < N; ++i) {
      EXPECT_LT(std::abs(imaginary(i)), Epsilon(1.0));
    }
  }
}
}

// Check if an invalid polynomial (no coefficient) is rejected.
TEST(PolynomialSolver, EmptyPoly) {
  // Vector poly(0) is an ambiguous constructor call, so
  // use the constructor with explicit column count.
  Vector poly(0, 1);
  Vector real;
  Vector imag;
  bool success = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(success, false);
}

// Check if no roots are returned for the constant polynomial
// p(x) = 1.23
TEST(PolynomialSolver, ConstantPoly) {
  Vector poly = ConstantPolynomial(1.23);
  Vector real;
  Vector imag;
  bool success = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(success, true);
  EXPECT_EQ(real.size(), 0);
  EXPECT_EQ(imag.size(), 0);
}

// Test p(x) = x - 42.42
TEST(PolynomialSolver, LinearPolyPositiveRoot) {
  const double roots[1] = { 42.42 };
  RunPolynomialTestRealRoots(roots, true, true);
}

// Test p(x) = x + 42.42
TEST(PolynomialSolver, LinearPolyNegativeRoot) {
  const double roots[1] = { -42.42 };
  RunPolynomialTestRealRoots(roots, true, true);
}

// Test p(x) = x - 42.42 only extracting the real parts.
TEST(PolynomialSolver, LinearPolyOnlyRealPartPositive) {
  const double roots[1] = { 42.42 };
  RunPolynomialTestRealRoots(roots, true, false);
}

// Test p(x) = x + 42.42 only extracting the real parts.
TEST(PolynomialSolver, LinearPolyOnlyRealPartNegative) {
  const double roots[1] = { -42.42 };
  RunPolynomialTestRealRoots(roots, true, false);
}

// Test p(x) = (x - 1.0)(x - 42.42)
TEST(PolynomialSolver, QuadraticPolyReal) {
  const double roots[2] = { 1.0, 42.42 };
  RunPolynomialTestRealRoots(roots, true, true);
}

// Test p(x) = (x - 1.0)(x + 42.42)
TEST(PolynomialSolver, QuadraticPolyRealMixedSign) {
  const double roots[2] = { -42.42, 1.0 };
  RunPolynomialTestRealRoots(roots, true, true);
}

// Test p(x) = (x - 1.0)(x - 42.42) only extracting real parts.
TEST(PolynomialSolver, QuadraticPolyRealOnlyRealPart) {
  const double roots[2] = { 1.0, 42.0 };
  RunPolynomialTestRealRoots(roots, true, false);
}

// Test p(x) = (x - 1.0)(x + 42.42) only extracting real parts.
TEST(PolynomialSolver, QuadraticPolyRealMixedSignOnlyRealPart) {
  const double roots[2] = { -42.0, 1.0 };
  RunPolynomialTestRealRoots(roots, true, false);
}

// Check if roots that are close to each other are correctly
// computed. As this is more difficult, we relax the floating
// point equality test a bit.
// The polynomial is p(x) = (x - 42.42)(x - 42.43)
TEST(PolynomialSolver, QuadraticPolyRealClose) {
  const double roots[2] = { 42.42, 42.43 };
  Vector real;
  Vector imag;

  Vector poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, roots[0]);
  poly = AddRealRoot(poly, roots[1]);
  bool success = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(success, true);
  EXPECT_EQ(real.size(), 2);
  EXPECT_EQ(imag.size(), 2);
  real = SortVector(real);
  // With close roots, we need to relax the tests.
  EXPECT_LT(std::abs(real(0) - roots[0]), Epsilon(roots[0], 1e5));
  EXPECT_LT(std::abs(imag(0)), Epsilon(roots[0], 1e5));
  EXPECT_LT(std::abs(real(1) - roots[1]), Epsilon(roots[1], 1e5));
  EXPECT_LT(std::abs(imag(1)), Epsilon(roots[1], 1e5));
}

// Check if complex roots are computed correctly.
// p(x) = (x - 42.42 - 4.2 i)(x - 42.42 + 4.2 i)
//      = x^2 - 84.84 x + 1817.0964
TEST(PolynomialSolver, QuadraticPolyComplex) {
  Vector real;
  Vector imag;

  Vector poly = ConstantPolynomial(1.23);
  poly = AddComplexRootPair(poly, 42.42, 4.2);
  bool success = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(success, true);
  EXPECT_EQ(real.size(), 2);
  EXPECT_EQ(imag.size(), 2);
  EXPECT_LT(std::abs(real(0) - 42.42), Epsilon(42.42));
  EXPECT_LT(std::min(std::abs(imag(0) - 4.2), std::abs(imag(0) + 4.2)), Epsilon(4.2));
  EXPECT_LT(std::abs(real(1) - 42.42), Epsilon(42.42));
  EXPECT_LT(std::min(std::abs(imag(1) - 4.2), std::abs(imag(1) + 4.2)), Epsilon(4.2));
  EXPECT_LT(std::abs(imag(0) + imag(1)), Epsilon(4.2));
}

// Test quartic polynomials where the roots are spread.
TEST(PolynomialSolver, QuarticPolyOnlyRealPart) {
  const double roots[4] = { 1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5 };
  RunPolynomialTestRealRoots(roots, true, false);
}

// Test quartic polynomial where the roots form two clusters.
TEST(PolynomialSolver, QuarticPolyTightOnlyRealPart) {
  const double roots[4] = { 1.23e-1, 2.46e-1, 1.23e+5, 2.46e+5 };
  RunPolynomialTestRealRoots(roots, true, false);
}

// Test quartic polynomial where two roots are zero.
TEST(PolynomialSolver, QuarticPolyTwoZerosOnlyRealPart) {
  const double roots[4] = { -42.42, 0.0, 0.0, 42.42 };
  RunPolynomialTestRealRoots(roots, true, false);
}

// Test the quartic monomial p(x) = x^4.
TEST(PolynomialSolver, QuarticPolyAllZerosOnlyRealPart) {
  const double roots[4] = { 0.0, 0.0, 0.0, 0.0 };
  RunPolynomialTestRealRoots(roots, true, false);
}

}  // namespace internal
}  // namespace ceres
