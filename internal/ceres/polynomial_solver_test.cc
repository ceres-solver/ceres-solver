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
  double epsilon(double reference, double units = 64.0) {
    return units * std::numeric_limits<double>::epsilon() * std::abs(reference);
  }

  Vector ConstantPolynomial(double value) {
    Vector poly(1);
    poly(0) = value;
    return poly;
  }

  Vector AddRealRoot(const Vector& poly, double root) {
    Vector poly2(poly.size() + 1);
    poly2.setZero();
    poly2.head(poly.size()) += poly;
    poly2.tail(poly.size()) -= root * poly;
    return poly2;
  }

  Vector AddComplexRootPair(const Vector& poly, double real, double imag) {
    Vector poly2(poly.size() + 2);
    poly2.setZero();
    // Multiply poly by x^2 - 2real + abs(real,imag)^2
    poly2.head(poly.size()) += poly;
    poly2.segment(1, poly.size()) -= 2 * real * poly;
    poly2.tail(poly.size()) += (real*real + imag*imag) * poly;
    return poly2;
  }

  Vector SortVector(const Vector& in) {
    Vector out(in);
    std::sort(out.data(), out.data() + out.size());
    return out;
  }
}

TEST(PolynomialSolver, EmptyPoly) {
  Vector poly(0, 1); // (0) is ambiguous, could be null pointer
  Vector real;
  Vector imag;
  int err = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_LT(err, 0);
}

TEST(PolynomialSolver, ConstantPoly) {
  Vector poly = ConstantPolynomial(1.23);
  Vector real;
  Vector imag;
  int err = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 0);
  EXPECT_EQ(imag.size(), 0);
}

TEST(PolynomialSolver, LinearPoly) {
  Vector poly;
  int err;
  Vector real;
  Vector imag;

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, 42.0);
  err = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 1);
  EXPECT_EQ(imag.size(), 1);
  EXPECT_LT(std::abs(real(0) - 42.0), epsilon(42.0));
  EXPECT_LT(std::abs(imag(0)), epsilon(42.0));

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, -42.0);
  err = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 1);
  EXPECT_EQ(imag.size(), 1);
  EXPECT_LT(std::abs(real(0) + 42.0), epsilon(42.0));
  EXPECT_LT(std::abs(imag(0)), epsilon(42.0));

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, -1.23);
  err = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 1);
  EXPECT_EQ(imag.size(), 1);
  EXPECT_LT(std::abs(real(0) + 1.23), epsilon(1.23));
  EXPECT_LT(std::abs(imag(0)), epsilon(1.23));

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, 0.0);
  err = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 1);
  EXPECT_EQ(imag.size(), 1);
  EXPECT_LT(std::abs(real(0)), epsilon(1.23));
  EXPECT_LT(std::abs(imag(0)), epsilon(1.23));
}

TEST(PolynomialSolver, LinearPolyOnlyRealPart) {
  Vector poly;
  int err;
  Vector real;

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, 42.0);
  err = FindPolynomialRoots(poly, &real, NULL);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 1);
  EXPECT_LT(std::abs(real(0) - 42.0), epsilon(42.0));

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, -42.0);
  err = FindPolynomialRoots(poly, &real, NULL);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 1);
  EXPECT_LT(std::abs(real(0) + 42.0), epsilon(42.0));

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, -1.23);
  err = FindPolynomialRoots(poly, &real, NULL);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 1);
  EXPECT_LT(std::abs(real(0) + 1.23), epsilon(1.23));

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, 0.0);
  err = FindPolynomialRoots(poly, &real, NULL);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 1);
  EXPECT_LT(std::abs(real(0)), epsilon(1.23));
}

TEST(PolynomialSolver, QuadraticPolyReal) {
  Vector poly;
  int err;
  Vector real;
  Vector imag;

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, 42.0);
  poly = AddRealRoot(poly, 1.0);
  err = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 2);
  EXPECT_EQ(imag.size(), 2);
  real = SortVector(real);
  EXPECT_LT(std::abs(real(0) - 1.0), epsilon(1.0));
  EXPECT_LT(std::abs(imag(0)), epsilon(42.0));
  EXPECT_LT(std::abs(real(1) - 42.0), epsilon(42.0));
  EXPECT_LT(std::abs(imag(1)), epsilon(42.0));

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, -42.0);
  poly = AddRealRoot(poly, 1.0);
  err = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 2);
  EXPECT_EQ(imag.size(), 2);
  real = SortVector(real);
  EXPECT_LT(std::abs(real(0) + 42.0), epsilon(42.0));
  EXPECT_LT(std::abs(imag(0)), epsilon(42.0));
  EXPECT_LT(std::abs(real(1) - 1.0), epsilon(1.0));
  EXPECT_LT(std::abs(imag(1)), epsilon(42.0));
}

TEST(PolynomialSolver, QuadraticPolyRealOnlyRealPart) {
  Vector poly;
  int err;
  Vector real;

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, 42.0);
  poly = AddRealRoot(poly, 1.0);
  err = FindPolynomialRoots(poly, &real, NULL);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 2);
  real = SortVector(real);
  EXPECT_LT(std::abs(real(0) - 1.0), epsilon(1.0));
  EXPECT_LT(std::abs(real(1) - 42.0), epsilon(42.0));

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, -42.0);
  poly = AddRealRoot(poly, 1.0);
  err = FindPolynomialRoots(poly, &real, NULL);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 2);
  real = SortVector(real);
  EXPECT_LT(std::abs(real(0) + 42.0), epsilon(42.0));
  EXPECT_LT(std::abs(real(1) - 1.0), epsilon(1.0));
}

TEST(PolynomialSolver, QuadraticPolyRealClose) {
  Vector poly;
  int err;
  Vector real;
  Vector imag;

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, 42.0);
  poly = AddRealRoot(poly, 42.01);
  err = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 2);
  EXPECT_EQ(imag.size(), 2);
  real = SortVector(real);
  // With close roots, we need to relax the tests.
  EXPECT_LT(std::abs(real(0) - 42.0), epsilon(42.0, 1e5));
  EXPECT_LT(std::abs(imag(0)), epsilon(42.01, 1e5));
  EXPECT_LT(std::abs(real(1) - 42.01), epsilon(42.01, 1e5));
  EXPECT_LT(std::abs(imag(1)), epsilon(42.01, 1e5));

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, -42.0);
  poly = AddRealRoot(poly, -42.01);
  err = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 2);
  EXPECT_EQ(imag.size(), 2);
  real = SortVector(real);
  EXPECT_LT(std::abs(real(0) + 42.01), epsilon(42.01, 1e5));
  EXPECT_LT(std::abs(imag(0)), epsilon(42.01, 1e5));
  EXPECT_LT(std::abs(real(1) + 42.0), epsilon(42.0, 1e5));
  EXPECT_LT(std::abs(imag(1)), epsilon(42.01, 1e5));
}

TEST(PolynomialSolver, QuadraticPolyComplex) {
  Vector poly;
  int err;
  Vector real;
  Vector imag;

  poly = ConstantPolynomial(1.23);
  poly = AddComplexRootPair(poly, 42.0, 4.2);
  err = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 2);
  EXPECT_EQ(imag.size(), 2);
  EXPECT_LT(std::abs(real(0) - 42.0), epsilon(42.0));
  EXPECT_LT(std::min(std::abs(imag(0) - 4.2), std::abs(imag(0) + 4.2)), epsilon(4.2));
  EXPECT_LT(std::abs(real(1) - 42.0), epsilon(42.0));
  EXPECT_LT(std::min(std::abs(imag(1) - 4.2), std::abs(imag(1) + 4.2)), epsilon(4.2));
  EXPECT_LT(std::abs(imag(0) + imag(1)), epsilon(4.2));
}

TEST(PolynomialSolver, QuarticPolyOnlyRealPart) {
  Vector poly;
  int err;
  Vector real;

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, 1.23e-4);
  poly = AddRealRoot(poly, 1.23e-1);
  poly = AddRealRoot(poly, 1.23e+2);
  poly = AddRealRoot(poly, 1.23e+5);
  err = FindPolynomialRoots(poly, &real, NULL);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 4);
  real = SortVector(real);
  EXPECT_LT(std::abs(real(0) - 1.23e-4), epsilon(1.23e-4));
  EXPECT_LT(std::abs(real(1) - 1.23e-1), epsilon(1.23e-1));
  EXPECT_LT(std::abs(real(2) - 1.23e+2), epsilon(1.23e+2));
  EXPECT_LT(std::abs(real(3) - 1.23e+5), epsilon(1.23e+5));

  poly = ConstantPolynomial(1.23);
  poly = AddRealRoot(poly, 1.23e-1);
  poly = AddRealRoot(poly, 2.46e-1);
  poly = AddRealRoot(poly, 1.23e+5);
  poly = AddRealRoot(poly, 2.46e+5);
  err = FindPolynomialRoots(poly, &real, NULL);

  EXPECT_EQ(err, 0);
  EXPECT_EQ(real.size(), 4);
  real = SortVector(real);
  EXPECT_LT(std::abs(real(0) - 1.23e-1), epsilon(1.23e-1));
  EXPECT_LT(std::abs(real(1) - 2.46e-1), epsilon(2.46e-1));
  EXPECT_LT(std::abs(real(2) - 1.23e+5), epsilon(1.23e+5));
  EXPECT_LT(std::abs(real(3) - 2.46e+5), epsilon(2.46e+5));
}
}  // namespace internal
}  // namespace ceres
