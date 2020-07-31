// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
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

#include "ceres/local_parameterization.h"

#include <cmath>
#include <limits>
#include <memory>

#include "Eigen/Geometry"
#include "ceres/autodiff_local_parameterization.h"
#include "ceres/internal/autodiff.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/householder_vector.h"
#include "ceres/random.h"
#include "ceres/rotation.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(IdentityParameterization, EverythingTest) {
  IdentityParameterization parameterization(3);
  EXPECT_EQ(parameterization.GlobalSize(), 3);
  EXPECT_EQ(parameterization.LocalSize(), 3);

  double x[3] = {1.0, 2.0, 3.0};
  double delta[3] = {0.0, 1.0, 2.0};
  double x_plus_delta[3] = {0.0, 0.0, 0.0};
  parameterization.Plus(x, delta, x_plus_delta);
  EXPECT_EQ(x_plus_delta[0], 1.0);
  EXPECT_EQ(x_plus_delta[1], 3.0);
  EXPECT_EQ(x_plus_delta[2], 5.0);

  double jacobian[9];
  parameterization.ComputeJacobian(x, jacobian);
  int k = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j, ++k) {
      EXPECT_EQ(jacobian[k], (i == j) ? 1.0 : 0.0);
    }
  }

  Matrix global_matrix = Matrix::Ones(10, 3);
  Matrix local_matrix = Matrix::Zero(10, 3);
  parameterization.MultiplyByJacobian(
      x, 10, global_matrix.data(), local_matrix.data());
  EXPECT_EQ((local_matrix - global_matrix).norm(), 0.0);
}

TEST(SubsetParameterization, EmptyConstantParameters) {
  std::vector<int> constant_parameters;
  SubsetParameterization parameterization(3, constant_parameters);
  EXPECT_EQ(parameterization.GlobalSize(), 3);
  EXPECT_EQ(parameterization.LocalSize(), 3);
  double x[3] = {1, 2, 3};
  double delta[3] = {4, 5, 6};
  double x_plus_delta[3] = {-1, -2, -3};
  parameterization.Plus(x, delta, x_plus_delta);
  EXPECT_EQ(x_plus_delta[0], x[0] + delta[0]);
  EXPECT_EQ(x_plus_delta[1], x[1] + delta[1]);
  EXPECT_EQ(x_plus_delta[2], x[2] + delta[2]);

  Matrix jacobian(3, 3);
  Matrix expected_jacobian(3, 3);
  expected_jacobian.setIdentity();
  parameterization.ComputeJacobian(x, jacobian.data());
  EXPECT_EQ(jacobian, expected_jacobian);

  Matrix global_matrix(3, 5);
  global_matrix.setRandom();
  Matrix local_matrix(3, 5);
  parameterization.MultiplyByJacobian(
      x, 5, global_matrix.data(), local_matrix.data());
  EXPECT_EQ(global_matrix, local_matrix);
}

TEST(SubsetParameterization, NegativeParameterIndexDeathTest) {
  std::vector<int> constant_parameters;
  constant_parameters.push_back(-1);
  EXPECT_DEATH_IF_SUPPORTED(
      SubsetParameterization parameterization(2, constant_parameters),
      "greater than equal to zero");
}

TEST(SubsetParameterization, GreaterThanSizeParameterIndexDeathTest) {
  std::vector<int> constant_parameters;
  constant_parameters.push_back(2);
  EXPECT_DEATH_IF_SUPPORTED(
      SubsetParameterization parameterization(2, constant_parameters),
      "less than the size");
}

TEST(SubsetParameterization, DuplicateParametersDeathTest) {
  std::vector<int> constant_parameters;
  constant_parameters.push_back(1);
  constant_parameters.push_back(1);
  EXPECT_DEATH_IF_SUPPORTED(
      SubsetParameterization parameterization(2, constant_parameters),
      "duplicates");
}

TEST(SubsetParameterization,
     ProductParameterizationWithZeroLocalSizeSubsetParameterization1) {
  std::vector<int> constant_parameters;
  constant_parameters.push_back(0);
  LocalParameterization* subset_param =
      new SubsetParameterization(1, constant_parameters);
  LocalParameterization* identity_param = new IdentityParameterization(2);
  ProductParameterization product_param(subset_param, identity_param);
  EXPECT_EQ(product_param.GlobalSize(), 3);
  EXPECT_EQ(product_param.LocalSize(), 2);
  double x[] = {1.0, 1.0, 1.0};
  double delta[] = {2.0, 3.0};
  double x_plus_delta[] = {0.0, 0.0, 0.0};
  EXPECT_TRUE(product_param.Plus(x, delta, x_plus_delta));
  EXPECT_EQ(x_plus_delta[0], x[0]);
  EXPECT_EQ(x_plus_delta[1], x[1] + delta[0]);
  EXPECT_EQ(x_plus_delta[2], x[2] + delta[1]);

  Matrix actual_jacobian(3, 2);
  EXPECT_TRUE(product_param.ComputeJacobian(x, actual_jacobian.data()));
}

TEST(SubsetParameterization,
     ProductParameterizationWithZeroLocalSizeSubsetParameterization2) {
  std::vector<int> constant_parameters;
  constant_parameters.push_back(0);
  LocalParameterization* subset_param =
      new SubsetParameterization(1, constant_parameters);
  LocalParameterization* identity_param = new IdentityParameterization(2);
  ProductParameterization product_param(identity_param, subset_param);
  EXPECT_EQ(product_param.GlobalSize(), 3);
  EXPECT_EQ(product_param.LocalSize(), 2);
  double x[] = {1.0, 1.0, 1.0};
  double delta[] = {2.0, 3.0};
  double x_plus_delta[] = {0.0, 0.0, 0.0};
  EXPECT_TRUE(product_param.Plus(x, delta, x_plus_delta));
  EXPECT_EQ(x_plus_delta[0], x[0] + delta[0]);
  EXPECT_EQ(x_plus_delta[1], x[1] + delta[1]);
  EXPECT_EQ(x_plus_delta[2], x[2]);

  Matrix actual_jacobian(3, 2);
  EXPECT_TRUE(product_param.ComputeJacobian(x, actual_jacobian.data()));
}

TEST(SubsetParameterization, NormalFunctionTest) {
  const int kGlobalSize = 4;
  const int kLocalSize = 3;

  double x[kGlobalSize] = {1.0, 2.0, 3.0, 4.0};
  for (int i = 0; i < kGlobalSize; ++i) {
    std::vector<int> constant_parameters;
    constant_parameters.push_back(i);
    SubsetParameterization parameterization(kGlobalSize, constant_parameters);
    double delta[kLocalSize] = {1.0, 2.0, 3.0};
    double x_plus_delta[kGlobalSize] = {0.0, 0.0, 0.0};

    parameterization.Plus(x, delta, x_plus_delta);
    int k = 0;
    for (int j = 0; j < kGlobalSize; ++j) {
      if (j == i) {
        EXPECT_EQ(x_plus_delta[j], x[j]);
      } else {
        EXPECT_EQ(x_plus_delta[j], x[j] + delta[k++]);
      }
    }

    double jacobian[kGlobalSize * kLocalSize];
    parameterization.ComputeJacobian(x, jacobian);
    int delta_cursor = 0;
    int jacobian_cursor = 0;
    for (int j = 0; j < kGlobalSize; ++j) {
      if (j != i) {
        for (int k = 0; k < kLocalSize; ++k, jacobian_cursor++) {
          EXPECT_EQ(jacobian[jacobian_cursor], delta_cursor == k ? 1.0 : 0.0);
        }
        ++delta_cursor;
      } else {
        for (int k = 0; k < kLocalSize; ++k, jacobian_cursor++) {
          EXPECT_EQ(jacobian[jacobian_cursor], 0.0);
        }
      }
    }

    Matrix global_matrix = Matrix::Ones(10, kGlobalSize);
    for (int row = 0; row < kGlobalSize; ++row) {
      for (int col = 0; col < kGlobalSize; ++col) {
        global_matrix(row, col) = col;
      }
    }

    Matrix local_matrix = Matrix::Zero(10, kLocalSize);
    parameterization.MultiplyByJacobian(
        x, 10, global_matrix.data(), local_matrix.data());
    Matrix expected_local_matrix =
        global_matrix * MatrixRef(jacobian, kGlobalSize, kLocalSize);
    EXPECT_EQ((local_matrix - expected_local_matrix).norm(), 0.0);
  }
}

// Functor needed to implement automatically differentiated Plus for
// quaternions.
struct QuaternionPlus {
  template <typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    const T squared_norm_delta =
        delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];

    T q_delta[4];
    if (squared_norm_delta > T(0.0)) {
      T norm_delta = sqrt(squared_norm_delta);
      const T sin_delta_by_delta = sin(norm_delta) / norm_delta;
      q_delta[0] = cos(norm_delta);
      q_delta[1] = sin_delta_by_delta * delta[0];
      q_delta[2] = sin_delta_by_delta * delta[1];
      q_delta[3] = sin_delta_by_delta * delta[2];
    } else {
      // We do not just use q_delta = [1,0,0,0] here because that is a
      // constant and when used for automatic differentiation will
      // lead to a zero derivative. Instead we take a first order
      // approximation and evaluate it at zero.
      q_delta[0] = T(1.0);
      q_delta[1] = delta[0];
      q_delta[2] = delta[1];
      q_delta[3] = delta[2];
    }

    QuaternionProduct(q_delta, x, x_plus_delta);
    return true;
  }
};

template <typename Parameterization, typename Plus>
void QuaternionParameterizationTestHelper(const double* x,
                                          const double* delta,
                                          const double* x_plus_delta_ref) {
  const int kGlobalSize = 4;
  const int kLocalSize = 3;

  const double kTolerance = 1e-14;

  double x_plus_delta[kGlobalSize] = {0.0, 0.0, 0.0, 0.0};
  Parameterization parameterization;
  parameterization.Plus(x, delta, x_plus_delta);
  for (int i = 0; i < kGlobalSize; ++i) {
    EXPECT_NEAR(x_plus_delta[i], x_plus_delta[i], kTolerance);
  }

  const double x_plus_delta_norm = sqrt(
      x_plus_delta[0] * x_plus_delta[0] + x_plus_delta[1] * x_plus_delta[1] +
      x_plus_delta[2] * x_plus_delta[2] + x_plus_delta[3] * x_plus_delta[3]);

  EXPECT_NEAR(x_plus_delta_norm, 1.0, kTolerance);

  double jacobian_ref[12];
  double zero_delta[kLocalSize] = {0.0, 0.0, 0.0};
  const double* parameters[2] = {x, zero_delta};
  double* jacobian_array[2] = {NULL, jacobian_ref};

  // Autodiff jacobian at delta_x = 0.
  internal::AutoDifferentiate<kGlobalSize,
                              StaticParameterDims<kGlobalSize, kLocalSize>>(
      Plus(), parameters, kGlobalSize, x_plus_delta, jacobian_array);

  double jacobian[12];
  parameterization.ComputeJacobian(x, jacobian);
  for (int i = 0; i < 12; ++i) {
    EXPECT_TRUE(IsFinite(jacobian[i]));
    EXPECT_NEAR(jacobian[i], jacobian_ref[i], kTolerance)
        << "Jacobian mismatch: i = " << i << "\n Expected \n"
        << ConstMatrixRef(jacobian_ref, kGlobalSize, kLocalSize)
        << "\n Actual \n"
        << ConstMatrixRef(jacobian, kGlobalSize, kLocalSize);
  }

  Matrix global_matrix = Matrix::Random(10, kGlobalSize);
  Matrix local_matrix = Matrix::Zero(10, kLocalSize);
  parameterization.MultiplyByJacobian(
      x, 10, global_matrix.data(), local_matrix.data());
  Matrix expected_local_matrix =
      global_matrix * MatrixRef(jacobian, kGlobalSize, kLocalSize);
  EXPECT_NEAR((local_matrix - expected_local_matrix).norm(),
              0.0,
              10.0 * std::numeric_limits<double>::epsilon());
}

template <int N>
void Normalize(double* x) {
  VectorRef(x, N).normalize();
}

TEST(QuaternionParameterization, ZeroTest) {
  double x[4] = {0.5, 0.5, 0.5, 0.5};
  double delta[3] = {0.0, 0.0, 0.0};
  double q_delta[4] = {1.0, 0.0, 0.0, 0.0};
  double x_plus_delta[4] = {0.0, 0.0, 0.0, 0.0};
  QuaternionProduct(q_delta, x, x_plus_delta);
  QuaternionParameterizationTestHelper<QuaternionParameterization,
                                       QuaternionPlus>(x, delta, x_plus_delta);
}

TEST(QuaternionParameterization, NearZeroTest) {
  double x[4] = {0.52, 0.25, 0.15, 0.45};
  Normalize<4>(x);

  double delta[3] = {0.24, 0.15, 0.10};
  for (int i = 0; i < 3; ++i) {
    delta[i] = delta[i] * 1e-14;
  }

  double q_delta[4];
  q_delta[0] = 1.0;
  q_delta[1] = delta[0];
  q_delta[2] = delta[1];
  q_delta[3] = delta[2];

  double x_plus_delta[4] = {0.0, 0.0, 0.0, 0.0};
  QuaternionProduct(q_delta, x, x_plus_delta);
  QuaternionParameterizationTestHelper<QuaternionParameterization,
                                       QuaternionPlus>(x, delta, x_plus_delta);
}

TEST(QuaternionParameterization, AwayFromZeroTest) {
  double x[4] = {0.52, 0.25, 0.15, 0.45};
  Normalize<4>(x);

  double delta[3] = {0.24, 0.15, 0.10};
  const double delta_norm =
      sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
  double q_delta[4];
  q_delta[0] = cos(delta_norm);
  q_delta[1] = sin(delta_norm) / delta_norm * delta[0];
  q_delta[2] = sin(delta_norm) / delta_norm * delta[1];
  q_delta[3] = sin(delta_norm) / delta_norm * delta[2];

  double x_plus_delta[4] = {0.0, 0.0, 0.0, 0.0};
  QuaternionProduct(q_delta, x, x_plus_delta);
  QuaternionParameterizationTestHelper<QuaternionParameterization,
                                       QuaternionPlus>(x, delta, x_plus_delta);
}

// Functor needed to implement automatically differentiated Plus for
// Eigen's quaternion.
struct EigenQuaternionPlus {
  template <typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    const T norm_delta =
        sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);

    Eigen::Quaternion<T> q_delta;
    if (norm_delta > T(0.0)) {
      const T sin_delta_by_delta = sin(norm_delta) / norm_delta;
      q_delta.coeffs() << sin_delta_by_delta * delta[0],
          sin_delta_by_delta * delta[1], sin_delta_by_delta * delta[2],
          cos(norm_delta);
    } else {
      // We do not just use q_delta = [0,0,0,1] here because that is a
      // constant and when used for automatic differentiation will
      // lead to a zero derivative. Instead we take a first order
      // approximation and evaluate it at zero.
      q_delta.coeffs() << delta[0], delta[1], delta[2], T(1.0);
    }

    Eigen::Map<Eigen::Quaternion<T>> x_plus_delta_ref(x_plus_delta);
    Eigen::Map<const Eigen::Quaternion<T>> x_ref(x);
    x_plus_delta_ref = q_delta * x_ref;
    return true;
  }
};

TEST(EigenQuaternionParameterization, ZeroTest) {
  Eigen::Quaterniond x(0.5, 0.5, 0.5, 0.5);
  double delta[3] = {0.0, 0.0, 0.0};
  Eigen::Quaterniond q_delta(1.0, 0.0, 0.0, 0.0);
  Eigen::Quaterniond x_plus_delta = q_delta * x;
  QuaternionParameterizationTestHelper<EigenQuaternionParameterization,
                                       EigenQuaternionPlus>(
      x.coeffs().data(), delta, x_plus_delta.coeffs().data());
}

TEST(EigenQuaternionParameterization, NearZeroTest) {
  Eigen::Quaterniond x(0.52, 0.25, 0.15, 0.45);
  x.normalize();

  double delta[3] = {0.24, 0.15, 0.10};
  for (int i = 0; i < 3; ++i) {
    delta[i] = delta[i] * 1e-14;
  }

  // Note: w is first in the constructor.
  Eigen::Quaterniond q_delta(1.0, delta[0], delta[1], delta[2]);

  Eigen::Quaterniond x_plus_delta = q_delta * x;
  QuaternionParameterizationTestHelper<EigenQuaternionParameterization,
                                       EigenQuaternionPlus>(
      x.coeffs().data(), delta, x_plus_delta.coeffs().data());
}

TEST(EigenQuaternionParameterization, AwayFromZeroTest) {
  Eigen::Quaterniond x(0.52, 0.25, 0.15, 0.45);
  x.normalize();

  double delta[3] = {0.24, 0.15, 0.10};
  const double delta_norm =
      sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);

  // Note: w is first in the constructor.
  Eigen::Quaterniond q_delta(cos(delta_norm),
                             sin(delta_norm) / delta_norm * delta[0],
                             sin(delta_norm) / delta_norm * delta[1],
                             sin(delta_norm) / delta_norm * delta[2]);

  Eigen::Quaterniond x_plus_delta = q_delta * x;
  QuaternionParameterizationTestHelper<EigenQuaternionParameterization,
                                       EigenQuaternionPlus>(
      x.coeffs().data(), delta, x_plus_delta.coeffs().data());
}

// Functor needed to implement automatically differentiated Plus for
// homogeneous vectors.
template <int Dim>
struct HomogeneousVectorParameterizationPlus {
  template <typename Scalar>
  bool operator()(const Scalar* p_x,
                  const Scalar* p_delta,
                  Scalar* p_x_plus_delta) const {
    Eigen::Map<const Eigen::Matrix<Scalar, Dim, 1>> x(p_x);
    Eigen::Map<const Eigen::Matrix<Scalar, Dim - 1, 1>> delta(p_delta);
    Eigen::Map<Eigen::Matrix<Scalar, Dim, 1>> x_plus_delta(p_x_plus_delta);

    const Scalar squared_norm_delta = delta.squaredNorm();

    Eigen::Matrix<Scalar, Dim, 1> y;
    Scalar one_half(0.5);
    if (squared_norm_delta > Scalar(0.0)) {
      Scalar norm_delta = sqrt(squared_norm_delta);
      Scalar norm_delta_div_2 = 0.5 * norm_delta;
      const Scalar sin_delta_by_delta =
          sin(norm_delta_div_2) / norm_delta_div_2;
      y.template head<Dim - 1>() = sin_delta_by_delta * one_half * delta;
      y[Dim - 1] = cos(norm_delta_div_2);

    } else {
      // We do not just use y = [0,0,0,1] here because that is a
      // constant and when used for automatic differentiation will
      // lead to a zero derivative. Instead we take a first order
      // approximation and evaluate it at zero.
      y.template head<Dim - 1>() = delta * one_half;
      y[Dim - 1] = Scalar(1.0);
    }

    Eigen::Matrix<Scalar, Dim, 1> v;
    Scalar beta;

    // NOTE: The explicit template arguments are needed here because
    // ComputeHouseholderVector is templated and some versions of MSVC
    // have trouble deducing the type of v automatically.
    internal::ComputeHouseholderVector<
        Eigen::Map<const Eigen::Matrix<Scalar, Dim, 1>>,
        Scalar,
        Dim>(x, &v, &beta);

    x_plus_delta = x.norm() * (y - v * (beta * v.dot(y)));

    return true;
  }
};

static void HomogeneousVectorParameterizationHelper(const double* x,
                                                    const double* delta) {
  const double kTolerance = 1e-14;

  HomogeneousVectorParameterization homogeneous_vector_parameterization(4);

  // Ensure the update maintains the norm.
  double x_plus_delta[4] = {0.0, 0.0, 0.0, 0.0};
  homogeneous_vector_parameterization.Plus(x, delta, x_plus_delta);

  const double x_plus_delta_norm = sqrt(
      x_plus_delta[0] * x_plus_delta[0] + x_plus_delta[1] * x_plus_delta[1] +
      x_plus_delta[2] * x_plus_delta[2] + x_plus_delta[3] * x_plus_delta[3]);

  const double x_norm =
      sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);

  EXPECT_NEAR(x_plus_delta_norm, x_norm, kTolerance);

  // Autodiff jacobian at delta_x = 0.
  AutoDiffLocalParameterization<HomogeneousVectorParameterizationPlus<4>, 4, 3>
      autodiff_jacobian;

  double jacobian_autodiff[12];
  double jacobian_analytic[12];

  homogeneous_vector_parameterization.ComputeJacobian(x, jacobian_analytic);
  autodiff_jacobian.ComputeJacobian(x, jacobian_autodiff);

  for (int i = 0; i < 12; ++i) {
    EXPECT_TRUE(ceres::IsFinite(jacobian_analytic[i]));
    EXPECT_NEAR(jacobian_analytic[i], jacobian_autodiff[i], kTolerance)
        << "Jacobian mismatch: i = " << i << ", " << jacobian_analytic[i] << " "
        << jacobian_autodiff[i];
  }
}

TEST(HomogeneousVectorParameterization, ZeroTest) {
  double x[4] = {0.0, 0.0, 0.0, 1.0};
  Normalize<4>(x);
  double delta[3] = {0.0, 0.0, 0.0};

  HomogeneousVectorParameterizationHelper(x, delta);
}

TEST(HomogeneousVectorParameterization, NearZeroTest1) {
  double x[4] = {1e-5, 1e-5, 1e-5, 1.0};
  Normalize<4>(x);
  double delta[3] = {0.0, 1.0, 0.0};

  HomogeneousVectorParameterizationHelper(x, delta);
}

TEST(HomogeneousVectorParameterization, NearZeroTest2) {
  double x[4] = {0.001, 0.0, 0.0, 0.0};
  double delta[3] = {0.0, 1.0, 0.0};

  HomogeneousVectorParameterizationHelper(x, delta);
}

TEST(HomogeneousVectorParameterization, AwayFromZeroTest1) {
  double x[4] = {0.52, 0.25, 0.15, 0.45};
  Normalize<4>(x);
  double delta[3] = {0.0, 1.0, -0.5};

  HomogeneousVectorParameterizationHelper(x, delta);
}

TEST(HomogeneousVectorParameterization, AwayFromZeroTest2) {
  double x[4] = {0.87, -0.25, -0.34, 0.45};
  Normalize<4>(x);
  double delta[3] = {0.0, 0.0, -0.5};

  HomogeneousVectorParameterizationHelper(x, delta);
}

TEST(HomogeneousVectorParameterization, AwayFromZeroTest3) {
  double x[4] = {0.0, 0.0, 0.0, 2.0};
  double delta[3] = {0.0, 0.0, 0};

  HomogeneousVectorParameterizationHelper(x, delta);
}

TEST(HomogeneousVectorParameterization, AwayFromZeroTest4) {
  double x[4] = {0.2, -1.0, 0.0, 2.0};
  double delta[3] = {1.4, 0.0, -0.5};

  HomogeneousVectorParameterizationHelper(x, delta);
}

TEST(HomogeneousVectorParameterization, AwayFromZeroTest5) {
  double x[4] = {2.0, 0.0, 0.0, 0.0};
  double delta[3] = {1.4, 0.0, -0.5};

  HomogeneousVectorParameterizationHelper(x, delta);
}

TEST(HomogeneousVectorParameterization, DeathTests) {
  EXPECT_DEATH_IF_SUPPORTED(HomogeneousVectorParameterization x(1), "size");
}

// Functor needed to implement automatically differentiated Plus for
// line parameterization.
template <int AmbientSpaceDim>
struct LineParameterizationPlus {
  template <typename Scalar>
  bool operator()(const Scalar* p_x,
                  const Scalar* p_delta,
                  Scalar* p_x_plus_delta) const {
    static constexpr int kTangetSpaceDim = AmbientSpaceDim - 1;
    Eigen::Map<const Eigen::Matrix<Scalar, AmbientSpaceDim, 1>> origin_point(
        p_x);
    Eigen::Map<const Eigen::Matrix<Scalar, AmbientSpaceDim, 1>> dir(
        p_x + AmbientSpaceDim);
    Eigen::Map<const Eigen::Matrix<Scalar, kTangetSpaceDim, 1>>
        delta_origin_point(p_delta);
    Eigen::Map<Eigen::Matrix<Scalar, AmbientSpaceDim, 1>>
        origin_point_plus_delta(p_x_plus_delta);

    HomogeneousVectorParameterizationPlus<AmbientSpaceDim> dir_plus;
    dir_plus(dir.data(),
             p_delta + kTangetSpaceDim,
             p_x_plus_delta + AmbientSpaceDim);

    Eigen::Matrix<Scalar, AmbientSpaceDim, 1> v;
    Scalar beta;

    // NOTE: The explicit template arguments are needed here because
    // ComputeHouseholderVector is templated and some versions of MSVC
    // have trouble deducing the type of v automatically.
    internal::ComputeHouseholderVector<
        Eigen::Map<const Eigen::Matrix<Scalar, AmbientSpaceDim, 1>>,
        Scalar,
        AmbientSpaceDim>(dir, &v, &beta);

    Eigen::Matrix<Scalar, AmbientSpaceDim, 1> y;
    y << 0.5 * delta_origin_point, Scalar(0.0);
    origin_point_plus_delta = origin_point + y - v * (beta * v.dot(y));

    return true;
  }
};

template <int AmbientSpaceDim>
static void LineParameterizationHelper(const double* x_ptr,
                                       const double* delta) {
  const double kTolerance = 1e-14;

  static constexpr int ParameterDim = 2 * AmbientSpaceDim;
  static constexpr int TangientParameterDim = 2 * (AmbientSpaceDim - 1);

  LineParameterization<AmbientSpaceDim> line_parameterization;

  using ParameterVector = Eigen::Matrix<double, ParameterDim, 1>;
  ParameterVector x_plus_delta = ParameterVector::Zero();
  line_parameterization.Plus(x_ptr, delta, x_plus_delta.data());

  // Ensure the update maintains the norm for the line direction.
  Eigen::Map<const ParameterVector> x(x_ptr);
  const double dir_plus_delta_norm =
      x_plus_delta.template tail<AmbientSpaceDim>().norm();
  const double dir_norm = x.template tail<AmbientSpaceDim>().norm();
  EXPECT_NEAR(dir_plus_delta_norm, dir_norm, kTolerance);

  // Ensure the update of the origin point is perpendicular to the line
  // direction.
  const double dot_prod_val = x.template tail<AmbientSpaceDim>().dot(
      x_plus_delta.template head<AmbientSpaceDim>() -
      x.template head<AmbientSpaceDim>());
  EXPECT_NEAR(dot_prod_val, 0.0, kTolerance);

  // Autodiff jacobian at delta_x = 0.
  AutoDiffLocalParameterization<LineParameterizationPlus<AmbientSpaceDim>,
                                ParameterDim,
                                TangientParameterDim>
      autodiff_jacobian;

  using JacobianMatrix = Eigen::
      Matrix<double, ParameterDim, TangientParameterDim, Eigen::RowMajor>;
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  JacobianMatrix jacobian_autodiff = JacobianMatrix::Constant(kNaN);
  JacobianMatrix jacobian_analytic = JacobianMatrix::Constant(kNaN);

  autodiff_jacobian.ComputeJacobian(x_ptr, jacobian_autodiff.data());
  line_parameterization.ComputeJacobian(x_ptr, jacobian_analytic.data());

  EXPECT_FALSE(jacobian_autodiff.hasNaN());
  EXPECT_FALSE(jacobian_analytic.hasNaN());
  EXPECT_TRUE(jacobian_autodiff.isApprox(jacobian_analytic))
      << "auto diff:\n"
      << jacobian_autodiff << "\n"
      << "analytic diff:\n"
      << jacobian_analytic;
}

TEST(LineParameterization, ZeroTest3D) {
  double x[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  double delta[4] = {0.0, 0.0, 0.0, 0.0};

  LineParameterizationHelper<3>(x, delta);
}

TEST(LineParameterization, ZeroTest4D) {
  double x[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  double delta[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  LineParameterizationHelper<4>(x, delta);
}

TEST(LineParameterization, ZeroOriginPointTest3D) {
  double x[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  double delta[4] = {0.0, 0.0, 1.0, 2.0};

  LineParameterizationHelper<3>(x, delta);
}

TEST(LineParameterization, ZeroOriginPointTest4D) {
  double x[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  double delta[6] = {0.0, 0.0, 0.0, 1.0, 2.0, 3.0};

  LineParameterizationHelper<4>(x, delta);
}

TEST(LineParameterization, ZeroDirTest3D) {
  double x[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  double delta[4] = {3.0, 2.0, 0.0, 0.0};

  LineParameterizationHelper<3>(x, delta);
}

TEST(LineParameterization, ZeroDirTest4D) {
  double x[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  double delta[6] = {3.0, 2.0, 1.0, 0.0, 0.0, 0.0};

  LineParameterizationHelper<4>(x, delta);
}

TEST(LineParameterization, AwayFromZeroTest3D1) {
  Eigen::Matrix<double, 6, 1> x;
  x.head<3>() << 1.54, 2.32, 1.34;
  x.tail<3>() << 0.52, 0.25, 0.15;
  x.tail<3>().normalize();

  double delta[4] = {4.0, 7.0, 1.0, -0.5};

  LineParameterizationHelper<3>(x.data(), delta);
}

TEST(LineParameterization, AwayFromZeroTest4D1) {
  Eigen::Matrix<double, 8, 1> x;
  x.head<4>() << 1.54, 2.32, 1.34, 3.23;
  x.tail<4>() << 0.52, 0.25, 0.15, 0.45;
  x.tail<4>().normalize();

  double delta[6] = {4.0, 7.0, -3.0, 0.0, 1.0, -0.5};

  LineParameterizationHelper<4>(x.data(), delta);
}

TEST(LineParameterization, AwayFromZeroTest3D2) {
  Eigen::Matrix<double, 6, 1> x;
  x.head<3>() << 7.54, -2.81, 8.63;
  x.tail<3>() << 2.52, 5.25, 4.15;

  double delta[4] = {4.0, 7.0, 1.0, -0.5};

  LineParameterizationHelper<3>(x.data(), delta);
}

TEST(LineParameterization, AwayFromZeroTest4D2) {
  Eigen::Matrix<double, 8, 1> x;
  x.head<4>() << 7.54, -2.81, 8.63, 6.93;
  x.tail<4>() << 2.52, 5.25, 4.15, 1.45;

  double delta[6] = {4.0, 7.0, -3.0, 2.0, 1.0, -0.5};

  LineParameterizationHelper<4>(x.data(), delta);
}

class ProductParameterizationTest : public ::testing::Test {
 protected:
  void SetUp() final {
    const int global_size1 = 5;
    std::vector<int> constant_parameters1;
    constant_parameters1.push_back(2);
    param1_.reset(
        new SubsetParameterization(global_size1, constant_parameters1));

    const int global_size2 = 3;
    std::vector<int> constant_parameters2;
    constant_parameters2.push_back(0);
    constant_parameters2.push_back(1);
    param2_.reset(
        new SubsetParameterization(global_size2, constant_parameters2));

    const int global_size3 = 4;
    std::vector<int> constant_parameters3;
    constant_parameters3.push_back(1);
    param3_.reset(
        new SubsetParameterization(global_size3, constant_parameters3));

    const int global_size4 = 2;
    std::vector<int> constant_parameters4;
    constant_parameters4.push_back(1);
    param4_.reset(
        new SubsetParameterization(global_size4, constant_parameters4));
  }

  std::unique_ptr<LocalParameterization> param1_;
  std::unique_ptr<LocalParameterization> param2_;
  std::unique_ptr<LocalParameterization> param3_;
  std::unique_ptr<LocalParameterization> param4_;
};

TEST_F(ProductParameterizationTest, LocalAndGlobalSize2) {
  LocalParameterization* param1 = param1_.release();
  LocalParameterization* param2 = param2_.release();

  ProductParameterization product_param(param1, param2);
  EXPECT_EQ(product_param.LocalSize(),
            param1->LocalSize() + param2->LocalSize());
  EXPECT_EQ(product_param.GlobalSize(),
            param1->GlobalSize() + param2->GlobalSize());
}

TEST_F(ProductParameterizationTest, LocalAndGlobalSize3) {
  LocalParameterization* param1 = param1_.release();
  LocalParameterization* param2 = param2_.release();
  LocalParameterization* param3 = param3_.release();

  ProductParameterization product_param(param1, param2, param3);
  EXPECT_EQ(product_param.LocalSize(),
            param1->LocalSize() + param2->LocalSize() + param3->LocalSize());
  EXPECT_EQ(product_param.GlobalSize(),
            param1->GlobalSize() + param2->GlobalSize() + param3->GlobalSize());
}

TEST_F(ProductParameterizationTest, LocalAndGlobalSize4) {
  LocalParameterization* param1 = param1_.release();
  LocalParameterization* param2 = param2_.release();
  LocalParameterization* param3 = param3_.release();
  LocalParameterization* param4 = param4_.release();

  ProductParameterization product_param(param1, param2, param3, param4);
  EXPECT_EQ(product_param.LocalSize(),
            param1->LocalSize() + param2->LocalSize() + param3->LocalSize() +
                param4->LocalSize());
  EXPECT_EQ(product_param.GlobalSize(),
            param1->GlobalSize() + param2->GlobalSize() + param3->GlobalSize() +
                param4->GlobalSize());
}

TEST_F(ProductParameterizationTest, Plus) {
  LocalParameterization* param1 = param1_.release();
  LocalParameterization* param2 = param2_.release();
  LocalParameterization* param3 = param3_.release();
  LocalParameterization* param4 = param4_.release();

  ProductParameterization product_param(param1, param2, param3, param4);
  std::vector<double> x(product_param.GlobalSize(), 0.0);
  std::vector<double> delta(product_param.LocalSize(), 0.0);
  std::vector<double> x_plus_delta_expected(product_param.GlobalSize(), 0.0);
  std::vector<double> x_plus_delta(product_param.GlobalSize(), 0.0);

  for (int i = 0; i < product_param.GlobalSize(); ++i) {
    x[i] = RandNormal();
  }

  for (int i = 0; i < product_param.LocalSize(); ++i) {
    delta[i] = RandNormal();
  }

  EXPECT_TRUE(product_param.Plus(&x[0], &delta[0], &x_plus_delta_expected[0]));
  int x_cursor = 0;
  int delta_cursor = 0;

  EXPECT_TRUE(param1->Plus(
      &x[x_cursor], &delta[delta_cursor], &x_plus_delta[x_cursor]));
  x_cursor += param1->GlobalSize();
  delta_cursor += param1->LocalSize();

  EXPECT_TRUE(param2->Plus(
      &x[x_cursor], &delta[delta_cursor], &x_plus_delta[x_cursor]));
  x_cursor += param2->GlobalSize();
  delta_cursor += param2->LocalSize();

  EXPECT_TRUE(param3->Plus(
      &x[x_cursor], &delta[delta_cursor], &x_plus_delta[x_cursor]));
  x_cursor += param3->GlobalSize();
  delta_cursor += param3->LocalSize();

  EXPECT_TRUE(param4->Plus(
      &x[x_cursor], &delta[delta_cursor], &x_plus_delta[x_cursor]));
  x_cursor += param4->GlobalSize();
  delta_cursor += param4->LocalSize();

  for (int i = 0; i < x.size(); ++i) {
    EXPECT_EQ(x_plus_delta[i], x_plus_delta_expected[i]);
  }
}

TEST_F(ProductParameterizationTest, ComputeJacobian) {
  LocalParameterization* param1 = param1_.release();
  LocalParameterization* param2 = param2_.release();
  LocalParameterization* param3 = param3_.release();
  LocalParameterization* param4 = param4_.release();

  ProductParameterization product_param(param1, param2, param3, param4);
  std::vector<double> x(product_param.GlobalSize(), 0.0);

  for (int i = 0; i < product_param.GlobalSize(); ++i) {
    x[i] = RandNormal();
  }

  Matrix jacobian =
      Matrix::Random(product_param.GlobalSize(), product_param.LocalSize());
  EXPECT_TRUE(product_param.ComputeJacobian(&x[0], jacobian.data()));
  int x_cursor = 0;
  int delta_cursor = 0;

  Matrix jacobian1(param1->GlobalSize(), param1->LocalSize());
  EXPECT_TRUE(param1->ComputeJacobian(&x[x_cursor], jacobian1.data()));
  jacobian.block(
      x_cursor, delta_cursor, param1->GlobalSize(), param1->LocalSize()) -=
      jacobian1;
  x_cursor += param1->GlobalSize();
  delta_cursor += param1->LocalSize();

  Matrix jacobian2(param2->GlobalSize(), param2->LocalSize());
  EXPECT_TRUE(param2->ComputeJacobian(&x[x_cursor], jacobian2.data()));
  jacobian.block(
      x_cursor, delta_cursor, param2->GlobalSize(), param2->LocalSize()) -=
      jacobian2;
  x_cursor += param2->GlobalSize();
  delta_cursor += param2->LocalSize();

  Matrix jacobian3(param3->GlobalSize(), param3->LocalSize());
  EXPECT_TRUE(param3->ComputeJacobian(&x[x_cursor], jacobian3.data()));
  jacobian.block(
      x_cursor, delta_cursor, param3->GlobalSize(), param3->LocalSize()) -=
      jacobian3;
  x_cursor += param3->GlobalSize();
  delta_cursor += param3->LocalSize();

  Matrix jacobian4(param4->GlobalSize(), param4->LocalSize());
  EXPECT_TRUE(param4->ComputeJacobian(&x[x_cursor], jacobian4.data()));
  jacobian.block(
      x_cursor, delta_cursor, param4->GlobalSize(), param4->LocalSize()) -=
      jacobian4;
  x_cursor += param4->GlobalSize();
  delta_cursor += param4->LocalSize();

  EXPECT_NEAR(jacobian.norm(), 0.0, std::numeric_limits<double>::epsilon());
}

}  // namespace internal
}  // namespace ceres
