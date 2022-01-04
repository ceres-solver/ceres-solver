#include "ceres/manifold.h"

#include <cmath>

#include "ceres/internal/eigen.h"
#include "ceres/internal/fixed_array.h"
#include "glog/logging.h"

namespace ceres {

bool Manifold::RightMultiplyByPlusJacobian(const double* x,
                                           const int num_rows,
                                           const double* ambient_matrix,
                                           double* tangent_matrix) const {
  const int tangent_size = TangentSize();
  if (tangent_size == 0) {
    return true;
  }

  const int ambient_size = AmbientSize();
  Matrix plus_jacobian(ambient_size, tangent_size);
  if (!PlusJacobian(x, plus_jacobian.data())) {
    return false;
  }

  MatrixRef(tangent_matrix, num_rows, tangent_size) =
      ConstMatrixRef(ambient_matrix, num_rows, ambient_size) * plus_jacobian;
  return true;
}

EuclideanManifold::EuclideanManifold(int size) : size_(size) {
  CHECK_GE(size, 0);
}

int EuclideanManifold::AmbientSize() const { return size_; }

int EuclideanManifold::TangentSize() const { return size_; }

bool EuclideanManifold::Plus(const double* x,
                             const double* delta,
                             double* x_plus_delta) const {
  for (int i = 0; i < size_; ++i) {
    x_plus_delta[i] = x[i] + delta[i];
  }
  return true;
}

bool EuclideanManifold::PlusJacobian(const double* x, double* jacobian) const {
  MatrixRef(jacobian, size_, size_).setIdentity();
  return true;
}

bool EuclideanManifold::RightMultiplyByPlusJacobian(
    const double* x,
    const int num_rows,
    const double* ambient_matrix,
    double* tangent_matrix) const {
  std::copy_n(ambient_matrix, num_rows * size_, tangent_matrix);
  return true;
}

bool EuclideanManifold::Minus(const double* y,
                              const double* x,
                              double* y_minus_x) const {
  for (int i = 0; i < size_; ++i) {
    y_minus_x[i] = y[i] - x[i];
  }
  return true;
};

bool EuclideanManifold::MinusJacobian(const double* x, double* jacobian) const {
  MatrixRef(jacobian, size_, size_).setIdentity();
  return true;
}

SubsetManifold::SubsetManifold(const int size,
                               const std::vector<int>& constant_parameters)

    : tangent_size_(size - constant_parameters.size()),
      constancy_mask_(size, false) {
  if (constant_parameters.empty()) {
    return;
  }

  std::vector<int> constant = constant_parameters;
  std::sort(constant.begin(), constant.end());
  CHECK_GE(constant.front(), 0) << "Indices indicating constant parameter must "
                                   "be greater than equal to zero.";
  CHECK_LT(constant.back(), size)
      << "Indices indicating constant parameter must be less than the size "
      << "of the parameter block.";
  CHECK(std::adjacent_find(constant.begin(), constant.end()) == constant.end())
      << "The set of constant parameters cannot contain duplicates";

  for (auto index : constant_parameters) {
    constancy_mask_[index] = true;
  }
}

int SubsetManifold::AmbientSize() const { return constancy_mask_.size(); }

int SubsetManifold::TangentSize() const { return tangent_size_; }

bool SubsetManifold::Plus(const double* x,
                          const double* delta,
                          double* x_plus_delta) const {
  const int ambient_size = AmbientSize();
  for (int i = 0, j = 0; i < ambient_size; ++i) {
    if (constancy_mask_[i]) {
      x_plus_delta[i] = x[i];
    } else {
      x_plus_delta[i] = x[i] + delta[j++];
    }
  }
  return true;
}

bool SubsetManifold::PlusJacobian(const double* x,
                                  double* plus_jacobian) const {
  if (tangent_size_ == 0) {
    return true;
  }

  const int ambient_size = AmbientSize();
  MatrixRef m(plus_jacobian, ambient_size, tangent_size_);
  m.setZero();
  for (int r = 0, c = 0; r < ambient_size; ++r) {
    if (!constancy_mask_[r]) {
      m(r, c++) = 1.0;
    }
  }
  return true;
}

bool SubsetManifold::RightMultiplyByPlusJacobian(const double* x,
                                                 const int num_rows,
                                                 const double* ambient_matrix,
                                                 double* tangent_matrix) const {
  if (tangent_size_ == 0) {
    return true;
  }

  const int ambient_size = AmbientSize();
  for (int r = 0; r < num_rows; ++r) {
    for (int idx = 0, c = 0; idx < ambient_size; ++idx) {
      if (!constancy_mask_[idx]) {
        tangent_matrix[r * tangent_size_ + c++] =
            ambient_matrix[r * ambient_size + idx];
      }
    }
  }
  return true;
}

bool SubsetManifold::Minus(const double* y,
                           const double* x,
                           double* y_minus_x) const {
  if (tangent_size_ == 0) {
    return true;
  }

  const int ambient_size = AmbientSize();
  for (int i = 0, j = 0; i < ambient_size; ++i) {
    if (!constancy_mask_[i]) {
      y_minus_x[j++] = y[i] - x[i];
    }
  }
  return true;
}

bool SubsetManifold::MinusJacobian(const double* x,
                                   double* minus_jacobian) const {
  const int ambient_size = AmbientSize();
  const int tangent_size = TangentSize();
  MatrixRef m(minus_jacobian, tangent_size_, ambient_size);
  m.setZero();
  for (int c = 0, r = 0; c < ambient_size; ++c) {
    if (!constancy_mask_[c]) {
      m(r++, c) = 1.0;
    }
  }
  return true;
}

int ProductManifold::AmbientSize() const { return ambient_size_; }

int ProductManifold::TangentSize() const { return tangent_size_; }

bool ProductManifold::Plus(const double* x,
                           const double* delta,
                           double* x_plus_delta) const {
  int ambient_cursor = 0;
  int tangent_cursor = 0;
  for (const auto& m : manifolds_) {
    if (!m->Plus(x + ambient_cursor,
                 delta + tangent_cursor,
                 x_plus_delta + ambient_cursor)) {
      return false;
    }
    tangent_cursor += m->TangentSize();
    ambient_cursor += m->AmbientSize();
  }

  return true;
}

bool ProductManifold::PlusJacobian(const double* x,
                                   double* jacobian_ptr) const {
  MatrixRef jacobian(jacobian_ptr, AmbientSize(), TangentSize());
  jacobian.setZero();
  internal::FixedArray<double> buffer(buffer_size_);

  int ambient_cursor = 0;
  int tangent_cursor = 0;
  for (const auto& m : manifolds_) {
    const int ambient_size = m->AmbientSize();
    const int tangent_size = m->TangentSize();

    if (!m->PlusJacobian(x + ambient_cursor, buffer.data())) {
      return false;
    }

    jacobian.block(ambient_cursor, tangent_cursor, ambient_size, tangent_size) =
        MatrixRef(buffer.data(), ambient_size, tangent_size);

    ambient_cursor += ambient_size;
    tangent_cursor += tangent_size;
  }

  return true;
}

bool ProductManifold::Minus(const double* y,
                            const double* x,
                            double* y_minus_x) const {
  int ambient_cursor = 0;
  int tangent_cursor = 0;
  for (const auto& m : manifolds_) {
    if (!m->Minus(y + ambient_cursor,
                  x + ambient_cursor,
                  y_minus_x + tangent_cursor)) {
      return false;
    }
    tangent_cursor += m->TangentSize();
    ambient_cursor += m->AmbientSize();
  }

  return true;
}

bool ProductManifold::MinusJacobian(const double* x,
                                    double* jacobian_ptr) const {
  MatrixRef jacobian(jacobian_ptr, TangentSize(), AmbientSize());
  jacobian.setZero();
  internal::FixedArray<double> buffer(buffer_size_);

  int ambient_cursor = 0;
  int tangent_cursor = 0;
  for (const auto& m : manifolds_) {
    const int ambient_size = m->AmbientSize();
    const int tangent_size = m->TangentSize();

    if (!m->MinusJacobian(x + ambient_cursor, buffer.data())) {
      return false;
    }

    jacobian.block(tangent_cursor, ambient_cursor, tangent_size, ambient_size) =
        MatrixRef(buffer.data(), tangent_size, ambient_size);

    ambient_cursor += ambient_size;
    tangent_cursor += tangent_size;
  }

  return true;
}

bool Quaternion::Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const {
  // x_plus_delta = QuaternionProduct(q_delta, x), where q_delta is the
  // quaternion constructed from delta.
  const double norm_delta = std::sqrt(
      delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
  if (norm_delta > 0.0) {
    const double sin_delta_by_delta = (std::sin(norm_delta) / norm_delta);
    double q_delta[4];
    q_delta[0] = std::cos(norm_delta);
    q_delta[1] = sin_delta_by_delta * delta[0];
    q_delta[2] = sin_delta_by_delta * delta[1];
    q_delta[3] = sin_delta_by_delta * delta[2];

    // clang-format off
    x_plus_delta[0] = q_delta[0] * x[0] - q_delta[1] * x[1] - q_delta[2] * x[2] - q_delta[3] * x[3];
    x_plus_delta[1] = q_delta[0] * x[1] + q_delta[1] * x[0] + q_delta[2] * x[3] - q_delta[3] * x[2];
    x_plus_delta[2] = q_delta[0] * x[2] - q_delta[1] * x[3] + q_delta[2] * x[0] + q_delta[3] * x[1];
    x_plus_delta[3] = q_delta[0] * x[3] + q_delta[1] * x[2] - q_delta[2] * x[1] + q_delta[3] * x[0];
    // clang-format on
  } else {
    for (int i = 0; i < 4; ++i) {
      x_plus_delta[i] = x[i];
    }
  }
  return true;
}

bool Quaternion::PlusJacobian(const double* x, double* jacobian) const {
  // clang-format off
  jacobian[0] = -x[1];  jacobian[1]  = -x[2];   jacobian[2]  = -x[3];
  jacobian[3] =  x[0];  jacobian[4]  =  x[3];   jacobian[5]  = -x[2];
  jacobian[6] = -x[3];  jacobian[7]  =  x[0];   jacobian[8]  =  x[1];
  jacobian[9] =  x[2];  jacobian[10] = -x[1];   jacobian[11] =  x[0];
  // clang-format on
  return true;
}

bool Quaternion::Minus(const double* y,
                       const double* x,
                       double* y_minus_x) const {
  // ambient_y_minus_x = QuaternionProduct(y, -x) where -x is the conjugate of
  // x.
  double ambient_y_minus_x[4];

  // clang-format off
  ambient_y_minus_x[0] =  y[0] * x[0] + y[1] * x[1] + y[2] * x[2] + y[3] * x[3];
  ambient_y_minus_x[1] = -y[0] * x[1] + y[1] * x[0] - y[2] * x[3] + y[3] * x[2];
  ambient_y_minus_x[2] = -y[0] * x[2] + y[1] * x[3] + y[2] * x[0] - y[3] * x[1];
  ambient_y_minus_x[3] = -y[0] * x[3] - y[1] * x[2] + y[2] * x[1] + y[3] * x[0];
  // clang-format on

  const double u_norm = std::sqrt(ambient_y_minus_x[1] * ambient_y_minus_x[1] +
                                  ambient_y_minus_x[2] * ambient_y_minus_x[2] +
                                  ambient_y_minus_x[3] * ambient_y_minus_x[3]);
  if (u_norm > 0.0) {
    const double theta = std::atan2(u_norm, ambient_y_minus_x[0]);
    y_minus_x[0] = theta * ambient_y_minus_x[1] / u_norm;
    y_minus_x[1] = theta * ambient_y_minus_x[2] / u_norm;
    y_minus_x[2] = theta * ambient_y_minus_x[3] / u_norm;
  } else {
    y_minus_x[0] = 0.0;
    y_minus_x[1] = 0.0;
    y_minus_x[2] = 0.0;
  }
  return true;
}

bool Quaternion::MinusJacobian(const double* x, double* jacobian) const {
  // clang-format off
  jacobian[0] = - x[1]; jacobian[1]=  x[0]; jacobian[2]  = -x[3]; jacobian[3]  =   x[2];
  jacobian[4] = - x[2]; jacobian[5]=  x[3]; jacobian[6]  =  x[0]; jacobian[7]  =  -x[1];
  jacobian[8] = - x[3]; jacobian[9]= -x[2]; jacobian[10] =  x[1]; jacobian[11] =   x[0];
  // clang-format on
  return true;
}

}  // namespace ceres
