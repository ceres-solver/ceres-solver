#include "ceres/manifold.h"

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

}  // namespace ceres
