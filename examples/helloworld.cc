#include "ceres/ceres.h"

void ThirdPartyProjectionFunction(const double*, const double*, double*);

struct IntrinsicProjection {
  IntrinsicProjection(const double* observation) {
    observation_[0] = observation[0];
    observation_[1] = observation[1];
  }

  bool operator()(const double* calibration,
                  const double* point,
                  double* residuals) const {
    double projection[2];
    ThirdPartyProjectionFunction(calibration, point, projection);
    residuals[0] = observation_[0] - projection[0];
    residuals[1] = observation_[1] - projection[1];
    return true;
  }
  double observation_[2];
};

struct CameraProjection {
  CameraProjection(double* observation)
      : intrinsic_projection_(
            new NumericDiffCostFunction<IntrinsicProjection, CENTRAL, 2, 5, 3>(
                new IntrinsicProjection(observation))) {}

  template <typename T>
  bool operator()(const T* rotation,
                  const T* translation,
                  const T* intrinsics,
                  const T* point,
                  T* residuals) const {
    T transformed_point[3];
    RotateAndTranslatePoint(rotation, translation, point, transformed_point);
    return intrinsic_projection_(intrinsics, transformed_point, residuals);
  }

 private:
  CostFunctionToFunctor<2, 5, 3> intrinsic_projection_;
};

class blaFunctor {
 public:
  blaFunctor(){};
  ~blaFunctor(){};

  bool operator()(const double* const x, double* residual) const {
    residual[0] = x[0];
    return true;
  }
};

int main() {
  auto cost_function =
      new ceres::NumericDiffCostFunction<blaFunctor, ceres::CENTRAL, 1, 1>(
          new blaFunctor);
  delete cost_function;
}
