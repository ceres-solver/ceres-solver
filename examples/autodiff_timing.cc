#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <ctime>
#include <sys/time.h>

double WallTimeInSeconds() {
  timeval time_val;
  gettimeofday(&time_val, NULL);
  return (time_val.tv_sec + time_val.tv_usec * 1e-6);
}

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);

    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = - p[0] / p[2];
    T yp = - p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    T r2 = xp*xp + yp*yp;
    T distortion = T(1.0) + r2  * (l1 + l2  * r2);

    // Compute final projected point position.
    const T& focal = camera[6];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};


int main(int argc, char** argv) {
  ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(1, 2);

  std::cout << "Hello thad5" << std::endl;

  ceres::Jet<double, 12> x;
  std::cout << sizeof(x) << std::endl;

  double point[] = {1, 2, 4};
  double camera[9] = {1, 2, 3, 4, 5, 6, 1.0, 1e-4, 1e-8};
  double point_jacobian[2 * 3];
  double camera_jacobian[2 * 9];
  double* parameters[] = {point, camera};
  double* jacobians[] = {camera_jacobian, point_jacobian};
  double residuals[2];

  const int num_iters = 1000000;
  double start_time = WallTimeInSeconds();
  for (int i = 0; i < num_iters; ++i) {
    cost_function->Evaluate(parameters, residuals, jacobians);
  }
  double end_time = WallTimeInSeconds();

  std::cout << "time per execution: " << (end_time - start_time) / num_iters << std::endl;
  delete cost_function;
  return 0;
};
