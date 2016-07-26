#include "pose_graph_2d_error_term.h"

#include <vector>

#include "Eigen/Dense"
#include "ceres/ceres.h"
#include "gtest/gtest.h"
#include "normalize_angle.h"

namespace {

template <typename Scalar>
Eigen::Matrix<Scalar, 2, 2> RotationMatrix(Scalar yaw_radians) {
  const Scalar cos_yaw = ceres::cos(yaw_radians);
  const Scalar sin_yaw = ceres::sin(yaw_radians);

  Eigen::Matrix<Scalar, 2, 2> rotation;
  rotation << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
  return rotation;
}

// The pose graph 2d error term implemented for use with Ceres's automatic
// differentiation capability.
class PoseGraph2dErrorTermAutoDiff {
 public:
  PoseGraph2dErrorTermAutoDiff(double A_x_B, double A_y_B,
                               double A_yaw_B_radians,
                               const Eigen::Matrix3d& sqrt_information)
      : A_p_B_(A_x_B, A_y_B),
        A_yaw_B_radians_(A_yaw_B_radians),
        sqrt_information_(sqrt_information) { }

  template <typename Scalar>
  bool operator()(const Scalar* const G_x_A, const Scalar* const G_y_A,
                  const Scalar* const yaw_A, const Scalar* const G_x_B,
                  const Scalar* const G_y_B, const Scalar* const yaw_B,
                  Scalar* residuals_ptr) const {
    CHECK(residuals_ptr != NULL);
    CHECK(G_x_A != NULL);
    CHECK(G_y_A != NULL);
    CHECK(yaw_A != NULL);
    CHECK(G_x_B != NULL);
    CHECK(G_y_B != NULL);
    CHECK(yaw_B != NULL);

    const Eigen::Matrix<Scalar, 2, 1> G_p_A(*G_x_A, *G_y_A);
    const Scalar& G_yaw_A = *yaw_A;

    const Eigen::Matrix<Scalar, 2, 2> A_R_G =
        RotationMatrix(G_yaw_A).transpose();

    const Eigen::Matrix<Scalar, 2, 1> G_p_B(*G_x_B, *G_y_B);
    const Scalar& G_yaw_B = *yaw_B;

    Eigen::Map<Eigen::Matrix<Scalar, 3, 1> > residuals_map(residuals_ptr);

    Eigen::Matrix<Scalar, 3, 1> residuals;
    residuals_map.template head<2>() =
        A_R_G * (G_p_B - G_p_A) - A_p_B_.cast<Scalar>();
    residuals_map(2) = ceres::examples::pose_graph_2d::NormalizeAngle(
        (G_yaw_B - G_yaw_A) - static_cast<Scalar>(A_yaw_B_radians_));

    // Scale the residuals by the sqrt information matrix.
    residuals_map = sqrt_information_.template cast<Scalar>() * residuals_map;

    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // The position of B relative to A in the A frame.
  Eigen::Vector2d A_p_B_;
  // The orientation of frame B relative to frame A.
  double A_yaw_B_radians_;
  // The square root of the measurement information matrix.
  Eigen::Matrix3d sqrt_information_;
};

void ErrorTermTestHelper(const double* G_x_A, const double* G_y_A,
                         const double* yaw_A, const double* G_x_B,
                         const double* G_y_B, const double* yaw_B,
                         const double A_x_B, const double A_y_B,
                         const double A_yaw_B,
                         const Eigen::Matrix3d& information) {
  CHECK(G_x_A != NULL);
  CHECK(G_y_A != NULL);
  CHECK(yaw_A != NULL);
  CHECK(G_x_B != NULL);
  CHECK(G_y_B != NULL);
  CHECK(yaw_B != NULL);

  const Eigen::Matrix3d sqrt_information = information.llt().matrixL();

  const double* parameters[6] = {G_x_A, G_y_A, yaw_A, G_x_B, G_y_B, yaw_B};

  double auto_1[3];
  double auto_2[3];
  double auto_3[3];
  double auto_4[3];
  double auto_5[3];
  double auto_6[3];
  double analytic_1[3];
  double analytic_2[3];
  double analytic_3[3];
  double analytic_4[3];
  double analytic_5[3];
  double analytic_6[3];
  double* jacobians_auto[6] = {auto_1, auto_2, auto_3, auto_4, auto_5, auto_6};
  double* jacobians_analytic[6] = {analytic_1, analytic_2, analytic_3,
                                   analytic_4, analytic_5, analytic_6};

  // Compute the Jacobian of the cost function via AutoDiff.
  ceres::AutoDiffCostFunction<PoseGraph2dErrorTermAutoDiff, 3, 1, 1, 1, 1, 1, 1>
      cost_function(new PoseGraph2dErrorTermAutoDiff(A_x_B, A_y_B, A_yaw_B,
                                                     sqrt_information));

  // Compute the Jacobian of the cost function via the analytical
  // derivatives.
  ceres::examples::pose_graph_2d::PoseGraph2dErrorTerm analytical_error_term(
      A_x_B, A_y_B, A_yaw_B, sqrt_information);
  double residuals_auto[3];
  double residuals_analytic[3];
  cost_function.Evaluate(parameters, residuals_auto, jacobians_auto);
  analytical_error_term.Evaluate(parameters, residuals_analytic,
                                 jacobians_analytic);

  // Tolerance for precision of the tests.
  const double kTolerance = 1e-14;

  // Check the residuals between the analytic and auto diff classes.
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(ceres::IsFinite(residuals_auto[i]));
    EXPECT_TRUE(ceres::IsFinite(residuals_analytic[i]));
    EXPECT_NEAR(residuals_auto[i], residuals_analytic[i], kTolerance)
        << "Residuals mismatch: i = " << i;
  }

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_TRUE(ceres::IsFinite(jacobians_auto[i][j]));
      EXPECT_TRUE(ceres::IsFinite(jacobians_analytic[i][j]));
      EXPECT_NEAR(jacobians_auto[i][j], jacobians_analytic[i][j], kTolerance)
          << "Jacobians mismatch: (i,j) = (" << i << ", " << j << ")";
    }
  }
}

struct Data {
  Data(double _G_x_A, double _G_y_A, double _yaw_A, double _G_x_B,
       double _G_y_B, double _yaw_B, double _A_x_B, double _A_y_B,
       double _A_yaw_B)
      : G_x_A(_G_x_A),
        G_y_A(_G_y_A),
        yaw_A(_yaw_A),
        G_x_B(_G_x_B),
        G_y_B(_G_y_B),
        yaw_B(_yaw_B),
        A_x_B(_A_x_B),
        A_y_B(_A_y_B),
        A_yaw_B(_A_yaw_B) {}

  double G_x_A;
  double G_y_A;
  double yaw_A;
  double G_x_B;
  double G_y_B;
  double yaw_B;
  double A_x_B;
  double A_y_B;
  double A_yaw_B;
};

// Tests the analytical and symbolic differentiation are equal for exact same
// poses with zero residuals.
TEST(PoseGraph2dErrorTerm, SamePoseZeroResidual) {
  Data params(/*G_x_A = */ 1, /*G_y_A = */ 1, /*yaw_A = */ 0,
              /*G_x_B = */ 1, /*G_y_B = */ 1, /*yaw_B = */ 0,
              /*A_x_B = */ 0, /*A_y_B = */ 0, /*A_yaw_B = */ 0);
  Eigen::Matrix3d information = Eigen::Matrix3d::Identity();

  ErrorTermTestHelper(&params.G_x_A, &params.G_y_A, &params.yaw_A,
                      &params.G_x_B, &params.G_y_B, &params.yaw_B, params.A_x_B,
                      params.A_y_B, params.A_yaw_B, information);
}

// Tests the analytical and symbolic differentiation are equal for different
// poses with zero residuals.
TEST(PoseGraph2dErrorTerm, ZeroResidual) {
  Data params(/*G_x_A = */ 1, /*G_y_A = */ 1, /*yaw_A = */ 0,
              /*G_x_B = */ 2, /*G_y_B = */ 3, /*yaw_B = */ 0,
              /*A_x_B = */ 1, /*A_y_B = */ 2, /*A_yaw_B = */ 0);
  Eigen::Matrix3d information = Eigen::Matrix3d::Identity();

  ErrorTermTestHelper(&params.G_x_A, &params.G_y_A, &params.yaw_A,
                      &params.G_x_B, &params.G_y_B, &params.yaw_B, params.A_x_B,
                      params.A_y_B, params.A_yaw_B, information);
}

// Tests the analytical and symbolic differentiation are equal for different
// poses with non-zero residuals.
TEST(PoseGraph2dErrorTerm, NonZeroResidualTest) {
  Data params(/*G_x_A = */ 1, /*G_y_A = */ 1, /*yaw_A = */ 0,
              /*G_x_B = */ 1, /*G_y_B = */ 1, /*yaw_B = */ 1,
              /*A_x_B = */ 0, /*A_y_B = */ 1, /*A_yaw_B = */ 2);
  Eigen::Matrix3d information = Eigen::Matrix3d::Identity();

  ErrorTermTestHelper(&params.G_x_A, &params.G_y_A, &params.yaw_A,
                      &params.G_x_B, &params.G_y_B, &params.yaw_B, params.A_x_B,
                      params.A_y_B, params.A_yaw_B, information);
}

// Tests the analytical and symbolic differentiation are equal for different
// poses with zero residuals.
TEST(PoseGraph2dErrorTerm, NonZeroResidualTest2) {
  Data params(/*G_x_A = */ 1, /*G_y_A = */ -1, /*yaw_A = */ 0,
              /*G_x_B = */ 1, /*G_y_B = */ 0, /*yaw_B = */ 1,
              /*A_x_B = */ 0, /*A_y_B = */ -2, /*A_yaw_B = */ -2);
  Eigen::Matrix3d information = Eigen::Matrix3d::Identity();

  ErrorTermTestHelper(&params.G_x_A, &params.G_y_A, &params.yaw_A,
                      &params.G_x_B, &params.G_y_B, &params.yaw_B, params.A_x_B,
                      params.A_y_B, params.A_yaw_B, information);
}

// Tests the analytical and symbolic differentiation are equal for different
// poses with zero residuals and non-identity information matrix.
TEST(PoseGraph2dErrorTerm, NonZeroResidualTestWithNonIdentityInformation) {
  Data params(/*G_x_A = */ 1, /*G_y_A = */ 1, /*yaw_A = */ 0,
              /*G_x_B = */ 2, /*G_y_B = */ 1, /*yaw_B = */ 1,
              /*A_x_B = */ 0, /*A_y_B = */ 1, /*A_yaw_B = */ 2);
  std::srand(1234);
  Eigen::Matrix3d random = Eigen::Matrix3d::Random();
  Eigen::Matrix3d information = random.transpose() * random;

  ErrorTermTestHelper(&params.G_x_A, &params.G_y_A, &params.yaw_A,
                      &params.G_x_B, &params.G_y_B, &params.yaw_B, params.A_x_B,
                      params.A_y_B, params.A_yaw_B, information);
}

}  // namespace
