// Author: joydeepb@ri.cmu.edu (Joydeep Biswas)
//
// This example demonstrates how to use the DynamicAutoDiffCostFunction
// variant of CostFunction. The DynamicAutoDiffCostFunction is meant to
// be used in cases where the number of parameter blocks or the sizes are not
// known in compile time.
//
// This example simulates a robot traversing down a 1-dimension hallway with
// noise odometry readings and noisy range readings of the end of the hallway.
// By fusing the noisy odometry and sensor readings this example demonstrates
// how to compute the maximum likelihood estimate (MLE) of the robot's pose at
// each timestep.
//
// The robot starts at the origin, and it is travels to the end of a corridor of
// fixed length specified by the "--pose_separation" flag. It executes a series
// of motion commands to move forward a fixed length, specified by the
// "--pose_separation" flag, at which pose it receives relative odometry
// measurements as well as a range reading of the distance to the end of the
// hallway. The odometry readings are drawn with Gaussian noise and standard
// deviation specified by the "--odometry_stddev" flag, and the range readings
// similarly with standard deviation specified by the "--range-stddev" flag.
//
// There are two types of residuals in this problem:
// 1) The OdometryConstraint residual, that accounts for the odometry readings
//    between successive pose estimatess of the robot.
// 2) The RangeConstraint residual, that accounts for the errors in the observed
//    range readings from each pose.
//
// The OdometryConstraint residual is modeled as an AutoDiffCostFunction with
// a fixed parameter block size of 1, which is the relative odometry being
// solved for, between a pair of successive poses of the robot. Differences
// between observed and computed relative odometry values are penalized weighted
// by the known standard deviation of the odometry readings.
//
// The RangeConstraint residual is modeled as a DynamicAutoDiffCostFunction
// which sums up the relative odometry estimates to compute the estimated
// global pose of the robot, and then computes the expected range reading.
// Differences between the observed and expected range readings are then
// penalized weighted by the standard deviation of readings of the sensor.
// Since the number of poses of the robot is not known at compile time, this
// cost function is implemented as a DynamicAutoDiffCostFunction.
//
// The outputs of the example are the initial values of the odometry and range
// readings, and the range and odometry errors for every pose of the robot.
// After computing the MLE, the computed poses and corrected odometry values
// are printed out, along with the corresponding range and odometry errors. Note
// that as an MLE of a noisy system the errors will not be reduced to zero, but
// the odometry estimates will be updated to maximize the joint likelihood of
// all odometry and range readings of the robot.
//
// TODO(joydeepb): Explain the mathematical cost-function based product of
// likelihoods model.

#include <cstdio>
#include <math.h>
#include <vector>

#include "ceres/ceres.h"
#include "ceres/dynamic_autodiff_cost_function.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::DynamicAutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
using std::min;
using std::vector;

DEFINE_double(corridor_length, 30.0, "Length of the corridor that the robot is "
              "travelling down.");

DEFINE_double(pose_separation, 0.5, "The distance that the robot traverses "
              "between successive odometry updates.");

DEFINE_double(odometry_stddev, 0.1, "The standard deviation of "
              "odometry error of the robot.");

DEFINE_double(range_stddev, 0.01, "The standard deviation of range readings of "
              "the robot.");

// The stride length of the dynamic_autodiff_cost_function evaluator.
static const size_t kStride = 10;

struct OdometryConstraint {
  OdometryConstraint(
      double _observed_odometry, double _odometry_stddev) :
      observed_odometry(_observed_odometry),
      odometry_stddev(_odometry_stddev) {}

  template <typename T>
  bool operator()(const T* const odometry_ptr, T* residual_ptr) const {
    const T& odometry = *odometry_ptr;
    T& residual = *residual_ptr;
    residual = (odometry - T(observed_odometry)) / T(odometry_stddev);
    return true;
  }

  const double observed_odometry;
  const double odometry_stddev;
};

struct RangeConstraint {
  RangeConstraint(
      size_t _pose_index, double _range_reading, double _range_stddev,
      double _corridor_length) :
      pose_index(_pose_index), range_reading(_range_reading),
      range_stddev(_range_stddev), corridor_length(_corridor_length) {}

  template <typename T>
  bool operator()(T const* const* relative_poses, T* residuals) const {
    T global_pose(0);
    for (size_t i = 0; i <= pose_index; ++i) {
      global_pose += relative_poses[i][0];
    }
    residuals[0] = (global_pose + T(range_reading) - T(corridor_length)) /
        T(range_stddev);
    return true;
  }

  const size_t pose_index;
  const double range_reading;
  const double range_stddev;
  const double corridor_length;
};

// Return a (pseudo)random number drawn from the Normal Dsitribution with
// specified standard deviation and mean.
template <typename num>
num randn(num stddev = 1.0, num mean = 0.0) {
  // Uses Box-Muller transform to turn a pair of uniform random numbers into a
  // pair of gaussian random numbers.
  num u1 = static_cast<num>(rand()) / static_cast<num>(RAND_MAX);
  num u2 = static_cast<num>(rand()) / static_cast<num>(RAND_MAX);
  num z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
  num x1 = z1 * stddev + mean;
  return x1;
}

void SimulateRobot(
    vector<double>* odometry_values_ptr, vector<double>* range_readings_ptr) {
  vector<double>& odometry_values = *odometry_values_ptr;
  vector<double>& range_readings = *range_readings_ptr;
  const size_t num_steps = static_cast<size_t>(
      ceil(FLAGS_corridor_length / FLAGS_pose_separation));

  // The robot starts out at the origin.
  double robot_location = 0.0;
  for (size_t i = 0; i < num_steps; ++i) {
    const double actual_odometry_value = min(
        FLAGS_pose_separation, FLAGS_corridor_length - robot_location);
    robot_location += actual_odometry_value;
    const double actual_range = FLAGS_corridor_length - robot_location;
    const double observed_odometry = randn<double>(
        FLAGS_odometry_stddev, actual_odometry_value);
    const double observed_range = randn<double>(
        FLAGS_range_stddev, actual_range);
    odometry_values.push_back(observed_odometry);
    range_readings.push_back(observed_range);
  }
}

void PrintState(const vector<double>& odometry_readings,
                const vector<double>& range_readings) {
  CHECK_EQ(odometry_readings.size(), range_readings.size());
  double robot_location = 0.0;
  printf("pose: location     odom    range  r.error  o.error\n");
  for (size_t i = 0; i < odometry_readings.size(); ++i) {
    robot_location += odometry_readings[i];
    const double range_error =
        robot_location + range_readings[i] - FLAGS_corridor_length;
    const double odometry_error =
        FLAGS_pose_separation - odometry_readings[i];
    printf("%4d: %8.3f %8.3f %8.3f %8.3f %8.3f\n",
           static_cast<int>(i), robot_location, odometry_readings[i],
           range_readings[i], range_error, odometry_error);
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Make sure that the arguments parsed are all positive.
  CHECK_GT(FLAGS_corridor_length, 0.0);
  CHECK_GT(FLAGS_pose_separation, 0.0);
  CHECK_GT(FLAGS_odometry_stddev, 0.0);
  CHECK_GT(FLAGS_range_stddev, 0.0);

  vector<double> odometry_values;
  vector<double> range_readings;
  SimulateRobot(&odometry_values, &range_readings);

  printf("Initial values:\n");
  PrintState(odometry_values, range_readings);
  ceres::Problem problem;

  for (size_t i = 0; i < odometry_values.size(); ++i) {
    RangeConstraint* range_constraint = new RangeConstraint(
        i, range_readings[i], FLAGS_range_stddev, FLAGS_corridor_length);
    DynamicAutoDiffCostFunction<RangeConstraint, kStride>*
        range_cost_function = new DynamicAutoDiffCostFunction<
            RangeConstraint, kStride>(range_constraint);
    // Add all the parameter blocks that affect this constraint.
    vector<double*> parameter_blocks;
    for (size_t j = 0; j <= i; ++j) {
      range_cost_function->AddParameterBlock(1);
      parameter_blocks.push_back(&(odometry_values[j]));
    }
    range_cost_function->SetNumResiduals(1);
    problem.AddResidualBlock(range_cost_function, NULL, parameter_blocks);

    OdometryConstraint* odometry_constraint = new OdometryConstraint(
        odometry_values[i], FLAGS_odometry_stddev);
    AutoDiffCostFunction<OdometryConstraint, 1, 1>* odometry_cost_function =
        new AutoDiffCostFunction<OdometryConstraint, 1, 1>(odometry_constraint);
    problem.AddResidualBlock(
        odometry_cost_function, NULL, &(odometry_values[i]));
  }

  ceres::Solver::Options solver_options;
  solver_options.minimizer_progress_to_stdout = true;
  solver_options.max_num_iterations = 2000;
  solver_options.function_tolerance = 0.000001;

  Solver::Summary summary;
  printf("Solving...\n");
  Solve(solver_options, &problem, &summary);
  printf("Done.\n");
  std::cout << summary.FullReport() << "\n";
  printf("Final values:\n");
  PrintState(odometry_values, range_readings);
  return 0;
}
