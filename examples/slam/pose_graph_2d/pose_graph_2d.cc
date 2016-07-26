#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "angle_local_parameterization.h"
#include "ceres/ceres.h"
#include "pose_graph_2d_error_term.h"
#include "read_g2o.h"
#include "types.h"

using ceres::examples::pose_graph_2d::Constraint2d;
using ceres::examples::pose_graph_2d::Pose2d;

// Constructs the nonlinear least squares optimization problem from the pose
// graph constraints and solves it.
void BuildOptimizationProblemAndSolve(
    const std::vector<Constraint2d>& constraints, std::map<int, Pose2d>* poses);

// Output the poses to the file with format: ID x y yaw_radians.
bool OutputPoses(const std::string& filename,
                 const std::map<int, Pose2d>& poses);

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Need to specify the filename to read as the first and only "
              << "argument.\n";
    return -1;
  }

  std::map<int, Pose2d> poses;
  std::vector<Constraint2d> constraints;

  if (!ceres::examples::pose_graph_2d::ReadG2oFile(argv[1], &poses,
                                                   &constraints)) {
    std::cerr << "Error reading the file: " << argv[1] << '\n';
    return -1;
  }

  std::cout << "Number of poses: " << poses.size() << '\n';
  std::cout << "Number of constraints: " << constraints.size() << '\n';

  if (!OutputPoses("poses_original.txt", poses)) {
    return -1;
  }

  BuildOptimizationProblemAndSolve(constraints, &poses);

  if (!OutputPoses("poses_optimized.txt", poses)) {
    return -1;
  }

  return 0;
}

void BuildOptimizationProblemAndSolve(
    const std::vector<Constraint2d>& constraints,
    std::map<int, Pose2d>* poses) {
  CHECK(poses != NULL);
  if (constraints.empty()) {
    std::cout << "No constraints, no problem to optimize.\n";
    return;
  }

  ceres::Problem problem;
  ceres::LossFunction* loss_function = NULL;
  ceres::LocalParameterization* angle_local_parameterization =
      new ceres::examples::pose_graph_2d::AngleLocalParameterization;

  for (std::vector<Constraint2d>::const_iterator constraints_iter =
           constraints.begin();
       constraints_iter != constraints.end(); ++constraints_iter) {
    const Constraint2d& constraint = *constraints_iter;

    std::map<int, Pose2d>::iterator pose_begin_iter =
        poses->find(constraint.id_begin);
    CHECK(pose_begin_iter != poses->end());
    std::map<int, Pose2d>::iterator pose_end_iter =
        poses->find(constraint.id_end);
    CHECK(pose_end_iter != poses->end());

    const Eigen::Matrix3d sqrt_information =
        constraint.information.llt().matrixL();
    // Ceres will take ownership of the pointer.
    ceres::CostFunction* cost_function =
        new ceres::examples::pose_graph_2d::PoseGraph2dErrorTerm(
            constraint.x, constraint.y, constraint.yaw_radians,
            sqrt_information);
    problem.AddResidualBlock(
        cost_function, loss_function, &pose_begin_iter->second.x,
        &pose_begin_iter->second.y, &pose_begin_iter->second.yaw_radians,
        &pose_end_iter->second.x, &pose_end_iter->second.y,
        &pose_end_iter->second.yaw_radians);

    problem.SetParameterization(&pose_begin_iter->second.yaw_radians,
                                angle_local_parameterization);
    problem.SetParameterization(&pose_end_iter->second.yaw_radians,
                                angle_local_parameterization);
  }

  // The pose graph optimization problem has three DOFs that are not fully
  // constrained. This is typically referred to as gauge freedom. You can apply
  // a rigid body transformation to all the nodes and the optimization problem
  // will still have the exact same cost. The Levenberg-Marquardt algorithm has
  // internal damping which mitigate this issue, but it is better to properly
  // constrain the gauge freedom. This can be done by setting one of the poses
  // as constant so the optimizer cannot change it.
  std::map<int, Pose2d>::iterator pose_start_iter =
      poses->begin();
  CHECK(pose_start_iter != poses->end());
  problem.SetParameterBlockConstant(&pose_start_iter->second.x);
  problem.SetParameterBlockConstant(&pose_start_iter->second.y);
  problem.SetParameterBlockConstant(&pose_start_iter->second.yaw_radians);

  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.num_threads = 1;
  options.num_linear_solver_threads = 1;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << '\n';
}

bool OutputPoses(const std::string& filename,
                 const std::map<int, Pose2d>& poses) {
  std::fstream outfile;
  outfile.open(filename.c_str(), std::istream::out);
  if (!outfile) {
    std::cerr << "Error opening the file: " << filename << '\n';
    return false;
  }
  for (std::map<int, Pose2d>::const_iterator poses_iter = poses.begin();
       poses_iter != poses.end(); ++poses_iter) {
    const std::map<int, Pose2d>::value_type& pair = *poses_iter;
    outfile <<  pair.first << " " << pair.second.x << " " << pair.second.y
            << ' ' << pair.second.yaw_radians << '\n';
  }
  return true;
}
