

#include <cmath>
#include <cstdio>
#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "gflags/gflags.h"

#include <TooN/TooN.h>
#include <TooN/se3.h>

#include <vector>

#include "BundlerProblem.h"

DEFINE_int32(num_models, 1, "The number of distinct fix camera models to model.");
DEFINE_bool(manual_ordering, false, "Use manual ordering. (Always used for stereo.)");

// The convention in this module is that (most of the time) a Vector<6> is not log(SE3), but rather [log(SO3) t]

using namespace std;
using namespace TooN;


struct SeparableReprojectionError {
  SeparableReprojectionError(TooN::Vector<2> const& observation)
      : observation(observation) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  const T* const camera_model,
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
    const T& l1 = camera_model[1];
    const T& l2 = camera_model[2];
    T r2 = xp*xp + yp*yp;
    T distortion = T(1.0) + r2  * (l1 + l2  * r2);

    // Compute final projected point position.
    const T& focal = camera_model[0];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observation[0]);
    residuals[1] = predicted_y - T(observation[1]);

    return true;
  }

  TooN::Vector<2> observation;
};



// This represents the problem where several cameras are assumed to have the same camera model
// num_unique_camera_models represents the number of different camera models
// and the camera models are assumed to be in the order so that camera[i] has model[i%num_unique_camera_models]
class FixedCameraModelProblem : public BundlerProblem {
public:
	FixedCameraModelProblem(int num_unique_camera_models)
	: num_unique_camera_models_(num_unique_camera_models)
	, unique_camera_models_(num_unique_camera_models)
	{

	}

	bool LoadFile(const char *filename) {
		if (!BundlerProblem::LoadFile(filename))
			return false;

		std::vector<int> num_models(num_unique_camera_models_);

		// Calculate mean camera models
		for (int i = 0; i < num_unique_camera_models_; i++) {
			unique_camera_models_[i] = Zeros;
		}

		for (int j = 0; j < camera_parameters_.size(); j++) {
			if (norm_sq(camera_parameters_[j].model_) > 0) {
				unique_camera_models_[j%num_unique_camera_models_] += camera_parameters_[j].model_;
				num_models[j%num_unique_camera_models_]++;
			}
		}

		for (int i = 0; i < num_unique_camera_models_; i++) {
			unique_camera_models_[i] /= num_models[i];
		}

		return true;
	}

	virtual int num_camera_models() { return num_unique_camera_models_; }
	virtual Vector<3> & camera_model(int camera_index) { return unique_camera_models_[camera_index%num_unique_camera_models_]; }

	virtual void AddResidualBlock(ceres::Problem & problem, int i) {
		ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<SeparableReprojectionError, 2, 6, 3, 3>
		( new SeparableReprojectionError(observations_[i].observation_) );

		int point_index = observations_[i].point_index_;
		int camera_index = observations_[i].camera_index_;

		problem.AddResidualBlock(cost_function,
								 NULL,//new ceres::HuberLoss(1.0),
								 ptr(camera_parameters_[camera_index].position_),
								 ptr(points_[point_index]),
								 ptr(unique_camera_models_[camera_index % num_unique_camera_models_]));

		if (camera_index == 0)
			problem.SetParameterBlockConstant(ptr(camera_parameters_[camera_index].position_));
	}

	virtual int SetupOrdering(ceres::ParameterBlockOrdering *ordering) {
		for (int i = 0; i < points_.size(); i++) {
			ordering->AddElementToGroup(ptr(points_[i]), 0);
		}
		for (int i = 0; i < camera_parameters_.size(); i++) {
			if (!isnan(camera_parameters_[i].position_)) {
				ordering->AddElementToGroup(ptr(camera_parameters_[i].position_), 1);
			}
		}
		for (int i = 0; i < num_unique_camera_models_; i++) {
			ordering->AddElementToGroup(ptr(unique_camera_models_[i]),1);
		}
		return 3; // Num blocks
	}

	virtual void print_cameras() {
		for (int i = 0; i < num_unique_camera_models_; i++) {
			cout << unique_camera_models_[i] << endl;
		}
	}


	int num_unique_camera_models_;

	vector<Vector<3> > unique_camera_models_;

};


std::string RemoveExtension(std::string const& filename) {
	size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
}


int main(int argc, char** argv) {
	google::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);

	if (argc != 2) {
		std::cerr << "usage: "<<argv[0]<<" <bundle-file>\n";
		return 1;
	}

	std::string base_filename = RemoveExtension(argv[1]);

	cout << "base_filename = " << base_filename << endl;

	BundlerProblem *bal_problem;

	bal_problem = new FixedCameraModelProblem(FLAGS_num_models);

	if (!bal_problem->LoadFile(argv[1])) {
		std::cerr << "ERROR: unable to open file " << argv[1] << "\n";
		return 1;
	}

	bal_problem->print_cameras();

	ceres::Problem problem;

	bal_problem->AddResidualBlocks(problem);

	ceres::Solver::Options options;

	if (FLAGS_manual_ordering) {
		ceres::ParameterBlockOrdering *ordering = new ceres::ParameterBlockOrdering;

		bal_problem->SetupOrdering(ordering);

		options.linear_solver_ordering = ordering;
	}

	options.linear_solver_type = ceres::DENSE_SCHUR;//ceres::ITERATIVE_SCHUR;
	options.gradient_tolerance = 1e-8;
	options.function_tolerance = 1e-5;

	// Single threaded
	options.num_threads=1;
	options.num_linear_solver_threads=1;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations=500;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	cout << "Sqrt(final cost^2) per observation = " << sqrt(summary.final_cost / summary.num_residual_blocks) << endl;

	cout << "Resulting camera model(s) =\n";
	bal_problem->print_cameras();

	bal_problem->print_PLY_file((base_filename+".ply").c_str());
	bal_problem->print_bundle_out_file((base_filename+"-refined.out").c_str());

	return 0;
}
