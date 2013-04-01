//
//  BundlerProblem.h
//
//
//  Created by Oskar Linde on 2013-03-24.
//
//

#ifndef _BundlerProblem_h
#define _BundlerProblem_h

template<typename T>
void FscanfOrDie(FILE *fptr, const char *format, T *value) {
	int num_scanned = fscanf(fptr, format, value);
	if (num_scanned != 1) {
		LOG(FATAL) << "Invalid UW data file.";
	}
}


template<int N>
TooN::Vector<N> FscanfVectorOrDie(FILE *fptr) {
	TooN::Vector<N> ret;
	for (int i = 0; i < N; i++) {
		FscanfOrDie(fptr, "%lf", &ret[i]);
	}
	return ret;
}


template<typename T, int N>
T* ptr(TooN::Vector<N,T> & v) {
	return &v[0];
}


// Reads and parses the format of Snavely's Bundler bundle.out-files

class BundlerProblem {
public:

	virtual bool LoadFile(const char* filename) {
		FILE* fptr = fopen(filename, "r");
		if (fptr == NULL) {
			return false;
		};

		double bundle_file_version;
		FscanfOrDie(fptr, "# Bundle file v%lf", &bundle_file_version);

		printf("Bundle file version %.1lf\n",bundle_file_version);

		int num_cameras;
		int num_points;

		FscanfOrDie(fptr, "%d", &num_cameras);
		FscanfOrDie(fptr, "%d", &num_points);

		for (int i = 0; i < num_cameras; i++) {
			TooN::Vector<3> camera_model = FscanfVectorOrDie<3>(fptr);

			TooN::Matrix<3> R;
			R[0] = FscanfVectorOrDie<3>(fptr);
			R[1] = FscanfVectorOrDie<3>(fptr);
			R[2] = FscanfVectorOrDie<3>(fptr);
			TooN::SO3<> so3(R);
			TooN::Vector<3> t = FscanfVectorOrDie<3>(fptr);
			TooN::Vector<6> v6;
			v6.slice<0,3>() = so3.ln();
			v6.slice<3,3>() = t;
			CameraParams params;
			params.position_ = v6;
			params.model_ = camera_model;
			camera_parameters_.push_back(params);

			if (std::isnan(norm_fro(so3.get_matrix())) && norm_sq(t) > 0) {
				std::cout << "Warning, camera " << i << " is left-handed\n";
			}
		}

                printf("size: %d %d\n", num_points, num_cameras);
		colors_.reserve(num_points);
		points_.reserve(num_points);

		for (int i = 0; i < num_points; i++) {
			TooN::Vector<3> pos = FscanfVectorOrDie<3>(fptr);
			points_.push_back(pos);

			TooN::Vector<3> color = FscanfVectorOrDie<3>(fptr);
			colors_.push_back(color);

			int n_observations;
			FscanfOrDie(fptr, "%d", &n_observations);

			for (int j = 0; j < n_observations; j++) {
				int camera_id;
				int feature_id;
				FscanfOrDie(fptr,"%d", &camera_id);
				FscanfOrDie(fptr,"%d", &feature_id);

				TooN::Vector<2> obs = FscanfVectorOrDie<2>(fptr);

				Observation o;
				o.point_index_ = i;
				o.camera_index_ = camera_id;
				o.observation_ = obs;
				o.feature_id_ = feature_id;
				observations_.push_back(o);
			}
		}

		return true;
	}

	void print_PLY_file(const char *filename) {
		FILE *f = fopen(filename,"w");

		fprintf(f, "ply\n");
		fprintf(f, "format ascii 1.0\n");
		fprintf(f, "comment made by 13th Lab\n");
		fprintf(f, "element vertex %d\n",(int)points_.size());
		fprintf(f, "property double x\n");
		fprintf(f, "property double y\n");
		fprintf(f, "property double z\n");
		fprintf(f, "property uchar red\n");
		fprintf(f, "property uchar green\n");
		fprintf(f, "property uchar blue\n");
		fprintf(f, "end_header\n");

		for (int i = 0; i < points_.size(); i++) {
			fprintf(f, "%lf %lf %lf %d %d %d\n",points_[i][0],points_[i][1],points_[i][2], colors_[i][0], colors_[i][1], colors_[i][2]);
		}

		fclose(f);

	}

	virtual void print_cameras() {
		for (int i = 0; i < camera_parameters_.size(); i++) {
			std::cout << camera_parameters_[i].model_ << std::endl;
		}
	}

	static void print_bundle_vector(FILE*f, TooN::Vector<3> const& v) {
		fprintf(f, "%.16g %.16g %.16g\n", v[0], v[1], v[2]);
	}

	void print_bundle_out_file(const char *filename) {
		FILE *f = fopen(filename, "w");

		fprintf(f, "# Bundle file v0.3\n");
		fprintf(f, "%d %d\n",(int)camera_parameters_.size(), (int)points_.size());

		for (int i = 0; i < camera_parameters_.size(); i++) {
			TooN::Vector<6> position = camera_position(i);
			TooN::SE3<> Rt(TooN::SO3<>(position.slice<0,3>()),position.slice<3,3>());

			if (std::isnan(norm_fro(Rt.get_rotation().get_matrix()))) {
				for (int i = 0; i < 5; i++) {
					fprintf(f, "0 0 0\n");
				}
			} else {
				print_bundle_vector(f, camera_model(i));
				print_bundle_vector(f, Rt.get_rotation().get_matrix()[0]);
				print_bundle_vector(f, Rt.get_rotation().get_matrix()[1]);
				print_bundle_vector(f, Rt.get_rotation().get_matrix()[2]);
				print_bundle_vector(f, Rt.get_translation());
			}
		}

		std::vector<Observation> sorted_observations;
		sorted_observations = observations_;
		std::sort(sorted_observations.begin(), sorted_observations.end(), sort_observation_by_point);

		std::vector<Observation>::const_iterator obs_i = sorted_observations.begin();

		for (int i = 0; i < points_.size(); i++) {
			fprintf(f, "%lf %lf %lf\n",points_[i][0],points_[i][1],points_[i][2]);
			fprintf(f, "%d %d %d\n",colors_[i][0], colors_[i][1], colors_[i][2]);

			std::vector<Observation> local;
			while(obs_i != sorted_observations.end() && obs_i->point_index_ == i) {
				local.push_back(*obs_i);
				obs_i++;
			}

			fprintf(f,"%d",(int)local.size());
			for (int j = 0; j < local.size(); j++) {
				fprintf(f, " %d %d %lf %lf",local[j].camera_index_,local[j].feature_id_,local[j].observation_[0],local[j].observation_[1]);
			}
			fprintf(f,"\n");
		}

		fclose(f);
	}

	virtual TooN::Vector<6> camera_position(int camera_index) {
		return camera_parameters_[camera_index].position_;
	}

	virtual int num_camera_models() {
		return camera_parameters_.size();
	}

	virtual TooN::Vector<3> & camera_model(int camera_index) {
		return camera_parameters_[camera_index].model_;
	}

	virtual void AddResidualBlock(ceres::Problem & problem, int i) = 0;

	virtual void AddResidualBlocks(ceres::Problem & problem) {
		for (int i = 0; i < observations_.size(); i++) {
			AddResidualBlock(problem, i);
		}
	}
	virtual int SetupOrdering(ceres::ParameterBlockOrdering *ordering) = 0;

	struct CameraParams {
		TooN::Vector<6> position_;
		TooN::Vector<3> model_;
	};

	struct Observation {
		int point_index_;
		int camera_index_;
		int feature_id_;
		TooN::Vector<2> observation_;
	};

	static bool sort_observation_by_point(BundlerProblem::Observation const& a,
										  BundlerProblem::Observation const& b) {
		return a.point_index_ < b.point_index_;
	}

	std::vector<Observation> observations_;

	std::vector<TooN::Vector<3> > points_;
	std::vector<TooN::Vector<3,unsigned char> > colors_;

	std::vector<CameraParams> camera_parameters_;

};


#endif
