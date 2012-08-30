#include <iostream>
#include <fstream>

#include "ceres/ceres.h"
#include "ceres/split.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "Eigen/Core"

using Eigen::Dynamic;
using Eigen::RowMajor;
typedef Eigen::Matrix<double, Dynamic, 1> Vector;
typedef Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> Matrix;

bool GetLineInPieces(std::ifstream& ifs, std::vector<std::string>* pieces) {
  pieces->clear();
  char buf[256];
  ifs.getline(buf, 256);
  ceres::SplitStringUsing(std::string(buf), " ", pieces);
  return true;
}

void SkipLines(std::ifstream& ifs, int num_lines) {
  char buf[256];
  for (int i = 0; i < num_lines; ++i) {
    ifs.getline(buf, 256);
  }
}
class NISTProblem {
 public:
  explicit NISTProblem(const std::string& filename) {
    std::ifstream ifs(filename.c_str(), std::ifstream::in);

    std::vector<std::string> pieces;
    SkipLines(ifs, 24);
    GetLineInPieces(ifs, &pieces);
    const int kNumResponses = std::atoi(pieces[1].c_str());

    GetLineInPieces(ifs, &pieces);
    const int kNumPredictors = std::atoi(pieces[0].c_str());

    GetLineInPieces(ifs, &pieces);
    const int kNumObservations = std::atoi(pieces[0].c_str());

    SkipLines(ifs, 4);
    GetLineInPieces(ifs, &pieces);
    const int kNumParameters = std::atoi(pieces[0].c_str());
    SkipLines(ifs, 8);

    GetLineInPieces(ifs, &pieces);
    const int kNumTries = pieces.size() - 4;

    predictor_.resize(kNumObservations, kNumPredictors);
    response_.resize(kNumObservations, kNumResponses);
    initial_parameters_.resize(kNumTries, kNumParameters);
    final_parameters_.resize(1, kNumParameters);

    int parameter_id = 0;
    for (int i = 0; i < kNumTries; ++i) {
      initial_parameters_(i, parameter_id) = std::atof(pieces[i + 2].c_str());
    }
    final_parameters_(0, parameter_id) = std::atof(pieces[2 + kNumTries].c_str());


    for (int parameter_id = 1; parameter_id < kNumParameters; ++parameter_id) {
      GetLineInPieces(ifs, &pieces);
      for (int i = 0; i < kNumTries; ++i) {
        initial_parameters_(i, parameter_id) = std::atof(pieces[i + 2].c_str());
      }
      final_parameters_(0, parameter_id) = std::atof(pieces[2 + kNumTries].c_str());
    }

    SkipLines(ifs, 20 - kNumParameters);
    for (int i = 0; i < kNumObservations; ++i) {
      GetLineInPieces(ifs, &pieces);
      for (int j = 0; j < kNumResponses; ++j) {
        response_(i, j) =  std::atof(pieces[j].c_str());
      }
      for (int j = 0; j < kNumPredictors; ++j) {
        predictor_(i, j) =  std::atof(pieces[j + kNumResponses].c_str());
      }
    }
  }

  Matrix initial_parameters(int start) const { return initial_parameters_.row(start); }
  Matrix final_parameters() const  { return final_parameters_; }
  Matrix predictor()        const { return predictor_;         }
  Matrix response()         const { return response_;          }
  int predictor_size()      const { return predictor_.cols();  }
  int num_observations()    const { return predictor_.rows();  }
  int response_size()       const { return response_.cols();   }
  int num_parameters()      const { return initial_parameters_.cols(); }
  int num_starts()          const { return initial_parameters_.rows(); }

 private:
  Matrix predictor_;
  Matrix response_;
  Matrix initial_parameters_;
  Matrix final_parameters_;
};

struct Bennet5 {
 public:
  Bennet5(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // y = b1 * (b2+x)**(-1/b3)  +  e
    residual[0] = T(y_) - b[0] * pow(b[1] + T(x_), T(-1.0) / b[2]);
    return true;
  }

 private:
  double x_;
  double y_;
};

struct BoxBOD {
 public:
  BoxBOD(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    //  y = b1*(1-exp[-b2*x])  +  e
    residual[0] = T(y_) - b[0] * (T(1.0) - exp(-b[1] * T(x_)));
    return true;
  }

 private:
  double x_;
  double y_;
};

struct Chwirut {
 public:
  Chwirut(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // y = exp[-b1*x]/(b2+b3*x)  +  e
    residual[0] = T(y_) - exp(-b[0] * T(x_)) / (b[1] + b[2] * T(x_));
    return true;
  }

 private:
  double x_;
  double y_;
};

struct DanWood {
 public:
  DanWood(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // y  = b1*x**b2  +  e
    residual[0] = T(y_) - b[0] * pow(T(x_), b[1]);
    return true;
  }

 private:
  double x_;
  double y_;
};

struct Gauss {
 public:
  Gauss(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    //  y = b1*exp( -b2*x ) + b3*exp( -(x-b4)**2 / b5**2 )
    //    + b6*exp( -(x-b7)**2 / b8**2 ) + e
    residual[0] = T(y_) - b[0] * exp(-b[1] * T(x_))
        - b[2] * exp(-(T(x_) - b[3]) * (T(x_) - b[3]) / (b[4] * b[4]))
        - b[5] * exp(-(T(x_) - b[6]) * (T(x_) - b[6]) / (b[7] * b[7]));
    return true;
  }

 private:
  double x_;
  double y_;
};

struct Lanczos {
 public:
  Lanczos(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)  +  e
    residual[0] = T(y_)
        - b[0] * exp(-b[1] * T(x_))
        - b[2] * exp(-b[3] * T(x_))
        - b[4] * exp(-b[5] * T(x_));
    return true;
  }

 private:
  double x_;
  double y_;
};

struct Hahn1 {
 public:
  Hahn1(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    //  y = (b1+b2*x+b3*x**2+b4*x**3) /
    //      (1+b5*x+b6*x**2+b7*x**3)  +  e

    residual[0] = T(y_)
        - (b[0] + b[1] * T(x_) + b[2] * T(x_ * x_) + b[3] * T(x_ * x_ * x_)) /
        (T(1.0) + b[4] * T(x_) + b[5] * T(x_ * x_) + b[6] * T(x_ * x_ * x_));
    return true;
  }

 private:
  double x_;
  double y_;
};

struct Kirby2 {
 public:
  Kirby2(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // y = (b1 + b2*x + b3*x**2) /
    //     (1 + b4*x + b5*x**2)  +  e


    residual[0] = T(y_)
        - (b[0] + b[1] * T(x_) + b[2] * T(x_ * x_)) /
        (T(1.0) + b[3] * T(x_) + b[4] * T(x_ * x_));
    return true;
  }

 private:
  double x_;
  double y_;
};

struct MGH09 {
 public:
  MGH09(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // y = b1*(x**2+x*b2) / (x**2+x*b3+b4)  +  e

    residual[0] = T(y_)
        - b[0] * (T(x_ * x_) + T(x_) * b[1]) / (T(x_ * x_) + T(x_) * b[2] + b[3]);
    return true;
  }

 private:
  double x_;
  double y_;
};

struct MGH10 {
 public:
  MGH10(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // y = b1 * exp[b2/(x+b3)]  +  e

    residual[0] = T(y_) - b[0] * exp(b[1] / (T(x_) + b[2]));
    return true;
  }

 private:
  double x_;
  double y_;
};


struct MGH17 {
 public:
  MGH17(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // y = b1 + b2*exp[-x*b4] + b3*exp[-x*b5]

    residual[0] = T(y_)
        - b[0]
        - b[1] * exp(-T(x_) * b[3])
        - b[2] * exp(-T(x_) * b[4]);
    return true;
  }

 private:
  double x_;
  double y_;
};

struct Misra1a {
 public:
  Misra1a(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // y = b1*(1-exp[-b2*x])  +  e
    residual[0] = T(y_) - b[0] * (T(1.0) - exp(-b[1] * T(x_)));
    return true;
  }

 private:
  double x_;
  double y_;
};

struct Misra1b {
 public:
  Misra1b(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // y = b1 * (1-(1+b2*x/2)**(-2))  +  e

    residual[0] = T(y_) - b[0] * (T(1.0) - T(1.0)/ ((T(1.0) + b[1] * T(x_ / 2.0)) * (T(1.0) + b[1] * T(x_ / 2.0))));
    return true;
  }

 private:
  double x_;
  double y_;
};

struct Misra1c {
 public:
  Misra1c(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // y = b1 * (1-(1+2*b2*x)**(-.5))  +  e

    residual[0] = T(y_) - b[0] * (T(1.0) - pow(T(1.0) + T(2.0) * b[1] * T(x_), 0.5));
    return true;
  }

 private:
  double x_;
  double y_;
};

struct Misra1d {
 public:
  Misra1d(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // y = b1*b2*x*((1+b2*x)**(-1))  +  e

    residual[0] = T(y_) - b[0] * b[1] * T(x_) / (T(1.0) + b[1] * T(x_));
    return true;
  }

 private:
  double x_;
  double y_;
};

struct Nelson {
 public:
  Nelson(const double* const x, const double* const y)
      : x1_(x[0]), x2_(x[1]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // log[y] = b1 - b2*x1 * exp[-b3*x2]  +  e

    residual[0] = T(log(y_))
        - b[0] + b[1] * T(x1_) * exp(-b[2] * T(x2_));
    return true;
  }

 private:
  double x1_;
  double x2_;
  double y_;
};

struct Roszman1 {
 public:
  Roszman1(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // pi = 3.141592653589793238462643383279E0
    // y =  b1 - b2*x - arctan[b3/(x-b4)]/pi  +  e
    residual[0] = T(y_)
        - b[0] + b[1] * T(x_) + atan2(b[2], (T(x_) - b[3]))/T(3.141592653589793238462643383279);
    return true;
  }

 private:
  double x_;
  double y_;
};

struct Rat42 {
 public:
 Rat42(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // y = b1 / (1+exp[b2-b3*x])  +  e
    residual[0] = T(y_) - b[0] / (T(1.0) + exp(b[1] - b[2] * T(x_)));
    return true;
  }

 private:
  double x_;
  double y_;
};

struct Rat43 {
 public:
  Rat43(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // y = b1 / ((1+exp[b2-b3*x])**(1/b4))  +  e
    residual[0] = T(y_) - b[0] / pow(T(1.0) + exp(b[1] - b[2] * T(x_)), T(1.0) / b[4]);
    return true;
  }

 private:
  double x_;
  double y_;
};

struct Thurber {
 public:
  Thurber(const double* const x, const double* const y)
      : x_(x[0]), y_(y[0]) {}

  template <typename T>
  bool operator()(const T* const b, T* residual) const {
    // y = (b1 + b2*x + b3*x**2 + b4*x**3) /
    //     (1 + b5*x + b6*x**2 + b7*x**3)  +  e

    residual[0] = T(y_) -
        (b[0] + b[1] * T(x_) + b[2] * T(x_ * x_)  + b[3] * T(x_ * x_ * x_)) /
        (T(1.0) + b[4] * T(x_) + b[5] * T(x_ * x_) + b[6] * T(x_ * x_ * x_));
    return true;
  }

 private:
  double x_;
  double y_;
};


template <typename Model, int num_residuals, int num_parameters>
void RegressionDriver(const std::string& filename,
                      const ceres::Solver::Options& options,
                      ceres::Solver::Summary* summary) {
  NISTProblem nist_problem(filename);
  Matrix predictor = nist_problem.predictor();
  Matrix response = nist_problem.response();
  Matrix final_parameters = nist_problem.final_parameters();
  std::vector<ceres::Solver::Summary> summaries(nist_problem.num_starts());
  for (int start = 0; start < nist_problem.num_starts(); ++start) {
    //std::cout << "try: " << start + 1 << "/" << nist_problem.num_starts() << std::endl;
    Matrix initial_parameters = nist_problem.initial_parameters(start);
    ceres::Problem problem;
    for (int i = 0; i < nist_problem.num_observations(); ++i) {
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<Model, num_residuals, num_parameters>(
              new Model(predictor.data() + nist_problem.predictor_size() * i,
                        response.data() + nist_problem.response_size() * i)),
          NULL,
          initial_parameters.data());
    }
    Solve(options, &problem, &summaries[start]);
  }


  Matrix initial_parameters = nist_problem.final_parameters();
  ceres::Problem problem;
  for (int i = 0; i < nist_problem.num_observations(); ++i) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Model, num_residuals, num_parameters>(
            new Model(predictor.data() + nist_problem.predictor_size() * i,
                      response.data() + nist_problem.response_size() * i)),
        NULL,
        initial_parameters.data());
  }
  Solve(options, &problem, summary);
  double certified_initial_cost = summary->initial_cost;

  std::cout << filename << std::endl;
  for (int i = 0; i < nist_problem.num_starts(); ++i) {
    std::cout << "start " << i + 1 << ": "
              << " relative difference : "
              << (summaries[i].final_cost - summary->initial_cost) / summary->initial_cost
              << " termination: "
              << ceres::SolverTerminationTypeToString(summaries[i].termination_type)
              << std::endl;
  }
}


int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);


  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 1000;
  options.function_tolerance *= 1e-10;
  options.gradient_tolerance *= 1e-10;
  options.parameter_tolerance *= 1e-10;
  options.jacobi_scaling = false;
  ceres::Solver::Summary summary;

  RegressionDriver<Bennet5,  1, 3>("/Users/sameeragarwal/ceres-solver/data/nist/Bennett5.dat", options, &summary);
  RegressionDriver<BoxBOD,   1, 2>("/Users/sameeragarwal/ceres-solver/data/nist/BoxBOD.dat",   options, &summary);
  RegressionDriver<Chwirut,  1, 3>("/Users/sameeragarwal/ceres-solver/data/nist/Chwirut1.dat", options, &summary);
  RegressionDriver<Chwirut,  1, 3>("/Users/sameeragarwal/ceres-solver/data/nist/Chwirut2.dat", options, &summary);
  RegressionDriver<DanWood,  1, 2>("/Users/sameeragarwal/ceres-solver/data/nist/DanWood.dat",  options, &summary);
  RegressionDriver<Gauss,    1, 8>("/Users/sameeragarwal/ceres-solver/data/nist/Gauss1.dat",   options, &summary);
  RegressionDriver<Gauss,    1, 8>("/Users/sameeragarwal/ceres-solver/data/nist/Gauss2.dat",   options, &summary);
  RegressionDriver<Gauss,    1, 8>("/Users/sameeragarwal/ceres-solver/data/nist/Gauss3.dat",   options, &summary);
  RegressionDriver<Lanczos,  1, 6>("/Users/sameeragarwal/ceres-solver/data/nist/Lanczos1.dat", options, &summary);
  RegressionDriver<Lanczos,  1, 6>("/Users/sameeragarwal/ceres-solver/data/nist/Lanczos2.dat", options, &summary);
  RegressionDriver<Lanczos,  1, 6>("/Users/sameeragarwal/ceres-solver/data/nist/Lanczos3.dat", options, &summary);
  RegressionDriver<Hahn1,    1, 7>("/Users/sameeragarwal/ceres-solver/data/nist/Hahn1.dat",    options, &summary);
  RegressionDriver<Kirby2,   1, 5>("/Users/sameeragarwal/ceres-solver/data/nist/Kirby2.dat",   options, &summary);
  RegressionDriver<MGH09,    1, 4>("/Users/sameeragarwal/ceres-solver/data/nist/MGH09.dat",    options, &summary);
  RegressionDriver<MGH10,    1, 3>("/Users/sameeragarwal/ceres-solver/data/nist/MGH10.dat",    options, &summary);
  RegressionDriver<MGH17,    1, 5>("/Users/sameeragarwal/ceres-solver/data/nist/MGH17.dat",    options, &summary);
  RegressionDriver<Misra1a,  1, 2>("/Users/sameeragarwal/ceres-solver/data/nist/Misra1a.dat",  options, &summary);
  RegressionDriver<Misra1b,  1, 2>("/Users/sameeragarwal/ceres-solver/data/nist/Misra1b.dat",  options, &summary);
  RegressionDriver<Misra1c,  1, 2>("/Users/sameeragarwal/ceres-solver/data/nist/Misra1c.dat",  options, &summary);
  RegressionDriver<Misra1d,  1, 2>("/Users/sameeragarwal/ceres-solver/data/nist/Misra1d.dat",  options, &summary);
  RegressionDriver<Nelson,   1, 3>("/Users/sameeragarwal/ceres-solver/data/nist/Nelson.dat",   options, &summary);
  RegressionDriver<Roszman1, 1, 4>("/Users/sameeragarwal/ceres-solver/data/nist/Roszman1.dat", options, &summary);
  RegressionDriver<Rat42,    1, 3>("/Users/sameeragarwal/ceres-solver/data/nist/Rat42.dat",    options, &summary);
  RegressionDriver<Rat43,    1, 4>("/Users/sameeragarwal/ceres-solver/data/nist/Rat43.dat",    options, &summary);
  RegressionDriver<Thurber,  1, 7>("/Users/sameeragarwal/ceres-solver/data/nist/Thurber.dat",  options, &summary);

  return 0;
};
