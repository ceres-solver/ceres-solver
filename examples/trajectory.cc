#include <fstream>
#include <unordered_map>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/constrained_problem.h>
#include <glog/logging.h>
#include <gflags/gflags.h>

using namespace ceres;
using namespace ceres::experimental;

DEFINE_string(input, "", "File containing problem data.");
DEFINE_string(compare, "", "File containing optimal solution to compare against.");

/**
 * forward-decl for functions that computes spline function and derivatives
 */
void compute_spline_derivatives(const double x0, const double xd0, const double xdd0,
                                const double x1, const double xd1, const double xdd1,
                                const double  t, const double   T,
                                Eigen::Matrix<double, 4, 1>& sample_derivs,
                                Eigen::Matrix<double, 4, 6>& sample_dx,
                                Eigen::Matrix<double, 4, 1>& sample_dT);
void compute_spline_derivatives(const double x0, const double xd0, const double xdd0,
                                const double x1, const double xd1, const double xdd1,
                                const double  t, const double   T,
                                Eigen::Matrix<double, 4, 1>& sample_derivs);


namespace
{
const double REGULARIZE_DT            = 1.0;
const double REGULARIZE_VEL           = 1e-8;
const double REGULARIZE_ACC           = 1e-8;
const int NUM_SAMPLES_PER_SEGMENT     = 5;
const int NUM_CONSTRAINTS_PER_SEGMENT = ((NUM_SAMPLES_PER_SEGMENT // N samples within the segment
                                        * 4)                      // pos, vel, acc, jerk
                                        + 2)                      // jerk at 0 and T, since it's not subject to bounds
                                             * 2;                 // lower and upper bound
}

typedef Eigen::Matrix<double, NUM_CONSTRAINTS_PER_SEGMENT, 1> VecCon;  // vector of size 'num_constraints'

struct ProblemData
{
  int num_dofs;
  int num_vias;
  int num_segments;

  double delta_t_lower_bound;
  double delta_t_upper_bound;

  std::vector<Eigen::MatrixXd> lower_limits; /**< [dof] (index, via) variable lower limits.
                                                   index refers to 0=pos, 1=vel, 2=acc, 3=jerk */
  std::vector<Eigen::MatrixXd> upper_limits; /**< [dof] (index, via) variable upper limits.
                                                   index refers to 0=pos, 1=vel, 2=acc, 3=jerk */
  std::vector<Eigen::MatrixXd> variables;    /**< [dof] (index, via) variable values.
                                                   index refers to 0=pos, 1=vel, 2=acc */
  Eigen::VectorXd dt_variables;              /**< time duration of each segment */

  std::vector<std::vector<Eigen::VectorXd> > slack_variables;      /**< [dof][segment] one slack variable per constraint */
  std::vector<std::vector<ceres::ConstraintBlockId> > constraints; /**< [dof][segment] pointers to constraint functions */

  void read_from_file(const std::string& filename)
  {
    std::ifstream f(filename.c_str());

    f >> num_dofs >> num_vias;
    num_segments = num_vias - 1;

    f >> delta_t_lower_bound >> delta_t_upper_bound;

    dt_variables = Eigen::VectorXd::Zero(num_segments);
    for (int i=0; i<num_segments; ++i)
    {
      f >> dt_variables(i);
    }

    variables.resize(num_dofs, Eigen::MatrixXd::Zero(3, num_vias));
    lower_limits.resize(num_dofs, Eigen::MatrixXd::Zero(4, num_vias));
    upper_limits.resize(num_dofs, Eigen::MatrixXd::Zero(4, num_vias));

    for (int i=0; i<num_vias; ++i)
    {
      for (int j=0; j<num_dofs; ++j)
      {
        for (int r=0; r<4; ++r)
        {
          if (r < 3) // jerk variables don't exist
          {
            f >> variables[j](r, i);
          }
          f >> lower_limits[j](r, i);
          f >> upper_limits[j](r, i);
        }
      }
    }
    f.close();
  }

};

/**
 * Quadratic cost on a scalar variable
 */
class RegularizationCost: public ceres::SizedCostFunction<1, 1>
{
public:
  RegularizationCost(double reg):
    k_(std::sqrt(reg))
  {
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const
  {
    residuals[0] = k_ * parameters[0][0];
    if (jacobians && jacobians[0])
      jacobians[0][0] = k_;
    return true;
  }

private:
  double k_;
};

class SegmentConstraints: public ceres::SizedCostFunction<NUM_CONSTRAINTS_PER_SEGMENT,
                                                          1, 1, 1, // pos, vel, acc of start
                                                          1, 1, 1, // pos, vel, acc of end
                                                          1,       // dt
                                                          NUM_CONSTRAINTS_PER_SEGMENT> // slack_lower_bound, slack_upper_bound
{
public:
  SegmentConstraints(int dof, int via, const ProblemData& data):
    dof_(dof),
    via_(via),
    data_(data)
  {
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const
  {
    Eigen::Matrix<double, 4, 1> sample;
    Eigen::Matrix<double, 4, 1> sample_dT;
    Eigen::Matrix<double, 4, 6> sample_dx;

    // zero out jacobians
    if (jacobians)
    {
      for (int i=0; i<7; ++i)
      {
        if (!jacobians[i])
          continue;
        for (int j=0; j<NUM_CONSTRAINTS_PER_SEGMENT; ++j)
          jacobians[i][j] = 0.0;
      }
      if (jacobians[7]) // slacks
      {
        for (int r=0; r<NUM_CONSTRAINTS_PER_SEGMENT * NUM_CONSTRAINTS_PER_SEGMENT; ++r)
        {
          jacobians[7][r] = 0.0;
        }
      }
    }

    // guard against getting negative inputs for dt
    double dt = parameters[6][0];
    if (dt < data_.delta_t_lower_bound)
      return false;

    // compute constraints
    int index = 0;
    for (int i=0; i<=NUM_SAMPLES_PER_SEGMENT+1; ++i)
    {
      double t = dt * (double(i) / double(NUM_SAMPLES_PER_SEGMENT+1));
      if (jacobians)
      {
        compute_spline_derivatives(parameters[0][0], parameters[1][0], parameters[2][0],
                                   parameters[3][0], parameters[4][0], parameters[5][0],
                                   t, dt, sample, sample_dx, sample_dT);
      }
      else
      {
        compute_spline_derivatives(parameters[0][0], parameters[1][0], parameters[2][0],
                                   parameters[3][0], parameters[4][0], parameters[5][0],
                                   t, dt, sample);
      }

      for (int r=0; r<4; ++r)
      {
        if ((i>0 && i<=NUM_SAMPLES_PER_SEGMENT) // pos vel and acc constraints are only needed "inside" the segment.
            || r==3)                            // jerk on the other hand needs to be checked at the borders as well.
        {
          fill_in_constraint(sample(r),
                             data_.lower_limits[dof_](r, via_),
                             data_.upper_limits[dof_](r, via_),
                             parameters[7][index], parameters[7][index+1],
                             sample_dx.row(r), sample_dT(r),
                             index, residuals, jacobians);
          index += 2;
        }
      }
    }
    assert(index == NUM_CONSTRAINTS_PER_SEGMENT);
    return true;
  }

  /**
   * Fills in lower and upper bound constraints (at positions index and index+1) on the given non-linear function
   */
  void fill_in_constraint(const double gx, const double lower_bound, const double upper_bound,
                          const double slack_lower, const double slack_upper,
                          const Eigen::Matrix<double, 1, 6>& jacobian_dx_row, const double jacobian_dt,
                          const int index, double* residuals, double** jacobians) const
  {
    // lower bound constraint is:
    // g(x) - lb - s_l = 0.0
    //             s_l > 0.0

    // upper bound constraint is:
    // ub - g(x) - s_u = 0.0
    //             s_u > 0.0

    const double jac_mult_lower = 1.0;
    const double jac_mult_upper = -1.0;

    double upper_bound_constraint = upper_bound - gx - slack_upper;
    double lower_bound_constraint = gx - lower_bound - slack_lower;

    residuals[index  ] = lower_bound_constraint;
    residuals[index+1] = upper_bound_constraint;

    if (!jacobians)
      return;

    for (int i=0; i<6; ++i)
    {
      if (!jacobians[i])
        continue;
      jacobians[i][index  ] = jacobian_dx_row(i) * jac_mult_lower;
      jacobians[i][index+1] = jacobian_dx_row(i) * jac_mult_upper;
    }
    if (jacobians[6])  // dt
    {
      jacobians[6][index  ] = jacobian_dt * jac_mult_lower;
      jacobians[6][index+1] = jacobian_dt * jac_mult_upper;
    }
    if (jacobians[7])  // slack
    {
      // this part is diagonal
      jacobians[7][index * NUM_CONSTRAINTS_PER_SEGMENT + index] = -1.0;           // slack_lower
      jacobians[7][(index+1) * NUM_CONSTRAINTS_PER_SEGMENT + (index+1)] = -1.0;   // slack_upper
    }
  }

private:
  int dof_;
  int via_;
  const ProblemData& data_;
};

void add_variables_and_reg_to_problem(ConstrainedProblem& problem, ProblemData& data)
{
  // add variables and regularization costs
  for (int i=0; i<data.num_vias; ++i)
  {
    bool fixed = (i == 0 || i == data.num_vias - 1);

    for (int j=0; j<data.num_dofs; ++j)
    {
      problem.AddParameterBlock(&data.variables[j].coeffRef(0, i), 1);
      // positions are always fixed
      problem.SetParameterBlockConstant(&data.variables[j].coeffRef(0, i));

      problem.AddParameterBlock(&data.variables[j].coeffRef(1, i), 1);
      problem.AddParameterBlock(&data.variables[j].coeffRef(2, i), 1);
      if (fixed)
      {
        problem.SetParameterBlockConstant(&data.variables[j].coeffRef(1, i));
        problem.SetParameterBlockConstant(&data.variables[j].coeffRef(2, i));
      }
      else
      {
        problem.AddResidualBlock(new RegularizationCost(REGULARIZE_VEL),
                                 NULL, &data.variables[j].coeffRef(1, i));
        problem.SetParameterLowerBound(&data.variables[j].coeffRef(1, i), 0,
                                       double(data.lower_limits[j](1, i)));
        problem.SetParameterUpperBound(&data.variables[j].coeffRef(1, i), 0,
                                       double(data.upper_limits[j](1, i)));

        problem.AddResidualBlock(new RegularizationCost(REGULARIZE_ACC),
                                 NULL, &data.variables[j].coeffRef(2, i));
        problem.SetParameterLowerBound(&data.variables[j].coeffRef(2, i), 0,
                                       double(data.lower_limits[j](2, i)));
        problem.SetParameterUpperBound(&data.variables[j].coeffRef(2, i), 0,
                                       double(data.upper_limits[j](2, i)));
      }
    }
  }
  for (int i=0; i<data.num_segments; ++i)
  {
    problem.AddParameterBlock(&data.dt_variables.coeffRef(i), 1);
    problem.AddResidualBlock(new RegularizationCost(REGULARIZE_DT),
                             NULL, &data.dt_variables.coeffRef(i));
    problem.SetParameterLowerBound(&data.dt_variables.coeffRef(i), 0,
                                   data.delta_t_lower_bound);
    problem.SetParameterUpperBound(&data.dt_variables.coeffRef(i), 0,
                                   data.delta_t_upper_bound);
  }
}

void add_slacks_and_constraints(ConstrainedProblem& problem, ProblemData& data)
{
  // add constraints per segment
  data.constraints.clear();
  data.slack_variables.clear();
  data.slack_variables.resize(data.num_dofs,
                              std::vector<Eigen::VectorXd>(data.num_segments,
                                                           Eigen::VectorXd::Zero(NUM_CONSTRAINTS_PER_SEGMENT)));
  data.constraints.resize(data.num_dofs,
                          std::vector<ConstraintBlockId>(data.num_segments));

  for (int j=0; j<data.num_dofs; ++j)
  {
    std::vector<ConstraintBlockId> constraints;
    for (int i=0; i<data.num_segments; ++i)
    {
      problem.AddParameterBlock(&data.slack_variables[j][i].coeffRef(0), NUM_CONSTRAINTS_PER_SEGMENT);
      // all slacks are lower-bounded by zero
      for (int s=0; s<NUM_CONSTRAINTS_PER_SEGMENT; ++s)
        problem.SetParameterLowerBound(&data.slack_variables[j][i].coeffRef(0), s, 0.0);
      data.constraints[j][i] = problem.AddConstraintBlock(
                                 new SegmentConstraints(j, i, data),
                                 NONLINEAR,
                                 &data.variables[j].coeffRef(0, i),
                                 &data.variables[j].coeffRef(1, i),
                                 &data.variables[j].coeffRef(2, i),
                                 &data.variables[j].coeffRef(0, i+1),
                                 &data.variables[j].coeffRef(1, i+1),
                                 &data.variables[j].coeffRef(2, i+1),
                                 &data.dt_variables.coeffRef(i),
                                 &data.slack_variables[j][i].coeffRef(0));
    }
  }
}

void ceres_solve_constrained(ProblemData& data)
{
  ConstrainedProblem problem;
  add_variables_and_reg_to_problem(problem, data);
  add_slacks_and_constraints(problem, data);

  // solve it
  ceres::Solver::Options options;
  options.function_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport();

  // check constraint violations
  //std::cout << "\nCeres solution:\n";
  //evaluate(problem, constraint_ids);
}

int main(int argc, char** argv)
{
  std::string usage("Runs the path_to_trajectory problem using Ceres. Usage:\n\t");
  usage += argv[0];
  usage += " --input=problem_data.txt [ --compare=problem_solution.txt ]";
  google::SetUsageMessage(usage);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  if (FLAGS_input.empty())
  {
    std::cerr << "Please provide an input file name.\n";
    return 1;
  }

  ProblemData data;
  data.read_from_file(FLAGS_input);

  ceres_solve_constrained(data);

//  // compare with KNITRO solution
//  if (!FLAGS_compare.empty())
//  {
//    ProblemData compare;
//    compare.read_from_file(FLAGS_compare);
//    std::vector<ceres::ResidualBlockId> constraint_ids;
//    ceres::Problem problem;
//    create_problem_quad_penalty(problem, compare, CONSTRAINT_PENALTY, constraint_ids);
//    std::cout << "\nKNITRO solution:\n";
//    evaluate(problem, constraint_ids);
//  }

  return 0;
}

// WARNING! DO NOT EDIT! This file is auto-generated from compute_spline_derivatives.py

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"

void compute_spline_derivatives(const double x0, const double xd0, const double xdd0,
                                const double x1, const double xd1, const double xdd1,
                                const double  t, const double   T,
                                Eigen::Matrix<double, 4, 1>& sample_derivs,
                                Eigen::Matrix<double, 4, 6>& sample_dx,
                                Eigen::Matrix<double, 4, 1>& sample_dT)
{
  // powers of t and T:
  const double t1     = t;
  const double T1     = T;
  const double T_inv1 = 1.0 / T;
  const double t_inv1 = 1.0 / t;
  const double t2 = t1 * t1;
  const double T2 = T1 * T1;
  const double t_inv2 = t_inv1 * t_inv1;
  const double T_inv2 = T_inv1 * T_inv1;
  const double t3 = t2 * t1;
  const double T3 = T2 * T1;
  const double t_inv3 = t_inv2 * t_inv1;
  const double T_inv3 = T_inv2 * T_inv1;
  const double t4 = t3 * t1;
  const double T4 = T3 * T1;
  const double t_inv4 = t_inv3 * t_inv1;
  const double T_inv4 = T_inv3 * T_inv1;
  const double t5 = t4 * t1;
  const double T5 = T4 * T1;
  const double t_inv5 = t_inv4 * t_inv1;
  const double T_inv5 = T_inv4 * T_inv1;
  const double t6 = t5 * t1;
  const double T6 = T5 * T1;
  const double t_inv6 = t_inv5 * t_inv1;
  const double T_inv6 = T_inv5 * T_inv1;
  const double t7 = t6 * t1;
  const double T7 = T6 * T1;
  const double t_inv7 = t_inv6 * t_inv1;
  const double T_inv7 = T_inv6 * T_inv1;
  const double t8 = t7 * t1;
  const double T8 = T7 * T1;
  const double t_inv8 = t_inv7 * t_inv1;
  const double T_inv8 = T_inv7 * T_inv1;
  const double t9 = t8 * t1;
  const double T9 = T8 * T1;
  const double t_inv9 = t_inv8 * t_inv1;
  const double T_inv9 = T_inv8 * T_inv1;

  // common subexpressions
  const double k0 = T2*xdd1;
  const double k1 = 1.0/T;
  const double k2 = 60*T_inv3;
  const double k3 = 6*T;
  const double k4 = 14*xd1;
  const double k5 = T_inv5*t3;
  const double k6 = T_inv4*t;
  const double k7 = t*xd0;
  const double k8 = 12*xd0;
  const double k9 = k2/120;
  const double k10 = t2*xdd0;
  const double k11 = 2*T;
  const double k12 = T_inv3*t4;
  const double k13 = k1*t3;
  const double k14 = T_inv2*t4;
  const double k15 = k1*t2;
  const double k16 = 36*T_inv2;
  const double k17 = 30*T_inv5;
  const double k18 = 12*t2;
  const double k19 = 3*T_inv3;
  const double k20 = T_inv3*t2;
  const double k21 = T_inv3*t;
  const double k22 = 9*k1;
  const double k23 = T_inv3*t3;
  const double k24 = 8*xd1;
  const double k25 = 3*k1;
  const double k26 = 16*xd0;
  const double k27 = T2*xdd0;
  const double k28 = t*xdd0;
  const double k29 = 24*T_inv2;
  const double k30 = T_inv2*t3;
  const double k31 = T_inv4*t2;
  const double k32 = 3*k27;
  const double k33 = 15*T_inv4*t4;
  const double k34 = 3*T_inv4*t5;
  const double k35 = k9*t5;
  const double k36 = k3*xdd0;
  const double k37 = k11*xdd1;
  const double k38 = 6*k30;
  const double k39 = 60*T_inv4*t3;
  const double k40 = 4*k30;
  const double k41 = 18*T_inv2*t2;
  const double k42 = 10*k23;
  const double k43 = 15*T_inv4*k18;
  const double k44 = T_inv2*k18;
  const double k45 = k16*t;
  const double k46 = 10*k19*t2;
  const double k47 = 360*k6;
  const double k48 = k29*t;
  const double k49 = k33/30;
  const double k50 = k19*t;
  const double k51 = k17*t5/60;
  const double k52 = 2*T_inv4*t3;
  const double k53 = 5*k9*t4;
  const double k54 = k17*t4/12;
  const double k55 = k17*k18 + k2;
  const double k56 = 6*T_inv5*t5 + k42;
  const double k57 = k17*t4 + k46;
  const double k58 = k2*t + 120*k5;
  const double k59 = -k11*xdd0 + k37 - 6*xd0 - 6*xd1;
  const double k60 = k26 + k36 - 2*k37 + k4;
  const double k61 = -k24 - k36 + k37 - k8;
  const double k62 = T_inv5*k59;
  const double k63 = k0 - k27 - k3*xd0 - k3*xd1 - 12*x0 + 12*x1;
  const double k64 = k0 - 4*k11*xd1 - 2*k3*xd0 - k32 - 20*x0 + 20*x1;
  const double k65 = T_inv6*k63;
  const double k66 = k63*t2;
  const double k67 = T_inv5*k63;
  const double k68 = T*k4 - 2*k0 + 8*k11*xd0 + k32 + 30*x0 - 30*x1;
  const double k69 = T_inv4*k64;
  const double k70 = T_inv4*k68;
  const double k71 = 12*k68;
  const double k72 = 5*k63*t4/2;
  const double k73 = 3*k64/2;
  const double k74 = T_inv5*k68;

  // derived quantities
  sample_dT(0) = k1*k10 + k1*k7 + k49*k60 + k51*k59 + k61*k9*t3;
  sample_dT(1) = k1*k28 - k43*k73/180 - 2*k5*k68 + k52*k60 + k54*k59 + 3*k61*k9*t2 - 5*k65*t4/2;
  sample_dT(2) = -k18*k74 + 6*k31*k60 + 10*k5*k59 + k50*k61 - 4*k6*k73 - 20*k65*t3;
  sample_dT(3) = -3*T_inv5*k71*t + k17*k59*t2 + k19*k61 + 12*k6*k60 - 90*k65*t2 - 9*k69;
  sample_derivs(0) = k10/2 + k49*k68 + k51*k63 + k64*k9*t3 + k7 + x0;
  sample_derivs(1) = k28 + k52*k68 + k54*k63 + 2*k73*k9*t2 + xd0;
  sample_derivs(2) = 6*k31*k68 + 10*k5*k63 + k50*k64 + xdd0;
  sample_derivs(3) = k17*k66 + k19*k64 + k6*k71;
  sample_dx(0,0) = k33 - k56 + 1;
  sample_dx(0,1) = 8*k12 - k34 - k38 + t;
  sample_dx(0,2) = -3*k13/2 + 3*k14/2 + k18/24 - k35;
  sample_dx(0,3) = -k33 + k56;
  sample_dx(0,4) = 7*k12 - k34 - k40;
  sample_dx(0,5) = k13/2 - k14 + k35;
  sample_dx(1,0) = k39 - k57;
  sample_dx(1,1) = 32*k23 - k33 - k41 + 1;
  sample_dx(1,2) = -9*k15/2 + k38 - k53 + t;
  sample_dx(1,3) = -k39 + k57;
  sample_dx(1,4) = 28*k23 - k33 - k44;
  sample_dx(1,5) = 3*k15/2 - k40 + k53;
  sample_dx(2,0) = k43 - k58;
  sample_dx(2,1) = 8*T_inv3*k18 - k39 - k45;
  sample_dx(2,2) = -k22*t + k41 - k42 + 1;
  sample_dx(2,3) = -k43 + k58;
  sample_dx(2,4) = 7*T_inv3*k18 - k39 - k48;
  sample_dx(2,5) = k25*t + k42 - k44;
  sample_dx(3,0) = k47 - k55;
  sample_dx(3,1) = -k16 - k43 + 64*k50;
  sample_dx(3,2) = -k22 + k45 - k46;
  sample_dx(3,3) = -k47 + k55;
  sample_dx(3,4) = -k29 - k43 + 56*k50;
  sample_dx(3,5) = k25 + k46 - k48;
}

void compute_spline_derivatives(const double x0, const double xd0, const double xdd0,
                                const double x1, const double xd1, const double xdd1,
                                const double  t, const double   T,
                                Eigen::Matrix<double, 4, 1>& sample_derivs)
{
  // powers of t and T:
  const double t1     = t;
  const double T1     = T;
  const double T_inv1 = 1.0 / T;
  const double t_inv1 = 1.0 / t;
  const double t2 = t1 * t1;
  const double T2 = T1 * T1;
  const double t_inv2 = t_inv1 * t_inv1;
  const double T_inv2 = T_inv1 * T_inv1;
  const double t3 = t2 * t1;
  const double T3 = T2 * T1;
  const double t_inv3 = t_inv2 * t_inv1;
  const double T_inv3 = T_inv2 * T_inv1;
  const double t4 = t3 * t1;
  const double T4 = T3 * T1;
  const double t_inv4 = t_inv3 * t_inv1;
  const double T_inv4 = T_inv3 * T_inv1;
  const double t5 = t4 * t1;
  const double T5 = T4 * T1;
  const double t_inv5 = t_inv4 * t_inv1;
  const double T_inv5 = T_inv4 * T_inv1;
  const double t6 = t5 * t1;
  const double T6 = T5 * T1;
  const double t_inv6 = t_inv5 * t_inv1;
  const double T_inv6 = T_inv5 * T_inv1;
  const double t7 = t6 * t1;
  const double T7 = T6 * T1;
  const double t_inv7 = t_inv6 * t_inv1;
  const double T_inv7 = T_inv6 * T_inv1;
  const double t8 = t7 * t1;
  const double T8 = T7 * T1;
  const double t_inv8 = t_inv7 * t_inv1;
  const double T_inv8 = T_inv7 * T_inv1;
  const double t9 = t8 * t1;
  const double T9 = T8 * T1;
  const double t_inv9 = t_inv8 * t_inv1;
  const double T_inv9 = T_inv8 * T_inv1;

  // common subexpressions
  const double k0 = T2*xdd1;
  const double k1 = T2*xdd0;
  const double k2 = T*xd1;
  const double k3 = T*xd0;
  const double k4 = 3*k1;
  const double k5 = k0 - k1 - 6*k2 - 6*k3 - 12*x0 + 12*x1;
  const double k6 = k0 - 8*k2 - 12*k3 - k4 - 20*x0 + 20*x1;
  const double k7 = T_inv5*k5;
  const double k8 = -2*k0 + 14*k2 + 16*k3 + k4 + 30*x0 - 30*x1;
  const double k9 = T_inv3*k6;
  const double k10 = T_inv4*k8;
  const double k11 = 3*k9;

  // derived quantities
  sample_derivs(0) = k10*t4/2 + k11*t3/6 + k7*t5/2 + t*xd0 + t2*xdd0/2 + x0;
  sample_derivs(1) = 2*k10*t3 + k11*t2/2 + 5*k7*t4/2 + t*xdd0 + xd0;
  sample_derivs(2) = 6*k10*t2 + k11*t + 10*k7*t3 + xdd0;
  sample_derivs(3) = 12*k10*t + k11 + 30*k7*t2;
}
#pragma GCC diagnostic pop
