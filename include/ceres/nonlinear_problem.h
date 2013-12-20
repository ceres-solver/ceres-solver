#ifndef CERES_PUBLIC_NONLINEAR_PROBLEM_H_
#define CERES_PUBLIC_NONLINEAR_PROBLEM_H_

// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.

namespace ceres {

// Instances of a Nonlinear problem evaluate the cost and the gradient
// of an objective function.
class NonlinearProblem {
 public:
  virtual ~NonlinearProblem() {}

  // cost is guaranteed never to be null.
  // gradient may or maynot be null.
  virtual bool Evaluate(const double* parameters,
                        double* cost,
                        double* gradient) const = 0;
  virtual int NumParameters() const = 0;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_NONLINEAR_PROBLEM_H_
