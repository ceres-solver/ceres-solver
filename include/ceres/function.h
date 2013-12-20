#ifndef CERES_PUBLIC_FUNCTION_H_
#define CERES_PUBLIC_FUNCTION_H_

namespace ceres {

struct Function {
  virtual ~Function() {}
  virtual bool operator()(const double* parameters,
                          double* cost,
                          double* gradient) const = 0;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_FUNCTION_SOLVER_H_
