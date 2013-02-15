#include "ceres/cost_function.h"
#include "ceres/sized_cost_function.h"
#include "ceres/types.h"

namespace ceres {
namespace internal {

// y1 = x1'x2      -> dy1/dx1 = x2,               dy1/dx2 = x1
// y2 = (x1'x2)^2  -> dy2/dx1 = 2 * x2 * (x1'x2), dy2/dx2 = 2 * x1 * (x1'x2)
// y3 = x2'x2      -> dy3/dx1 = 0,                dy3/dx2 = 2 * x2
class EasyFunctor {
 public:
  bool operator()(const double* x1, const double* x2, double* residuals) const;
  void ExpectCostFunctionEvaluationIsNearlyCorrect(
      const CostFunction& cost_function,
      NumericDiffMethod method) const;
};

class EasyCostFunction : public SizedCostFunction<3, 5, 5> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** /* not used */) const {
    return functor_(parameters[0], parameters[1], residuals);
  }

 private:
  EasyFunctor functor_;
};

// y1 = sin(x1'x2)
// y2 = exp(-x1'x2 / 10)
//
// dy1/dx1 =  x2 * cos(x1'x2),            dy1/dx2 =  x1 * cos(x1'x2)
// dy2/dx1 = -x2 * exp(-x1'x2 / 10) / 10, dy2/dx2 = -x2 * exp(-x1'x2 / 10) / 10
class TranscendentalFunctor {
 public:
  bool operator()(const double* x1, const double* x2, double* residuals) const;
  void ExpectCostFunctionEvaluationIsNearlyCorrect(
      const CostFunction& cost_function,
      NumericDiffMethod method) const;
};

class TranscendentalCostFunction : public SizedCostFunction<2, 5, 5> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** /* not used */) const {
    return functor_(parameters[0], parameters[1], residuals);
  }
 private:
  TranscendentalFunctor functor_;
};

}  // namespace internal
}  // namespace ceres
