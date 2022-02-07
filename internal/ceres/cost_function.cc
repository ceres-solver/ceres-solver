#include <ceres/cost_function.h>

namespace ceres {

CostFunction::CostFunction() : num_residuals_(0) {}

CostFunction::~CostFunction() {}

}  // namespace ceres
