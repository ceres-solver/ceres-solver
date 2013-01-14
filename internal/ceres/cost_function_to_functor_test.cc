#include "ceres/cost_function_to_functor.h"
#include "ceres/autodiff_cost_function.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

const double kTolerance = 1e-18;

void ExpectCostFunctionsAreEqual(const CostFunction& cost_function,
                                 const CostFunction& actual_cost_function) {
  EXPECT_EQ(cost_function.num_residuals(), actual_cost_function.num_residuals());
  const int num_residuals = cost_function.num_residuals();
  const vector<int16>& parameter_block_sizes = cost_function.parameter_block_sizes();
  const vector<int16>& actual_parameter_block_sizes  = actual_cost_function.parameter_block_sizes();
  EXPECT_EQ(parameter_block_sizes.size(), actual_parameter_block_sizes.size());

  int num_parameters = 0;
  for (int i = 0; i < parameter_block_sizes.size(); ++i) {
    EXPECT_EQ(parameter_block_sizes[i], actual_parameter_block_sizes[i]);
    num_parameters += parameter_block_sizes[i];
  }

  scoped_array<double> parameters(new double[num_parameters]);
  for (int i = 0; i < num_parameters; ++i) {
    parameters[i] = static_cast<double>(i) + 1.0;
  }

  scoped_array<double> residuals(new double[num_residuals]);
  scoped_array<double> jacobians(new double[num_parameters * num_residuals]);

  scoped_array<double> actual_residuals(new double[num_residuals]);
  scoped_array<double> actual_jacobians(new double[num_parameters * num_residuals]);

  scoped_array<double*> parameter_blocks(new double*[parameter_block_sizes.size()]);
  scoped_array<double*> jacobian_blocks(new double*[parameter_block_sizes.size()]);
  scoped_array<double*> actual_jacobian_blocks(new double*[parameter_block_sizes.size()]);

  num_parameters = 0;
  for (int i = 0; i < parameter_block_sizes.size(); ++i) {
    parameter_blocks[i] = parameters.get() + num_parameters;
    jacobian_blocks[i] = jacobians.get() + num_parameters * num_residuals;
    actual_jacobian_blocks[i] = actual_jacobians.get() + num_parameters * num_residuals;
    num_parameters += parameter_block_sizes[i];
  }

  EXPECT_TRUE(cost_function.Evaluate(parameter_blocks.get(), residuals.get(), NULL));
  EXPECT_TRUE(actual_cost_function.Evaluate(parameter_blocks.get(), actual_residuals.get(), NULL));
  for (int i = 0; i < num_residuals; ++i) {
    EXPECT_NEAR(residuals[i], actual_residuals[i], kTolerance)
        << "residual id: " << i;
  }


  EXPECT_TRUE(cost_function.Evaluate(parameter_blocks.get(),
                                     residuals.get(),
                                     jacobian_blocks.get()));
  EXPECT_TRUE(actual_cost_function.Evaluate(parameter_blocks.get(),
                                            actual_residuals.get(),
                                            actual_jacobian_blocks.get()));
  for (int i = 0; i < num_residuals; ++i) {
    EXPECT_NEAR(residuals[i], actual_residuals[i], kTolerance)
        << "residual : " << i;
  }

  for (int i = 0; i < num_residuals * num_parameters; ++i) {
    EXPECT_NEAR(jacobians[i], actual_jacobians[i], kTolerance)
        << "jacobian : " << i << " " << jacobians[i] << " " << actual_jacobians[i];
  }
};

struct OneParameterBlockFunctor {
 public:
  template <typename T>
  bool operator()(const T* x1, T* residuals) const {
    residuals[0] = x1[0] * x1[0];
    residuals[1] = x1[1] * x1[1];
    return true;
  }
};

struct TwoParameterBlockFunctor {
 public:
  template <typename T>
  bool operator()(const T* x1, const T* x2, T* residuals) const {
    residuals[0] = x1[0] * x1[0]  + x2[0] * x2[0];
    residuals[1] = x1[1] * x1[1]  + x2[1] * x2[1];
    return true;
  }
};

struct ThreeParameterBlockFunctor {
 public:
  template <typename T>
  bool operator()(const T* x1, const T* x2, const T* x3, T* residuals) const {
    residuals[0] = x1[0] * x1[0]  + x2[0] * x2[0] + x3[0] * x3[0];
    residuals[1] = x1[1] * x1[1]  + x2[1] * x2[1] + x3[1] * x3[1];
    return true;
  }
};

struct FourParameterBlockFunctor {
 public:
  template <typename T>
  bool operator()(const T* x1, const T* x2, const T* x3, const T* x4,
                  T* residuals) const {
    residuals[0] = x1[0] * x1[0]  + x2[0] * x2[0] + x3[0] * x3[0]
        + x4[0] * x4[0];
    residuals[1] = x1[1] * x1[1]  + x2[1] * x2[1] + x3[1] * x3[1]
        + x4[1] * x4[1];
    return true;
  }
};

struct FiveParameterBlockFunctor {
 public:
  template <typename T>
  bool operator()(const T* x1, const T* x2, const T* x3, const T* x4,
                  const T* x5, T* residuals) const {
    residuals[0] = x1[0] * x1[0]  + x2[0] * x2[0] + x3[0] * x3[0]
        + x4[0] * x4[0] + x5[0] * x5[0];
    residuals[1] = x1[1] * x1[1]  + x2[1] * x2[1] + x3[1] * x3[1]
        + x4[1] * x4[1] + x5[1] * x5[1];
    return true;
  }
};

struct SixParameterBlockFunctor {
 public:
  template <typename T>
  bool operator()(const T* x1, const T* x2, const T* x3, const T* x4,
                  const T* x5, const T* x6,  T* residuals) const {
    residuals[0] = x1[0] * x1[0]  + x2[0] * x2[0] + x3[0] * x3[0]
        + x4[0] * x4[0] + x5[0] * x5[0] + x6[0] * x6[0];
    residuals[1] = x1[1] * x1[1]  + x2[1] * x2[1] + x3[1] * x3[1]
        + x4[1] * x4[1] + x5[1] * x5[1] + x6[1] * x6[1];
    return true;
  }
};

struct SevenParameterBlockFunctor {
 public:
  template <typename T>
  bool operator()(const T* x1, const T* x2, const T* x3, const T* x4,
                  const T* x5, const T* x6, const T* x7, T* residuals) const {
    residuals[0] = x1[0] * x1[0]  + x2[0] * x2[0] + x3[0] * x3[0]
        + x4[0] * x4[0] + x5[0] * x5[0] + x6[0] * x6[0] + x7[0] * x7[0];
    residuals[1] = x1[1] * x1[1]  + x2[1] * x2[1] + x3[1] * x3[1]
        + x4[1] * x4[1] + x5[1] * x5[1] + x6[1] * x6[1] + x7[1] * x7[1];
    return true;
  }
};

struct EightParameterBlockFunctor {
 public:
  template <typename T>
  bool operator()(const T* x1, const T* x2, const T* x3, const T* x4,
                  const T* x5, const T* x6, const T* x7, const T* x8,
                  T* residuals) const {
    residuals[0] = x1[0] * x1[0]  + x2[0] * x2[0] + x3[0] * x3[0]
        + x4[0] * x4[0] + x5[0] * x5[0] + x6[0] * x6[0] + x7[0] * x7[0]
        + x8[0] * x8[0];
    residuals[1] = x1[1] * x1[1]  + x2[1] * x2[1] + x3[1] * x3[1]
        + x4[1] * x4[1] + x5[1] * x5[1] + x6[1] * x6[1] + x7[1] * x7[1]
        + x8[1] * x8[1];
    return true;
  }
};

struct NineParameterBlockFunctor {
 public:
  template <typename T>
  bool operator()(const T* x1, const T* x2, const T* x3, const T* x4,
                  const T* x5, const T* x6, const T* x7, const T* x8,
                  const T* x9, T* residuals) const {
    residuals[0] = x1[0] * x1[0]  + x2[0] * x2[0] + x3[0] * x3[0]
        + x4[0] * x4[0] + x5[0] * x5[0] + x6[0] * x6[0] + x7[0] * x7[0]
        + x8[0] * x8[0] + x9[0] * x9[0];
    residuals[1] = x1[1] * x1[1]  + x2[1] * x2[1] + x3[1] * x3[1]
        + x4[1] * x4[1] + x5[1] * x5[1] + x6[1] * x6[1] + x7[1] * x7[1]
        + x8[1] * x8[1] + x9[1] * x9[1];
    return true;
  }
};

struct TenParameterBlockFunctor {
 public:
  template <typename T>
  bool operator()(const T* x1, const T* x2, const T* x3, const T* x4,
                  const T* x5, const T* x6, const T* x7, const T* x8,
                  const T* x9, const T* x10, T* residuals) const {
    residuals[0] = x1[0] * x1[0]  + x2[0] * x2[0] + x3[0] * x3[0]
        + x4[0] * x4[0] + x5[0] * x5[0] + x6[0] * x6[0] + x7[0] * x7[0]
        + x8[0] * x8[0] + x9[0] * x9[0] + x10[0] * x10[0];

    residuals[1] = x1[1] * x1[1]  + x2[1] * x2[1] + x3[1] * x3[1]
        + x4[1] * x4[1] + x5[1] * x5[1] + x6[1] * x6[1] + x7[1] * x7[1]
        + x8[1] * x8[1] + x9[1] * x9[1] + x10[1] * x10[1];
    return true;
  }
};

#define TEST_BODY(NAME)                                                 \
TEST(CostFunctionToFunctor, NAME){                                      \
  scoped_ptr<CostFunction> cost_function(                               \
      new AutoDiffCostFunction<CostFunctionToFunctor<2, PARAMETER_BLOCK_SIZES >, 2, PARAMETER_BLOCK_SIZES>( \
          new CostFunctionToFunctor<2, PARAMETER_BLOCK_SIZES >(         \
              new AutoDiffCostFunction<NAME##Functor, 2, PARAMETER_BLOCK_SIZES >( \
                  new NAME##Functor))));                     \
                                                                        \
  scoped_ptr<CostFunction> actual_cost_function(                        \
      new AutoDiffCostFunction<NAME##Functor, 2, PARAMETER_BLOCK_SIZES >( \
          new NAME##Functor));                               \
  ExpectCostFunctionsAreEqual(*cost_function, *actual_cost_function);   \
}

#define PARAMETER_BLOCK_SIZES 2
TEST_BODY(OneParameterBlock)
#undef PARAMETER_BLOCK_SIZES

#define PARAMETER_BLOCK_SIZES 2,2
TEST_BODY(TwoParameterBlock)
#undef PARAMETER_BLOCK_SIZES

#define PARAMETER_BLOCK_SIZES 2,2,2
TEST_BODY(ThreeParameterBlock)
#undef PARAMETER_BLOCK_SIZES

#define PARAMETER_BLOCK_SIZES 2,2,2,2
TEST_BODY(FourParameterBlock)
#undef PARAMETER_BLOCK_SIZES

#define PARAMETER_BLOCK_SIZES 2,2,2,2,2
TEST_BODY(FiveParameterBlock)
#undef PARAMETER_BLOCK_SIZES

#define PARAMETER_BLOCK_SIZES 2,2,2,2,2,2
TEST_BODY(SixParameterBlock)
#undef PARAMETER_BLOCK_SIZES

#define PARAMETER_BLOCK_SIZES 2,2,2,2,2,2,2
TEST_BODY(SevenParameterBlock)
#undef PARAMETER_BLOCK_SIZES

#define PARAMETER_BLOCK_SIZES 2,2,2,2,2,2,2,2
TEST_BODY(EightParameterBlock)
#undef PARAMETER_BLOCK_SIZES

#define PARAMETER_BLOCK_SIZES 2,2,2,2,2,2,2,2,2
TEST_BODY(NineParameterBlock)
#undef PARAMETER_BLOCK_SIZES

#define PARAMETER_BLOCK_SIZES 2,2,2,2,2,2,2,2,2,2
TEST_BODY(TenParameterBlock)
#undef PARAMETER_BLOCK_SIZES

#undef TEST_BODY
}  // namespace internal
}  // namespace ceres
