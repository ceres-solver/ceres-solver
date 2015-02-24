// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: richie.stebbing@gmail.com (Richard Stebbing)
//
// All functions, derivatives, `expected_residuals`, and
// `expected_jacobians_data` generated with the following Python code
// (requires Sympy).
//   import sympy as sp
//   from operator import add
//
//   _M = sp.Matrix
//
//   # `_D` computes the flattend row-major Jacobian for function `F`.
//   _D = lambda F, *V: _M([sp.diff(f, v) for f in F for v in V])
//
//   # Define functions `F`, `G`, and `H` ...
//   x, y0, y1, z = sp.symbols('x y0 y1 z')
//   F = _M([x * x + x * y0, x * y0 * y1])
//   G = _M([z * z])
//   H = _M([z, z * z * z])
//
//   # ... and determine their derivatives.
//   Fx, Fy = _D(F, x), _D(F, y0, y1)
//   Gz = _D(G, z)
//   Hz = _D(H, z)
//
//   # Define compositions `FG`, `FH`, and `FGH` ...
//   FG = F.subs({x : G[0]})
//   y_to_H = {y0 : H[0], y1 : H[1]}
//   FH = F.subs(y_to_H)
//   FGH = FG.subs(y_to_H)
//
//   # ... and determine their derivatives.
//   FGy, FGz = _D(FG, y0, y1), _D(FG, z)
//   FHx, FHz = _D(FH, x), _D(FH, z)
//   FGHz = _D(FGH, z)
//
//   # Set `x` and `y` as results of `G` and `H` so that the residuals are
//   # identical under all compositions.
//   _z = 2
//   _S = lambda d: (lambda e: tuple(e.subs(d)))
//   (_x,), (_y0, _y1) = list(map(_S({z : _z}), [G, H]))
//
//   # Evaluate all residuals and derivatives for each function and
//   # composition.
//   exprs = [(F, (Fx, Fy)),
//            (G, (Gz,)),
//            (H, (Hz,)),
//            # Order swapped to match `CostFunctionComposition`.
//            (FG, (FGz, FGy)),
//            (FH, (FHx, FHz)),
//            (FGH, (FGHz,))]
//
//   values = {x : _x, y0 : _y0, y1 : _y1, z : _z}
//   Sv = lambda e, _Sv=_S(values): tuple(float(v) for v in _Sv(e))
//   eval_exprs = [(Sv(e), reduce(add, map(Sv, ds))) for e, ds in exprs]

#include "ceres/cost_function_composition.h"

#include "ceres/cost_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/macros.h"

#include "gtest/gtest.h"

namespace ceres {
namespace internal {

using std::vector;

class CostFunctionF : public CostFunction {
 public:
  CostFunctionF() {
    set_num_residuals(2);
    mutable_parameter_block_sizes()->push_back(1);
    mutable_parameter_block_sizes()->push_back(2);
  }

  virtual bool Evaluate(const double* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    const double& x = parameters[0][0],
                  y0 = parameters[1][0],
                  y1 = parameters[1][1];

    residuals[0] = x * x + x * y0;
    residuals[1] = x * y0 * y1;

    if (jacobians != NULL) {
      if (jacobians[0] != NULL) {
        jacobians[0][0] = 2.0 * x + y0;
        jacobians[0][1] = y0 * y1;
      }
      if (jacobians[1] != NULL) {
        jacobians[1][0] = x;
        jacobians[1][1] = 0.0;
        jacobians[1][2] = x * y1;
        jacobians[1][3] = x * y0;
      }
    }

    return true;
  }
};

class CostFunctionG : public CostFunction {
 public:
  CostFunctionG() {
    set_num_residuals(1);
    mutable_parameter_block_sizes()->push_back(1);
  }

  virtual bool Evaluate(const double* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    const double& x = parameters[0][0];

    residuals[0] = x * x;

    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = 2.0 * x;
    }

    return true;
  }
};

class CostFunctionH : public CostFunction {
 public:
  CostFunctionH() {
    set_num_residuals(2);
    mutable_parameter_block_sizes()->push_back(1);
  }

  virtual bool Evaluate(const double* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    const double& x = parameters[0][0];

    residuals[0] = x;
    residuals[1] = x * x * x;

    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = 1.0;
      jacobians[0][1] = 3.0 * x * x;
    }

    return true;
  }
};

class CostFunctionCompositionTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    x[0] = 4.0;
    y[0] = 2.0; y[1] = 8.0;
    z[0] = 2.0;

    z_.push_back(z);

    f.reset(new CostFunctionF());
    g.reset(new CostFunctionG());
    h.reset(new CostFunctionH());
    q.reset(new CostFunctionComposition(f.get(), DO_NOT_TAKE_OWNERSHIP));
  }

  template <int M, int N>
  void ExpectArraysEqual(const double (&a) [M], const double (&b) [N]) {
    EXPECT_TRUE(M == N);
    EXPECT_TRUE(ConstVectorRef(a, M) == ConstVectorRef(b, N));
  }

  double x[1];
  double y[2];
  double z[1];
  vector<double*> z_;

  scoped_ptr<CostFunction> f;
  scoped_ptr<CostFunction> g;
  scoped_ptr<CostFunction> h;
  scoped_ptr<CostFunctionComposition> q;
};

TEST_F(CostFunctionCompositionTest, F) {
  static const double expected_residuals[2] = {24.0, 64.0};
  static const double expected_jacobians_data[6] = {
    10.0, 16.0, 4.0, 0.0, 32.0, 8.0};

  // f(x, y).
  const double* parameters[2] = {x, y};
  double residuals[2] = {0.0}, jacobians_data[6] = {0.0};
  double* jacobians[2] = {&jacobians_data[0], &jacobians_data[2]};
  ASSERT_TRUE(f->Evaluate(parameters, residuals, jacobians));

  ExpectArraysEqual(residuals, expected_residuals);
  ExpectArraysEqual(jacobians_data, expected_jacobians_data);
}

TEST_F(CostFunctionCompositionTest, G) {
  static const double expected_residuals[1] = {4.0};
  static const double expected_jacobians_data[1] = {4.0};

  // g(z).
  const double* parameters[1] = {z};
  double residuals[1] = {0.0}, jacobians_data[1] = {0.0};
  double* jacobians[1] = {&jacobians_data[0]};
  ASSERT_TRUE(g->Evaluate(parameters, residuals, jacobians));

  ExpectArraysEqual(residuals, expected_residuals);
  ExpectArraysEqual(jacobians_data, expected_jacobians_data);
}

TEST_F(CostFunctionCompositionTest, H) {
  static const double expected_residuals[2] = {2.0, 8.0};
  static const double expected_jacobians_data[2] = {1.0, 12.0};

  // h(z).
  const double* parameters[1] = {z};
  double residuals[2] = {0.0}, jacobians_data[2] = {0.0};
  double* jacobians[1] = {&jacobians_data[0]};
  ASSERT_TRUE(h->Evaluate(parameters, residuals, jacobians));

  ExpectArraysEqual(residuals, expected_residuals);
  ExpectArraysEqual(jacobians_data, expected_jacobians_data);
}

TEST_F(CostFunctionCompositionTest, FG) {
  static const double expected_residuals[2] = {24.0, 64.0};
  static const double expected_jacobians_data[6] = {
    40.0, 64.0, 4.0, 0.0, 32.0, 8.0};

  // q(z, y) = f(g(z), y).
  q->AddInputCostFunction(g.get(), z_, DO_NOT_TAKE_OWNERSHIP);
  q->AddInputParameterBlock(y);
  q->Finalize();

  double residuals[2] = {0.0}, jacobians_data[6] = {0.0};
  double* jacobians[2] = {&jacobians_data[0], &jacobians_data[2]};
  ASSERT_TRUE(q->Evaluate(&q->parameter_blocks()[0], residuals, jacobians));

  ExpectArraysEqual(residuals, expected_residuals);
  ExpectArraysEqual(jacobians_data, expected_jacobians_data);
}

TEST_F(CostFunctionCompositionTest, FH) {
  static const double expected_residuals[2] = {24.0, 64.0};
  static const double expected_jacobians_data[4] = {
    10.0, 16.0, 4.0, 128.0};

  // q(x, z) = f(x, h(z)).
  q->AddInputParameterBlock(x);
  q->AddInputCostFunction(h.get(), z_, DO_NOT_TAKE_OWNERSHIP);
  q->Finalize();

  double residuals[2] = {0.0}, jacobians_data[4] = {0.0};
  double* jacobians[2] = {&jacobians_data[0], &jacobians_data[2]};
  ASSERT_TRUE(q->Evaluate(&q->parameter_blocks()[0], residuals, jacobians));

  ExpectArraysEqual(residuals, expected_residuals);
  ExpectArraysEqual(jacobians_data, expected_jacobians_data);
}

TEST_F(CostFunctionCompositionTest, FGH) {
  static const double expected_residuals[2] = {24.0, 64.0};
  static const double expected_jacobians_data[2] = {44.0, 192.0};

  // q(z) = f(g(z), h(z)).
  q->AddInputCostFunction(g.get(), z_, DO_NOT_TAKE_OWNERSHIP);
  q->AddInputCostFunction(h.get(), z_, DO_NOT_TAKE_OWNERSHIP);
  q->Finalize();

  double residuals[2] = {0.0}, jacobians_data[2] = {0.0};
  double* jacobians[1] = {&jacobians_data[0]};
  ASSERT_TRUE(q->Evaluate(&q->parameter_blocks()[0], residuals, jacobians));

  ExpectArraysEqual(residuals, expected_residuals);
  ExpectArraysEqual(jacobians_data, expected_jacobians_data);
}

}  // namespace internal
}  // namespace ceres
