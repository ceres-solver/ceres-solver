// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2020 Google Inc. All rights reserved.
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
// Author: darius.rueckert@fau.de (Darius Rueckert)
//
// This file includes unit test functors for every supported expression type.
// This is similar to expression_ref_test and codegeneration_test, but for the
// complete pipeline including automatic differentation. For each of the structs
// below, the Evaluate function is generated using GenerateCodeForFunctor. After
// that this function is executed with random parameters. The result of the
// residuals and jacobians is then compared to AutoDiff (without code
// generation). Of course, the correctness of this module depends on the
// correctness of autodiff.
//
#include <cmath>
#include <limits>

#include "ceres/codegen/codegen_cost_function.h"
namespace test {

struct InputOutputAssignment : public ceres::CodegenCostFunction<7, 4, 2, 1> {
  template <typename T>
  bool operator()(const T* x0, const T* x1, const T* x2, T* y) const {
    y[0] = x0[0];
    y[1] = x0[1];
    y[2] = x0[2];
    y[3] = x0[3];

    y[4] = x1[0];
    y[5] = x1[1];

    y[6] = x2[0];
    return true;
  }
#include "tests/inputoutputassignment.h"
};

struct CompileTimeConstants : public ceres::CodegenCostFunction<7, 1> {
  template <typename T>
  bool operator()(const T* x0, T* y) const {
    y[0] = T(0);
    y[1] = T(1);
    y[2] = T(-1);
    y[3] = T(1e-10);
    y[4] = T(1e10);
    y[5] = T(std::numeric_limits<double>::infinity());
    y[6] = T(std::numeric_limits<double>::quiet_NaN());

    return true;
  }
#include "tests/compiletimeconstants.h"
};

struct Assignments : public ceres::CodegenCostFunction<8, 2> {
  template <typename T>
  bool operator()(const T* x0, T* y) const {
    T a = x0[0];
    T b = x0[1];
    y[0] = a;
    y[1] = b;
    y[2] = y[3] = a;

    T c = a;
    y[4] = c;

    T d(b);
    y[5] = d;

    y[6] = std::move(c);

    y[7] = std::move(T(T(std::move(T(a)))));
    return true;
  }
#include "tests/assignments.h"
};

struct BinaryArithmetic : public ceres::CodegenCostFunction<9, 2> {
  template <typename T>
  bool operator()(const T* x0, T* y) const {
    T a = x0[0];
    T b = x0[1];
    y[0] = a + b;
    y[1] = a - b;
    y[2] = a * b;
    y[3] = a / b;

    y[4] = a;
    y[4] += b;
    y[5] = a;
    y[5] -= b;
    y[6] = a;
    y[6] *= b;
    y[7] = a;
    y[7] /= b;

    y[8] = a + b * a / a - b + b / a;
    return true;
  }
#include "tests/binaryarithmetic.h"
};

struct UnaryArithmetic : public ceres::CodegenCostFunction<3, 1> {
  template <typename T>
  bool operator()(const T* x0, T* y) const {
    T a = x0[0];
    y[0] = -a;
    y[1] = +a;
    y[2] = a;
    return true;
  }
#include "tests/unaryarithmetic.h"
};

struct BinaryComparison : public ceres::CodegenCostFunction<12, 2> {
  template <typename T>
  bool operator()(const T* x0, T* y) const {
    T a = x0[0];
    T b = x0[1];

    // For each operator we swap the inputs so both branches are evaluated once.
    CERES_IF(a < b) { y[0] = T(0); }
    CERES_ELSE { y[0] = T(1); }
    CERES_ENDIF
    CERES_IF(b < a) { y[1] = T(0); }
    CERES_ELSE { y[1] = T(1); }
    CERES_ENDIF

    CERES_IF(a > b) { y[2] = T(0); }
    CERES_ELSE { y[2] = T(1); }
    CERES_ENDIF
    CERES_IF(b > a) { y[3] = T(0); }
    CERES_ELSE { y[3] = T(1); }
    CERES_ENDIF

    CERES_IF(a <= b) { y[4] = T(0); }
    CERES_ELSE { y[4] = T(1); }
    CERES_ENDIF
    CERES_IF(b <= a) { y[5] = T(0); }
    CERES_ELSE { y[5] = T(1); }
    CERES_ENDIF

    CERES_IF(a >= b) { y[6] = T(0); }
    CERES_ELSE { y[6] = T(1); }
    CERES_ENDIF
    CERES_IF(b >= a) { y[7] = T(0); }
    CERES_ELSE { y[7] = T(1); }
    CERES_ENDIF

    CERES_IF(a == b) { y[8] = T(0); }
    CERES_ELSE { y[8] = T(1); }
    CERES_ENDIF
    CERES_IF(b == a) { y[9] = T(0); }
    CERES_ELSE { y[9] = T(1); }
    CERES_ENDIF

    CERES_IF(a != b) { y[10] = T(0); }
    CERES_ELSE { y[10] = T(1); }
    CERES_ENDIF
    CERES_IF(b != a) { y[11] = T(0); }
    CERES_ELSE { y[11] = T(1); }
    CERES_ENDIF

    return true;
  }
#include "tests/binarycomparison.h"
};

struct LogicalOperators : public ceres::CodegenCostFunction<8, 3> {
  template <typename T>
  bool operator()(const T* x0, T* y) const {
    T a = x0[0];
    T b = x0[1];
    T c = x0[2];
    auto r1 = a < b;
    auto r2 = a < c;

    CERES_IF(r1) { y[0] = T(0); }
    CERES_ELSE { y[0] = T(1); }
    CERES_ENDIF
    CERES_IF(r2) { y[1] = T(0); }
    CERES_ELSE { y[1] = T(1); }
    CERES_ENDIF
    CERES_IF(!r1) { y[2] = T(0); }
    CERES_ELSE { y[2] = T(1); }
    CERES_ENDIF
    CERES_IF(!r2) { y[3] = T(0); }
    CERES_ELSE { y[3] = T(1); }
    CERES_ENDIF

    CERES_IF(r1 && r2) { y[4] = T(0); }
    CERES_ELSE { y[4] = T(1); }
    CERES_ENDIF
    CERES_IF(!r1 && !r2) { y[5] = T(0); }
    CERES_ELSE { y[5] = T(1); }
    CERES_ENDIF

    CERES_IF(r1 || r2) { y[6] = T(0); }
    CERES_ELSE { y[6] = T(1); }
    CERES_ENDIF
    CERES_IF(!r1 || !r2) { y[7] = T(0); }
    CERES_ELSE { y[7] = T(1); }
    CERES_ENDIF

    return true;
  }
#include "tests/logicaloperators.h"
};

struct ScalarFunctions : public ceres::CodegenCostFunction<20, 22> {
  template <typename T>
  bool operator()(const T* x0, T* y) const {
    y[0] = abs(x0[0]);
    y[1] = acos(x0[1]);
    y[2] = asin(x0[2]);
    y[3] = atan(x0[3]);
    y[4] = cbrt(x0[4]);
    y[5] = ceil(x0[5]);
    y[6] = cos(x0[6]);
    y[7] = cosh(x0[7]);
    y[8] = exp(x0[8]);
    y[9] = exp2(x0[9]);
    y[10] = floor(x0[10]);
    y[11] = log(x0[11]);
    y[12] = log2(x0[12]);
    y[13] = sin(x0[13]);
    y[14] = sinh(x0[14]);
    y[15] = sqrt(x0[15]);
    y[16] = tan(x0[16]);
    y[17] = tanh(x0[17]);
    y[18] = atan2(x0[18], x0[19]);
    y[19] = pow(x0[20], x0[21]);
    return true;
  }
#include "tests/scalarfunctions.h"
};

struct LogicalFunctions : public ceres::CodegenCostFunction<4, 4> {
  template <typename T>
  bool operator()(const T* x0, T* y) const {
    using std::isfinite;
    using std::isinf;
    using std::isnan;
    using std::isnormal;
    T a = x0[0];
    auto r1 = isfinite(a);
    auto r2 = isinf(a);
    auto r3 = isnan(a);
    auto r4 = isnormal(a);

    CERES_IF(r1) { y[0] = T(0); }
    CERES_ELSE { y[0] = T(1); }
    CERES_ENDIF
    CERES_IF(r2) { y[1] = T(0); }
    CERES_ELSE { y[1] = T(1); }
    CERES_ENDIF
    CERES_IF(r3) { y[2] = T(0); }
    CERES_ELSE { y[2] = T(1); }
    CERES_ENDIF
    CERES_IF(r4) { y[3] = T(0); }
    CERES_ELSE { y[3] = T(1); }
    CERES_ENDIF

    return true;
  }
#include "tests/logicalfunctions.h"
};

struct Branches : public ceres::CodegenCostFunction<4, 3> {
  template <typename T>
  bool operator()(const T* x0, T* y) const {
    T a = x0[0];
    T b = x0[1];
    T c = x0[2];
    auto r1 = a < b;
    auto r2 = a < c;
    auto r3 = b < c;

    // If without else
    y[0] = T(0);
    CERES_IF(r1) { y[0] += T(1); }
    CERES_ENDIF

    // If else
    y[1] = T(0);
    CERES_IF(r1) { y[1] += T(-1); }
    CERES_ELSE { y[1] += T(1); }
    CERES_ENDIF

    // Nested if
    y[2] = T(0);
    CERES_IF(r1) {
      y[2] += T(1);
      CERES_IF(r2) {
        y[2] += T(4);
        CERES_IF(r2) { y[2] += T(8); }
        CERES_ENDIF
      }
      CERES_ENDIF
    }
    CERES_ENDIF

    // Nested if-else
    y[3] = T(0);
    CERES_IF(r1) {
      y[3] += T(1);
      CERES_IF(r2) {
        y[3] += T(2);
        CERES_IF(r3) { y[3] += T(4); }
        CERES_ELSE { y[3] += T(8); }
        CERES_ENDIF
      }
      CERES_ELSE {
        y[3] += T(16);
        CERES_IF(r3) { y[3] += T(32); }
        CERES_ELSE { y[3] += T(64); }
        CERES_ENDIF
      }
      CERES_ENDIF
    }
    CERES_ELSE {
      y[3] += T(128);
      CERES_IF(r2) { y[3] += T(256); }
      CERES_ELSE { y[3] += T(512); }
      CERES_ENDIF
    }
    CERES_ENDIF

    return true;
  }
#include "tests/branches.h"
};

}  // namespace test
