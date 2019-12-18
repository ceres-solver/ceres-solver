// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2019 Google Inc. All rights reserved.
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
// This file defines the required macros to use local variables and if-else
// branches in the AutoDiffCodeGen system. This is also the only file that
// should be included by a user-defined cost functor.
//
// To generate code for your cost functor the following steps have to be
// implemented (see below for a full example):
//
// 1. Include this file
// 2. Wrap accesses to local variables in the CERES_LOCAL_VARIABLE macro.
// 3. Replace if, else by CERES_IF, CERES_ELSE and add CERES_ENDIF
// 4. Add a default constructor
//
// Example - my_cost_functor.h
// ========================================================
//  #include "ceres/rotation.h"
//  #include "ceres/autodiff_codegen_macros.h"
//
//  struct MyReprojectionError {
//    MyReprojectionError(double observed_x, double observed_y)
//        : observed_x(observed_x), observed_y(observed_y) {}
//
//    // The cost functor must be default constructible!
//    MyReprojectionError() = default;
//
//    template <typename T>
//    bool operator()(const T* const camera,
//                    const T* const point,
//                    T* residuals) const {
//      T p[3];
//      AngleAxisRotatePoint(camera, point, p);
//      p[0] += camera[3];
//      p[1] += camera[4];
//      p[2] += camera[5];
//
//      // The if block is written using the macros!
//      CERES_IF(p[2] < T(0)) {
//        p[0] = -p[0];
//        p[1] = -p[1];
//        p[2] = -p[2];
//      } CERES_ELSE {
//        p[0] += T(1.0);
//      }CERES_ENDIF;
//
//      const T& focal = camera[6];
//      const T predicted_x = focal * p[0];
//      const T predicted_y = focal * p[1];
//
//      // The read-access to the local variables observed_x and observed_y are
//      // wrapped in the CERES_LOCAL_VARIABLE macro!
//      residuals[0] = predicted_x - CERES_LOCAL_VARIABLE(T, observed_x);
//      residuals[1] = predicted_y - CERES_LOCAL_VARIABLE(T, observed_y);
//      return true;
//    }
//    double observed_x;
//    double observed_y;
//  };
//
// ========================================================
//
// This file defines the following macros:
//
// CERES_LOCAL_VARIABLE
// CERES_IF
// CERES_ELSE
// CERES_ENDIF
//
#ifndef CERES_PUBLIC_CODEGEN_MACROS_H_
#define CERES_PUBLIC_CODEGEN_MACROS_H_

// The CERES_CODEGEN macro is defined by the build system only during code
// generation.
#ifndef CERES_CODEGEN
#define CERES_LOCAL_VARIABLE(_template_type, _local_variable) (_local_variable)
#define CERES_IF(condition_) if (condition_)
#define CERES_ELSE else
#define CERES_ENDIF
#define CERES_COMMENT(comment_)
#else
#define CERES_LOCAL_VARIABLE(_template_type, _local_variable)            \
  ceres::internal::InputAssignment<_template_type>::Get(_local_variable, \
                                                        #_local_variable)
#define CERES_IF(condition_) \
  AddExpressionToGraph(ceres::internal::Expression::CreateIf((condition_).id));
#define CERES_ELSE \
  AddExpressionToGraph(ceres::internal::Expression::CreateElse());
#define CERES_ENDIF \
  AddExpressionToGraph(ceres::internal::Expression::CreateEndIf());
#define CERES_COMMENT(comment_) \
  AddExpressionToGraph(ceres::internal::Expression::CreateComment(comment_))
#endif

namespace ceres {
// A function equivalent to the ternary ?-operator.
// This function is required, because in the context of code generation a
// comparison returns an expression type which is not convertible to bool.
inline double Ternary(bool c, double a, double b) { return c ? a : b; }
}  // namespace ceres

#endif  // CERES_PUBLIC_CODEGEN_MACROS_H_
