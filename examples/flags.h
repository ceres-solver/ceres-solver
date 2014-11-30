// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2014 Google Inc. All rights reserved.
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
// Author: mierle@gmail.com (Keir Mierle)

#ifndef CERES_EXAMPLES_FLAGS_H_
#define CERES_EXAMPLES_FLAGS_H_

#include "gflags/gflags.h"
#include "ceres/solver.h"

DECLARE_string(trust_region_strategy);
DECLARE_string(dogleg);
DECLARE_bool(inner_iterations);
DECLARE_string(linear_solver);
DECLARE_bool(explicit_schur_complement);
DECLARE_string(preconditioner);
DECLARE_string(visibility_clustering);
DECLARE_string(sparse_linear_algebra_library);
DECLARE_string(dense_linear_algebra_library);
DECLARE_string(ordering);
DECLARE_double(eta);
DECLARE_int32(num_threads);
DECLARE_int32(num_iterations);
DECLARE_double(max_solver_time);
DECLARE_bool(nonmonotonic_steps);
DECLARE_bool(line_search);

namespace ceres {
namespace examples {

void SetMinimizerOptionsFromFlags(Solver::Options* options);
void SetLinearSolverOptionsFromFlags(Solver::Options* options);
void SetSolverOptionsFromFlags(Solver::Options* options);

}  // namespace ceres
}  // namespace examples

#endif  // CERES_EXAMPLES_FLAGS_H_
