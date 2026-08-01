// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2026 Google Inc. All rights reserved.
// http://ceres-solver.org/
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
// Author: sergiu.deitsch@gmail.com (Sergiu Deitsch)

#include "bal_problem.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

namespace ceres::examples {
namespace {

TEST(BALProblem, PerturbsRotationWhenPointPerturbationIsDisabled) {
  constexpr const char* kProblemFile =
      CERES_TEST_DATA_DIR "/problem-16-22106-pre.txt";
  BALProblem baseline(kProblemFile, false);
  BALProblem problem(kProblemFile, false);
  const int num_camera_parameters =
      problem.num_cameras() * problem.camera_block_size();

  constexpr double kRotationSigma = 1.0;
  constexpr double kNoRotationSigma = 0.0;
  constexpr double kNoTranslationSigma = 0.0;
  constexpr double kNoPointSigma = 0.0;
  baseline.Perturb(kNoRotationSigma, kNoTranslationSigma, kNoPointSigma);
  problem.Perturb(kRotationSigma, kNoTranslationSigma, kNoPointSigma);

  double maximum_change = 0.0;
  for (int i = 0; i < num_camera_parameters; ++i) {
    maximum_change = std::max(
        maximum_change, std::abs(baseline.cameras()[i] - problem.cameras()[i]));
  }
  EXPECT_GT(maximum_change, 0.0);
}

}  // namespace
}  // namespace ceres::examples
