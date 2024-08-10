// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/solver_utils.h"

#include "Eigen/Core"
#include "ceres/internal/config.h"
#include "ceres/internal/export.h"
#include "ceres/version.h"
#ifndef CERES_NO_CUDA
#include "cuda_runtime.h"
#ifndef CERES_NO_CUDSS
#include "cudss.h"
#endif  // CERES_NO_CUDSS
#endif  // CERES_NO_CUDA

namespace ceres::internal {

constexpr char kVersion[] =
    // clang-format off
  CERES_VERSION_STRING "-eigen-("
  CERES_SEMVER_VERSION(EIGEN_WORLD_VERSION,
                       EIGEN_MAJOR_VERSION,
                       EIGEN_MINOR_VERSION) ")"

#ifdef CERES_NO_LAPACK
  "-no_lapack"
#else
  "-lapack"
#endif

#ifndef CERES_NO_SUITESPARSE
  "-suitesparse-(" CERES_SUITESPARSE_VERSION ")"
#endif

#if !defined(CERES_NO_EIGEN_METIS) || !defined(CERES_NO_CHOLMOD_PARTITION)
  "-metis-(" CERES_METIS_VERSION ")"
#endif

#ifndef CERES_NO_ACCELERATE_SPARSE
  "-acceleratesparse"
#endif

#ifdef CERES_USE_EIGEN_SPARSE
  "-eigensparse"
#endif

#ifdef CERES_RESTRUCT_SCHUR_SPECIALIZATIONS
  "-no_schur_specializations"
#endif

#ifdef CERES_NO_CUSTOM_BLAS
  "-no_custom_blas"
#endif

#ifndef CERES_NO_CUDA
  "-cuda-(" CERES_TO_STRING(CUDART_VERSION) ")"
#ifndef CERES_NO_CUDSS
  "-cudss-(" CERES_SEMVER_VERSION(CUDSS_VERSION_MAJOR,
                                  CUDSS_VERSION_MINOR,
                                  CUDSS_VERSION_PATCH) ")"
#endif // CERES_NO_CUDSS
#endif
  ;
// clang-format on

std::string VersionString() { return kVersion; }

}  // namespace ceres::internal
