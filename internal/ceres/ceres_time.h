// Copyright 2011 Google Inc. All Rights Reserved.
// Author: keir@google.com (Keir Mierle)

#ifndef CERES_INTERNAL_CERES_TIME_H_
#define CERES_INTERNAL_CERES_TIME_H_

#ifdef CERES_USE_OPENMP
#include <omp.h>
#else
#include <ctime>
#endif

namespace ceres {
namespace internal {
namespace {

// Returns time, in seconds, from some arbitrary starting point. Has very
// high precision if OpenMP is available, otherwise only second granularity.
double CeresTime() {
#ifdef CERES_USE_OPENMP
  return omp_get_wtime();
#else
  return static_cast<double>(std::time(NULL));
#endif
}

}  // namespace
}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_CERES_TIME_H_
