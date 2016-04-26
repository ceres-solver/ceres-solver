#include "ceres/thread_token_provider.h"

#ifdef CERES_USE_OPENMP
#include <omp.h>
#endif

namespace ceres {
namespace internal {

#if defined(CERES_USE_OPENMP) || defined(CERES_NO_THREADS)
ThreadTokenProvider::ThreadTokenProvider(int /* num_threads */) {}

int ThreadTokenProvider::Acquire() {
#ifdef CERES_USE_OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif  // CERES_USE_OPENMP
}

void ThreadTokenProvider::Release(int /* thread_id */) {}
#endif  // CERES_USE_OPENMP || CERES_NO_THREADS

#ifdef CERES_USE_TBB
ThreadTokenProvider::ThreadTokenProvider(int num_threads) {
  pool_.set_capacity(num_threads);
  for (int i = 0; i < num_threads; i++) {
    pool_.push(i);
  }
}

int ThreadTokenProvider::Acquire() {
  int thread_id;
  pool_.pop(thread_id);
  return thread_id;
}

void ThreadTokenProvider::Release(int thread_id) { pool_.push(thread_id); }
#endif  // CERES_USE_TBB

}  // namespace ceres
}  // namespace ceres
