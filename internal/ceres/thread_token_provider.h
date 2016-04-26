#ifndef CERES_INTERNAL_THREAD_TOKEN_PROVIDER_H_
#define CERES_INTERNAL_THREAD_TOKEN_PROVIDER_H_

#include "ceres/internal/config.h"

#ifdef CERES_USE_TBB
#include <tbb/concurrent_queue.h>
#endif

namespace ceres {
namespace internal {

// Helper for TBB thread number identification with similar to
// omp_get_thread_num() behaviour. The sequence of tokens varies form 0 to
// num_threads-1 that can be acquired to identify the thread in a thread pool.
//
// If CERES_NO_THREADS is defined, Acquire() always returns 0 and Release()
// takes no action.
//
// If CERES_USE_OPENMP, omp_get_thread_num() is uses to Acquire() with no action
// in Release()
class ThreadTokenProvider {
 public:
  ThreadTokenProvider(int num_threads);

  // Returns the first tocken from the queue. The acquired value must be
  // given back by Release().
  //
  // With TBB this function will block if all the tokens are aquired by
  // concurent threads.
  int Acquire();

  // Make previously acquired token available for other threads.
  void Release(int thread_id);

 private:
#ifdef CERES_USE_TBB
  // This queue initially holds a sequence from 0..num_threads-1. Every
  // Acquire() call the first number is removed from here. When the token is not
  // needed anymore it shall be given back with corresponding Release() call.
  tbb::concurrent_bounded_queue<int> pool_;
#endif

  ThreadTokenProvider(ThreadTokenProvider&);
  ThreadTokenProvider& operator=(ThreadTokenProvider&);
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_THREAD_TOKEN_PROVIDER_H_
