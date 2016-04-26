#ifndef CERES_INTERNAL_THREAD_ID_H_
#define CERES_INTERNAL_THREAD_ID_H_

#include "ceres/internal/config.h"

#ifdef CERES_USE_TBB
#include <tbb/atomic.h>
#include <tbb/enumerable_thread_specific.h>
#endif

namespace ceres {
namespace internal {

// Common for all threading options interface obtaining thread Id. If compiled
// with CERES_NO_THREAD it always returns 0. For OpenMP it simply wraps
// omp_get_thread_num(). In case of TBB it simulates omp_get_thread_num() with
// TLS.
class ThreadId {
 public:
  ThreadId();
  int id();

 private:
  ThreadId(ThreadId&) {}
  ThreadId& operator=(ThreadId&) { return *this; }

#ifdef CERES_USE_TBB
  // Wrapper for int with -1 default to know that it was not initialized.
  class IdValue {
   public:
    IdValue() : value_(kUninitialized) {}

    bool initialized() const { return value_ != kUninitialized; }

    int value() const { return value_; }
    void set_value(int value) { value_ = value; }

   private:
    enum { kUninitialized = -1 };
    int value_;
  };

  tbb::atomic<int> id_counter_;
  tbb::enumerable_thread_specific<IdValue> id_;
#endif
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_THREAD_ID_H_
