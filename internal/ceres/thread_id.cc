#include "ceres/thread_id.h"

#ifdef CERES_USE_OPENMP
#include <omp.h>
#endif

namespace ceres {
namespace internal {

#ifdef CERES_NO_THREADS
ThreadId::ThreadId() {}

int ThreadId::id() { return 0; }
#endif

#ifdef CERES_USE_OPENMP
ThreadId::ThreadId() {}

int ThreadId::id() { return omp_get_thread_num(); }
#endif

#ifdef CERES_USE_TBB
ThreadId::ThreadId() : id_counter_(0) {}

int ThreadId::id() {
  ThreadId::IdValue &id = id_.local();
  if (!id.initialized()) {
    id.set_value(id_counter_.fetch_and_increment());
  }
  return id.value();
}
#endif

}  // namespace ceres
}  // namespace ceres
