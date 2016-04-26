#ifndef CERES_INTERNAL_THREAD_TOKEN_H_
#define CERES_INTERNAL_THREAD_TOKEN_H_

#include "ceres/thread_token_provider.h"

namespace ceres {
namespace internal {

// Helper class for ThreadTokenProvider. This object acquires a token in its
// constructor and puts that token back with destruction.
class ScopedThreadToken {
 public:
  ScopedThreadToken(ThreadTokenProvider* provider)
      : provider_(provider), token_(provider->Acquire()) {}

  ~ScopedThreadToken() { provider_->Release(token_); }

  int token() const { return token_; }

 private:
  ThreadTokenProvider* provider_;
  int token_;

  ScopedThreadToken(ScopedThreadToken&);
  ScopedThreadToken& operator=(ScopedThreadToken&);
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_THREAD_TOKEN_H_
