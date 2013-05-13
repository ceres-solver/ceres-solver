// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2013 Google Inc. All rights reserved.
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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_PUBLIC_COVARIANCE_H_
#define CERES_PUBLIC_COVARIANCE_H_

#include <utility>
#include <vector>
#include "ceres/internal/port.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/types.h"

namespace ceres {

class Problem;

namespace internal {
class CovarianceImpl;
}  // namespace internal

// WARNINGS
// ========
//
// 1. This is experimental code and the API WILL CHANGE before
//    release.
//
// 2. WARNING: It is very easy to use this class incorrectly without
//    understanding the underlying mathematics. Please read and
//    understand the documentation completely before attempting to use
//    this class.
//
// One way to the quality of the solution returned by a non-linear
// least squares solve is to analyze the covariance of the solution.
//
// Let us consider the non-linear regression problem
//
//   y = f(x) + N(0, I)
//
// i.e., the observation y is a random non-linear function of the
// independent variable x with mean f(x) and identity covariance. Then
// the maximum likelihood estimate of x given observations y is the
// solution to the non-linear least squares problem:
//
//  x* = arg min_x min_x |f(x)|^2
//
// And the covariance of x^* is given by
//
//  C(x*) = inverse[J'(x*)J(x*)]
//
// Here J(x*) is the Jacobian of f at x*. The above formula assumes
// that J(x'*) has full column rank.
//
// If J(x*) is rank deficient, then the covariance matrix C(x*) is
// also rank deficient and is given by
//
//  C(x*) =  pseudoinverse[J'(x*)J(x*)]
//
// WARNING
// =======
//
// Note that in the above, we assumed that the covariance
// matrix for y was identity. This is an important assumption. If this
// is not the case and we have
//
//  y = f(x) + N(0, S)
//
// Where S is a positive semi-definite matrix denoting the covariance
// of S, then the maximum likelihood problem to be solved is
//
//  x* = arg min_x min_x f'(x) inverse[S] f(x)
//
// and the corresponding covariance estimate of x^* is given by
//
//   C(x*) = inverse[J'(x*) inverse[S] J(x*)]
//
// So, if it is the case that the observations being fitted to have a
// covariance matrix not equal to identity, then it is the user's
// responsibility that the corresponding cost functions are correctly
// scaled, e.g. in the above case the cost function for this problem
// should evaluate S^{-1/2} f(x) instead of just f(x), where S^{-1/2}
// is the inverse square root of the covariance matrix S.
//
// This class allows the user to evaluate the covariance for a
// non-linear least squares problem and provides random access to its
// blocks. Since the computation of the covariance matrix involves
// computing the inverse of a potentially large matrix, this can
// involve a rather large amount of time and memory. However, it is
// usually the case that the user is only interested in a small part
// of the covariance matrix. Quite often just the block diagonal. This
// class allows the user to specify the parts of the covariance matrix
// that she is interested in and then uses this information to only
// compute and store those parts of the covariance matrix.
//
// LIMITATIONS
// ===========
//
// There are many.
//
// 1. SuiteSparse is required to compile this code. Preferably version
// 4.2.1 or later. More recent versions of SuiteSparse allow for a
// more efficient implementation of the matrix inversion algorithm.
//
// 2. Full rank Jacobian. As we noted above. If the jacobian is rank
// deficient, then the inverse of J'J is not defined and instead a
// pseudo inverse needs to be computed.
//
// The rank deficiency in J can be structural -- columns which are
// always known to be zero or numerical -- depending on the exact
// values in the Jacobian. This happens when the problem contains
// parameter blocks that are constant. This class correctly handles
// structural rank deficiency like that.
//
// Numerical rank deficiency, where the rank of the matrix cannot be
// predicted by its sparsity structure and requires looking at its
// numerical values is more complicated. Handling numerical rank
// deficiency requires the computation of the so called Singular Value
// Decomposition (SVD) of J'J. We do not know how to do this for large
// sparse matrices efficiently. We plan to add support for this using
// dense linear algebra, but in that case only small to moderate sized
// problems can be handled.
//
// 3. Gauge Invariance. This is related to rank deficiency of the
// Jacobian. In structure from motion (3D reconstruction) problems,
// the reconstruction is ambiguous upto a similarity transform. This
// is known as a Gauge Ambiguity. Covariance estimation in the
// presence of a gauge ambiguity also requires the use of a SVD and
// thus is not supported right now. See Morris, Kanatani & Kanade's
// work the subject.
//
// 4. Speed. This is a first implementation and the focus is on
// correctness. We do not expect the speed to be particularly good. We
// plan to work on improving the speed as we understand the common
// usage patterns of this class.
//
// Example usage
//
//  double x[3];
//  double y[2];
//
//  Problem problem;
//  problem.AddParameterBlock(x, 3);
//  problem.AddParameterBlock(y, 2);
//  <Build Problem>
//  <Solve Problem>
//
//  Covariance::Options options;
//  Covariance covariance(options);
//
//  vector<pair<const double*, const double*> > covariance_blocks;
//  covariance_blocks.push_back(make_pair(x, x));
//  covariance_blocks.push_back(make_pair(y, y));
//  covariance_blocks.push_back(make_pair(x, y));
//
//  CHECK(covariance.Compute(covariance_blocks, &problem));
//
//  double covariance_xx[3 * 3];
//  double covariance_yy[2 * 2];
//  double covariance_xy[3 * 2];
//  covariance.GetCovarianceBlock(x, x, covariance_xx)
//  covariance.GetCovarianceBlock(y, y, covariance_yy)
//  covariance.GetCovarianceBlock(x, y, covariance_xy)
class Covariance {
 public:
  struct Options {
    Options()
        : num_threads(1),
          apply_loss_function(true) {
    }

    // Number of threads to be used for evaluating the Jacobian and
    // estimation of covariance. Currently only the Jacobian
    // evaluation is multi-threaded when the compiler supports OpenMP.
    int num_threads;

    // Even though the residual blocks in the problem may contain loss
    // functions, setting apply_loss_function to false will turn off
    // the application of the loss function to the output of the cost
    // function. Loss functions are useful for robustifying the cost
    // functions during optimization. (I do not think they should be
    // applied to the Jacobian for covariance estimation).
    //
    // The user should generally speaking set this to false.
    //
    // TODO(sameeragarwal): Figure out the definitive answer to this.
    bool apply_loss_function;
  };

  explicit Covariance(const Options& options);

  bool Compute(
      const vector<pair<const double*, const double*> >& covariance_blocks,
      Problem* problem);

  // Compute must be called before the first call to
  // GetCovarianceBlock. covariance_block must point to a memory
  // location that can store a parameter_block1_size x
  // parameter_block2_size matrix. The returned covariance will be a
  // row-major matrix.
  bool GetCovarianceBlock(const double* parameter_block1,
                          const double* parameter_block2,
                          double* covariance_block) const;

 private:
  internal::scoped_ptr<internal::CovarianceImpl> impl_;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_COVARIANCE_H_
