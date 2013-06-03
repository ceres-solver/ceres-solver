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
// One way to assess the quality of the solution returned by a
// non-linear least squares solve is to analyze the covariance of the
// solution.
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
//  x* = arg min_x |f(x)|^2
//
// And the covariance of x* is given by
//
//  C(x*) = inverse[J'(x*)J(x*)]
//
// Here J(x*) is the Jacobian of f at x*. The above formula assumes
// that J(x*) has full column rank.
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
// of y, then the maximum likelihood problem to be solved is
//
//  x* = arg min_x f'(x) inverse[S] f(x)
//
// and the corresponding covariance estimate of x* is given by
//
//  C(x*) = inverse[J'(x*) inverse[S] J(x*)]
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
// blocks. The computation assumes that the CostFunctions compute
// residuals such that their covariance is identity.
//
// Since the computation of the covariance matrix involves computing
// the inverse of a potentially large matrix, this can involve a
// rather large amount of time and memory. However, it is usually the
// case that the user is only interested in a small part of the
// covariance matrix. Quite often just the block diagonal. This class
// allows the user to specify the parts of the covariance matrix that
// she is interested in and then uses this information to only compute
// and store those parts of the covariance matrix.
//
// Rank of the Jacobian
// ====================
// As we noted above, if the jacobian is rank deficient, then the
// inverse of J'J is not defined and instead a pseudo inverse needs to
// be computed.
//
// The rank deficiency in J can be structural -- columns which are
// always known to be zero or numerical -- depending on the exact
// values in the Jacobian. This happens when the problem contains
// parameter blocks that are constant. This class correctly handles
// structural rank deficiency like that.
//
// Numerical rank deficiency, where the rank of the matrix cannot be
// predicted by its sparsity structure and requires looking at its
// numerical values is more complicated. Here again there are two
// cases.
//
//   a. The rank deficiency arises from overparameterization. e.g., a
//   four dimensional quaternion used to parameterize SO(3), which is
//   a three dimensional manifold. In cases like this, the user should
//   use an appropriate LocalParameterization. Not only will this lead
//   to better numerical behaviour of the Solver, it will also expose
//   the rank deficiency to the Covariance object so that it can
//   handle it correctly.
//
//   b. More general numerical rank deficiency in the Jacobian
//   requires the computation of the so called Singular Value
//   Decomposition (SVD) of J'J. We do not know how to do this for
//   large sparse matrices efficiently. For small and moderate sized
//   problems this is done using dense linear algebra.
//
// Gauge Invariance
// ----------------
// In structure from motion (3D reconstruction) problems, the
// reconstruction is ambiguous upto a similarity transform. This is
// known as a Gauge Ambiguity. Handling Gauges correctly requires the
// use of SVD or custom inversion algorithms. For small problems the
// user can use the dense algorithm. For more details see Morris,
// Kanatani & Kanade's work the subject.
//
// Example Usage
// =============
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
//
class Covariance {
 public:
  struct Options {
    Options()
        : num_threads(1),
#ifndef CERES_NO_SUITESPARSE
          use_dense_linear_algebra(false),
#else
          use_dense_linear_algebra(true),
#endif
          min_reciprocal_condition_number(1e-16),
          null_space_rank(-1),
          apply_loss_function(true) {
    }

    // Number of threads to be used for evaluating the Jacobian and
    // estimation of covariance.
    int num_threads;

    // Use Eigen's JacobiSVD algorithm to compute the covariance
    // instead of SuiteSparse. This is a very accurate but slow
    // algorithm. The up side is that it can handle numerically rank
    // deficient jacobians. This option only makes sense for small to
    // moderate sized problems.
    bool use_dense_linear_algebra;

    // If the Jacobian matrix is near singular, then inverting J'J
    // will result in unreliable results.
    //
    // The reciprocal condition number of a matrix is a measure of
    // ill-conditioning or how close the matrix is to being
    // singular/rank deficient. It is defined as the ratio of the
    // smallest eigenvalue of the matrix to the largest eigenvalue.
    //
    // Matrices with condition number lower than
    // min_reciprocal_condition_number are considered rank deficient.
    //
    // Depending on the value of use_dense_linear_algebra this may
    // have further consequences on the covariance estimation process.
    //
    //  1. use_dense_linear_algebra = false
    //
    //  If the reciprocal_condition_number of J'J is less than
    //  min_reciprocal_condition_number, Covariance::Compute() will
    //  fail and return false.
    //
    //  2. use_dense_linear_algebra = true
    //
    //  When dense covariance estimation is being used, then rank
    //  deficiency/singularity of the Jacobian can be handled in a
    //  more sophisticated manner.
    //
    //  If null_space_rank = -1, then instead of computing the inverse
    //  of J'J, the Moore-Penrose Pseudoinverse is computed. If
    //  (lambda_i, e_i) are eigenvalue and eigenvector pairs of J'J.
    //
    //    pseudoinverse[J'J] = sum_i e_i e_i' / lambda_i
    //
    //  if lambda_i / lambda_max >= min_reciprocal_condition_number.
    //
    //  If null_space_rank is non-negative, then the smallest
    //  null_space_rank eigenvalue/eigenvectors are dropped
    //  irrespective of the magnitude of lambda_i. If the ratio of the
    //  smallest non-zero eigenvalue to the largest eigenvalue in the
    //  truncated matrix is still below
    //  min_reciprocal_condition_number, then the
    //  Covariance::Compute() will fail and return false.
    double min_reciprocal_condition_number;

    // When use_dense_linear_algebra is true, null_space_rank
    // determines how many of the smallest eigenvectors of J'J are
    // dropped when computing the pseudoinverse.
    //
    //  If null_space_rank = -1, then instead of computing the inverse
    //  of J'J, the Moore-Penrose Pseudoinverse is computed. If
    //  (lambda_i, e_i) are eigenvalue and eigenvector pairs of J'J.
    //
    //    pseudoinverse[J'J] = sum_i e_i e_i' / lambda_i
    //
    //  if lambda_i / lambda_max >= min_reciprocal_condition_number.
    //
    //  If null_space_rank is non-negative, then the smallest
    //  null_space_rank eigenvalue/eigenvectors are dropped
    //  irrespective of the magnitude of lambda_i. If the ratio of the
    //  smallest non-zero eigenvalue to the largest eigenvalue in the
    //  truncated matrix is still below
    //  min_reciprocal_condition_number, then the
    //  Covariance::Compute() will fail and return false.
    int null_space_rank;

    // Even though the residual blocks in the problem may contain loss
    // functions, setting apply_loss_function to false will turn off
    // the application of the loss function to the output of the cost
    // function and in turn its effect on the covariance.
    //
    // TODO(sameergaarwal): Expand this based on Jim's experiments.
    bool apply_loss_function;
  };

  explicit Covariance(const Options& options);
  ~Covariance();

  // Compute a part of the covariance matrix.
  //
  // The vector covariance_blocks, indexes into the covariance matrix
  // block-wise using pairs of parameter blocks. This allows the
  // covariance estimation algorithm to only compute and store these
  // blocks.
  //
  // Since the covariance matrix is symmetric, if the user passes
  // (block1, block2), then GetCovarianceBlock can be called with
  // block1, block2 as well as block2, block1.
  //
  // covariance_blocks cannot contain duplicates. Bad things will
  // happen if they do.
  //
  // Note that the list of covariance_blocks is only used to determine
  // what parts of the covariance matrix are computed. The full
  // Jacobian is used to do the computation, i.e. they do not have an
  // impact on what part of the Jacobian is used for computation.
  //
  // The return value indicates the success or failure of the
  // covariance computation.
  bool Compute(
      const vector<pair<const double*, const double*> >& covariance_blocks,
      Problem* problem);

  // Return the block of the covariance matrix corresponding to
  // parameter_block1 and parameter_block2.
  //
  // Compute must be called before the first call to
  // GetCovarianceBlock and the pair <parameter_block1,
  // parameter_block2> OR the pair <parameter_block2,
  // parameter_block1> must have been present in the vector
  // covariance_blocks when Compute was called. Otherwise
  // GetCovarianceBlock will return false.
  //
  // covariance_block must point to a memory location that can store a
  // parameter_block1_size x parameter_block2_size matrix. The
  // returned covariance will be a row-major matrix.
  bool GetCovarianceBlock(const double* parameter_block1,
                          const double* parameter_block2,
                          double* covariance_block) const;

 private:
  internal::scoped_ptr<internal::CovarianceImpl> impl_;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_COVARIANCE_H_
