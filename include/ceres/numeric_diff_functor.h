// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2014 Google Inc. All rights reserved.
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
//
// ====================
// NOTICE NOTICE NOTICE
// ====================
//
// This class has been removed from Ceres Solver. Its API was broken,
// and the implementation was adding a layer of abstraction for no
// good reason.
//
// If you have a functor which evaluates a function, and you wish to
// use it as part of automatic differentiation, then the right way to
// do this is in two steps.
//
// a. Create NumericDiffCostFunction using the functor.
// b. Wrap the NumericDiffCostFunction using a CostFunctionToFunctor object.
//
// For example, let us assume that
//
//  struct IntrinsicProjection
//    IntrinsicProjection(const double* observations);
//    bool operator()(const double* calibration,
//                    const double* point,
//                    double* residuals);
//  };
//
// is a functor that implements the projection of a point in its local
// coordinate system onto its image plane and subtracts it from the
// observed point projection.
//
// Now we would like to compose the action of this functor with the
// action of camera extrinsics, i.e., rotation and translation, which
// is given by the following templated function
//
//   template<typename T>
//   void RotateAndTranslatePoint(const T* rotation,
//                                const T* translation,
//                                const T* point,
//                                T* result);
//
// To compose the extrinsics and intrinsics, we can construct a
// CameraProjection functor as follows.
//
// struct CameraProjection {
//    typedef CostFunctionToFunctor<2, 5, 3> IntrinsicProjectionFunctor;
//
//   CameraProjection(double* observation) {
//     intrinsic_projection_(
//       new NumericDiffCostFunction<IntrinsicProjection, CENTRAL, 2, 5, 3>(
//         new IntrinsicProjection(observations)) {
//   }
//
//   template <typename T>
//   bool operator()(const T* rotation,
//                   const T* translation,
//                   const T* intrinsics,
//                   const T* point,
//                   T* residuals) const {
//     T transformed_point[3];
//     RotateAndTranslatePoint(rotation, translation, point, transformed_point);
//     return intrinsic_projection_(intrinsics, transformed_point, residual);
//   }
//
//  private:
//   IntrinsicProjectionFunctor intrinsic_projection_;
// };
//
// Here, we made the choice of using CENTRAL differences to compute
// the jacobian of IntrinsicProjection.
//
// Now, we are ready to construct an automatically differentiated cost
// function as
//
// CostFunction* cost_function =
//    new AutoDiffCostFunction<CameraProjection, 2, 3, 3, 5>(
//        new CameraProjection(observations));
//
// cost_function now seamlessly integrates automatic differentiation
// of RotateAndTranslatePoint with a numerically differentiated
// version of IntrinsicProjection.
