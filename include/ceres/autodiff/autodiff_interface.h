// Author: thadh@google.com (Thad Hughes)

// This automatic differentiation (autodiff) library is designed to
// automatically and exactly compute the partial derivatives of arbitrarily
// complex arithmetic C++ code.  The user of this library is expected to
// write C++ code that performs a mathematical computation, doing math
// on a template type instead of on floats or doubles, and the library can
// compute the partial derivatives of the resulting values with respect to
// each of the variables of interest.  It does this by instantiating the
// templated code with a special type called a "jet," which keeps track
// of the derivative relationships between the variables while the computation
// is executed.
//
// This file provides the "interface" to the automatic differentiation
// framework without actually providing automatic differentiation.
// It defines a set of functions that will work on floats and doubles,
// and which will also work on the jet types used for autodiff.
// Including this file does not introduce a dependency on any particular
// jet type, but allows you to write templated functions that can
// be instantiated on both floating point types and jet types.
//
// In addition to the standard arithmetic and comparison operators, jet types
// support these standard mathematical functions:
// * abs
// * log
// * exp
// * sqrt
// * cos
// * acos
// * sin
// * asin
// * tanh
// * pow
//
// This file also provides a few additional common functions whose derivatives
// can be algebraically simplified for faster execuation:
// * square
// * logistic
//
// As an example, after including this file, you can write the following
// function (copied from jet_tester.h):
//
//  // ComputeAngle returns the angle between two vectors, in radians.
//  // It is an example of a templated function that can be instantiated
//  // with either T=double (for normal computation), or with T being some Jet
//  // type (for automatic differentiation).  Note that it is somewhat
//  // nontrivial to differentiate by hand w.r.t. the entries of the
//  // two vectors.
//  template <typename T>
//  T ComputeAngle(const vector<T>& v1, const vector<T>& v2) {
//    CHECK_EQ(v1.size(), v2.size());
//    T dot_product(0), v1_squared_norm(0), v2_squared_norm(0);
//    for (int i = 0; i < v1.size(); ++i) {
//      dot_product += v1[i] * v2[i];
//
//      // Below, we could have written this:
//      // v1_norm += v1[i] * v1[i];
//      // But using speech::math::autodiff::square results in faster
//      // computation of the derivatives.
//      using speech::math::autodiff::square;
//      v1_squared_norm += square(v1[i]);
//      v2_squared_norm += square(v2[i]);
//    }
//
//    // Autodiffed functions can have comparison operators, although this can
//    // make them only piecewise differentiable.
//    const double kEpsilon = 1e-7;
//    if (v1_squared_norm < kEpsilon || v2_squared_norm < kEpsilon) {
//      return T(0);
//    }
//    // MakeIntermediateVar is optional, but may reduce the cost of using
//    // the variables in future expressions (while increasing the cost
//    // of backpropagation through the intermediate quantities).
//    // See comments on declaration of MakeIntermediateVar in
//    // autodiff_interface.h for more information.
//    using speech::math::autodiff::MakeIntermediateVar;
//    MakeIntermediateVar(&dot_product);
//    MakeIntermediateVar(&v1_squared_norm);
//    MakeIntermediateVar(&v2_squared_norm);
//    // We can write "using std::sqrt;" below because when this function
//    // is instantiated with Jet arguments, C++'s argument-dependent
//    // name lookup kicks in and finds the speech::math::autodiff::sqrt.
//    // See: http://en.wikipedia.org/wiki/Argument-dependent_name_lookup
//    // Note that we CANNOT write this: "std::sqrt(v1_squared_norm)" because
//    // then argument dependent name lookup does not apply.
//    using std::sqrt;
//    T cos_theta = dot_product /
//        (sqrt(v1_squared_norm) * sqrt(v2_squared_norm));
//
//    // Mathematically, cos_theta should always lie in the range [-1, 1],
//    // but floating point roundoff can result in values slightly outside
//    // that range, which cause the acos function to return NaN.
//    using std::min;
//    using std::max;
//    cos_theta = min(T(1), max(T(-1), cos_theta));
//    MakeIntermediateVar(&cos_theta);
//    using std::acos;
//    return acos(cos_theta);
//  }
//
// Once this function is defined, it can be instantiated with T=double,
// but it can also be instantiated with a jet type to enable autodiff of the
// value it computes w.r.t. other variables.

#ifndef SPEECH_PORTABLE_MATH_AUTODIFF_AUTODIFF_INTERFACE_H_
#define SPEECH_PORTABLE_MATH_AUTODIFF_AUTODIFF_INTERFACE_H_

#include <cmath>

namespace ceres {

// MakeIntermediateVar(&x) is a special function that signals to the autodiff
// framework that the argument x is finished receiving fan-in from other
// variables and should be replaced by an intermediate variable through which
// partial derivatives should be back-propagated.  Introducing an intermediate
// variable to represent the Jet's current value can reduce the cost of
// expressions involving the Jet, since the Jet will have exactly one
// parent immediately following the call to MakeIntermediateVar.  However,
// this speedup in the forward direction comes at a cost of more expensive
// back-propagation in the reverse direction, since there are more
// intermediate variables to back-propagate through.
//
// MakeIntermediateVar is a necessary part of the API because
// deciding which quantities in a computational graph should become
// intermediate variables is an NP-complete problem in general:
// See: http://en.wikipedia.org/wiki/Automatic_differentiation#Beyond_forward_and_reverse_accumulation
//
// However, a reasonable intuition is that MakeIntermediateVar should be
// invoked after a Jet is finished receiving "fan-in" (accumulating
// results from other Jets).  This typically happens after an accumulation
// loop finishes executing.  The loop accumulates fan-in, and after the
// loop completes is an appropriate place for MakeIntermediateVar.
//
// MakeIntermediateVar is a no-op for doubles and for Jet types that don't
// support reverse mode autodiff.
//
// MakeIntermediateVar returns true if an intermediate variable was actually
// constructed.
template<typename T> struct IntermediateVar {
  // Don't instantiate on random types.
};
template<> struct IntermediateVar<double> {
  static bool Make(double* x) { return false; }
};
template<> struct IntermediateVar<float> {
  static bool Make(float* x) { return false; }
};
template <typename T>
inline bool MakeIntermediateVar(T* t) {
  return IntermediateVar<T>::Make(t);
}

// ZeroOrder provides an interface to inspect the zero-order part of
// a Jet type. For floats, and doubles, it simply provides their actual
// value, since they are purely zero-order values.  The ZeroOrder template
// is also specialized by Jet implementations, providing an API to
// directly examine and modify the zero-order (value) part of the Jet.
template <typename T>
struct ZeroOrder {
  // Don't instantiate on random types.
};
template <>
struct ZeroOrder<double> {
  // The typedef below is a part of the public API, and allows API users
  // to query the zero-order type of the Jets they are using (which is
  // typically float or double).
  typedef double ZeroOrderType;
  static const ZeroOrderType& Get(const double& x) { return x; }
  static void Set(double* x, const ZeroOrderType& zero_order) {
    *x = zero_order;
  }
};
template <>
struct ZeroOrder<float> {
  typedef float ZeroOrderType;
  static const ZeroOrderType& Get(const float& x) { return x; }
  static void Set(float* x, const ZeroOrderType& zero_order) {
    *x = zero_order;
  }
};
template <typename T>
inline const typename ZeroOrder<T>::ZeroOrderType& GetZeroOrder(const T& t) {
  return ZeroOrder<T>::Get(t);
}
template <typename T>
inline void SetZeroOrder(
    T* t, const typename ZeroOrder<T>::ZeroOrderType& zero_order) {
  ZeroOrder<T>::Set(t, zero_order);
}

// Users of the autodiff API do not need to use the square function
// defined below, since autodiff can correctly compute the derivative for
// "x * x".  However, since square is a commonly used operation, and the
// derivative of square can be computed more efficiently than the general case
// of "x * x", we define a function that can be used on primitive types.
// This function is then overloaded for Jet arguments by the autodiff library
// to perform the more efficient computation.
// Note that square is a unary function.  If we have: y = square(x);
// Then: y' = 2*x*x'
// However, if we expressed the same thing with operator* (a binary function),
// we would have: y = x * x;
// And thus: y' = x*x' + x'*x
// This is algebraically equivalent, but more expensive to compute.
inline double square(double x) { return x * x; }
inline float square(float x) { return x * x; }

// Similar to square, defined above, the logistic function can be
// differentiated more efficiently than by applying autodiff to the
// expression that defines it. This function is also overloaded for Jet
// types, resulting in a faster computation of the derivative.
inline double logistic(double x) { return 1 / (1 + exp(-x)); }
inline float logistic(float x) { return 1 / (1 + exp(-x)); }

}  // namespace autodiff

#endif  // SPEECH_PORTABLE_MATH_AUTODIFF_AUTODIFF_INTERFACE_H_
