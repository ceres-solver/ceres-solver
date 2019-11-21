// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2019 Google Inc. All rights reserved.
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
// Author: darius.rueckert@fau.de (Darius Rueckert)
//
// TODO: Documentation
#ifndef CERES_PUBLIC_EXPRESSION_REF_H_
#define CERES_PUBLIC_EXPRESSION_REF_H_

#include <string>
#include "ceres/jet.h"
#include "expression.h"

namespace ceres {
namespace internal {

// This class represents a scalar value that creates new expressions during
// evaluation. ExpressionRef can be used as template parameter for cost functors
// and Jets.
//
// ExpressionRef should be passed by value.
struct ExpressionRef {
  ExpressionRef() = default;

  // Create a compile time constant expression directly from a double value.
  // This is important so that we can write T(3.14) in our code and
  // it's automatically converted to the correct expression.
  //
  // This constructor is implicit, because the line
  //    T a(0);
  // must work for T = Jet<ExpressionRef>.
  ExpressionRef(double compile_time_constant);

  // By adding this deleted constructor we can detect invalid usage of
  // ExpressionRef. ExpressionRef must only be created from constexpr doubles.
  //
  // If you get a compile error here, you have probably written something like:
  //   T x = local_variable_;
  // Change this into:
  //   T x = CERES_LOCAL_VARIABLE(local_variable_);
  ExpressionRef(double&) = delete;

  // Create an ASSIGNMENT expression from other to this.
  //
  // For example:
  //   a = b;        // With a.id = 5 and b.id = 3
  // will generate the following assignment:
  //   v_5 = v_3;
  //
  // If this (lhs) ExpressionRef is currently not pointing to a variable
  // (id==invalid), then we can eliminate the assignment by just letting "this"
  // point to the same variable as "other".
  //
  // Example:
  //   a = b;       // With a.id = invalid and b.id = 3
  // will generate NO expression, but after this line the following will be
  // true:
  //    a.id == b.id == 3
  //
  // If 'other' is not pointing to a variable (id==invalid), we found an
  // uninitialized assignment, which is handled as an error.
  ExpressionRef(const ExpressionRef& other);
  ExpressionRef& operator=(const ExpressionRef& other);

  // Compound operators
  ExpressionRef& operator+=(ExpressionRef x);
  ExpressionRef& operator-=(ExpressionRef x);
  ExpressionRef& operator*=(ExpressionRef x);
  ExpressionRef& operator/=(ExpressionRef x);

  bool IsInitialized() const { return id != kInvalidExpressionId; }

  // The index into the ExpressionGraph data array.
  ExpressionId id = kInvalidExpressionId;

  static ExpressionRef Create(ExpressionId id);
};

// Arithmetic Operators
ExpressionRef operator-(ExpressionRef x);
ExpressionRef operator+(ExpressionRef x);
ExpressionRef operator+(ExpressionRef x, ExpressionRef y);
ExpressionRef operator-(ExpressionRef x, ExpressionRef y);
ExpressionRef operator*(ExpressionRef x, ExpressionRef y);
ExpressionRef operator/(ExpressionRef x, ExpressionRef y);

// Functions

// Helper function to create a function call expression.
// Users can generate code for their own custom functions by adding an overload
// for ExpressionRef that maps to MakeFunctionCall. See below for examples.
ExpressionRef MakeFunctionCall(const std::string& name,
                               const std::vector<ExpressionRef>& params);

#define CERES_DEFINE_UNARY_FUNCTION_CALL(name) \
  inline ExpressionRef name(ExpressionRef x) { \
    return MakeFunctionCall(#name, {x});       \
  }
#define CERES_DEFINE_BINARY_FUNCTION_CALL(name)                 \
  inline ExpressionRef name(ExpressionRef x, ExpressionRef y) { \
    return MakeFunctionCall(#name, {x, y});                     \
  }
CERES_DEFINE_UNARY_FUNCTION_CALL(abs);
CERES_DEFINE_UNARY_FUNCTION_CALL(acos);
CERES_DEFINE_UNARY_FUNCTION_CALL(asin);
CERES_DEFINE_UNARY_FUNCTION_CALL(atan);
CERES_DEFINE_UNARY_FUNCTION_CALL(cbrt);
CERES_DEFINE_UNARY_FUNCTION_CALL(ceil);
CERES_DEFINE_UNARY_FUNCTION_CALL(cos);
CERES_DEFINE_UNARY_FUNCTION_CALL(cosh);
CERES_DEFINE_UNARY_FUNCTION_CALL(exp);
CERES_DEFINE_UNARY_FUNCTION_CALL(exp2);
CERES_DEFINE_UNARY_FUNCTION_CALL(floor);
CERES_DEFINE_UNARY_FUNCTION_CALL(log);
CERES_DEFINE_UNARY_FUNCTION_CALL(log2);
CERES_DEFINE_UNARY_FUNCTION_CALL(sin);
CERES_DEFINE_UNARY_FUNCTION_CALL(sinh);
CERES_DEFINE_UNARY_FUNCTION_CALL(sqrt);
CERES_DEFINE_UNARY_FUNCTION_CALL(tan);
CERES_DEFINE_UNARY_FUNCTION_CALL(tanh);

CERES_DEFINE_BINARY_FUNCTION_CALL(atan2);
CERES_DEFINE_BINARY_FUNCTION_CALL(pow);

#undef CERES_DEFINE_UNARY_FUNCTION_CALL
#undef CERES_DEFINE_BINARY_FUNCTION_CALL

// This additonal type is required, so that we can detect invalid conditions
// during compile time. For example, the following should create a compile time
// error:
//
//   ExpressionRef a(5);
//   CERES_IF(a){           // Error: Invalid conversion
//   ...
//
// Following will work:
//
//   ExpressionRef a(5), b(7);
//   ComparisonExpressionRef c = a < b;
//   CERES_IF(c){
//   ...
struct ComparisonExpressionRef {
  ExpressionId id;
  explicit ComparisonExpressionRef(ExpressionRef ref) : id(ref.id) {}
};

ExpressionRef Ternary(ComparisonExpressionRef c,
                      ExpressionRef a,
                      ExpressionRef b);

// Comparison operators
ComparisonExpressionRef operator<(ExpressionRef a, ExpressionRef b);
ComparisonExpressionRef operator<=(ExpressionRef a, ExpressionRef b);
ComparisonExpressionRef operator>(ExpressionRef a, ExpressionRef b);
ComparisonExpressionRef operator>=(ExpressionRef a, ExpressionRef b);
ComparisonExpressionRef operator==(ExpressionRef a, ExpressionRef b);
ComparisonExpressionRef operator!=(ExpressionRef a, ExpressionRef b);

// Logical Operators
ComparisonExpressionRef operator&&(ComparisonExpressionRef a,
                                   ComparisonExpressionRef b);
ComparisonExpressionRef operator||(ComparisonExpressionRef a,
                                   ComparisonExpressionRef b);
ComparisonExpressionRef operator!(ComparisonExpressionRef a);

// This struct is used to mark numbers which are constant over
// multiple invocations but can differ between instances.
template <typename T>
struct InputAssignment {
  using ReturnType = T;
  static inline ReturnType Get(double v, const char* /* unused */) { return v; }
};

template <>
struct InputAssignment<ExpressionRef> {
  using ReturnType = ExpressionRef;
  static inline ReturnType Get(double /* unused */, const char* name) {
    return ExpressionRef::Create(Expression::CreateInputAssignment(name));
  }
};

template <typename G, int N>
struct InputAssignment<Jet<G, N>> {
  using ReturnType = Jet<G, N>;
  static inline Jet<G, N> Get(double v, const char* /* unused */) {
    return Jet<G, N>(v);
  }
};

template <int N>
struct InputAssignment<Jet<ExpressionRef, N>> {
  using ReturnType = Jet<ExpressionRef, N>;
  static inline ReturnType Get(double /* unused */, const char* name) {
    // Note: The scalar value of v will be thrown away, because we don't need it
    // during code generation.
    return Jet<ExpressionRef, N>(
        ExpressionRef::Create(Expression::CreateInputAssignment(name)));
  }
};

template <typename T>
inline typename InputAssignment<T>::ReturnType MakeInputAssignment(
    double v, const char* name) {
  return InputAssignment<T>::Get(v, name);
}

// This macro should be used for local variables in cost functors. Using local
// variables directly, will compile their current value into the code.
// Example:
//  T x = CERES_LOCAL_VARIABLE(observed_x_);
#define CERES_LOCAL_VARIABLE(_v) \
  ceres::internal::MakeInputAssignment<T>(_v, #_v)

inline ExpressionRef MakeParameter(const std::string& name) {
  return ExpressionRef::Create(Expression::CreateInputAssignment(name));
}
inline ExpressionRef MakeOutput(ExpressionRef v, const std::string& name) {
  return ExpressionRef::Create(Expression::CreateOutputAssignment(v.id, name));
}

// The CERES_CODEGEN macro is defined by the build system only during code
// generation. In all other cases the CERES_IF/ELSE macros just expand to the
// if/else keywords.
#ifdef CERES_CODEGEN
#define CERES_IF(condition_) Expression::CreateIf((condition_).id);
#define CERES_ELSE Expression::CreateElse();
#define CERES_ENDIF Expression::CreateEndIf();
#else
// clang-format off
#define CERES_IF(condition_) if (condition_) {
#define CERES_ELSE } else {
#define CERES_ENDIF }
// clang-format on
#endif

}  // namespace internal

// See jet.h for more info on this type.
template <>
struct ComparisonReturnType<internal::ExpressionRef> {
  using type = internal::ComparisonExpressionRef;
};

}  // namespace ceres
#endif
