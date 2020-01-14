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

#include "ceres/codegen/internal/expression.h"
#include "ceres/codegen/internal/types.h"

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

  // Copy construction/assignment creates an ASSIGNMENT expression from
  // 'other' to 'this'.
  //
  // For example:
  //   a = b;        // With a.id = 5 and b.id = 3
  // will generate the following assignment:
  //   v_5 = v_3;
  //
  //  If 'this' ExpressionRef is currently not pointing to a variable
  // (id==invalid), then an assignment to a new variable is generated. Example:
  //    T a = 5;
  //    T b;
  //    b = a;  // During the assignment 'b' is invalid
  //
  // The right hand side of the assignment (= the argument 'other') must be
  // valid in every case. The following code will result in an error.
  //   T a;
  //   T b = a;  // Error: Uninitialized assignment
  ExpressionRef(const ExpressionRef& other);
  ExpressionRef& operator=(const ExpressionRef& other);

  // Compound operators
  ExpressionRef& operator+=(const ExpressionRef& x);
  ExpressionRef& operator-=(const ExpressionRef& x);
  ExpressionRef& operator*=(const ExpressionRef& x);
  ExpressionRef& operator/=(const ExpressionRef& x);

  bool IsInitialized() const { return id != kInvalidExpressionId; }

  // The index into the ExpressionGraph data array.
  ExpressionId id = kInvalidExpressionId;

  static ExpressionRef Create(ExpressionId id);
};

// A helper function which calls 'InsertBack' on the currently active graph.
// This wrapper also checks if StartRecordingExpressions was called. See
// ExpressionGraph::InsertBack for more information.
ExpressionRef AddExpressionToGraph(const Expression& expression);

// Arithmetic Operators
ExpressionRef operator-(const ExpressionRef& x);
ExpressionRef operator+(const ExpressionRef& x);
ExpressionRef operator+(const ExpressionRef& x, const ExpressionRef& y);
ExpressionRef operator-(const ExpressionRef& x, const ExpressionRef& y);
ExpressionRef operator*(const ExpressionRef& x, const ExpressionRef& y);
ExpressionRef operator/(const ExpressionRef& x, const ExpressionRef& y);

// Functions
#define CERES_DEFINE_UNARY_FUNCTION_CALL(name)                \
  inline ExpressionRef name(const ExpressionRef& x) {         \
    return AddExpressionToGraph(                              \
        Expression::CreateScalarFunctionCall(#name, {x.id})); \
  }
#define CERES_DEFINE_BINARY_FUNCTION_CALL(name)                               \
  inline ExpressionRef name(const ExpressionRef& x, const ExpressionRef& y) { \
    return AddExpressionToGraph(                                              \
        Expression::CreateScalarFunctionCall(#name, {x.id, y.id}));           \
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
  explicit ComparisonExpressionRef(const ExpressionRef& ref) : id(ref.id) {}
};

ExpressionRef Ternary(const ComparisonExpressionRef& c,
                      const ExpressionRef& x,
                      const ExpressionRef& y);

// Comparison operators
ComparisonExpressionRef operator<(const ExpressionRef& x,
                                  const ExpressionRef& y);
ComparisonExpressionRef operator<=(const ExpressionRef& x,
                                   const ExpressionRef& y);
ComparisonExpressionRef operator>(const ExpressionRef& x,
                                  const ExpressionRef& y);
ComparisonExpressionRef operator>=(const ExpressionRef& x,
                                   const ExpressionRef& y);
ComparisonExpressionRef operator==(const ExpressionRef& x,
                                   const ExpressionRef& y);
ComparisonExpressionRef operator!=(const ExpressionRef& x,
                                   const ExpressionRef& y);

// Logical Operators
ComparisonExpressionRef operator&&(const ComparisonExpressionRef& x,
                                   const ComparisonExpressionRef& y);
ComparisonExpressionRef operator||(const ComparisonExpressionRef& x,
                                   const ComparisonExpressionRef& y);
ComparisonExpressionRef operator&(const ComparisonExpressionRef& x,
                                  const ComparisonExpressionRef& y);
ComparisonExpressionRef operator|(const ComparisonExpressionRef& x,
                                  const ComparisonExpressionRef& y);
ComparisonExpressionRef operator!(const ComparisonExpressionRef& x);

#define CERES_DEFINE_UNARY_LOGICAL_FUNCTION_CALL(name)          \
  inline ComparisonExpressionRef name(const ExpressionRef& x) { \
    return ComparisonExpressionRef(AddExpressionToGraph(        \
        Expression::CreateLogicalFunctionCall(#name, {x.id}))); \
  }

CERES_DEFINE_UNARY_LOGICAL_FUNCTION_CALL(isfinite);
CERES_DEFINE_UNARY_LOGICAL_FUNCTION_CALL(isinf);
CERES_DEFINE_UNARY_LOGICAL_FUNCTION_CALL(isnan);
CERES_DEFINE_UNARY_LOGICAL_FUNCTION_CALL(isnormal);

#undef CERES_DEFINE_UNARY_LOGICAL_FUNCTION_CALL

template <>
struct InputAssignment<ExpressionRef> {
  using ReturnType = ExpressionRef;
  static inline ReturnType Get(double /* unused */, const char* name) {
    // Note: The scalar value of v will be thrown away, because we don't need it
    // during code generation.
    return AddExpressionToGraph(Expression::CreateInputAssignment(name));
  }
};

template <typename T>
inline typename InputAssignment<T>::ReturnType MakeInputAssignment(
    double v, const char* name) {
  return InputAssignment<T>::Get(v, name);
}

inline ExpressionRef MakeParameter(const std::string& name) {
  return AddExpressionToGraph(Expression::CreateInputAssignment(name));
}
inline ExpressionRef MakeOutput(const ExpressionRef& v,
                                const std::string& name) {
  return AddExpressionToGraph(Expression::CreateOutputAssignment(v.id, name));
}

}  // namespace internal

template <>
struct ComparisonReturnType<internal::ExpressionRef> {
  using type = internal::ComparisonExpressionRef;
};

}  // namespace ceres
#endif
