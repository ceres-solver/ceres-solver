

#ifndef CERES_PUBLIC_CONDITIONAL_EXPRESSIONS_H_
#define CERES_PUBLIC_CONDITIONAL_EXPRESSIONS_H_

#include "ceres/internal/expression_operators.h"

// This file is only here to discuss how we want to handle conditions.
namespace ceres {

// The original pow function from Jet.h as reference
template <typename T, int N>
inline Jet<T, N> pow_original(const Jet<T, N>& f, const Jet<T, N>& g) {
  if (f.a == 0 && g.a >= 1) {
    // Handle cases 2 and 3.
    if (g.a > 1) {
      return Jet<T, N>(T(0.0));
    }
    return f;
  }
  if (f.a < 0 && g.a == floor(g.a)) {
    // Handle cases 7 and 8.
    T const tmp = g.a * pow(f.a, g.a - T(1.0));
    Jet<T, N> ret(pow(f.a, g.a), tmp * f.v);
    for (int i = 0; i < N; i++) {
      if (g.v[i] != T(0.0)) {
        // Return a NaN when g.v != 0.
        ret.v[i] = std::numeric_limits<T>::quiet_NaN();
      }
    }
    return ret;
  }
  // Handle the remaining cases. For cases 4,5,6,9 we allow the log() function
  // to generate -HUGE_VAL or NaN, since those cases result in a nonfinite
  // derivative.
  T const tmp1 = pow(f.a, g.a);
  T const tmp2 = g.a * pow(f.a, g.a - T(1.0));
  T const tmp3 = tmp1 * log(f.a);
  return Jet<T, N>(tmp1, tmp2 * f.v + tmp3 * g.v);
}

// Option 1
// Convert all branches to PHIs
// -> Compute all conditions
// -> Compute the potential results
// -> Create a nested PHI to assign the correct result
template <int N>
inline Jet<Expression, N> pow_phi(const Jet<Expression, N>& f,
                                  const Jet<Expression, N>& g) {
  using T = Expression;
  using J = Jet<Expression, N>;

  // The 3 possible conditions
  auto c1 = (f.a == T(0)) && (g.a >= T(1));
  auto c2 = g.a > 1;
  auto c3 = f.a < 0 && g.a == floor(g.a);

  // The 4 results
  J r1 = J(T(0.0));
  J r2 = f;

  T const tmp = g.a * pow(f.a, g.a - T(1.0));
  J r3(pow(f.a, g.a), tmp * f.v);
  for (int i = 0; i < N; i++) {
    r3.v[i] =
        PHI(g.v[i] != T(0.0), r3.v[i], std::numeric_limits<T>::quiet_NaN());
  }

  // Handle the remaining cases. For cases 4,5,6,9 we allow the log() function
  // to generate -HUGE_VAL or NaN, since those cases result in a nonfinite
  // derivative.
  T const tmp1 = pow(f.a, g.a);
  T const tmp2 = g.a * pow(f.a, g.a - T(1.0));
  T const tmp3 = tmp1 * log(f.a);
  J r4(tmp1, tmp2 * f.v + tmp3 * g.v);

  return PHI(c1, PHI(c2, r1, r2), PHI(c3, r3, r4));
}

// Option 2
// Use a special function "condition", which takes a comparison expression as
// first argument and two lambdas as second and third. This should work
// recursively. Not implemented yet, but 'should' be possible.
//
// void condition(ComparisonExpressions c, Lambda trueBranch, Lambda
// falseBranch);
//
// The following code will be generated:
//
//  if(ComparisonExpressions)
//    trueBranch
//  else
//    falseBranch

template <typename TrueOperator, typename FalseOperator>
void condition(ComparisonExpression c, TrueOperator t1, FalseOperator t2) {}

template <int N>
inline Jet<Expression, N> pow_lambda(const Jet<Expression, N>& f,
                                     const Jet<Expression, N>& g) {
  using T = Expression;
  using J = Jet<Expression, N>;

  J result;
  // clang-format off
  condition((f.a == T(0)) && (g.a >= T(1)),
            [&]() -> void {
                  condition(g.a > 1,
                    [&]()  -> void { result = J(T(0.0)); },
                    [&]()  -> void {result = f; });
                },
            [&]()  -> void{
                  condition(f.a < 0 && g.a == floor(g.a),
                    [&]()  -> void{
                       // Handle cases 7 and 8.
                          T const tmp = g.a * pow(f.a, g.a - T(1.0));
                         result  = J(pow(f.a, g.a), tmp * f.v);
                          for (int i = 0; i < N; i++) {
                            if (g.v[i] != T(0.0)) {
                              // Return a NaN when g.v != 0.
                              result.v[i] = std::numeric_limits<T>::quiet_NaN();
                            }
                          }
                      },
                    [&]()  -> void{
                      // Handle the remaining cases. For cases 4,5,6,9 we allow the log() function
                      // to generate -HUGE_VAL or NaN, since those cases result in a nonfinite
                      // derivative.
                      T const tmp1 = pow(f.a, g.a);
                      T const tmp2 = g.a * pow(f.a, g.a - T(1.0));
                      T const tmp3 = tmp1 * log(f.a);
                      result= Jet<T, N>(tmp1, tmp2 * f.v + tmp3 * g.v);
                    });
               }
           );
  // clang-format on
  return result;
}

#define CERES_IF(_condition) condition((_condition), [&]() -> void
#define CERES_ELSE , [&]() -> void
#define CERES_ENDIF )

// Option 3
// Same as option 3,but trying to hide the lambda symbols [&]() using macros.
// It almost looks like "normal" code except that we have to add CERES_ENDIF at
// the end to close the function call. An additional advantage would be that we
// can convert it to "normal" if-else when we are not generating code.
template <int N>
inline Jet<Expression, N> pow_lambda_macros(const Jet<Expression, N>& f,
                                            const Jet<Expression, N>& g) {
  using T = Expression;
  using J = Jet<Expression, N>;

  J result;

  // This version works fine with clang-format
  CERES_IF((f.a == T(0)) && (g.a >= T(1))) {
    CERES_IF(g.a > 1) { result = J(T(0.0)); }
    CERES_ELSE { result = f; }
    CERES_ENDIF;
  }
  CERES_ELSE {
    CERES_IF(f.a < 0 && g.a == floor(g.a)) {
      // Handle cases 7 and 8.
      T const tmp = g.a * pow(f.a, g.a - T(1.0));
      result = J(pow(f.a, g.a), tmp * f.v);
      for (int i = 0; i < N; i++) {
        result.v[i] = PHI(
            g.v[i] != T(0.0), result.v[i], std::numeric_limits<T>::quiet_NaN());
      }
    }
    CERES_ELSE {
      // Handle the remaining cases. For cases 4,5,6,9 we allow the log()
      // function to generate -HUGE_VAL or NaN, since those cases result in
      // a nonfinite derivative.
      T const tmp1 = pow(f.a, g.a);
      T const tmp2 = g.a * pow(f.a, g.a - T(1.0));
      T const tmp3 = tmp1 * log(f.a);
      result = Jet<T, N>(tmp1, tmp2 * f.v + tmp3 * g.v);
    }
    CERES_ENDIF;
  }
  CERES_ENDIF;
  return result;
}

}  // namespace ceres

#endif
