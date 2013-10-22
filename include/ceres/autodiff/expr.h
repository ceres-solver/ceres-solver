// Author: thadh@google.com (Thad Hughes)

// ExprBase is the core class implementing automatic differentiation using
// expression templates. All Jet types (the types used instead of float or
// double to evaluate partial derivatives) are derived from ExprBase using the
// CRTP (with SIZE=1, indicating an expression of 1 variable).
//
// In addition to its zero-order value, an ExprBase of a particular SIZE stores
// references back to the SIZE Jets involved in the expression, and the linear
// weight with which each Jet in the expression contributes to the partial
// derivative of the entire expression. As Jets interact in C++ expressions,
// they combine to form ExprBases of increasing SIZE. Notably, ExprBase is
// entirely agnostic to the Jets' internal representation of the partial
// derivatives, meaning that the same ExprBase template can be used as the
// parent for many types of Jets.
//
// It is possible to think of ExprBase as performing a form of forward autodiff,
// in which the calculation of the actual partial derivatives is deferred until
// the ExprBase is evaluated back into its JetT type. This can often eliminate
// the need for temporary Jet variables, which can be expensive, since Jets must
// store partial derivative information.
//
// UNARY OPERATIONS: Unary operations on expressions can only scale the partial
// derivatives of the Jets involved (but cannot add more Jets). Therefore, a
// unary operation does not change the SIZE of the ExprBase or the contents of
// the jets array, but only the jet_weights. For example, if an ExprBase for
// quantity "x" is derived from two Jets "a" and "b" (say by taking x = 2*a +
// 5*b), then the partials of x w.r.t. its parents (at the current point of
// interest) are:
//
// x.jets[0] = a, x.jet_weights[0] = dx/da = 2
// x.jets[1] = b, x.jet_weights[1] = dx/db = 5
//
// As an ExprBase, x would be represented as an ExprBase<Jet, 2> where the
// x.jets array contains pointers to "a" and "b", and the x.jet_weights contains
// 2 and 5.
//
// Then we compute the unary operation z = sqrt(x). This results in a new
// expression with the same SIZE: ExprBase<Jet, 2> and the same jets. We only
// need to scale the jet_weights in x using the chain rule:
//
// dz = 1/(2*sqrt(x)) * dx
//
// So by the chain rule, we have:
//
// z.jets[0] = a, z.jet_weights[0] = dz/da = dz/dx * dx/da = 1/(2*sqrt(x)) * 2
// z.jets[1] = b, z.jet_weights[1] = dz/db = dz/dx * dx/db = 1/(2*sqrt(x)) * 5
//
// So z.jet_weights is simply 1/(2*sqrt(x))*x.jet_weights. Note that the partial
// derivatives of any unary operation are simply the partial derivatives of the
// unary operation's input, all scaled by a factor that depends solely on the
// value of the operation's input. This operation is factored in the method
// ExprBase::Scale().
//
// BINARY OPERATIONS: Binary operations on expressions join together the
// dependencies from both parent expressions. For example, suppose we have two
// input expressions "x" (SIZE=2) and "y" (SIZE=1) that together depend on 3
// Jets (a, b, and c) with these jet_weights:
//
// x.jets[0] = a, x.jet_weights[0] = dx/da = 4
// x.jets[1] = b, x.jet_weights[1] = dx/db = 5
// y.jets[0] = c, y.jet_weights[0] = dy/dc = 3
//
// Then we compute the binary expression: z = x * y. Then:
//
// dz/dx = ZeroOrder(y)
// dz/dy = ZeroOrder(x)
//
// So by the chain rule:
//
// z.jets[0] = a, z.jet_weights[0] = dz/da = dz/dx * dx/da = ZeroOrder(y) * 4
// z.jets[1] = b, z.jet_weights[1] = dz/db = dz/dx * dx/db = ZeroOrder(y) * 5
// z.jets[2] = c, z.jet_weights[2] = dz/dc = dz/dy * dy/dc = ZeroOrder(x) * 3
//
// Thus the resulting ExprBase representing z brings together x's dependence on
// a and b and y's dependence on c, and thus has SIZE=3, with z.jets pointing to
// Jets a, b, and c with z.jet_weights containing y*4, y*5, and x*3. This
// operation is implemented in ExprBase::Combine().
//
// Note that binary operations can combine 2 Jets with overlapping sets of
// parents.  For example, in the above example, if Jet y also had Jet b as
// a parent, then both Jet x and Jet y would have Jet b as a parent, and
// the formation of "z = x * y" would result in 2 different entries in
// z.jets, both of which point at Jet a, and for which there are different
// jet_weights.  This is not a problem, and the operator= that assigns a
// Jet from an Expr will handle this naturally, and merge the duplicates
// together.  Note that we don't do that de-duping here inside ExprBase
// because SIZE needs to be a compile-time constant so that nothing
// involving ExprBases touches the heap and ExprBase expressions can
// hopefully be optimized away.

#ifndef SPEECH_PORTABLE_MATH_AUTODIFF_EXPR_H_
#define SPEECH_PORTABLE_MATH_AUTODIFF_EXPR_H_

#include <ostream>  // NOLINT

//#include "base/logging.h"
#include "ceres/autodiff/autodiff_interface.h"

namespace ceres {

// Here we will be using CRTP: ExprBase<JetT, 1> will be an ancestor
// class for JetT, through the intermediate specialization Expr<JetT, 1>,
// which will be JetT's direct parent class.
template <typename JetT, int SIZE>
struct ExprBase {
  enum { size = SIZE };
  typedef JetT JetType;
  typedef typename ZeroOrder<JetT>::ZeroOrderType ZeroOrderType;

  // Construct an ExprBase with the specified value.
  inline explicit ExprBase(const ZeroOrderType& v) : a(v) {}

  // "Evaluates" this ExprBase by coercing it into a JetType.
  // Implies that JetT must be constructible from ExprBase<JetT, SIZE>.
  // This operator is used during implicit conversions, for example when
  // an ExprBase is passed to a function expecting a JetType, and can thus
  // result in a temporary variable being constructed.
  // If your code starts doing infinite recursion here, it might be that JetT
  // doesn't define a constructor like:
  //     template <int SIZE> JetT(const ExprBase<JetT, SIZE>& expr)
  // The infinite recursion happens right here, since C++ cannot invoke
  // JetT's constructor as requested, but it thinks it knows how to convert
  // "*this" into a JetT, using this very method below.
  operator JetT() const {
    // If JetT doesn't define the appropriate constructor, this becomes a
    // recursive call.
    return JetT(*this);
  }

  // Scale is a helper function that creates a new ExprBase with a zero-order
  // value of new_value, and with the same SIZE and parent jets as x, but
  // with x's weights scaled by x_scale.  This is a common pattern when
  // implementing unary operations on ExprBase.
  friend ExprBase<JetT, SIZE> Scale(
      const ZeroOrderType& new_value, const ExprBase<JetT, SIZE>& x,
      const ZeroOrderType& x_scale) {
    ExprBase<JetT, SIZE> res(new_value);
    for (int i = 0; i < SIZE; ++i) {
      res.jets[i] = x.jets[i];
      res.jet_partials[i] = x_scale * x.jet_partials[i];
    }
    return res;
  }

  // Combine is a helper function that creates a new ExprBase with SIZE
  // increased by SIZE2.  The new ExprBase has a value equal to new_value,
  // and has jet parents from x scaled by x_scale, and jets from y scaled by
  // y_scale. This is a common pattern when implementing binary operations on
  // two ExprBase objects.
  template <int SIZE2>
  friend ExprBase<JetT, SIZE + SIZE2> Combine(
      const ZeroOrderType& new_value,
      const ExprBase<JetT, SIZE>& x, const ZeroOrderType& x_scale,
      const ExprBase<JetT, SIZE2>& y, const ZeroOrderType& y_scale) {
    ExprBase<JetT, SIZE + SIZE2> res(new_value);
    for (int i = 0; i < SIZE; ++i) {
      res.jets[i] = x.jets[i];
      res.jet_partials[i] = x_scale * x.jet_partials[i];
    }
    for (int i = 0; i < SIZE2; ++i) {
      res.jets[SIZE + i] = y.jets[i];
      res.jet_partials[SIZE + i] = y_scale * y.jet_partials[i];
    }
    // Note that res.jets may contain duplicates, but this is not a problem.
    return res;
  }
  template <int SIZE2>
  friend ExprBase<JetT, SIZE + SIZE2> CombinePlus(
      const ZeroOrderType& new_value,
      const ExprBase<JetT, SIZE>& x, const ZeroOrderType& x_scale,
      const ExprBase<JetT, SIZE2>& y, const ZeroOrderType& y_scale) {
    ExprBase<JetT, SIZE + SIZE2> res(new_value);
    for (int i = 0; i < SIZE; ++i) {
      res.jets[i] = x.jets[i];
      res.jet_partials[i] = x.jet_partials[i];
    }
    for (int i = 0; i < SIZE2; ++i) {
      res.jets[SIZE + i] = y.jets[i];
      res.jet_partials[SIZE + i] = y.jet_partials[i];
    }
    // Note that res.jets may contain duplicates, but this is not a problem.
    return res;
  }
  // Implement all the differentiation math.

  // Implementing these as inline friends means they aren't explicitly
  // templated based on the type JetT (although this entire class is
  // templated on JetT, once the class is instantiated, these aren't
  // function templates, they are actual functions).  This is important,
  // since the compiler won't do any type conversions when instantiating a
  // function template parameter (not even an upcast from JetT to to its parent
  // class ExprBase<JetT, 1>!!!).  Since these functions all take as an
  // argument a const ExprBase<JetT, SIZE>&, they wouldn't be instantiated for
  // a JetT argument, meaning you couldn't apply them to actual JetTs if they
  // had been written as function templates.  Friend functions defined inline
  // are also implicitly declared inline, see C++ standard section 11.4/5.

  // First the unary operators and functions.
  friend const ExprBase<JetT, SIZE>& operator+(const ExprBase<JetT, SIZE>& x) {
    return x;
  }

  friend ExprBase<JetT, SIZE> operator-(const ExprBase<JetT, SIZE>& x) {
    return Scale(-GetZeroOrder(x), x, -1);
  }

#define DEFINE_EXPR_BASE_UNARY_FN(name, use, dx_impl) \
  friend ExprBase<JetT, SIZE> name(const ExprBase<JetT, SIZE>& x) { \
    use; \
    const ZeroOrderType& x_s = GetZeroOrder(x); \
    const ZeroOrderType res = name(x_s); \
    return Scale(res, x, dx_impl); \
  }
DEFINE_EXPR_BASE_UNARY_FN(abs, using std::abs, x < 0 ? -1 : 1)
DEFINE_EXPR_BASE_UNARY_FN(log, using std::log, 1 / x_s)
DEFINE_EXPR_BASE_UNARY_FN(exp, using std::exp, res)
DEFINE_EXPR_BASE_UNARY_FN(sqrt, using std::sqrt, 1 / (2 * res))
DEFINE_EXPR_BASE_UNARY_FN(cos, using std::cos, -sin(x_s))
DEFINE_EXPR_BASE_UNARY_FN(acos, using std::acos, -1 / sqrt(1 - x_s * x_s))
DEFINE_EXPR_BASE_UNARY_FN(sin, using std::sin, cos(x_s))
DEFINE_EXPR_BASE_UNARY_FN(asin, using std::asin, 1 / sqrt(1 - x_s * x_s))
DEFINE_EXPR_BASE_UNARY_FN(tan, using std::tan, 1 + res * res)
DEFINE_EXPR_BASE_UNARY_FN(atan, using std::atan, 1 / (1 + x_s * x_s))
DEFINE_EXPR_BASE_UNARY_FN(sinh, using std::sinh, cosh(x_s))
DEFINE_EXPR_BASE_UNARY_FN(cosh, using std::cosh, sinh(x_s))
DEFINE_EXPR_BASE_UNARY_FN(tanh, using std::tanh, 1 - res * res)
DEFINE_EXPR_BASE_UNARY_FN(logistic, using std::pow, res * (1 - res))
DEFINE_EXPR_BASE_UNARY_FN(square, using std::pow, 2 * x_s)
#undef DEFINE_EXPR_BASE_UNARY_FN

  // Binary arithmetic operators and functions.

  // They must be implemented for 3 different pairs of argument types:
  // * JetT, JetT
  // * JetT, ZeroOrderType
  // * ZeroOrderType, JetT
  // Luckily, though, since again these are inline friends not templated
  // using the ZeroOrderType as a template argument, the compiler is free
  // to do implicit conversions to the ZeroOrderType, meaning, for example
  // that even if ZeroOrderType=double, you can still use int literals as
  // operands and the compiler will do the implicit conversion.  This would
  // not have worked had the ZeroOrderType argument been declared using a
  // template parameter (ie. using a non-member template function.)

#define DEFINE_EXPR_BASE_BINARY_FN(\
    name, value_impl, combine_fn, dx_impl, dy_impl) \
  template <int SIZE2> friend ExprBase<JetT, SIZE + SIZE2> \
  name(const ExprBase<JetT, SIZE>& x, const ExprBase<JetT, SIZE2>& y) { \
    using std::pow; \
    using std::atan2; \
    const ZeroOrderType& x_s = GetZeroOrder(x); \
    const ZeroOrderType& y_s = GetZeroOrder(y); \
    const ZeroOrderType res = value_impl; \
    return combine_fn(res, x, dx_impl, y, dy_impl); \
  } \
  friend ExprBase<JetT, SIZE> \
  name(const ZeroOrderType& x, const ExprBase<JetT, SIZE>& y) { \
    using std::pow; \
    using std::atan2; \
    const ZeroOrderType& x_s = x; \
    const ZeroOrderType& y_s = GetZeroOrder(y); \
    const ZeroOrderType res = value_impl; \
    return Scale(res, y, dy_impl); \
  } \
  friend ExprBase<JetT, SIZE> \
  name(const ExprBase<JetT, SIZE>& x, const ZeroOrderType& y) { \
    using std::pow; \
    using std::atan2; \
    const ZeroOrderType& x_s = GetZeroOrder(x); \
    const ZeroOrderType& y_s = y; \
    const ZeroOrderType res = value_impl; \
    return Scale(res, x, dx_impl); \
  }

DEFINE_EXPR_BASE_BINARY_FN(operator+, x_s + y_s, CombinePlus, 1, 1)
DEFINE_EXPR_BASE_BINARY_FN(operator-, x_s - y_s, Combine, 1, -1)
DEFINE_EXPR_BASE_BINARY_FN(operator*, x_s * y_s, Combine, y_s, x_s)
DEFINE_EXPR_BASE_BINARY_FN(operator/, x_s / y_s, Combine, 1 / y_s, -res / y_s)
DEFINE_EXPR_BASE_BINARY_FN(pow,
                           pow(x_s, y_s), Combine,
                           y_s * pow(x_s, y_s - 1),
                           res * log(x_s))
DEFINE_EXPR_BASE_BINARY_FN(atan2,  \
                           atan2(x_s, y_s),  \
                           Combine,
                           y_s / (x_s * x_s + y_s * y_s), \
                           -x_s / (x_s * x_s + y_s * y_s))
#undef DEFINE_EXPR_BASE_BINARY_FN

  // Binary comparison operators.

#define DEFINE_EXPR_BASE_COMPARISON_OPERATOR(op) \
  template <int SIZE2> friend bool operator op( \
        const ExprBase<JetT, SIZE>& x, const ExprBase<JetT, SIZE2>& y) { \
    return GetZeroOrder(x) op GetZeroOrder(y); \
  } \
  friend bool operator op( \
      const ZeroOrderType& x, const ExprBase<JetT, SIZE>& y) { \
    return x op GetZeroOrder(y); \
  } \
  friend bool operator op( \
      const ExprBase<JetT, SIZE>& x, const ZeroOrderType& y) { \
    return GetZeroOrder(x) op y; \
  }
DEFINE_EXPR_BASE_COMPARISON_OPERATOR( <  )  // NOLINT
DEFINE_EXPR_BASE_COMPARISON_OPERATOR( <= )  // NOLINT
DEFINE_EXPR_BASE_COMPARISON_OPERATOR( >  )  // NOLINT
DEFINE_EXPR_BASE_COMPARISON_OPERATOR( >= )  // NOLINT
DEFINE_EXPR_BASE_COMPARISON_OPERATOR( == )  // NOLINT
DEFINE_EXPR_BASE_COMPARISON_OPERATOR( != )  // NOLINT
#undef DEFINE_EXPR_BASE_COMPARISON_OPERATOR

  // The zero-order value of this ExprBase.
  ZeroOrderType a;

  // Pointers to the JetTs involved in this ExprBase.
  // Note that this list may contain duplicate pointers to the same JetT.
  const JetType* jets[size];

  // A parallel array to jets containing the partial derivatives of this
  // ExprBase w.r.t. each of the JetTs in this->jets.
  ZeroOrderType jet_partials[size];

 private:
  void operator=(const ExprBase<JetT, SIZE>& that);
  // It would be nice to make the copy-constructor private as well, but
  // Exprs of other SIZEs need to invoke it to return by value.
};

// How to get the ZeroOrder from an ExprBase.
template <typename JetT, int SIZE>
struct ZeroOrder<ExprBase<JetT, SIZE> > {
  typedef typename ZeroOrder<JetT>::ZeroOrderType ZeroOrderType;
  inline static const ZeroOrderType& Get(const ExprBase<JetT, SIZE>& x) {
    return x.a;
  }
};

// Expr: basically just a copy of ExprBase, but we're going to specialize
// it below for SIZE=1, and in that specialization with SIZE=1, we will
// provide the in-place arithmetic operators, which can only be defined once.
// Most of the implementation for this template and its specialization for
// SIZE=1 is factored into ExprBase.
template <typename JetT, int SIZE>
struct Expr : public ExprBase<JetT, SIZE> {
  // C++ does not inherit typedefs from templated base classes:
  // See C++ standard section: 14.6.2/3.
  typedef typename ExprBase<JetT, SIZE>::ZeroOrderType ZeroOrderType;
};

// Template specialization for Expr with SIZE=1 that provides in-place
// operators: operator+=, operator*=, etc. This is meant to be used as the
// parent class for JetT, which will then inherit all the operations specified
// here and in ExprBase.  Note that if this code were inside ExprBase, these
// declarations would be specified for ExprBases of ALL SIZEs, and thus they
// would be redefinitions of each other, since their left operand is a JetT
// and not an ExprBase<JetT, SIZE>.
template <typename JetT>
struct Expr<JetT, 1> : public ExprBase<JetT, 1> {
  typedef typename ExprBase<JetT, 1>::ZeroOrderType ZeroOrderType;
  explicit Expr(const ZeroOrderType& v) : ExprBase<JetT, 1>(v) {}

  Expr(const ZeroOrderType& v, const JetT* jet) : ExprBase<JetT, 1>(v) {
    this->jets[0] = jet;
    this->jet_partials[0] = 1;
  }

  // In-place arithmetic operators. These are slightly different from the
  // operators defined in ExprBase in that they take an actual JetT& on
  // the left, which is why they must be here and not in ExprBase.
#define DEFINE_EXPR_BASE_IN_PLACE_FN(name, impl) \
  template <int SIZE2> friend \
  JetT& name(JetT& x, const ExprBase<JetT, SIZE2>& y) { \
    impl; \
  } \
  friend JetT& name(JetT& x, const ZeroOrderType& y) { \
    impl; \
  }
DEFINE_EXPR_BASE_IN_PLACE_FN(operator+=, return x = x + y)
DEFINE_EXPR_BASE_IN_PLACE_FN(operator-=, return x = x - y)
DEFINE_EXPR_BASE_IN_PLACE_FN(operator*=, return x = x * y)
DEFINE_EXPR_BASE_IN_PLACE_FN(operator/=, return x = x / y)
#undef DEFINE_EXPR_BASE_IN_PLACE_FN

 private:
  // Force child classes to implement a proper copy-constructor by making the
  // default one private and thus not inheritable.
  Expr(const Expr<JetT, 1>& that);
};

// How to get the ZeroOrder from any Expr.
template <typename JetT, int SIZE>
struct ZeroOrder<Expr<JetT, SIZE> > {
  typedef typename ZeroOrder<JetT>::ZeroOrderType ZeroOrderType;
  inline static const ZeroOrderType& Get(const Expr<JetT, SIZE>& x) {
    return x.a;
  }
};

}  // namespace autodiff

#endif  // SPEECH_PORTABLE_MATH_AUTODIFF_EXPR_H_
