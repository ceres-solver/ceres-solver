#ifndef SPEECH_MATH_AUTODIFF_EIGEN_JET_H_
#define SPEECH_MATH_AUTODIFF_EIGEN_JET_H_

#include <cmath>
#include <ostream>  // NOLINT
#include <vector>

#include "Eigen/Core"
//#include "base/logging.h"
#include "ceres/autodiff/expr.h"
//#include "strings/strcat.h"
//#include "util/gtl/stl_util.h"

namespace ceres {

template <typename T, int N>
class Jet : public Expr<Jet<T, N>, 1> {
 public:
  typedef T ZeroOrderType;

  // Default constructor.
  inline Jet() : Expr<Jet<T, N>, 1>(T(), this) {
    v.setZero();
  }

  // Construct with the value of a T.  This EigenJet
  // represents a "constant" value, that is, a value whose derivatives w.r.t.
  // anything are all zero.
  // Certain Eigen algorithms (Eigenvalues/EigenSolver.h) require that this
  // not be marked as explicit.
  inline Jet(const T& value) : Expr<Jet<T, N>, 1>(value, this) {
    v.setZero();
  }

  // Construct a EigenJet for which derivatives can be computed.
  // The initial value of this EigenJet is specified by value,
  // and partial_goes_here specifies where the partial derivative of the
  // resulting computation w.r.t. this value should be stored.
  // Unlike EigenJets formed using the constant value constructor
  // above, expressions derived from this EigenJet may
  // have a non-zero derivative w.r.t. this EigenJet, since
  // this constructor signifies that the partial derivative of this
  // EigenJet w.r.t. itself is 1.0.
  inline Jet(const T& value, int n)
      : Expr<Jet<T, N>, 1>(value, this) {
    v.setZero();
    v[n] = 1;
  }

  // Copy constructor.
  Jet(const Jet<T, N>& that)
      : Expr<Jet<T, N>, 1>(that.a, this),
        v(that.v) {}

  // Construct from ExprBase.  This finishes evaluating the partial
  // derivatives (w.r.t. this EigenJet's direct parents,
  // not all of its ancestors) in the expression.
  template <int SIZE> inline
  Jet(const ExprBase<Jet<T, N>, SIZE>& expr)
      : Expr<Jet<T, N>, 1>(expr.a, this) {
    AssignFromAndEvaluatePartials<false, SIZE>(expr);
  }

  // Copy assignment.
  inline Jet<T, N>& operator=(const Jet<T, N>& that) {
    this->a = that.a;
    v = that.v;
    return *this;
  }

  // Assign from ZeroOrder.  This EigenJet is now a "constant"
  // value , in the sense that all its partial derivatives w.r.t. anything
  // are now zero.
  inline Jet<T, N>& operator=(const T& value) {
    this->a = value;
    v.setZero();
    return *this;
  }

  // Assign from ExprBase.  This finishes evaluating the partial
  // derivatives in the expression.
  template <int SIZE> inline
  Jet<T, N>& operator=(const ExprBase<Jet<T, N>, SIZE>& expr) {
    AssignFromAndEvaluatePartials<true, SIZE>(expr);
    return *this;
  }

  // Overload std::isfinite, std::isinf, and std::isnan for our Jets.
  // Returns true if the value and ALL the derivaties are finite.
  friend bool isfinite(const Jet<T, N>& jet) {
    using std::isfinite;
    if (!isfinite(jet.a)) {
      return false;
    }
    for (int i = 0; i < N; ++i) {
      if (!isfinite(jet.v[i])) {
        return false;
      }
    }
    return true;
  }

  // Returns true if the value or ANY the derivaties are infinite.
  friend bool isinf(const Jet<T, N>& jet) {
    using std::isinf;
    if (isinf(jet.a)) {
      return true;
    }
    for (int i = 0; i < N; ++i) {
      if (isinf(jet.v[i])) {
        return true;
      }
    }
    return false;
  }

  // Returns true if the value or ANY the derivaties are NaN.
  friend bool isnan(const Jet<T, N>& jet) {
    using std::isnan;
    if (isnan(jet.a)) {
      return true;
    }
    for (int i = 0; i < N; ++i) {
      if (isnan(jet.v[i])) {
        return true;
      }
    }
    return false;
  }

  friend bool IsNaN(const Jet<T, N>& jet) { return isnan(jet); }
  friend bool IsFinite(const Jet<T, N>& jet) { return isfinite(jet); }
  friend bool IsInfinite(const Jet<T, N>& jet) { return isinf(jet); }
  friend bool IsNormal(const Jet<T, N>& jet) { return isfinite(jet); }

  // operator<< is necessary in order for code like this to compile:
  // CHECK_LT(my_jet, 2.0);
  // (And more generally, because lots of code expects to be able to use
  // operator<< on doubles.)
  friend std::ostream& operator<<(
      std::ostream& s, const Jet<T, N>& jet) {  // NOLINT
    s << GetZeroOrder(jet) << " [";
    for (int i = 0; i < N; ++i) {
      s << " " << jet.v[i];
    }
    s << " ]";
    return s;
  }

  Eigen::Matrix<T, N, 1, Eigen::DontAlign> v;

 private:
  // Assign the result of the expression into this JetT.  This evaluates
  // the deferred calculation of partial derivative information represented
  // inside expr.
  template <bool ALIASING, int SIZE>
  inline void AssignFromAndEvaluatePartials(
      const ExprBase<Jet<T, N>, SIZE>& expr) {
    this->a = expr.a;

    if (!ALIASING) {
      if (SIZE == 1) {
        v = expr.jet_partials[0] * expr.jets[0]->v;
      } else if (SIZE == 2) {
        v = expr.jet_partials[0] * expr.jets[0]->v +
            expr.jet_partials[1] * expr.jets[1]->v;
      } else if (SIZE == 3) {
        v = expr.jet_partials[0] * expr.jets[0]->v +
            expr.jet_partials[1] * expr.jets[1]->v +
            expr.jet_partials[2] * expr.jets[2]->v;
      } else if (SIZE == 4) {
        v = expr.jet_partials[0] * expr.jets[0]->v +
            expr.jet_partials[1] * expr.jets[1]->v +
            expr.jet_partials[2] * expr.jets[2]->v +
            expr.jet_partials[3] * expr.jets[3]->v;
      } else if (SIZE == 5) {
        v = expr.jet_partials[0] * expr.jets[0]->v +
            expr.jet_partials[1] * expr.jets[1]->v +
            expr.jet_partials[2] * expr.jets[2]->v +
            expr.jet_partials[3] * expr.jets[3]->v +
            expr.jet_partials[4] * expr.jets[4]->v;
      } else if (SIZE == 6) {
        v = expr.jet_partials[0] * expr.jets[0]->v +
            expr.jet_partials[1] * expr.jets[1]->v +
            expr.jet_partials[2] * expr.jets[2]->v +
            expr.jet_partials[3] * expr.jets[3]->v +
            expr.jet_partials[4] * expr.jets[4]->v +
            expr.jet_partials[5] * expr.jets[5]->v;
      } else {
        CHECK(false) << SIZE;
        v = expr.jet_partials[0] * expr.jets[0]->v;
        for (int i = 1; i < SIZE; ++i) {
          v += expr.jet_partials[i] * expr.jets[i]->v;
        }
      }
      return;
    }

    Eigen::Matrix<T, N, 1, Eigen::DontAlign> new_partials;
    if (SIZE == 1) {
      new_partials = expr.jet_partials[0] * expr.jets[0]->v;
    } else if (SIZE == 2) {
      new_partials = expr.jet_partials[0] * expr.jets[0]->v +
          expr.jet_partials[1] * expr.jets[1]->v;
    } else if (SIZE == 3) {
      new_partials = expr.jet_partials[0] * expr.jets[0]->v +
          expr.jet_partials[1] * expr.jets[1]->v +
          expr.jet_partials[2] * expr.jets[2]->v;
    } else if (SIZE == 4) {
      new_partials = expr.jet_partials[0] * expr.jets[0]->v +
          expr.jet_partials[1] * expr.jets[1]->v +
          expr.jet_partials[2] * expr.jets[2]->v +
          expr.jet_partials[3] * expr.jets[3]->v;
    } else if (SIZE == 5) {
      new_partials = expr.jet_partials[0] * expr.jets[0]->v +
          expr.jet_partials[1] * expr.jets[1]->v +
          expr.jet_partials[2] * expr.jets[2]->v +
          expr.jet_partials[3] * expr.jets[3]->v +
          expr.jet_partials[4] * expr.jets[4]->v;
    } else if (SIZE == 6) {
      new_partials = expr.jet_partials[0] * expr.jets[0]->v +
          expr.jet_partials[1] * expr.jets[1]->v +
          expr.jet_partials[2] * expr.jets[2]->v +
          expr.jet_partials[3] * expr.jets[3]->v +
          expr.jet_partials[4] * expr.jets[4]->v +
          expr.jet_partials[5] * expr.jets[5]->v;
    } else if (SIZE == 8) {
      new_partials = expr.jet_partials[0] * expr.jets[0]->v +
          expr.jet_partials[1] * expr.jets[1]->v +
          expr.jet_partials[2] * expr.jets[2]->v +
          expr.jet_partials[3] * expr.jets[3]->v +
          expr.jet_partials[4] * expr.jets[4]->v +
          expr.jet_partials[5] * expr.jets[5]->v +
          expr.jet_partials[6] * expr.jets[6]->v +
          expr.jet_partials[7] * expr.jets[7]->v;
    } else {
      CHECK(false) << SIZE;
      new_partials = expr.jet_partials[0] * expr.jets[0]->v;
      for (int i = 1; i < SIZE; ++i) {
        new_partials += expr.jet_partials[i] * expr.jets[i]->v;
      }
    }
    v = new_partials;
  }  // AssignFromAndEvaluatePartials
};  // EigenJet

template <typename T, int N>
struct ZeroOrder<Jet<T, N> > {
  typedef T ZeroOrderType;
  static const T& Get(const Jet<T, N>& x) {
    return x.a;
  }
  static void Set(Jet<T, N>* x, const T& value) {
    x->a = value;
  }
};

template<typename T, int N>
struct IntermediateVar<Jet<T, N> > {
  static bool Make(Jet<T, N>* x) {
    return false;
  }
};

}  // namespace autodiff

#endif  // SPEECH_MATH_AUTODIFF_EIGEN_JET_H_
