.. default-domain:: cpp

.. cpp:namespace:: ceres

.. _chapter-on_derivatives:

==============
On Derivatives
==============

Introduction
============

Ceres Solver like all gradient based optimization algorithms, depends
on being able to evaluate the objective function and its derivatives
at arbitrary points in its domain. Indeed, defining the objective
function and its `Jacobian
<https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_ is
the principal task that the user is required to perform when solving
an optimization problem using Ceres Solver. The correct and efficient
computation of the Jacobian is the key to good performance.

Ceres Solver offers considerable flexibility in how the user can
provide derivatives to the solver. She can compute them using:

1. Analytic/Symbolic Differentiation
2. Numerical Differentiation
3. Automatic Differentiation

Which of these three approaches (alone or in combination) should be
used depends on the situation and the tradeoffs the user is willing to
make. Unfortunately, numerical optimization textbooks rarely discuss
these issues in detail and the user is left to her own devices.

The aim of this article is to fill this gap and describe each of these
three approaches in the context of Ceres Solver with sufficient detail
that the user can make an informed choice.

tl;dr
-----

And for the impatient amongst you, here is some high level advice:

1. Use automatic differentiation.
2. In rare cases it maybe worth using analytic derivatives.
3. Avoid numeric differentiation, use it as a measure of last resort,
   mostly to interface with external libraries.


Spivak Notation
===============

To preserve our collective sanities, we will use Spivak's notation for
derivatives. It is a functional notation that makes reading and
reasoning about expressions involving derivatives simple.

For a univariate function :math:`f`, :math:`f(a)` denotes its value at
:math:`a`. :math:`Df` denotes its first derivative, and
:math:`Df(a)` is the derivative evaluated at :math:`a`, i.e

.. math::
   Df(a) = \left . \frac{d}{dx} f(x) \right |_{x = a}

:math:`D^nf` denotes the :math:`n^{\text{th}}` derivative of :math:`f`.

For a bi-variate function :math:`g(x,y)`. :math:`D_1g` and
:math:`D_2g` denote the partial derivatives of :math:`g` w.r.t the
first and second variable respectively. In the classical notation this
is equivalent to saying:

.. math::

   D_1 g = \frac{\partial}{\partial x}g(x,y) \text{ and }  D_2 g  = \frac{\partial}{\partial y}g(x,y).


:math:`Dg` denotes the Jacobian of `g`, i.e.,

.. math::

  Dg = \begin{bmatrix} D_1g & D_2g \end{bmatrix}

More generally for for a multivariate function :math:`g:\mathbb{R}^m
\rightarrow \mathbb{R}^n`, :math:`Dg` denotes the Jacobian matrix and
:math:`D_i g` is partial derivative of :math:`g` w.r.t the
:math:`i^{\text{th}}` coordinate and the :math:`i^{\text{th}}` column
of :math:`Dg`.

Finally, :math:`D^2_1g, D_1D_2g` have the obvious meaning as higher
partial order partial derivatives derivatives.

For more see Michael Spivak's book `Calculus on Manifolds
<https://www.amazon.com/Calculus-Manifolds-Approach-Classical-Theorems/dp/0805390219>`_
or a brief discussion of the `merits of this notation
<http://www.vendian.org/mncharity/dir3/dxdoc/>`_ by
Mitchell N. Charity.


Analytic Derivatives
====================

Consider the problem of fitting the following curve (`Rat43
<http://www.itl.nist.gov/div898/strd/nls/data/ratkowsky3.shtml>`_) to
data:

.. math::
  y = \frac{b_1}{(1+e^{b_2-b_3x})^{1/b_4}}

That is, give some data :math:`\{x_i, y_i\}\quad \forall i=1,... ,n`,
determine parameters :math:`b_1, b_2, b_3` and :math:`b_4` that best
fit this data. So, assuming that the noise is Gaussian, find
:math:`b_1, b_2, b_3` and :math:`b_4` that minimize the following
objective function:

.. math::
   \begin{align}
   E(b_1, b_2, b_3, b_4)
   &= \sum_i f^2(b_1, b_2, b_3, b_4 ; x_i, y_i)\\
   &= \sum_i \left(\frac{b_1}{(1+e^{b_2-b_3x_i})^{1/b_4}} - y_i\right)^2\\
   \end{align}

To solve this problem using Ceres Solver, we need to define a
:class:`CostFunction` that computes the residual :math:`f` for a given
:math:`x` and :math:`y` and its derivatives with respect to
:math:`b_1, b_2, b_3` and :math:`b_4`.

The most direct way to do this is to derive using the rules of
differential calculus, the algebraic expressions for the derivatives
and implement the code to evaluate them. So lets do that:

.. math::
  \begin{align}
  D_1 f(b_1, b_2, b_3, b_4; x,y) &= \frac{1}{(1+e^{b_2-b_3x})^{1/b_4}}\\
  D_2 f(b_1, b_2, b_3, b_4; x,y) &=
  \frac{-b_1e^{b_2-b_3x}}{b_4(1+e^{b_2-b_3x})^{1/b_4 + 1}} \\
  D_3 f(b_1, b_2, b_3, b_4; x,y) &=
  \frac{b_1xe^{b_2-b_3x}}{b_4(1+e^{b_2-b_3x})^{1/b_4 + 1}} \\
  D_4 f(b_1, b_2, b_3, b_4; x,y) & = \frac{b_1  \log\left(1+e^{b_2-b_3x}\right) }{b_4^2(1+e^{b_2-b_3x})^{1/b_4}}
  \end{align}

With these derivatives in hand, we can now implement the
:class:`CostFunction`: as

.. code-block:: c++

  class Rat43CostFunction : public SizedCostFunction<1,4> {
     public:
       Rat43CostFunction(const double x, double const y) : x_(x), y_(y) {}
       virtual ~Rat43CostFunction() {}
       virtual bool Evaluate(double const* const* parameters,
                             double* residuals,
			     double** jacobians) const {
	 const double b1 = parameters[0][0];
	 const double b2 = parameters[0][1];
	 const double b3 = parameters[0][2];
	 const double b4 = parameters[0][3];

	 residuals[0] = b1 *  pow(1 + exp(b2 -  b3 * x_), -1.0 / b4) - y_;

         if (!jacobians) return true;
	 double* jacobian = jacobians[0];
	 if (!jacobian) return true;

         jacobian[0] = pow(1 + exp(b2 - b3 * x_), -1.0 / b4);
         jacobian[1] = -b1 * exp(b2 - b3 * x_) *
                       pow(1 + exp(b2 - b3 * x_), -1.0 / b4 - 1) / b4;
	 jacobian[2] = x_ * b1 * exp(b2 - b3 * x_) *
                       pow(1 + exp(b2 - b3 * x_), -1.0 / b4 - 1) / b4;
         jacobian[3] = b1 * log(1 + exp(b2 - b3 * x_)) *
                       pow(1 + exp(b2 - b3 * x_), -1.0 / b4) / (b4 * b4);
         return true;
       }

      private:
       const double x_;
       const double y_;
   };

This is rather tedious looking code which is hard to read with a lot
of redundancy, so in practice we will cache some subexpressions to
improve its efficiency, which gives us:

.. code-block:: c++

  class Rat43CostFunction : public SizedCostFunction<1,4> {
     public:
       Rat43CostFunction(const double x, double const y) : x_(x), y_(y) {}
       virtual ~Rat43CostFunction() {}
       virtual bool Evaluate(double const* const* parameters,
                             double* residuals,
			     double** jacobians) const {
	 const double b1 = parameters[0][0];
	 const double b2 = parameters[0][1];
	 const double b3 = parameters[0][2];
	 const double b4 = parameters[0][3];

	 const double t1 = exp(b2 -  b3 * x_);
         const double t2 = 1 + t1;
	 const double t3 = pow(t2, -1.0 / b4);
	 residuals[0] = b1 * t3 - y_;

         if (!jacobians) return true;
	 double* jacobian = jacobians[0];
	 if (!jacobian) return true;

	 const double t4 = pow(t2, -1.0 / b4 - 1);
	 jacobian[0] = t3;
	 jacobian[1] = -b1 * t1 * t4 / b4;
	 jacobian[2] = -x_ * jacobian[1];
	 jacobian[3] = b1 * log(t2) * t3 / (b4 * b4);
	 return true;
       }

     private:
       const double x_;
       const double y_;
   };

As can be seen, even in the case of ``Rat43`` which is a fairly simple
curve, the symbolic differentiation and efficient implementation of
the cost function is a tedious and error prone process.

Pitfalls
--------

It is a common mistake to believe that hand written symbolic
derivatives result in the most efficient code. This is not
true. Automatic differentiation that will talk about in more detail
below usually has performance comparable to that of symbolic
differentiation at a fraction of the development cost.

Another thing to be careful about when working with deriving and
implementing symbolic derivatives is the possibility of `indeterminate
forms <https://en.wikipedia.org/wiki/Indeterminate_form>`_,
i.e. expressions of the form :math:`0/0, 0 \times \infty, \infty -
\infty, 0^0, 1^\infty` and :math:`\infty^0`. In these cases, special
care needs to be taken (e.g. `L'Hopital's rule
<https://en.wikipedia.org/wiki/L'H%C3%B4pital's_rule>`_). e.g.,

When should I use analytical derivatives?
-----------------------------------------

#. The expressions are simple, e.g. mostly linear.

#. A computer algebra system like `Maple
   <https://www.maplesoft.com/products/maple/>`_ , `Mathematica
   <https://www.wolfram.com/mathematica/>`_, or `SymPy
   <http://www.sympy.org/en/index.html>`_ can be used to symbolically
   differentiate the objective function and generate the ``C++`` to
   evaluate them.

#. Performance is of utmost concern and there is algebraic structure
   in the terms that you can exploit to get better performance than
   automatic differentiation. But performance of analytic
   differentiation is a tricky thing, and getting a substatial
   performance improvement over automatic differentiation commensurate
   with the development cost is usually not easy.

   One particular case where the performance of automatic
   differentiation is really large parameter blocks with relatively
   simple operations.


#. There is no other way to compute the derivatives, e.g. you
   wish to compute the derivative of the root of a polynomial:

   .. math::
     a_3(x,y)z^3 + a_2(x,y)z^2 + a_1(x,y)z + a_0(x,y) = 0


   with respect to :math:`x` and :math:`y`. This requires the use of
   the *Inverse Function Theorem*. (We will have more to say about
   this later in this section).

#. You love the chain rule and actually enjoy doing all the algebra by
   hand.


Numeric derivatives
===================

The other extreme from using analytic dervatives is to use numeric
differentiation to compute the derivatives. The key observation here
is that the process of differentiating a function :math:`f(x)` w.r.t
:math:`x` can be written as the limiting process:

.. math::
   Df(x) = \lim_{h \rightarrow 0} \frac{f(x + h) - f(x)}{h}

Now of course one cannot perform the limiting operation numerically on
a computer so we do the next best thing, which is choose a fixed small
value of :math:`h` and approximate the derivative as

.. math::
   Df(x) \approx \frac{f(x + h) - f(x)}{h}


The above formula is the simplest most basic form of numeric
differentiation. It is known as the *Forward Difference* formula.

So how would one go about constructing a numerically differentiated
version of ``Rat43CostFunction``. The first step is to define a
*Functor* that given the parameter values will evaluate the residual
for a given :math:`(x,y)`.

.. code-block:: c++

  struct Rat43CostFunctor {
    Rat43CostFunctor(const double x, const double y) : x(x), y(y) {}
    bool operator()(const double* parameters, double* residuals) const {
      const double b1 = parameters[0][0];
      const double b2 = parameters[0][1];
      const double b3 = parameters[0][2];
      const double b4 = parameters[0][3];
      residuals[0] = b1 * pow(1.0 + exp(b2 -  b3 * x_), -1.0 / b4) - y_;
      return true;
    }

    const double x_;
    const double y_;
  }


The next step is to construct a :class:`CostFunction` by using
:class:`NumericDiffCostFunction` to wrap an instance of
``Rat43CostFunctor`` as follows:

.. code-block:: c++

  typedef NumericDiffCostFunction<Rat43CostFunctor, FORWARD, 1, 4> Rat43CostFunction;
  CostFunction* cost_function = new Rat43CostFunction(new Rat43CostFunctor(x, y));

Compared to computing the Jacobian by hand, this is about the minimum
amount of work one can expect to do to define the cost function. The
only thing that the user really needs to do is to make sure that the
evaluation of the residual is implemented correctly and efficiently.

:class:`NumericDiffCostFunction` implements a generic algorithm to
numerically differentiate a given functor. While the actual
implementation of :class:`NumericDiffCostFunction` is complicated, the
net result is a ``CostFunction`` that roughly looks something like the
following:

.. code-block:: c++

  class Rat43CostFunction {
     public:
       Rat43CostFunction(const Rat43Functor* functor) : functor_(functor) {}
       virtual ~Rat43CostFunction() {}
       virtual bool Evaluate(double const* const* parameters,
                             double* residuals,
			     double** jacobians) const {
 	 functor_(parameters[0], residuals);
	 if (!jacobians) return true;
	 double* jacobian = jacobians[0];
	 if (!jacobian) return true;

	 const double f = residuals[0];
	 double parameters_plus_h[4];
	 for (int i = 0; i < 4; ++i) {
	   std::copy(parameters, parameters + 4, parameters_plus_h);
	   const double h = parameters[i] * 1e-6;
	   parameters_plus_h[i] += h;
           double f_plus;
  	   functor_(parameters_plus_h, &f_plus);
	   jacobian[i] = (f_plus - f) / h;
         }
	 return true;
       }

     private:
       scoped_ptr<Rat43Functor> functor_;
   };


Note the choice of step size in the above code:

.. math::
   h = x \times 10^{-6}.

Instead of an absolute step size, a relative step size of
:math:`10^{-6}` is used. This is the default when :math:`x` is away
from zero.  Near zero, the code uses to a fixed step size. The user
can control the relative step size by setting it in
:class:`NumericDiffOptions`. We have skipped over these details to
keep the code above simple.

Before going further, it is instructive to get an estimate of the
error in the forward difference formula. To that we start by
considering the `Taylor expansion
<https://en.wikipedia.org/wiki/Taylor_series>`_  of :math:`f` near
:math:`x`.

.. math::
   \begin{align}
   f(x+h) &= f(x) + h Df(x) + \frac{h^2}{2!} D^2f(x) +
   \frac{h^3}{3!}D^3f(x) + \cdots \\
   Df(x) &= \frac{f(x + h) - f(x)}{h} - \left [\frac{h}{2!}D^2f(x) +
   \frac{h^2}{3!}D^3f(x) + \cdots  \right]\\
   Df(x) &= \frac{f(x + h) - f(x)}{h} + O(h)
   \end{align}

So the error of the forward difference formula is :math:`O(h)`.

Forward differencing is a simple but not particularly good way of
approximating derivatives. A better method is to use the *Central
Difference* formula:

.. math::
   Df(x) \approx \frac{f(x + h) - f(x - h)}{2h}

Notice that if the value of :math:`f(x)` is known, the forward
difference formula only requires one extra evaluation, but the central
difference formula requires two evaluations, making it twice as
expensive.

Using central differences instead of forward differences in Ceres
Solver is a simple matter of changing a template argument to
:class:`NumericDiffCostFunction` as follows:

.. code-block:: c++

  typedef NumericDiffCostFunction<Rat43CostFunctor, CENTRAL, 1, 4> Rat43CostFunction;
  CostFunction* cost_function = new Rat43CostFunction(new Rat43CostFunctor(x, y));

But is the extra evaluation worth it? How much better is the Central
Difference formula compared to the Forward Difference formula?

Lets start by comparing the errors in approximation.

.. math::
   \begin{align}
  f(x + h) &= f(x) + h Df(x) + \frac{h^2}{2!}
  D^2f(x) + \frac{h^3}{3!} D^3f(x) + \frac{h^4}{4!} D^4f(x) + \cdots\\
    f(x - h) &= f(x) - h Df(x) + \frac{h^2}{2!}
  D^2f(x) - \frac{h^3}{3!} D^3f(c_2) + \frac{h^4}{4!} D^4f(x) +
  \cdots\\
  Df(x) & =  \frac{f(x + h) - f(x - h)}{2h} + \frac{h^2}{3!}
  D^3f(x) +  \frac{h^4}{5!}
  D^5f(x) + \cdots
   \end{align}

So the error of the Central Difference formula is
:math:`O(h^2)`. Recall that the error in the Forward Difference
formula is :math:`O(h)`.

To get a sense of the difference in performance of the two methods,
consider the problem of evaluating the derivative of the function
:math:`f(x) = \frac{e^x}{\sin x - x^2}` at :math:`x = 1.0`. It is
straightforward to see that :math:`Df(1.0) =
140.73773557129658`. Using this value as reference, we can now
compute the error in the forward and central difference formulae and
plot them.

.. figure:: forward_central_error.png
   :figwidth: 500px
   :height: 400px
   :align: center

   The red line is the error in the forward difference formula and the
   blue line is the error in the central difference formula.


Two things stand out in the above graph.

The forward difference formula is not a great method for evaluating
derivatives. Central differences converges much more quickly and is
more accurate. So unless the evaluation of :math:`f(x)` is so
expensive that you absolutely cannot afford the extra evaluation
required by central differences, **do not use the forward difference
formula**.

But even more important than that is the fact that neither formula
works well for a poorly chosen value of :math:`h`. The graph for both
modes have two distinct regions. At first, starting from a large value
of :math:`h` the error goes down as the effect of truncating the
Taylor series dominates, but as the value of :math:`h` continues to
decrease, the error starts increasing again as roundoff error starts
to dominate the computation.

Can we do better? Indeed we can, and there is a fairly large
literature on numeric differentiation. One approach that works quite
well is to apply *Richardson's deffered approach to the limit* to
problem of differentiation, which is also known as *Ridder's
Method*. The idea is quite simple.

Let us recall, the error in the central differences formula.

.. math::
   \begin{align}
   Df)(x) & =  \frac{f(x + h) - f(x - h)}{2h} + \frac{h^2}{3!}
   D^3f(x) +  \frac{h^4}{5!}
   D^5f(x) + \cdots\\
           & =  \frac{f(x + h) - f(x - h)}{2h} + K_2 h^2 + K_4 h^4 + \cdots
   \end{align}

Let us now define:

.. math::

   A(i,1) = \frac{f(x + h/2^{i-1}) - f(x - h/2^{i-1})}{2h/2^{i-1}}.

Then,

.. math::

   Df(x) & = A(1,1) + K_2 h^2 + K_4 h^4 + \cdots \\
   Df(x) & = A(2,1) + K_2 (h/2)^2 + K_4 (h/2)^4 + \cdots \\
   Df(x) & = \frac{4 A(2,1) - A(1,1)}{4 - 1} + O(h^4)

So by combining two finite difference estimates obtained by halfing
the step size, we have obtained an estimate for the derivative whose
error goes down as :math:`O(h^4)`. But we do not have to stop here, we
can iterate this process to obtain even more accurate estimates. We
define

.. math::

   A(1,2) =  \frac{4 A(2,1) - A(1,1)}{4 - 1}

or more generally

.. math::

   A(m, n) =  \frac{4 A(m + 1, n-1) - A(m,n)}{4^{n-1} - 1},\ \forall n
   > 1.

The error in :math:`A(1,n)` is :math:`O(h^{2n})`. By structuring the
computation in a tableau as

.. math::
   \begin{array}{ccccc}
   A(1,1) & A(2, 1) & A(3, 1) & A(4, 1) & \cdots\\
          & A(1, 2) & A(2, 2) & A(3, 2) & \cdots\\
	  &         & A(1, 3) & A(2, 3) & \cdots\\
	  &         & \vdots  &         &
   \end{array}

We can compute :math:`A(1,n)` for increasing values of :math:`n` by
moving from the left to the right. Applying this method to :math:`f(x)
= \frac{e^x}{\sin x - x^2}` starting with a fairly large step size
:math:`h = 0.1`, we get the tableau:

.. math::
   \begin{array}{rrrrr}
   141.678097131 &140.971663667 &140.7961454 &140.752333523 &140.741384778\\
   &140.736185846 &140.737639311 &140.737729564 &140.737735196\\
   & &140.737736209 &140.737735581 &140.737735571\\
   & & &140.737735571 &140.737735571\\
   & & & &140.737735571\\
   \end{array}

Compared to the *correct* value :math:`Df(1.0) =
140.73773557129658`,  :math:`A(1,5)` has a relative error of
:math:`10^{-13}`.

This tableau is Ridders' method. The resulting algorithm is an
multi-step adaptive scheme that stops automatically when the error in
the estimate of the derivative falls below a threshold. As you may
imagine, this is considerably more expensive than the central
differences formula. It is however significantly more robust and
accurate and frees the user from worrying about the right value of
:math:`h` at the expense of increased evaluation time.

Using Ridder's method instead of forward or central differences in
Ceres is again a simple matter of changing a template argument to
:class:`NumericDiffCostFunction` as follows:

.. code-block:: c++

  typedef NumericDiffCostFunction<Rat43CostFunctor, RIDDERS, 1, 4> Rat43CostFunction;
  CostFunction* cost_function = new Rat43CostFunction(new Rat43CostFunctor(x, y));

If you must use numeric differentiation, Ridders' method is an
good choice either if execution time is not a concern or the
objective function is such that determining a good static relative
step size is hard.

.. NOTE::
   Talk about the illconditioning
   Lowering of convergence rates
   Pitfalls of the power of numeric differentiation.

Automatic derivatives
=====================

Quick intro to what is Automatic Differentiation and how it differs
from symbolic/analytic and numeric differentiaton.


What is a Jet?
--------------

Let us begin by considering the concept of **Dual Number**. Dual
numbers are extensions of the real numbers analogous to complex
numbers: whereas complex numbers augment the reals by introducing an
imaginary unit :math:`\iota` such that :math:`\iota^2 = -1`, dual
numbers introduce an *infinitesimal* unit :math:`\epsilon` such that
:math:`\epsilon^2 = 0`. Dual numbers have two components: the *real*
component and the *infinitesimal* component, generally written as
:math:`x + y\epsilon`. Surprisingly, this leads to a convenient method
for computing exact derivatives without needing to manipulate
complicated symbolic expressions.

For example, consider the function

.. math::

   f(x) = x^2 ,

evaluated at :math:`10`. Using normal arithmetic, :math:`f(10) = 100`,
and :math:`Df(10) = 20`.  Next, augument :math:`10` with an
infinitesimal to get:

.. math::

   \begin{align}
   f(10 + \epsilon) &= (10 + \epsilon)^2\\
            &= 100 + 20 \epsilon + \epsilon^2\\
            &= 100 + 20 \epsilon
   \end{align}

Observe that the derivative of :math:`f(x)` with respect to :math:`x`
is simply the infinitesimal component of the value of :math:`f(x +
\epsilon)`.

Indeed this generalizes to functions which are not
polynomial. Consider an arbitrary differetiable function
:math:`f(x)`. Then we can evaluate :math:`f(a + \epsilon)` by
considering the Taylor expansion of :math:`f` near :math:`a`, which
gives us the infinite series

.. math::
   \begin{align}
   f(a + \epsilon) &= f(a) + Df(a) \epsilon + D^2f(a)
   \frac{\epsilon^2}{2} + D^3f(a) \frac{\epsilon^2}{6} + \cdots\\
   f(a + \epsilon) &= f(a) + Df(a) \epsilon
   \end{align}

Here we are using the fact that :math:`\epsilon^n = 0,\ \forall n >
1`.

Now, one does not usually evaluate functions by evaluating their
Taylor expansions, so in order for the above to work in practice, we
will need the ability to evaluate function :math:`f` not just on real
numbers but also on dual numbers. But before we get into that, what
about functions that depend on more than one variable? Say for example
a scalar function of two scalar parameters :math:`x` and :math:`y`:

.. math::

   f(x, y) = x^2 + xy

and we are interested in the partial derivative of this function at
:math:`x=1` and :math:`y = 3`, i.e., :math:`D_1f(1, 3)` and
:math:`D_2f(1, 3)`.

One method would be to do two evaluations. the first time replacing
:math:`x` with :math:`x + \epsilon`, the second time replacing
:math:`y` with :math:`y + \epsilon`. Which leads to:

.. math::

  \begin{align}
  f(1 + \epsilon, y) &= (1 + \epsilon)^2 + (1 + \epsilon)  3\\
                     &= 1 + 2  \epsilon + 3 + 3  \epsilon\\
                     &= 4 + 5  \epsilon\\
  f(1, 3 + \epsilon) &= 1^2 + 1 (3 + \epsilon)\\
                     &= 1 + 3 + \epsilon\\
                     &= 4 + \epsilon
  \end{align}

And we get :math:`D_1f(1,3)` and :math:`D_2f(1,3)` as the
coefficients of :math:`\epsilon` as expected. But we can do better and
introduce two infinitesimal symbols :math:`\epsilon_1` and
:math:`\epsilon_2` to go with :math:`x` and :math:`y` respectively
with the property that :math:`\forall i, j\ \epsilon_i \epsilon_j =
0`. Then we can get both the partial derivatives using a single
evaluation:

.. math::

  \begin{align}
  f(1 + \epsilon_1, 3 + \epsilon_2) &= (1 + \epsilon_1)^2 + (1 +
  \epsilon_1)  (3 + \epsilon_2)\\
              &= 1 + 2  \epsilon_1 +  3 +  3\epsilon_1 + \epsilon_2\\
              &= 4 + 5 \epsilon_1 + \epsilon_2
  \end{align}

As you may imagine this idea can be generalized for functions that
depend on any number of variables. A **Jet** is a
:math:`n`-dimensional dual number. It consists of a *real* part, we
will call it :math:`a` and a :math:`n`-dimensional *infinitesimal*
part :math:`\mathbf{v}`. So

.. math::
   x = a + \sum_i v_i \epsilon_i

The summation notation gets tedius, so we will also just write

.. math::
   x = a + \mathbf{v}.

Then, using the same Taylor series expansion used above, it is
straightforward to see that

.. math::

  f(a + \mathbf{v}) = f(a) + Df(a) \mathbf{v}.


.. NOTE::

   Everything below this needs to be re-done.

.. code-block:: c++

  struct Rat43CostFunctor {
    Rat43CostFunctor(const double x, const double y) : x_(x), y_(y) {}
    template <typename T>
    bool operator()(const T* parameters, T* residuals) const {
      const T b1 = parameters[0][0];
      const T b2 = parameters[0][1];
      const T b3 = parameters[0][2];
      const T b4 = parameters[0][3];
      residuals[0] = b1 * pow(1.0 + exp(b2 -  b3 * x_), -1.0 / b4) - y_;
      return true;
    }

    private:
      const double x_;
      const double y_;
  }


.. code-block:: c++

  CostFunction* cost_function =
        new AutoDiffCostFunction<Rat43CostFunctor, 1, 4>(
	  new Rat43CostFunctor(x, y));



.. code-block:: c++

  class Rat43CostFunction {
     public:
       Rat43CostFunction(const Rat43Functor* functor) : functor_(functor) {}
       virtual ~Rat43CostFunction() {}
       virtual bool Evaluate(double const* const* parameters,
                             double* residuals,
			     double** jacobians) const {
	 if (!jacobians) return functor_(parameters[0], residuals);

	 typedef Jet<double, 4> JetT;
	 JetT jets[4];
	 for (int i = 0; i < 4; ++i) {
	   jets[i].a = parameters[0][i];
	   jets[i].v.setZero();
	   jets[i].v[j] = 1.0;
	 }

	 T result;
	 functor_(jets, result);

	 residuals[0] = result.a;
	 for (int i = 0; i < 4; ++i) {
	   jacobians[0][i] = result.v[i];
	 }
	 return true;
       }

     private:
       scoped_ptr<Rat43Functor> functor_;
   };


but this doesn't really explain the magic of jets, as it is hidden
behind a bunch of operator overloading.

Let us assume that the parameter values at which we want to evaluate
the jacobian are :math:`b_1 = 0.5, b_2 = 1.2, b_3 = -1` and
:math:`b_4 = 1.3`.

Then, we will begin by defining four *Jets* of size 4 each, and to
make our subsequent derivation simple, we will also refer to them as :math:`b_1,
b_2, b_3` and :math:`b_4`.

.. math::
   \begin{align}
   b_1 &= \begin{bmatrix} 0.5 & ; & 1 & 0 & 0 & 0\end{bmatrix}\\
   b_2 &= \begin{bmatrix} 1.2 & ; & 0 & 1 & 0 & 0\end{bmatrix}\\
   b_3 &= \begin{bmatrix} -1 & ; & 0 & 0 & 1 & 0\end{bmatrix}\\
   b_4 &= \begin{bmatrix} 1.3 & ; & 0 & 0 & 0 & 1\end{bmatrix}\\
   \end{align}

So each Jet contains, five numbers. The first number is called the
zeroth order or the value part of the Jet. It carries the the value of
the function and the remaining numbers carry the derivatives w.r.t
each of the variables that we are interested in.


.. math::

   \begin{align}
   b_2 - b_3 x & = \begin{bmatrix} p_2 - p_3 x & ; & 0 & 1 & -x & 0\end{bmatrix}\\
   e^{b_2 - b_3x} & = \begin{bmatrix} e^{p_2 - p_3 x} & ; & 0 &
   e^{p_2 - p_3 x} & -xe^{p_2 - p_3 x} & 0\end{bmatrix}\\
   1 +  e^{b_2 - b_3x} & = \begin{bmatrix} 1 + e^{p_2 - p_3 x} & ;
   & 0 & e^{p_2 - p_3 x} & -xe^{p_2 - p_3 x} & 0\end{bmatrix}\\
   -\frac{1}{b_4} & = \begin{bmatrix} -\frac{1}{p_4} & ; & 0 & 0 & 0 &
   \frac{1}{p_4^2}\end{bmatrix}\\
   \left(1 +  e^{b_2 - b_3x}\right)^{-1/b_4} & = \begin{bmatrix} \left(1 + e^{p_2 - p_3 x}\right)^{-1/b_4} & ;
   & 0 & e^{p_2 - p_3 x} & -xe^{p_2 - p_3 x} & 0\end{bmatrix}
   \end{align}

.. code-block:: c++

   struct Rat43CostFunctor {
    Rat43CostFunctor(const double x, const double y) : x_(x), y_(y) {}
    bool operator()(const Jet<double, 4>* parameters, Jet<double, 4>* residuals) const {
      const Jet<double, 4> b1 = parameters[0][0];
      const Jet<double, 4> b2 = parameters[0][1];
      const Jet<double, 4> b3 = parameters[0][2];
      const Jet<double, 4> b4 = parameters[0][3];
      residuals[0] = b1 * pow(1.0 + exp(b2 -  b3 * x_), -1.0 / b4) - y_;
      return true;
    }

    private:
      const double x_;
      const double y_;
  }


Dumping below and will come back to these later on.


This of course begs the question,
When should one use analytic derivatives? A few situations come to
mind:


FAQs
----

:Q: I want to call another library from my automatically
    differentiated function?

:Q: When defining ``Rat43CostFunction``, you used
    :class:`SizedCostFunction`, but this requires knowing the number of
    parameter blocks and the size of each parameter block and the number
    of residuals at compile time. What if I do not know this information
    at compile time?

:A: I used :class:`SizedCostFunction` for
   convenience. :class:`SizedCostFunction` is a statically sized
   subclass of :class:`CostFunction`. :class:`CostFunction` is fully
   dynamic, in that all of its properties can be set at runtime. So if
   you are constructing cost functions whose structure is not known at
   compile time, use :class:`CostFunction` instead.

:Q: How does :class:`NumericDiffCostFunction` choose the step size
    :math:`h`? How do I control it?

:Q: Looking at ``Rat43CostFunctor::operator()``, the interface seems to
    place little to no restrictions on what can happen there. What
    should I not do inside ``Rat43CostFunctor::operator()``?

:A: One, don't do non-differentiable operations. The
    formulae used for approximating the derivatives using numerical
    differentiation assume the existence of the derivative. If
    function being differentiated is not differentiable at the point
    of interest, then all bets are off.

    Two, do not call iterative procedures or other *solver* routines.

    The derivative of the solver is not the same thing as
    differentiating the function being evaluated.

    It can usually be done much faster using alternate methods.

    Pay attention to the curvature of your function? what does that
    even mean?

:Q: All the size arguments in :class:`NumericDiffCostFunction` are
    specified via templates, what if I do not know the number of
    residuals at compile time? What if I do not know the number and/or
    the size of the parameter blocks at compile time ?

:A: If the number and size of parameter blocks is known at compile time
    but number of residuals is only known at run time, then you can
    still use :class:`NumericDiffCostFunction` as follows:

    .. code-block:: c++

      int num_residuals = 1;
      CostFunction* cost_function =
         new NumericDiffCostFunction<Rat43CostFunctor, CENTRAL, DYNAMIC, 4> (
 	   new Rat43CostFunctor(x, y), TAKE_OWNERSHIP, num_residuals);

    If the number and size of parameter blocks is also not known at
    compile time, then use :class:`DynamicNumericDiffCostFunction`
    instead of :class:`NumericDiffCostFunction`.


TODO
====

1. pit falls.
2. inverse function theorem
3. Numerically differentiating other libraries.
4. Grids, analytic and numerical derivatives.
5. Add references in the various sections about the things to
   do. NIST, RIDDER's METHOD, Numerical Recipes.
