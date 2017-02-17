.. default-domain:: cpp

.. cpp:namespace:: ceres

.. _chapter-spivak_notation:

===============
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

More generally for a multivariate function :math:`g:\mathbb{R}^m
\rightarrow \mathbb{R}^n`, :math:`Dg` denotes the :math:`n\times m`
Jacobian matrix. :math:`D_i g` is the partial derivative of :math:`g`
w.r.t the :math:`i^{\text{th}}` coordinate and the
:math:`i^{\text{th}}` column of :math:`Dg`.

Finally, :math:`D^2_1g, D_1D_2g` have the obvious meaning as higher
order partial derivatives derivatives.

For more see Michael Spivak's book `Calculus on Manifolds
<https://www.amazon.com/Calculus-Manifolds-Approach-Classical-Theorems/dp/0805390219>`_
or a brief discussion of the `merits of this notation
<http://www.vendian.org/mncharity/dir3/dxdoc/>`_ by
Mitchell N. Charity.
