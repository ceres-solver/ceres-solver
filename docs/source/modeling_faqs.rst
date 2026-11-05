.. _chapter-modeling_faqs:

.. default-domain:: cpp

.. cpp:namespace:: ceres

========
Modeling
========

Use analytical or automatic derivatives
=======================================

Avoid numeric differentiation whenever possible. Numeric differentiation is slow,
sensitive to step-size choices and floating-point roundoff, and prone to poor
convergence.

Ceres can automatically differentiate templated cost functors, which is the
recommended approach for most problems. When the derivatives have a simple
closed form or the performance overhead of automatic differentiation is too
high, analytic derivatives are worth implementing directly.

Reserve numerical derivatives for cases where a templated implementation of the
cost function is impossible, such as when calling an external library. Even
then, you can often isolate the non-templated evaluation in a helper and
combine numeric differentiation of that part with automatic or analytic
differentiation of the rest using :class:`CostFunctionToFunctor`.

Do not return the norm or squared norm of a residual vector from a :class:`CostFunction`
========================================================================================

A :class:`CostFunction` should output the signed components of the residual
vector :math:`r(x) \in \mathbb{R}^k` directly, never its Euclidean norm
:math:`\|r(x)\|` or squared norm :math:`\|r(x)\|^2`. Ceres already forms the
sum of squared components :math:`\frac{1}{2} \|r(x)\|^2 = \frac{1}{2} \sum_{i=1}^k r_i(x)^2`
internally.

Collapsing a :math:`k`-dimensional residual vector into a scalar :math:`\tilde{r}(x) = \|r(x)\|`
leaves the objective value unchanged, but replaces the :math:`k \times n`
Jacobian :math:`J(x)` with the single row :math:`\frac{r(x)^\top}{\|r(x)\|} J(x)`.
The Gauss-Newton Hessian approximation then drops from :math:`J(x)^\top J(x)` to
the rank-1 matrix :math:`\frac{1}{\|r(x)\|^2} J(x)^\top r(x) r(x)^\top J(x)`,
discarding curvature information in all directions orthogonal to :math:`r(x)`
and stalling convergence. Worse, :math:`\|r(x)\|` is not differentiable at
:math:`r(x) = 0`, producing ``NaN`` Jacobians at the exact point the solver is
trying to reach. Returning :math:`\|r(x)\|^2` avoids the ``NaN`` but changes the
objective to :math:`\frac{1}{2}\|r(x)\|^4`, whose Jacobian :math:`2 r(x)^\top J(x)`
vanishes as :math:`r(x) \to 0`, making the Gauss-Newton system singular near the
solution.

Do not pack multiple independent measurements into a single residual block when using a :class:`LossFunction`
=============================================================================================================

A :class:`LossFunction` operates on the squared norm
:math:`s = \|f_i(x)\|^2 = \sum_{j=1}^{k_i} f_{ij}(x)^2` of an entire residual
block, not on its individual scalar components :math:`f_{ij}(x)` separately.
For a multi-dimensional measurement such as a 2D image reprojection error, this
is desirable: :math:`\rho(\|f_i(x)\|^2)` is isotropic (rotationally invariant
in measurement space) and down-weights both coordinates together whenever the
feature match is an outlier.

Conversely, you should not pack multiple independent measurements into a single
:class:`CostFunction` if you intend to use a non-trivial :class:`LossFunction`.
A single outlier component will inflate :math:`\|f_i(x)\|^2` and cause the
:class:`LossFunction` to down-weight every other inlier measurement in the same
residual block. Independent measurements that can fail independently should be
added as separate residual blocks.

When using Quaternions, consider using :class:`QuaternionManifold`
==================================================================

`Quaternions <https://en.wikipedia.org/wiki/Quaternion>`_ are a
four-dimensional parameterization of the three-dimensional rotation group
:math:`SO(3)`. Because :math:`SO(3)` is three-dimensional, the tangent space of
a unit quaternion is also three-dimensional, so it is often beneficial to
attach a :class:`Manifold` to quaternion parameter blocks. For parameter blocks
stored in :math:`w, x, y, z` order, use :class:`QuaternionManifold`.

.. NOTE::

 If you are using `Eigen's Quaternion
 <https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1Quaternion.html>`_,
 whose memory layout is :math:`x, y, z, w`, use
 :class:`EigenQuaternionManifold` instead.


How do I solve problems with general linear & non-linear **inequality** constraints with Ceres Solver?
======================================================================================================

Ceres Solver directly supports only lower and upper bound constraints on
parameter blocks.

A crude workaround for general inequality constraints is to have one or more
cost functions check whether the inequalities are satisfied and return ``false``
when they are violated. Returning ``false`` rejects the candidate step and
prevents the solver from entering the infeasible region, provided the initial
parameter values are feasible. Because the solver cannot see the constraint
boundary until a step crosses it, however, this approach can cause premature
convergence.

How do I solve problems with general linear & non-linear **equality** constraints with Ceres Solver?
====================================================================================================

Ceres Solver does not support general equality constraints directly; it supports
only lower and upper bound constraints on parameter blocks. The step-rejection
workaround described above for inequality constraints does **not** work for
equality constraints.

How do I set one or more components of a parameter block constant?
==================================================================

Attach a :class:`SubsetManifold` to the parameter block to hold the desired
coordinates constant during optimization.
