.. _chapter-modeling_faqs:

.. default-domain:: cpp

.. cpp:namespace:: ceres

========
Modeling
========

Use analytical/automatic derivatives
====================================

This is the single most important piece of advice we can give to you. It is
tempting to take the easy way out and use numeric differentiation. This is a bad
idea. Numeric differentiation is slow, ill-behaved, hard to get right, and
results in poor convergence behaviour.

Ceres allows the user to define templated functors which will be automatically
differentiated. For most situations this is enough and we recommend using this
facility. In some cases the derivatives are simple enough or the performance
considerations are such that the overhead of automatic differentiation is too
much. In such cases, analytic derivatives are recommended.

The use of numerical derivatives should be a measure of last resort, where it is
simply not possible to write a templated implementation of the cost function.

In many cases it is not possible to do analytic or automatic differentiation of
the entire cost function, but it is generally the case that it is possible to
decompose the cost function into parts that need to be numerically
differentiated and parts that can be automatically or analytically
differentiated.

To this end, Ceres has extensive support for mixing analytic, automatic and
numeric differentiation. See :class:`CostFunctionToFunctor`.

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

When using Quaternions, consider using :class:`QuaternionManifold`
==================================================================

`Quaternions <https://en.wikipedia.org/wiki/Quaternion>`_ are a four dimensional
parameterization of the space of three dimensional rotations :math:`SO(3)`.
However, the :math:`SO(3)` is a three dimensional set, and so is the tangent
space of a Quaternion. Therefore, it is sometimes (not always) beneficial to
associate a local parameterization with parameter blocks representing a
Quaternion. Assuming that the order of entries in your parameter block is
:math:`w,x,y,z`, you can use :class:`QuaternionManifold`.

.. NOTE::

 If you are using `Eigen's Quaternion
 <https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1Quaternion.html>`_
 object, whose layout is :math:`x,y,z,w`, then you should use
 :class:`EigenQuaternionManifold`.


How do I solve problems with general linear & non-linear **inequality** constraints with Ceres Solver?
======================================================================================================

Currently, Ceres Solver only supports upper and lower bounds constraints on the
parameter blocks.

A crude way of dealing with inequality constraints is have one or more of your
cost functions check if the inequalities you are interested in are satisfied,
and if not return false instead of true. This will prevent the solver from ever
stepping into an infeasible region.

This requires that the starting point for the optimization be a feasible point.
You also risk pre-mature convergence using this method.

How do I solve problems with general linear & non-linear **equality** constraints with Ceres Solver?
====================================================================================================

There is no built in support in ceres for solving problems with equality
constraints.  Currently, Ceres Solver only supports upper and lower bounds
constraints on the parameter blocks.

The trick described above for dealing with inequality constraints will **not**
work for equality constraints.

How do I set one or more components of a parameter block constant?
==================================================================

Using :class:`SubsetManifold`.
