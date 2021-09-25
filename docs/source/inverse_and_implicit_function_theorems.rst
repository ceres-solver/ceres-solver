.. default-domain:: cpp

.. cpp:namespace:: ceres

.. _chapter-inverse_function_theorem:

==========================================
Using Inverse & Implicit Function Theorems
==========================================

In this chapter we will see, how we can use two basic results from
multi-variate calculus to compute derivatives where the usual approach
to computing analytic or automatic derivatives fails.

Inverse Function Theorem
========================

Suppose we wish to evaluate the derivative of a function :math:`f(x)`,
but evaluating :math:`f(x)` is not easy. Say it involves running an
iterative algorithm. You could try automatically differentiating the
iterative algorithm, but even if that is possible, it can become quite
expensive.

In some cases however, we get lucky, and computing the inverse of
:math:`f(x)` is an easy operation. In these cases, the we can use the
`Inverse Function Theorem
<http://en.wikipedia.org/wiki/Inverse_function_theorem>`_ to compute
the derivative exactly. Here is now.

The key result here is. Assuming that :math:`y=f(x)` is continuously
differentiable in the neighborhood of a point :math:`x` and
:math:`Df(x)` is the invertible Jacobian of :math:`f` at :math:`x`,
then :math:`Df^{-1}(y) = [Df(x)]^{-1}`, i.e., the Jacobian of
:math:`f^{-1}` is the inverse of the Jacobian of :math:`f`.

You maybe wondering why the above is true. A smoothly differentiable
function in a small neighborhood is well approximated by a linear
function. Indeed this is a good way to think about the Jacobian, it is
the matrix that best approximates the function linearly. Once you do
that, it is straightforward to see that the *locally* :math:`f^-1(y)`
is best approximated linearly by the inverse of the Jacobian of
:math:`f(x)`. Here is a practical example.

Geodetic Coordinate System Conversion
-------------------------------------

When working with data related to the Earth, one can use two different
coordinate system. The familiar (latitude, longitude, height)
Latitude-Longitude-Altitude coordinate system or the `ECEF
<http://en.wikipedia.org/wiki/ECEF>`_ coordinate system. The former is
familiar but is not terribly convenient analytically. The latter is a
Cartesian system but not particularly intuitive. So system that
process earth related data have to go back and forth between these
coordinate systems.

The conversion between the LLA and the ECEF coordinate system requires
a model of the Earth, the most commonly used one being `WGS84
<https://en.wikipedia.org/wiki/World_Geodetic_System#1984_version>`_.

Going from the spherical :math:`(\phi,\lambda,h)` to the ECEF
:math:`(x,y,z)` coordinates is easy.

.. math::

   \chi &= \sqrt{1 - e^2 \sin^2 \phi}

   X &= \left( \frac{a}{\chi} + h \right) \cos \phi \cos \lambda

   Y &= \left( \frac{a}{\chi} + h \right) \cos \phi \sin \lambda

   Z &= \left(\frac{a(1-e^2)}{\chi}  +h \right) \sin \phi

Here :math:`a` and :math:`e^2` are constants defined by `WGS84
<https://en.wikipedia.org/wiki/World_Geodetic_System#1984_version>`_.

Going from ECEF to LLA coordinates requires an iterative algorithm. So
to compute the derivative of the this transformation we invoke the
Inverse Function Theorem as follows:

.. code-block:: c++

   Eigen::Vector3d ecef; // Fill some values
   // Iterative computation.
   Eigen::Vector3d lla = ECEFToLLA(ecef);
   // Analytic derivatives
   Eigen::Matrix3d lla_to_ecef_jacobian = LLAToECEFJacobian(lla);
   bool invertible;
   Eigen::Matrix3d ecef_to_lla_jacobian;
   lla_to_ecef_jacobian.computeInverseWithCheck(ecef_to_lla_jacobian, invertible);


Implicit Function Theorem
=========================

An even more interesting case is when, we have two variables :math:`x
\in \mathbb{R}^m` and :math:`y \in \mathbb{R}^m` and a function
:math:`F:\mathbb{R}^m \times \mathbb{R}^n \rightarrow \mathbb{R}^n`
such that :math:`F(x,y) = 0` and we wish to calculate the Jacobian of
:math:`y` with respect to `x`. How do we do this?

If for a given value of :math:`(x,y)`, the partial Jacobian
:math:`D_2F(x,y)` is full rank, then the `Implicit Function Theorem
<https://en.wikipedia.org/wiki/Implicit_function_theorem>`_ tells us
that there exists a neighborhood of :math:`x` and a function :math:`G`
such :math:`y = G(x)` in this neighborhood. Differentiating
:math:`F(x,G(x))` gives us

.. math::

   D_1F(x,y) + D_2F(x,y)DG(x) &= 0

                        DG(x) &= -(D_2F(x,y))^{-1} D_1 F(x,y)

			D_x y &= -(D_2F(x,y))^{-1} D_1 F(x,y)

This means that we can compute the derivative of :math:`y` with
respect to :math:`x` by multiplying the Jacobian of :math:`F` w.r.t
:math:`x` by the inverse of the Jacobian of :math:`F` w.r.t :math:`y`.

Lets see an example.

Let :math:`\theta \in \mathbb{R}^m` be a vector, :math:`A(\theta) \in
\mathbb{R}^{k\times n}` be a matrix whose entries are a function of
:math:`\theta` with :math:`k \ge n` and let :math:`b \in \mathbb{R}^k`
be a constant vector, then consider the linear least squares problem:

.. math::

   x^* = \arg \min_x \|A(\theta) x - b\|_2^2

How do we compute :math:`D_\theta x^*(\theta)`?

One approach would be observe that :math:`x^*(\theta) =
[A^\top(\theta)A(\theta]^{-1}A^\top(\theta)b` and then differentiate
this w.r.t :math:`\theta`. But his would require differentiating
through the inverse of the matrix
:math:`[A^\top(\theta)A(\theta]^{-1}`. Not exactly easy. Lets use the
Implicit Function Theorem instead.

The first step is to observe that :math:`x^*` satisfies the so called
*normal equations*.

.. math::

   A^\top(\theta)A(\theta)x^* - A^\top(\theta)b = 0

We will compute :math:`D_\theta x^*` column-wise, treating
:math:`A(\theta)` as a function of one coordinate (:math:`\theta_i`)
of :math:`\theta` at a time.

So using the normal equations, lets define :math:`F(\theta_i, x^*) =
A^\top(\theta_i)A(\theta_i)x^* - A^\top(\theta_)b = 0`.

.. math::

   D_1F(\theta_i, x^*) &= DA^\top(\theta_i) (DA(\theta_i)x^* - b)

   D_2F(\theta_i, x^*) &= A^\top(\theta_i)A(\theta_i)

   D_{\theta_i} x^* & = (A^\top(\theta_i)A(\theta_i))^{-1} DA^\top(\theta_i) (b - DA(\theta_i)x^*)

   D_{\theta} x^* & = -(A^\top(\theta_i)A(\theta_i))^{-1} \left
   [DA^\top(\theta_1)(b - DA(\theta_1)x^*), \dots , DA^\top(\theta_n)(b - DA(\theta_m)x^*) \right]

Observe that we only need to compute one matrix inverse, to compute
the Jacobian, which we needed to compute anyways to compute
:math:`x^*`.

https://www.mail-archive.com/eigen@lists.tuxfamily.org/msg00460.html
