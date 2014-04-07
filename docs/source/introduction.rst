.. _chapter-introduction:

============
Introduction
============

What is Ceres Solver?
---------------------
Ceres is an industrial-grade C++ library for modeling and solving large and
small nonlinear least squares problems of the form

.. math:: \frac{1}{2}\sum_{i} \rho_i\left(\left\|f_i\left(x_{i_1}, ... ,x_{i_k}\right)\right\|^2\right).

For a brief introduction to nonlinear solving in general, see the
:ref:`chapter-tutorial`.

Who uses Ceres Solver?
----------------------

* `Google Street View`_ panorama poses are computed with Ceres (`see video`_)
* `Google Photo Tours`_ employ Ceres to pose all the photos
* `Google Maps and Earth`_ imagery spatial alignment and satellite sensor calibration is done with Ceres
* `Project Tango`_ uses Ceres as part of the SLAM pipeline
* `Willow Garage's`_ SLAM pipeline uses Ceres for realtime bundle adjustment
* `Android`_ uses Ceres for image processing and stitching, including for `Photo Sphere`_
* `Blender's`_ `motion tracking module`_ depends critically on Ceres, using it
  for 2D tracking, 3D reconstruction, panorama tracking, and more; see the
  results in `Tears of Steel`_

.. _Google Street View: http://www.google.com/maps/about/behind-the-scenes/streetview/
.. _see video: https://www.youtube.com/watch?v=z00ORu4bU-A
.. _Google Photo Tours: http://googlesystem.blogspot.com/2012/04/photo-tours-in-google-maps.html
.. _Google Maps and Earth: http://www.google.com/earth/
.. _Project Tango: https://www.google.com/atap/projecttango/
.. _Willow Garage's: https://www.willowgarage.com/blog/2013/08/09/enabling-robots-see-better-through-improved-camera-calibration
.. _Android: https://android.googlesource.com/platform/external/ceres-solver/
.. _Photo Sphere: http://www.google.com/maps/about/contribute/photosphere/
.. _Blender's: http://blender.org
.. _motion tracking module: http://wiki.blender.org/index.php/Doc:2.6/Manual/Motion_Tracking
.. _Tears of Steel: http://mango.blender.org/

Why use Ceres Solver?
---------------------
* Ceres has an **integrated modelling layer**, making it easy and intutive to
  model large, complex cost functions with interacting terms, such as a moving
  vehicle with multiple sensors and tricky dynamics
* Ceres has **integrated automatic differentiation**, avoiding the error-prone
  task of manually computing derivatives
* Ceres can model a **wide variety of problems**, beyond simple nonlinear least
  squares, thanks to robust loss functions and local parameterizations (e.g.
  for quaternions)
* Ceres is **very fast**, thanks to threaded cost function evaluators, threaded linear
  solvers, and generous amounts of engineering time spent optimizing
* Ceres has **multiple nonlinear solvers** including trust region (fast, uses
  more memory) and line search (slower, uses less memory)
* Ceres has **multiple linear solvers** for both sparse and dense systems,
  leveraging Eigen or MKL for dense solving, CHOLMOD or CXSparse for sparse
  solving, and specialized solvers bundle adjustment
* Ceres has **thorough automated tests** ensuring it is high-quality
* Ceres is **industrial grade** thanks to **many compute-years** spent
  running its code, analyzing the results, and improving it
* Ceres has **world-class solution quality**, with the best known results of
  any least squares solver on the `NIST least squares precision benchmark`_
* Ceres has an **active community** encouraging contributions and mentoring
  those starting out
* Ceres runs on **many platforms** including Linux, Windows, Mac OS X, Android, and
  iOS (sort of)
* Ceres is **liberally licensed (BSD)** so that you can use it freely in
  commercial applications without releasing your code

.. _NIST least squares precision benchmark: https://groups.google.com/forum/#!topic/ceres-solver/UcicgMPgbXw
