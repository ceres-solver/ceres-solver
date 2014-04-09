.. Ceres Solver documentation master file, created by
   sphinx-quickstart on Sat Jan 19 00:07:33 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

============
Ceres Solver
============

.. toctree::
   :maxdepth: 3
   :hidden:

   building
   tutorial
   modeling
   solving
   faqs
   contributing
   version_history
   about
   bibliography
   license

Ceres Solver is an industrial-grade open source C++ library for
modeling and solving `nonlinear least squares`_ problems. It is used
in Google `Street View`_, Google `PhotoTours`_, Google `PhotoSphere`_,
`Project Tango`_, `Willow Garage`_, `Blender`_, and more.

.. _nonlinear least squares: http://en.wikipedia.org/wiki/Non-linear_least_squares
.. _fitting curves: http://en.wikipedia.org/wiki/Nonlinear_regression
.. _3D models from photographs: http://en.wikipedia.org/wiki/Structure_from_motion
.. _Street View: http://youtu.be/z00ORu4bU-A
.. _PhotoTours: http://google-latlong.blogspot.com/2012/04/visit-global-landmarks-with-photo-tours.html
.. _PhotoSphere: http://www.google.com/maps/about/contribute/photosphere/
.. _Project Tango: https://www.google.com/atap/projecttango/
.. _Blender: http://mango.blender.org/development/planar-tracking-preview/
.. _Willow Garage: https://www.willowgarage.com/blog/2013/08/09/enabling-robots-see-better-through-improved-camera-calibration

Features
--------

* Modeling API - Build your objective function one term at a
  time with arbitrary interactions between variables.
* Automatic and numeric differentiation - Never compute (unless you
  really want to) derivatives by hand again.
* Robust loss functions to handle outliers.
* Local Parameterizations to handle parameters that lie on
  manifolds like rotations.
* Speed - well optimized code with and multi-threaded Jacobian
  evaluators and linear solvers.
* Excellent `solution quality`_
* Multiple non-linear solvers - Trust region (Levenberg-Marquardt
  and Dogleg (Powell & Subspace)), Line Search (L-BFGS, Non-linear CG).
* Multiple linear solvers - Dense QR and Cholesky factorization (using
  `Eigen`_ or `LAPACK`_), sparse Cholesky factorization (using
  `SuiteSparse`_ or `CXSparse`) for large sparse problems.
* Covariance estimation - evaluate the uncertainty/sensitivity of your
  solutions.
* Specialized direct and iterative solvers for `bundle adjustment`_
  problems.
* Well documented and tested production quality code.
* Portable - runs on *Linux*, *Windows*, *Mac OS X*, *Android* and
  *iOS*.
* Actively developer community.
* BSD Licensed

.. _solution quality: https://groups.google.com/forum/#!topic/ceres-solver/UcicgMPgbXw
.. _bundle adjustment: http://en.wikipedia.org/wiki/Bundle_adjustment
.. _SuiteSparse: http://www.cise.ufl.edu/research/sparse/SuiteSparse/
.. _Eigen: http://eigen.tuxfamily.org/
.. _LAPACK: http://www.netlib.org/lapack/
.. _CXSparse: https://www.cise.ufl.edu/research/sparse/CXSparse/

Getting started
---------------

* Download the `latest stable release
  <http://ceres-solver.org/ceres-solver-1.8.0.tar.gz>`_
  or, for those wanting the latest
* Clone the development version or `browse the source
  <https://ceres-solver.googlesource.com/ceres-solver>`_

  .. code-block:: bash

       git clone https://ceres-solver.googlesource.com/ceres-solver

* Read the :ref:`chapter-tutorial`, browse :ref:`chapter-modeling` and :ref:`chapter-solving`.
* Join the `mailing list
  <https://groups.google.com/forum/?fromgroups#!forum/ceres-solver>`_
  and ask questions.
* File bugs, feature requests in the `issue tracker
  <https://code.google.com/p/ceres-solver/issues/list>`_.


Cite Us
-------
If you use Ceres Solver for a publication, please cite it as::

    @misc{ceres-solver,
      author = "Sameer Agarwal and Keir Mierle and Others",
      title = "Ceres Solver",
      howpublished = "\url{http://ceres-solver.org}",
    }
