========
Features
========
.. _chapter-features:

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
