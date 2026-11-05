.. default-domain:: cpp

.. cpp:namespace:: ceres

====
Why?
====
.. _chapter-features:

* **Code Quality.** Ceres Solver has been used in production at Google
  since 2010. It is clean, extensively tested, and well-documented code
  that is actively developed and supported.

* **Modeling API.** One rarely starts with the exact, final formulation
  of an optimization problem. Ceres's modeling API lets you build and
  modify the objective function one term at a time, without worrying
  about how the solver will handle the resulting changes in sparsity or
  problem structure.

  - **Derivatives.** Supplying derivatives is often the most tedious
    and error-prone part of using an optimization library. Ceres ships
    with `automatic`_ and `numeric`_ differentiation, so you never have
    to compute derivatives by hand unless you want to. Ceres also lets
    you mix automatic, numeric, and analytical derivatives in any
    combination.

  - **Robust Loss Functions.** Most non-linear least squares problems
    fit models to data, and real data contain outliers. Ceres lets you
    shape residuals with a :class:`LossFunction` to reduce the
    influence of outliers.

  - **Manifolds.** Parameters often lie on a nonlinear manifold rather
    than in Euclidean space, such as rotation matrices in :math:`SO(3)`.
    You can specify the geometry of the local tangent space using a
    :class:`Manifold` object.

* **Solver Choice.** Different problem sizes, sparsity structures, and
  time or memory budgets call for different algorithms. Ceres Solver
  provides both trust-region and line-search optimizers:

  - **Trust Region Solvers.** Ceres supports Levenberg-Marquardt,
    Powell's Dogleg, and Subspace Dogleg methods. The dominant
    computational cost in each iteration is solving a linear least
    squares system. Ceres ships with linear solvers tailored to major
    problem classes: dense QR and dense Cholesky factorization (via
    `Eigen`_, `LAPACK`_, or `CUDA`_) for dense problems; sparse
    Cholesky factorization (via `SuiteSparse`_, `Apple's Accelerate`_,
    `Eigen`_, or `NVIDIA cuDSS`_) for general sparse problems; and
    Schur complement based dense, sparse, and iterative linear solvers
    for `bundle adjustment`_ problems.

  - **Line Search Solvers.** When the problem is so large that storing
    and factoring the Jacobian is infeasible, or when a low-accuracy
    solution is needed quickly, Ceres offers line search algorithms,
    including several variants of Non-linear Conjugate Gradients, BFGS,
    and L-BFGS.

* **Speed.** Ceres Solver is extensively optimized using C++ templates,
  hand-tuned linear algebra kernels, and multithreaded Jacobian
  evaluation and linear solvers.

* **GPU Acceleration.** On systems with an NVIDIA GPU, Ceres Solver can
  use `CUDA`_ to accelerate both dense and sparse linear solvers,
  including mixed-precision solves and sparse Cholesky factorization
  via `NVIDIA cuDSS`_.

* **Solution Quality.** Ceres is the `best performing`_ solver on the
  NIST problem set used by Mondragon and Borchers to benchmark
  non-linear least squares solvers.

* **Covariance Estimation.** You can evaluate the sensitivity and
  uncertainty of a solution by computing all or part of its covariance
  matrix. Ceres is one of the few solvers capable of performing this
  analysis at scale.

* **Community.** Since its release as open-source software, Ceres has
  developed an active developer community that contributes new
  features, bug fixes, and user support.

* **Portability.** Runs on Linux, Windows, macOS, Android, and iOS.

* **BSD Licensed.** The 3-clause BSD license gives you the flexibility
  to ship Ceres in commercial and open-source applications.

.. _best performing: https://groups.google.com/forum/#!topic/ceres-solver/UcicgMPgbXw
.. _bundle adjustment: https://en.wikipedia.org/wiki/Bundle_adjustment
.. _SuiteSparse: https://people.engr.tamu.edu/davis/suitesparse.html
.. _Eigen: https://libeigen.gitlab.io/
.. _LAPACK: https://www.netlib.org/lapack/
.. _automatic: https://en.wikipedia.org/wiki/Automatic_differentiation
.. _numeric: https://en.wikipedia.org/wiki/Numerical_differentiation
.. _CUDA : https://developer.nvidia.com/cuda/toolkit
.. _NVIDIA cuDSS: https://developer.nvidia.com/cudss
.. _Apple's Accelerate: https://developer.apple.com/documentation/accelerate/sparse-solvers-library
