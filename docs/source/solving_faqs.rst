.. _chapter-solving_faqs:

.. default-domain:: cpp

.. cpp:namespace:: ceres

=======
Solving
=======

How do I evaluate the Jacobian for a solved problem?
====================================================

Call :func:`Problem::Evaluate` after solving to compute the residuals, gradient,
and Jacobian at the solution.

How do I choose the right linear solver?
========================================

When using the ``TRUST_REGION`` minimizer, the choice of linear solver strongly
affects both solution quality and runtime:

1. For small problems (a few hundred parameters) or dense problems, use
   ``DENSE_QR`` or ``DENSE_NORMAL_CHOLESKY``. On systems with an
   NVIDIA GPU, setting
   :member:`Solver::Options::dense_linear_algebra_library_type` to
   ``CUDA`` can accelerate larger dense problems.

2. For general sparse problems (where the Jacobian matrix has a
   substantial number of zeros), use
   ``SPARSE_NORMAL_CHOLESKY`` (backed by ``SUITE_SPARSE``,
   ``ACCELERATE_SPARSE``, ``EIGEN_SPARSE``, or ``CUDA_SPARSE`` via
   NVIDIA ``cuDSS``).

3. For bundle adjustment problems with up to a hundred or so
   cameras, use ``DENSE_SCHUR``.

4. For larger bundle adjustment problems with sparse Schur
   complement (reduced camera) matrices, use ``SPARSE_SCHUR``.

   If you do not have access to a sparse Cholesky library,
   ``ITERATIVE_SCHUR`` with ``SCHUR_JACOBI`` is an
   excellent alternative.

5. For large bundle adjustment problems (a few thousand cameras or
   more), use the ``ITERATIVE_SCHUR`` solver. Among its preconditioners,
   ``SCHUR_JACOBI`` and ``SCHUR_POWER_SERIES_EXPANSION`` offer an
   excellent balance of speed and accuracy. They are also recommended for
   medium-sized problems when ``DENSE_SCHUR`` is too slow and
   ``SuiteSparse`` is unavailable.

   .. NOTE::

     For small to medium-sized problems, setting
     ``Solver::Options::use_explicit_schur_complement`` to ``true`` can
     substantially improve performance.

   If ``SCHUR_JACOBI`` is not fast enough, try
   ``CLUSTER_JACOBI`` and ``CLUSTER_TRIDIAGONAL`` in that
   order. Both require a sparse Cholesky library such as
   ``SuiteSparse`` and a clustering algorithm; try ``SINGLE_LINKAGE``
   before ``CANONICAL_VIEWS``.

Use :func:`Solver::Summary::FullReport` to diagnose performance problems
=========================================================================

When diagnosing Ceres runtime or convergence issues, start by inspecting the
output of :func:`Solver::Summary::FullReport`. Here is an example:

.. code-block:: console

   ./bin/bundle_adjuster --input ../data/problem-16-22106-pre.txt

   iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
      0  4.185660e+06    0.00e+00    2.16e+07   0.00e+00   0.00e+00  1.00e+04       0    7.50e-02    3.58e-01
      1  1.980525e+05    3.99e+06    5.34e+06   2.40e+03   9.60e-01  3.00e+04       1    1.84e-01    5.42e-01
      2  5.086543e+04    1.47e+05    2.11e+06   1.01e+03   8.22e-01  4.09e+04       1    1.53e-01    6.95e-01
      3  1.859667e+04    3.23e+04    2.87e+05   2.64e+02   9.85e-01  1.23e+05       1    1.71e-01    8.66e-01
      4  1.803857e+04    5.58e+02    2.69e+04   8.66e+01   9.93e-01  3.69e+05       1    1.61e-01    1.03e+00
      5  1.803391e+04    4.66e+00    3.11e+02   1.02e+01   1.00e+00  1.11e+06       1    1.49e-01    1.18e+00

   Solver Summary (v 2.3.0-eigen-(3.4.0)-lapack-suitesparse-(7.1.0)-metis-(5.1.0)-acceleratesparse-eigensparse)
   ----------------------------------
                                        Original                  Reduced
   Parameter blocks                        22122                    22122
   Parameters                              66462                    66462
   Residual blocks                         83718                    83718
   Residuals                              167436                   167436

   Minimizer                        TRUST_REGION

   Sparse linear algebra library    SUITE_SPARSE
   Trust region strategy     LEVENBERG_MARQUARDT

                                           Given                     Used
   Linear solver                    SPARSE_SCHUR             SPARSE_SCHUR
   Threads                                     1                        1
   Linear solver ordering              AUTOMATIC                22106, 16

   Cost:
   Initial                          4.185660e+06
   Final                            1.803391e+04
   Change                           4.167626e+06

   Minimizer iterations                        5
   Successful steps                            5
   Unsuccessful steps                          0

   Time (in seconds):
   Preprocessor                            0.283

     Residual only evaluation              0.061 (5)
     Jacobian & residual evaluation        0.361 (6)
     Linear solver                         0.382 (5)
   Minimizer                               0.895

   Postprocessor                           0.002
   Total                                   1.220

   Termination:                   NO_CONVERGENCE (Maximum number of iterations reached.)

Let us focus on runtime performance. The timing breakdown is:

.. code-block:: console

   Time (in seconds):
   Preprocessor                            0.283

     Residual only evaluation              0.061 (5)
     Jacobian & residual evaluation        0.361 (6)
     Linear solver                         0.382 (5)
   Minimizer                               0.895

   Postprocessor                           0.002
   Total                                   1.220

Of the total 1.220 seconds, 0.382 seconds was spent in the linear solver, while
most of the remaining time went to preprocessing and Jacobian evaluation.

Preprocessing is particularly expensive here. Looking back at the report, we see:

.. code-block:: console

   Linear solver ordering              AUTOMATIC                22106, 16

This line shows that ``SPARSE_SCHUR`` is computing the elimination ordering
automatically, which can be expensive. Supplying the ordering manually avoids
this overhead; in ``bundle_adjuster``, pass ``-ordering_type=user``. The timing
block then becomes:

.. code-block:: console

   Time (in seconds):
   Preprocessor                            0.051

     Residual only evaluation              0.053 (5)
     Jacobian & residual evaluation        0.344 (6)
     Linear solver                         0.372 (5)
   Minimizer                               0.854

   Postprocessor                           0.002
   Total                                   0.935

The preprocessor time drops by a factor of more than 5.5.
