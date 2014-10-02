.. highlight:: c++

.. default-domain:: cpp

.. _chapter-gradient_problem_solver:

=========================================
Gradient Based Unconstrained Optimization
=========================================

While much of Ceres Solver is devoted to solving non-linear least
squares problems, internally it contains a more solver that can solve
general unconstrained optimization problems using just their objective
function value and gradients. The ``GradientProblem`` and
``GradientProblemSolver`` objects give the user access to this
solver.

So without much further ado, let us look at how one goes about using
them. We consider the minimization of the famous `Rosenbrock's
function <http://en.wikipedia.org/wiki/Rosenbrock_function>`_ [#f1]_.

We begin by defining an instance of the ``FirstOrderFunction``
interface. This is the object that is responsible for computing the
objective function value and the gradient (if required). This is the
analog of the :class:`CostFunction` when defining non-linear least
squares problems in Ceres.

.. code::

  class Rosenbrock : public ceres::FirstOrderFunction {
   public:
    virtual bool Evaluate(const double* parameters,
                          double* cost,
                          double* gradient) const {
      const double x = parameters[0];
      const double y = parameters[1];

      cost[0] = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
      if (gradient != NULL) {
        gradient[0] = -2.0 * (1.0 - x) - 200.0 * (y - x * x) * 2.0 * x;
        gradient[1] = 200.0 * (y - x * x);
      }
      return true;
    }

    virtual int NumParameters() const { return 2; }
  };


Minimizing it then is a straightforward matter of constructing a
:class:`GradientProblem` object and calling :func:`Solve` on it.

.. code::

    double parameters[2] = {-1.2, 1.0};

    ceres::GradientProblem problem(new Rosenbrock());

    ceres::GradientProblemSolver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(options, problem, parameters, &summary);

    std::cout << summary.FullReport() << "\n";

Executing this code results, solve the problem using limited memory
`BFGS
<http://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm>`_
algorithm.

.. code-block:: bash

     0: f: 2.420000e+01 d: 0.00e+00 g: 2.16e+02 h: 0.00e+00 s: 0.00e+00 e:  0 it: 2.00e-05 tt: 2.00e-05
     1: f: 4.280493e+00 d: 1.99e+01 g: 1.52e+01 h: 2.01e-01 s: 8.62e-04 e:  2 it: 7.32e-05 tt: 2.19e-04
     2: f: 3.571154e+00 d: 7.09e-01 g: 1.35e+01 h: 3.78e-01 s: 1.34e-01 e:  3 it: 2.50e-05 tt: 2.68e-04
     3: f: 3.440869e+00 d: 1.30e-01 g: 1.73e+01 h: 1.36e-01 s: 1.00e+00 e:  1 it: 4.05e-06 tt: 2.92e-04
     4: f: 3.213597e+00 d: 2.27e-01 g: 1.55e+01 h: 1.06e-01 s: 4.59e-01 e:  1 it: 2.86e-06 tt: 3.14e-04
     5: f: 2.839723e+00 d: 3.74e-01 g: 1.05e+01 h: 1.34e-01 s: 5.24e-01 e:  1 it: 2.86e-06 tt: 3.36e-04
     6: f: 2.448490e+00 d: 3.91e-01 g: 1.29e+01 h: 3.04e-01 s: 1.00e+00 e:  1 it: 4.05e-06 tt: 3.58e-04
     7: f: 1.943019e+00 d: 5.05e-01 g: 4.00e+00 h: 8.81e-02 s: 7.43e-01 e:  1 it: 4.05e-06 tt: 3.79e-04
     8: f: 1.731469e+00 d: 2.12e-01 g: 7.36e+00 h: 1.71e-01 s: 4.60e-01 e:  2 it: 9.06e-06 tt: 4.06e-04
     9: f: 1.503267e+00 d: 2.28e-01 g: 6.47e+00 h: 8.66e-02 s: 1.00e+00 e:  1 it: 3.81e-06 tt: 4.33e-04
    10: f: 1.228331e+00 d: 2.75e-01 g: 2.00e+00 h: 7.70e-02 s: 7.90e-01 e:  1 it: 3.81e-06 tt: 4.54e-04
    11: f: 1.016523e+00 d: 2.12e-01 g: 5.15e+00 h: 1.39e-01 s: 3.76e-01 e:  2 it: 1.00e-05 tt: 4.82e-04
    12: f: 9.145773e-01 d: 1.02e-01 g: 6.74e+00 h: 7.98e-02 s: 1.00e+00 e:  1 it: 3.10e-06 tt: 5.03e-04
    13: f: 7.508302e-01 d: 1.64e-01 g: 3.88e+00 h: 5.76e-02 s: 4.93e-01 e:  1 it: 2.86e-06 tt: 5.25e-04
    14: f: 5.832378e-01 d: 1.68e-01 g: 5.56e+00 h: 1.42e-01 s: 1.00e+00 e:  1 it: 3.81e-06 tt: 5.47e-04
    15: f: 3.969581e-01 d: 1.86e-01 g: 1.64e+00 h: 1.17e-01 s: 1.00e+00 e:  1 it: 4.05e-06 tt: 5.68e-04
    16: f: 3.171557e-01 d: 7.98e-02 g: 3.84e+00 h: 1.18e-01 s: 3.97e-01 e:  2 it: 9.06e-06 tt: 5.94e-04
    17: f: 2.641257e-01 d: 5.30e-02 g: 3.27e+00 h: 6.14e-02 s: 1.00e+00 e:  1 it: 3.10e-06 tt: 6.16e-04
    18: f: 1.909730e-01 d: 7.32e-02 g: 5.29e-01 h: 8.55e-02 s: 6.82e-01 e:  1 it: 4.05e-06 tt: 6.42e-04
    19: f: 1.472012e-01 d: 4.38e-02 g: 3.11e+00 h: 1.20e-01 s: 3.47e-01 e:  2 it: 1.00e-05 tt: 6.69e-04
    20: f: 1.093558e-01 d: 3.78e-02 g: 2.97e+00 h: 8.43e-02 s: 1.00e+00 e:  1 it: 3.81e-06 tt: 6.91e-04
    21: f: 6.710346e-02 d: 4.23e-02 g: 1.42e+00 h: 9.64e-02 s: 8.85e-01 e:  1 it: 3.81e-06 tt: 7.12e-04
    22: f: 3.993377e-02 d: 2.72e-02 g: 2.30e+00 h: 1.29e-01 s: 4.63e-01 e:  2 it: 9.06e-06 tt: 7.39e-04
    23: f: 2.911794e-02 d: 1.08e-02 g: 2.55e+00 h: 6.55e-02 s: 1.00e+00 e:  1 it: 4.05e-06 tt: 7.62e-04
    24: f: 1.457683e-02 d: 1.45e-02 g: 2.77e-01 h: 6.37e-02 s: 6.14e-01 e:  1 it: 3.81e-06 tt: 7.84e-04
    25: f: 8.577515e-03 d: 6.00e-03 g: 2.86e+00 h: 1.40e-01 s: 1.00e+00 e:  1 it: 4.05e-06 tt: 8.05e-04
    26: f: 3.486574e-03 d: 5.09e-03 g: 1.76e-01 h: 1.23e-02 s: 1.00e+00 e:  1 it: 4.05e-06 tt: 8.27e-04
    27: f: 1.257570e-03 d: 2.23e-03 g: 1.39e-01 h: 5.08e-02 s: 1.00e+00 e:  1 it: 4.05e-06 tt: 8.48e-04
    28: f: 2.783568e-04 d: 9.79e-04 g: 6.20e-01 h: 6.47e-02 s: 1.00e+00 e:  1 it: 4.05e-06 tt: 8.69e-04
    29: f: 2.533399e-05 d: 2.53e-04 g: 1.68e-02 h: 1.98e-03 s: 1.00e+00 e:  1 it: 3.81e-06 tt: 8.91e-04
    30: f: 7.591572e-07 d: 2.46e-05 g: 5.40e-03 h: 9.27e-03 s: 1.00e+00 e:  1 it: 3.81e-06 tt: 9.12e-04
    31: f: 1.902460e-09 d: 7.57e-07 g: 1.62e-03 h: 1.89e-03 s: 1.00e+00 e:  1 it: 2.86e-06 tt: 9.33e-04
    32: f: 1.003030e-12 d: 1.90e-09 g: 3.50e-05 h: 3.52e-05 s: 1.00e+00 e:  1 it: 3.10e-06 tt: 9.54e-04
    33: f: 4.835994e-17 d: 1.00e-12 g: 1.05e-07 h: 1.13e-06 s: 1.00e+00 e:  1 it: 4.05e-06 tt: 9.81e-04
    34: f: 1.885250e-22 d: 4.84e-17 g: 2.69e-10 h: 1.45e-08 s: 1.00e+00 e:  1 it: 4.05e-06 tt: 1.00e-03

  Solver Summary (v 1.10.0-lapack-suitesparse-cxsparse-no_openmp)

  Parameters                                  2
  Line search direction              LBFGS (20)
  Line search type                  CUBIC WOLFE


  Cost:
  Initial                          2.420000e+01
  Final                            1.885250e-22
  Change                           2.420000e+01

  Minimizer iterations                       35

  Time (in seconds):

    Cost evaluation                       0.000
    Gradient evaluation                   0.000
  Total                                   0.003

  Termination:                      CONVERGENCE (Gradient tolerance reached. Gradient max norm: 9.032775e-13 <= 1.000000e-10)

.. rubric:: Footnotes

.. [#f1] `examples/rosenbrock.cc
   <https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/rosenbrock.cc>`_


.. _section-api_reference:

API Reference
=============

:class:`FirstOrderFunction`
---------------------------

.. class:: FirstOrderFunction

  Instances of :class:`FirstOrderFunction` implement the evaluation of
  a function and its gradient.

  .. code-block:: c++

   class FirstOrderFunction {
     public:
      virtual ~FirstOrderFunction() {}
      virtual bool Evaluate(const double* const parameters,
                            double* cost,
                            double* gradient) const = 0;
      virtual int NumParameters() const = 0;
   };

.. function:: bool FirstOrderFunction::Evaluate(const double* const parameters, double* cost, double* gradient) const

   Evaluate the cost/value of the function. If ``gradient`` is not
   ``NULL`` then evaluate the gradient too. If evaluation is
   successful return, ``true`` else return ``false``.

   ``cost`` guaranteed to be never ``NULL``, ``gradient`` can be ``NULL``.

.. function:: int FirstOrderFunction::NumParameters() const

   Number of parameters in the domain of the function.


:class:`GradientProblem`
------------------------

.. class:: GradientProblem

.. code-block:: c++

  class GradientProblem {
   public:
    explicit GradientProblem(FirstOrderFunction* function);
    GradientProblem(FirstOrderFunction* function,
                    LocalParameterization* parameterization);
    int NumParameters() const;
    int NumLocalParameters() const;
    bool Evaluate(const double* parameters, double* cost, double* gradient) const;
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const;
  };

Instances of :class:`GradientProblem` represent general non-linear
optimization problems that must be solved using just the value of the
objective function and its gradient. Unlike the :class:`Problem`
class, which can only be used to model non-linear least squares
problems, instances of :class:`GradientProblem` not restricted in the
form of the objective function.

Structurally :class:`GradientProblem` is a composition of a
:class:`FirstOrderFunction` and optionally a
:class:`LocalParameterization`.

The :class:`FirstOrderFunction` is responsible for evaluating the cost
and gradient of the objective function.

The :class:`LocalParameterization` is responsible for going back and
forth between the ambient space and the local tangent space. When a
:class:`LocalParameterization` is not provided, then the tangent space
is assumed to coincide with the ambient Euclidean space that the
gradient vector lives in.

The constructor takes ownership of the :class:`FirstOrderFunction` and
:class:`LocalParamterization` objects passed to it.

:function`Solve`
----------------

.. class:: GradientProblemSolver

.. function: void Solve(const GradientProblemSolver::Options& options, const GradientProblem& problem, double* parameters, GradientProblemSolver::Summary* summary)

   Solve the given :class:`GradientProblem` using the values in
   ``parameters`` as the initial guess of the solution.

:class:`GradientProblemSolver::Options`
---------------------------------------

.. class:: GradientProblemSolver::Options

   :class:`GradientProblemSolver::Options` controls the overall
   behavior of the solver. We list the various settings and their
   default values below.

.. function:: bool GradientProblemSolver::Options::IsValid(string* error) const

   Validate the values in the options struct and returns true on
   success. If there is a problem, the method returns false with
   ``error`` containing a textual description of the cause.

.. member:: LineSearchDirectionType GradientProblemSolver::Options::line_search_direction_type

   Default: ``LBFGS``

   Choices are ``STEEPEST_DESCENT``, ``NONLINEAR_CONJUGATE_GRADIENT``,
   ``BFGS`` and ``LBFGS``.

.. member:: LineSearchType GradientProblemSolver::Options::line_search_type

   Default: ``WOLFE``

   Choices are ``ARMIJO`` and ``WOLFE`` (strong Wolfe conditions).
   Note that in order for the assumptions underlying the ``BFGS`` and
   ``LBFGS`` line search direction algorithms to be guaranteed to be
   satisifed, the ``WOLFE`` line search should be used.

.. member:: NonlinearConjugateGradientType GradientProblemSolver::Options::nonlinear_conjugate_gradient_type

   Default: ``FLETCHER_REEVES``

   Choices are ``FLETCHER_REEVES``, ``POLAK_RIBIERE`` and
   ``HESTENES_STIEFEL``.

.. member:: int GradientProblemSolver::Options::max_lbfs_rank

   Default: 20

   The L-BFGS hessian approximation is a low rank approximation to the
   inverse of the Hessian matrix. The rank of the approximation
   determines (linearly) the space and time complexity of using the
   approximation. Higher the rank, the better is the quality of the
   approximation. The increase in quality is however is bounded for a
   number of reasons.

     1. The method only uses secant information and not actual
        derivatives.

     2. The Hessian approximation is constrained to be positive
        definite.

   So increasing this rank to a large number will cost time and space
   complexity without the corresponding increase in solution
   quality. There are no hard and fast rules for choosing the maximum
   rank. The best choice usually requires some problem specific
   experimentation.

.. member:: bool GradientProblemSolver::Options::use_approximate_eigenvalue_bfgs_scaling

   Default: ``false``

   As part of the ``BFGS`` update step / ``LBFGS`` right-multiply
   step, the initial inverse Hessian approximation is taken to be the
   Identity.  However, [Oren]_ showed that using instead :math:`I *
   \gamma`, where :math:`\gamma` is a scalar chosen to approximate an
   eigenvalue of the true inverse Hessian can result in improved
   convergence in a wide variety of cases.  Setting
   ``use_approximate_eigenvalue_bfgs_scaling`` to true enables this
   scaling in ``BFGS`` (before first iteration) and ``LBFGS`` (at each
   iteration).

   Precisely, approximate eigenvalue scaling equates to

   .. math:: \gamma = \frac{y_k' s_k}{y_k' y_k}

   With:

  .. math:: y_k = \nabla f_{k+1} - \nabla f_k
  .. math:: s_k = x_{k+1} - x_k

  Where :math:`f()` is the line search objective and :math:`x` the
  vector of parameter values [NocedalWright]_.

  It is important to note that approximate eigenvalue scaling does
  **not** *always* improve convergence, and that it can in fact
  *significantly* degrade performance for certain classes of problem,
  which is why it is disabled by default.  In particular it can
  degrade performance when the sensitivity of the problem to different
  parameters varies significantly, as in this case a single scalar
  factor fails to capture this variation and detrimentally downscales
  parts of the Jacobian approximation which correspond to
  low-sensitivity parameters. It can also reduce the robustness of the
  solution to errors in the Jacobians.

.. member:: LineSearchIterpolationType GradientProblemSolver::Options::line_search_interpolation_type

   Default: ``CUBIC``

   Degree of the polynomial used to approximate the objective
   function. Valid values are ``BISECTION``, ``QUADRATIC`` and
   ``CUBIC``.

.. member:: double GradientProblemSolver::Options::min_line_search_step_size

   The line search terminates if:

   .. math:: \|\Delta x_k\|_\infty < \text{min_line_search_step_size}

   where :math:`\|\cdot\|_\infty` refers to the max norm, and
   :math:`\Delta x_k` is the step change in the parameter values at
   the :math:`k`-th iteration.

.. member:: double GradientProblemSolver::Options::line_search_sufficient_function_decrease

   Default: ``1e-4``

   Solving the line search problem exactly is computationally
   prohibitive. Fortunately, line search based optimization algorithms
   can still guarantee convergence if instead of an exact solution,
   the line search algorithm returns a solution which decreases the
   value of the objective function sufficiently. More precisely, we
   are looking for a step size s.t.

   .. math:: f(\text{step_size}) \le f(0) + \text{sufficient_decrease} * [f'(0) * \text{step_size}]

   This condition is known as the Armijo condition.

.. member:: double GradientProblemSolver::Options::max_line_search_step_contraction

   Default: ``1e-3``

   In each iteration of the line search,

   .. math:: \text{new_step_size} >= \text{max_line_search_step_contraction} * \text{step_size}

   Note that by definition, for contraction:

   .. math:: 0 < \text{max_step_contraction} < \text{min_step_contraction} < 1

.. member:: double GradientProblemSolver::Options::min_line_search_step_contraction

   Default: ``0.6``

   In each iteration of the line search,

   .. math:: \text{new_step_size} <= \text{min_line_search_step_contraction} * \text{step_size}

   Note that by definition, for contraction:

   .. math:: 0 < \text{max_step_contraction} < \text{min_step_contraction} < 1

.. member:: int GradientProblemSolver::Options::max_num_line_search_step_size_iterations

   Default: ``20``

   Maximum number of trial step size iterations during each line
   search, if a step size satisfying the search conditions cannot be
   found within this number of trials, the line search will stop.

   As this is an 'artificial' constraint (one imposed by the user, not
   the underlying math), if ``WOLFE`` line search is being used, *and*
   points satisfying the Armijo sufficient (function) decrease
   condition have been found during the current search (in :math:`<=`
   ``max_num_line_search_step_size_iterations``).  Then, the step size
   with the lowest function value which satisfies the Armijo condition
   will be returned as the new valid step, even though it does *not*
   satisfy the strong Wolfe conditions.  This behaviour protects
   against early termination of the optimizer at a sub-optimal point.

.. member:: int GradientProblemSolver::Options::max_num_line_search_direction_restarts

   Default: ``5``

   Maximum number of restarts of the line search direction algorithm
   before terminating the optimization. Restarts of the line search
   direction algorithm occur when the current algorithm fails to
   produce a new descent direction. This typically indicates a
   numerical failure, or a breakdown in the validity of the
   approximations used.

.. member:: double GradientProblemSolver::Options::line_search_sufficient_curvature_decrease

   Default: ``0.9``

   The strong Wolfe conditions consist of the Armijo sufficient
   decrease condition, and an additional requirement that the
   step size be chosen s.t. the *magnitude* ('strong' Wolfe
   conditions) of the gradient along the search direction
   decreases sufficiently. Precisely, this second condition
   is that we seek a step size s.t.

   .. math:: \|f'(\text{step_size})\| <= \text{sufficient_curvature_decrease} * \|f'(0)\|

   Where :math:`f()` is the line search objective and :math:`f'()` is the derivative
   of :math:`f` with respect to the step size: :math:`\frac{d f}{d~\text{step size}}`.

.. member:: double GradientProblemSolver::Options::max_line_search_step_expansion

   Default: ``10.0``

   During the bracketing phase of a Wolfe line search, the step size
   is increased until either a point satisfying the Wolfe conditions
   is found, or an upper bound for a bracket containing a point
   satisfying the conditions is found.  Precisely, at each iteration
   of the expansion:

   .. math:: \text{new_step_size} <= \text{max_step_expansion} * \text{step_size}

   By definition for expansion

   .. math:: \text{max_step_expansion} > 1.0

.. member:: int GradientProblemSolver::Options::max_num_iterations

   Default: ``50``

   Maximum number of iterations for which the solver should run.

.. member:: double GradientProblemSolver::Options::max_solver_time_in_seconds

   Default: ``1e6``
   Maximum amount of time for which the solver should run.

.. member:: double GradientProblemSolver::Options::function_tolerance

   Default: ``1e-6``

   Solver terminates if

   .. math:: \frac{|\Delta \text{cost}|}{\text{cost}} < \text{function_tolerance}

   where, :math:`\Delta \text{cost}` is the change in objective
   function value (up or down) in the current iteration of
   Levenberg-Marquardt.

.. member:: double GradientProblemSolver::Options::gradient_tolerance

   Default: ``1e-10``

   Solver terminates if

   .. math:: \|x - \Pi \boxplus(x, -g(x))\|_\infty < \text{gradient_tolerance}

   where :math:`\|\cdot\|_\infty` refers to the max norm, :math:`\Pi`
   is projection onto the bounds constraints and :math:`\boxplus` is
   Plus operation for the overall local parameterization associated
   with the parameter vector.

.. member:: LoggingType GradientProblemSolver::Options::logging_type

   Default: ``PER_MINIMIZER_ITERATION``

.. member:: bool GradientProblemSolver::Options::minimizer_progress_to_stdout

   Default: ``false``

   By default the :class:`Minimizer` progress is logged to ``STDERR``
   depending on the ``vlog`` level. If this flag is set to true, and
   :member:`GradientProblemSolver::Options::logging_type` is not
   ``SILENT``, the logging output is sent to ``STDOUT``.

   The progress display looks like

   .. code-block:: bash

      0: f: 2.317806e+05 d: 0.00e+00 g: 3.19e-01 h: 0.00e+00 s: 0.00e+00 e:  0 it: 2.98e-02 tt: 8.50e-02
      1: f: 2.312019e+05 d: 5.79e+02 g: 3.18e-01 h: 2.41e+01 s: 1.00e+00 e:  1 it: 4.54e-02 tt: 1.31e-01
      2: f: 2.300462e+05 d: 1.16e+03 g: 3.17e-01 h: 4.90e+01 s: 2.54e-03 e:  1 it: 4.96e-02 tt: 1.81e-01

   Here

   #. ``f`` is the value of the objective function.
   #. ``d`` is the change in the value of the objective function if
      the step computed in this iteration is accepted.
   #. ``g`` is the max norm of the gradient.
   #. ``h`` is the change in the parameter vector.
   #. ``s`` is the optimal step length computed by the line search.
   #. ``it`` is the time take by the current iteration.
   #. ``tt`` is the total time taken by the minimizer.

.. member:: vector<IterationCallback> GradientProblemSolver::Options::callbacks

   Callbacks that are executed at the end of each iteration of the
   :class:`Minimizer`. They are executed in the order that they are
   specified in this vector. See the documentation for
   :class:`IterationCallback` for more details.

   The solver does NOT take ownership of these pointers.


:class:`GradientProblemSolver::Summary`
---------------------------------------

.. class:: GradientProblemSolver::Summary

   Summary of the various stages of the solver after termination.

.. function:: string GradientProblemSolver::Summary::BriefReport() const

   A brief one line description of the state of the solver after
   termination.

.. function:: string GradientProblemSolver::Summary::FullReport() const

   A full multiline description of the state of the solver after
   termination.

.. function:: bool GradientProblemSolver::Summary::IsSolutionUsable() const

   Whether the solution returned by the optimization algorithm can be
   relied on to be numerically sane. This will be the case if
   `GradientProblemSolver::Summary:termination_type` is set to `CONVERGENCE`,
   `USER_SUCCESS` or `NO_CONVERGENCE`, i.e., either the solver
   converged by meeting one of the convergence tolerances or because
   the user indicated that it had converged or it ran to the maximum
   number of iterations or time.

.. member:: TerminationType GradientProblemSolver::Summary::termination_type

   The cause of the minimizer terminating.

.. member:: string GradientProblemSolver::Summary::message

   Reason why the solver terminated.

.. member:: double GradientProblemSolver::Summary::initial_cost

   Cost of the problem (value of the objective function) before the
   optimization.

.. member:: double GradientProblemSolver::Summary::final_cost

   Cost of the problem (value of the objective function) after the
   optimization.

.. member:: vector<IterationSummary> GradientProblemSolver::Summary::iterations

   :class:`IterationSummary` for each minimizer iteration in order.

.. member:: double GradientProblemSolver::Summary::total_time_in_seconds

   Time (in seconds) spent in the solver.

.. member:: double GradientProblemSolver::Summary::cost_evaluation_time_in_seconds

   Time (in seconds) spent evaluating the cost vector.

.. member:: double GradientProblemSolver::Summary::gradient_evaluation_time_in_seconds

   Time (in seconds) spent evaluating the gradient vector.

.. member:: int GradientProblemSolver::Summary::num_parameters

   Number of parameters in the problem.

.. member:: int GradientProblemSolver::Summary::num_local_parameters

   Dimension of the tangent space of the problem. This is different
   from :member:`GradientProblemSolver::Summary::num_parameters` if a
   :class:`LocalParameterization` object is used.

.. member:: LineSearchDirectionType GradientProblemSolver::Summary::line_search_direction_type

   Type of line search direction used.

.. member:: LineSearchType GradientProblemSolver::Summary::line_search_type

   Type of the line search algorithm used.

.. member:: LineSearchInterpolationType GradientProblemSolver::Summary::line_search_interpolation_type

   When performing line search, the degree of the polynomial used to
   approximate the objective function.

.. member:: NonlinearConjugateGradientType GradientProblemSolver::Summary::nonlinear_conjugate_gradient_type

   If the line search direction is `NONLINEAR_CONJUGATE_GRADIENT`,
   then this indicates the particular variant of non-linear conjugate
   gradient used.

.. member:: int GradientProblemSolver::Summary::max_lbfgs_rank

   If the type of the line search direction is `LBFGS`, then this
   indicates the rank of the Hessian approximation.
