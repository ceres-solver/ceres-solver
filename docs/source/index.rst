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

   introduction
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

Solving `nonlinear least squares`_ problems comes up in a broad range of areas
across science and engineering - from `fitting curves`_ in statistics, to
constructing `3D models from photographs`_ in computer vision.

.. _nonlinear least squares: http://en.wikipedia.org/wiki/Non-linear_least_squares
.. _fitting curves: http://en.wikipedia.org/wiki/Nonlinear_regression
.. _3D models from photographs: http://en.wikipedia.org/wiki/Structure_from_motion

What is Ceres Solver?
---------------------
Ceres is an industrial-grade C++ library for modeling and solving large and
small nonlinear least squares problems of the form

.. math:: \frac{1}{2}\sum_{i} \rho_i\left(\left\|f_i\left(x_{i_1}, ... ,x_{i_k}\right)\right\|^2\right).

For a brief introduction to nonlinear solving in general, see the
:ref:`chapter-tutorial`.

Who uses Ceres Solver?
----------------------
There are many users of Ceres, including Google Street View, Google Maps,
several SLAM pipelines, Blender, and more. See the :ref:`chapter-introduction`
for more users.

Why use Ceres Solver?
---------------------
Ceres is a world-class least squares solver for a variety of reasons, including
an integrated modelling layer, automatic differentiation, optimized code,
extensive tests, and more. See the :ref:`chapter-introduction` for a detailed
list.

Getting started
---------------

* Download the `latest stable release
  <http://ceres-solver.org/ceres-solver-1.8.0.tar.gz>`_
  or, for those wanting the latest
* Clone the development version or `browse the source
  <https://ceres-solver.googlesource.com/ceres-solver>`_

  .. code-block:: bash

       git clone https://ceres-solver.googlesource.com/ceres-solver

* Read the :ref:`chapter-tutorial`
* Browse the :ref:`chapter-modeling` and :ref:`chapter-solving`.
* Join the `mailing list
  <https://groups.google.com/forum/?fromgroups#!forum/ceres-solver>`_
  and ask questions.
* File bugs, feature requests in the `issue tracker
  <https://code.google.com/p/ceres-solver/issues/list>`_.
* Improve Ceres by :ref:`chapter-contributing`

Cite Us
-------
If you use Ceres Solver for a publication, you must cite it as::

    @misc{ceres-solver,
      author = "Sameer Agarwal and Keir Mierle and Others",
      title = "Ceres Solver",
      howpublished = "\url{http://ceres-solver.org}",
    }
