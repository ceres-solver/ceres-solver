.. _chapter-introduction:

============
Introduction
============

Ceres Solver [#f1]_ is a non-linear least squares solver developed at
Google. It is designed to solve small and large sparse problems
accurately and efficiently [#f2]_ . Amongst its various features is a
simple but expressive API with support for automatic differentiation,
robust norms, local parameterizations, automatic gradient checking,
multithreading and automatic problem structure detection.

The key computational cost when solving a non-linear least squares
problem is the solution of a linear least squares problem in each
iteration. To this end Ceres supports a number of different linear
solvers suited for different needs. This includes dense QR
factorization (using `Eigen
<http://eigen.tuxfamily.org/index.php?title=Main_Page>`_) for small
scale problems, sparse Cholesky factorization (using `SuiteSparse
<http://www.cise.ufl.edu/research/sparse/SuiteSparse/>`_) for general
sparse problems and specialized Schur complement based solvers for
problems that arise in multi-view geometry.

Ceres has been used for solving a variety of problems in computer
vision and machine learning at Google with sizes that range from a
tens of variables and objective functions with a few hundred terms to
problems with millions of variables and objective functions with tens
of millions of terms.


What's in a name?
-----------------

While there is some debate as to who invented of the method of Least
Squares [Stigler]_. There is no debate that it was Carl Friedrich
Gauss's prediction of the orbit of the newly discovered asteroid Ceres
based on just 41 days of observations that brought it to the attention
of the world [Tenenbaum-Director]_. We named our solver after Ceres to
celebrate this seminal event in the history of astronomy, statistics
and optimization.

Contributing to Ceres Solver
----------------------------

We welcome contributions to Ceres, whether they are new features, bug
fixes or tests. The Ceres mailing list [#f3]_ is the best place for
all development related discussions. Please consider joining it. If
you have ideas on how you would like to contribute to Ceres, it is a
good idea to let us know on the mailinglist before you start
development. We may have suggestions that will save effort when trying
to merge your work into the main branch. If you are looking for ideas,
please let us know about your interest and skills and we will be happy
to make a suggestion or three.

We follow Google's C++ Style Guide [#f4]_ and use ``git`` for version
control.

Citing Ceres Solver
-------------------

If you use Ceres Solver for an academic publication, please cite this
manual. e.g., ::

  @manual{ceres-manual,
  	  Author = {Sameer Agarwal and Keir Mierle},
	  Title = {Ceres Solver: Tutorial \& Reference},
	  Organization = {Google Inc.}
  }


Acknowledgements
----------------

A number of people have helped with the development and open sourcing
of Ceres.

Fredrik Schaffalitzky when he was at Google started the development of
Ceres, and even though much has changed since then, many of the ideas
from his original design are still present in the current code.

Amongst Ceres' users at Google two deserve special mention: William
Rucklidge and James Roseborough. William was the first user of
Ceres. He bravely took on the task of porting production code to an
as-yet unproven optimization library, reporting bugs and helping fix
them along the way. James is perhaps the most sophisticated user of
Ceres at Google. He has reported and fixed bugs and helped evolve the
API for the better.

Since the initial release of Ceres, a number of people have
contributed to Ceres by porting it to new platforms, reporting bugs,
fixing bugs and adding new functionality. We acknowledge all of these
contributions in :ref:`chapter-version-history`.

.. rubric:: Footnotes
.. [#f1] For brevity, in the rest of this document we will just use the term Ceres.
.. [#f2] For a gentle but brief introduction to non-linear least
         squares problems, please start by reading the
         :ref:`chapter-tutorial`.
.. [#f3] http://groups.google.com/group/ceres-solver
.. [#f4] http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml
