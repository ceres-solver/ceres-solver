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

Ceres Solver is an industrial-grade C++ library for modeling and
solving `nonlinear least squares`_ problems. These problems comes up
in a broad range of areas across science and engineering - from
`fitting curves`_ in statistics, to constructing `3D models from
photographs`_ in computer vision.

Ceres Solver features an integrated modeling layer with automatic
differentiation (you can also use numeric and/or analytic
derivatives), well optimized code with extensive tests and state of
the art performance on a variety of problems.

Ceres Solver is used in Google `Street View`_, Google `PhotoTours`_,
Google `PhotoSphere`_, `Project Tango`_, `Blender`_, and more.

.. _nonlinear least squares: http://en.wikipedia.org/wiki/Non-linear_least_squares
.. _fitting curves: http://en.wikipedia.org/wiki/Nonlinear_regression
.. _3D models from photographs: http://en.wikipedia.org/wiki/Structure_from_motion
.. _Street View: http://youtu.be/z00ORu4bU-A
.. _PhotoTours: http://google-latlong.blogspot.com/2012/04/visit-global-landmarks-with-photo-tours.html
.. _PhotoSphere: http://www.google.com/maps/about/contribute/photosphere/
.. _Project Tango: https://www.google.com/atap/projecttango/
.. _Blender: http://mango.blender.org/development/planar-tracking-preview/

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
