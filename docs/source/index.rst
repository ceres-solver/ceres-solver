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

   features
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

Ceres Solver is an open source C++ library for modeling and solving
large complicated `nonlinear least squares`_ problems. It is a feature
rich, mature and performant library which has been used in production
since 2010.

Ceres Solver is used at Google to estimate the pose of `Street View`_
cars, aircrafts, and satellites; to build `3D models`_ for
`PhotoTours`_; to estimate satellite image sensor characteristics;
stitch `panoramas`_ on cellphones; `Project Tango`_ and more. Outside
Google, Ceres is used for solving problems in computer vision,
computer graphics, astronomy and finance. e.g., `Willow Garage`_ uses
it to solve SLAM problems and `Blender`_ uses it for for planar
tracking and bundle adjustment

.. _nonlinear least squares: http://en.wikipedia.org/wiki/Non-linear_least_squares
.. _fitting curves: http://en.wikipedia.org/wiki/Nonlinear_regression
.. _3D models: http://en.wikipedia.org/wiki/Structure_from_motion
.. _Street View: http://youtu.be/z00ORu4bU-A
.. _PhotoTours: http://google-latlong.blogspot.com/2012/04/visit-global-landmarks-with-photo-tours.html
.. _panoramas: http://www.google.com/maps/about/contribute/photosphere/
.. _Project Tango: https://www.google.com/atap/projecttango/
.. _Blender: http://mango.blender.org/development/planar-tracking-preview/
.. _Willow Garage: https://www.willowgarage.com/blog/2013/08/09/enabling-robots-see-better-through-improved-camera-calibration

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
