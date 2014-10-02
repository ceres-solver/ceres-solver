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
   bibliography
   license

Ceres Solver [#f1]_ is an open source C++ library for modeling and solving
large complicated optimization problem. While much of Ceres Solver
functionality is aimed at solving `nonlinear least squares`_ problems,
it can also solve more general unconstrained optimization problems. It
is a feature rich, mature and performant library which has been used
in production at Google since 2010.

At Google, Ceres Solver is used to:

* Estimate the pose of `Street View`_ cars, aircrafts, and satellites.
* Build 3D models for `PhotoTours`_.
* Estimate satellite image sensor characteristics.
* Stitch `panoramas`_ or apply `Lens Blur`_ on Android.
* Solve `bundle adjustment`_ and SLAM problems in `Project Tango`_.

Outside Google, Ceres is used for solving problems in computer vision,
computer graphics, astronomy and physics. For example, `Willow
Garage`_ uses it to solve SLAM problems and `Blender`_ uses it for for
planar tracking and bundle adjustment.

.. _nonlinear least squares: http://en.wikipedia.org/wiki/Non-linear_least_squares
.. _fitting curves: http://en.wikipedia.org/wiki/Nonlinear_regression
.. _bundle adjustment: http://en.wikipedia.org/wiki/Structure_from_motion
.. _Street View: http://youtu.be/z00ORu4bU-A
.. _PhotoTours: http://google-latlong.blogspot.com/2012/04/visit-global-landmarks-with-photo-tours.html
.. _panoramas: http://www.google.com/maps/about/contribute/photosphere/
.. _Project Tango: https://www.google.com/atap/projecttango/
.. _Blender: http://mango.blender.org/development/planar-tracking-preview/
.. _Willow Garage: https://www.willowgarage.com/blog/2013/08/09/enabling-robots-see-better-through-improved-camera-calibration
.. _Lens Blur: http://googleresearch.blogspot.com/2014/04/lens-blur-in-new-google-camera-app.html

Getting started
---------------

* Download the `latest stable release
  <http://ceres-solver.org/ceres-solver-1.9.0.tar.gz>`_ or clone the
  Git repository for the latest development version.

  .. code-block:: bash

       git clone https://ceres-solver.googlesource.com/ceres-solver

* Read the :ref:`chapter-tutorial`, browse the chapters on the
  :ref:`chapter-modeling` API and the :ref:`chapter-solving` API.
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


.. rubic:: Footnotes

.. [#f1] While there is some debate as to who invented the method of
         Least Squares [Stigler]_, there is no debate that it was
         `Carl Friedrich Gauss
         <http://www-groups.dcs.st-and.ac.uk/~history/Biographies/Gauss.html>`_
         who brought it to the attention of the world. Using just 22
         observations of the newly discovered asteroid `Ceres
         <http://en.wikipedia.org/wiki/Ceres_(dwarf_planet)>`_, Gauss
         used the method of least squares to correctly predict when
         and where the asteroid will emerge from behind the Sun
         [TenenbaumDirector]_. We named our solver after Ceres to
         celebrate this seminal event in the history of astronomy,
         statistics and optimization.
