.. _chapter-users:

=====
Users
=====

* At `Google <https://www.google.com>`_, Ceres is used to:

  * Estimate the pose of `Street View`_ cars, aircraft, and satellites.
  * Build 3D models for `PhotoTours`_.
  * Estimate satellite image sensor characteristics.
  * Stitch `panoramas`_ on Android and iOS.
  * Apply `Lens Blur`_ on Android.
  * Solve `bundle adjustment`_ and `SLAM`_ problems in `ARCore`_.

* `Willow Garage`_ uses Ceres to solve `SLAM`_ problems.
* `Southwest Research Institute <https://www.swri.org/>`_ uses Ceres for
  `calibrating robot-camera systems`_.
* `Blender <https://www.blender.org>`_ uses Ceres for `planar
  tracking`_ and `bundle adjustment`_.
* `OpenMVG <https://github.com/openMVG/openMVG>`_, an open-source
  multi-view geometry library, uses Ceres for `bundle adjustment`_.
* `Microsoft Research <https://www.microsoft.com/en-us/research/>`_ uses
  Ceres for non-linear optimization of objectives involving subdivision
  surfaces under `skinned control meshes`_.
* `Matterport <https://matterport.com/>`_ uses Ceres for global
  alignment of 3D point clouds and for pose graph optimization.
* The `Autonomous Systems Lab <https://asl.ethz.ch/>`_ at ETH
  Zurich uses Ceres for:

  * Camera and camera-IMU calibration.
  * Large-scale optimization of visual, inertial, GPS, and
    wheel-odometry data for long-term autonomy.

* `OpenPTrack <https://github.com/OpenPTrack/open_ptrack>`_ uses Ceres for camera
  calibration.
* The `Intelligent Autonomous Systems Lab <https://robotics.dei.unipd.it/>`_
  at the University of Padova, Italy, uses Ceres for:

  * Camera and depth-sensor network calibration.
  * Depth sensor distortion map estimation.

* `Theia <https://github.com/sweeneychris/TheiaSfM>`_ is an open-source
  Structure from Motion library that uses Ceres for `bundle adjustment`_
  and camera pose estimation.

* The `Applied Research Laboratory <https://www.arl.psu.edu/>`_ at
  Pennsylvania State University uses Ceres in its synthetic aperture
  sonar beamforming engine, ASASIN, to estimate platform kinematics.

* `COLMAP <https://github.com/colmap/colmap>`_ is an open-source
  Structure from Motion library that makes heavy use of Ceres for
  bundle adjustment with support for many camera models and for other
  non-linear least squares problems (such as relative and absolute pose
  refinement).



.. _bundle adjustment: https://en.wikipedia.org/wiki/Structure_from_motion
.. _Street View: https://youtu.be/z00ORu4bU-A
.. _PhotoTours: https://maps.googleblog.com/2012/04/visit-global-landmarks-with-photo-tours.html
.. _panoramas: https://www.google.com/streetview/contribute/
.. _ARCore: https://developers.google.com/ar
.. _planar tracking: https://mango.blender.org/development/planar-tracking-preview/
.. _Willow Garage: https://www.willowgarage.com/blog/2013/08/09/enabling-robots-see-better-through-improved-camera-calibration
.. _Lens Blur: https://research.google/blog/lens-blur-in-the-new-google-camera-app/
.. _SLAM: https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping
.. _calibrating robot-camera systems:
   https://rosindustrial.org/news/2014/9/24/industrial-calibration-library-update-and-presentation
.. _skinned control meshes: https://www.microsoft.com/en-us/research/publication/user-specific-hand-modeling-from-monocular-depth-sequences/
