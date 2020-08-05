.. _chapter-installation:

============
Installation
============

Getting the source code
=======================
.. _section-source:

You can start with the `latest stable release
<http://ceres-solver.org/ceres-solver-2.0.0.tar.gz>`_ . Or if you want
the latest version, you can clone the git repository

.. code-block:: bash

       git clone https://ceres-solver.googlesource.com/ceres-solver

.. _section-dependencies:

Dependencies
============

  .. NOTE ::

    Starting with v2.0 Ceres requires a **fully C++14-compliant**
    compiler.  In versions <= 1.14, C++11 was an optional requirement.

Ceres relies on a number of open source libraries, some of which are
optional. For details on customizing the build process, see
:ref:`section-customizing` .

- `Eigen <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_
  3.3 or later **required**.

  .. NOTE ::

    Ceres can also use Eigen as a sparse linear algebra
    library. Please see the documentation for ``EIGENSPARSE`` for
    more details.

- `CMake <http://www.cmake.org>`_ 3.5 or later.
  **Required on all platforms except for legacy Android.**

- `glog <https://github.com/google/glog>`_ 0.3.1 or
  later. **Recommended**

  ``glog`` is used extensively throughout Ceres for logging detailed
  information about memory allocations and time consumed in various
  parts of the solve, internal error conditions etc. The Ceres
  developers use it extensively to observe and analyze Ceres's
  performance. `glog <https://github.com/google/glog>`_ allows you to
  control its behaviour from the command line. Starting with
  ``-logtostderr`` you can add ``-v=N`` for increasing values of ``N``
  to get more and more verbose and detailed information about Ceres
  internals.

  Ceres also ships with a minimal replacement of ``glog`` called
  ``miniglog`` that can be enabled with the ``MINIGLOG`` build option.
  ``miniglog`` is supplied for platforms which do not support the full
  version of ``glog``.

  In an attempt to reduce dependencies, it may be tempting to use
  ``miniglog`` on platforms which already support ``glog``. While
  there is nothing preventing the user from doing so, we strongly
  recommend against it. ``miniglog`` has worse performance than
  ``glog`` and is much harder to control and use.

  .. NOTE ::

     If you are compiling ``glog`` from source, please note that
     currently, the unit tests for ``glog`` (which are enabled by
     default) do not compile against a default build of ``gflags`` 2.1
     as the gflags namespace changed from ``google::`` to
     ``gflags::``.  A patch to fix this is available from `here
     <https://code.google.com/p/google-glog/issues/detail?id=194>`_.

- `gflags <https://github.com/gflags/gflags>`_. Needed to build
  examples and tests.

- `SuiteSparse
  <http://faculty.cse.tamu.edu/davis/suitesparse.html>`_. Needed for
  solving large sparse linear systems. **Optional; strongly recomended
  for large scale bundle adjustment**

- `CXSparse <http://faculty.cse.tamu.edu/davis/suitesparse.html>`_.
  Similar to ``SuiteSparse`` but simpler and slower. CXSparse has
  no dependencies on ``LAPACK`` and ``BLAS``. This makes for a simpler
  build process and a smaller binary. **Optional**

- `Apple's Accelerate sparse solvers <https://developer.apple.com/documentation/accelerate/sparse_solvers>`_.
  As of Xcode 9.0, Apple's Accelerate framework includes support for
  solving sparse linear systems across macOS, iOS et al. **Optional**

- `BLAS <http://www.netlib.org/blas/>`_ and `LAPACK
  <http://www.netlib.org/lapack/>`_ routines are needed by
  ``SuiteSparse``, and optionally used by Ceres directly for some
  operations.

  On ``UNIX`` OSes other than Mac OS X we recommend `ATLAS
  <http://math-atlas.sourceforge.net/>`_, which includes ``BLAS`` and
  ``LAPACK`` routines. It is also possible to use `OpenBLAS
  <https://github.com/xianyi/OpenBLAS>`_ . However, one needs to be
  careful to `turn off the threading
  <https://github.com/xianyi/OpenBLAS/wiki/faq#wiki-multi-threaded>`_
  inside ``OpenBLAS`` as it conflicts with use of threads in Ceres.

  Mac OS X ships with an optimized ``LAPACK`` and ``BLAS``
  implementation as part of the ``Accelerate`` framework. The Ceres
  build system will automatically detect and use it.

  For Windows things are much more complicated. `LAPACK For
  Windows <http://icl.cs.utk.edu/lapack-for-windows/lapack/>`_
  has detailed instructions..

  **Optional but required for** ``SuiteSparse``.

.. _section-linux:

Linux
=====

We will use `Ubuntu <http://www.ubuntu.com>`_ as our example linux
distribution.

.. NOTE::

 Up to at least Ubuntu 14.04, the SuiteSparse package in the official
 package repository (built from SuiteSparse v3.4.0) **cannot** be used
 to build Ceres as a *shared* library.  Thus if you want to build
 Ceres as a shared library using SuiteSparse, you must perform a
 source install of SuiteSparse or use an external PPA (see `bug report
 here
 <https://bugs.launchpad.net/ubuntu/+source/suitesparse/+bug/1333214>`_).
 It is recommended that you use the current version of SuiteSparse
 (4.2.1 at the time of writing).


Start by installing all the dependencies.

.. code-block:: bash

     # CMake
     sudo apt-get install cmake
     # google-glog + gflags
     sudo apt-get install libgoogle-glog-dev
     # BLAS & LAPACK
     sudo apt-get install libatlas-base-dev
     # Eigen3
     sudo apt-get install libeigen3-dev
     # SuiteSparse and CXSparse (optional)
     # - If you want to build Ceres as a *static* library (the default)
     #   you can use the SuiteSparse package in the main Ubuntu package
     #   repository:
     sudo apt-get install libsuitesparse-dev
     # - However, if you want to build Ceres as a *shared* library, you must
     #   add the following PPA:
     sudo add-apt-repository ppa:bzindovic/suitesparse-bugfix-1319687
     sudo apt-get update
     sudo apt-get install libsuitesparse-dev

We are now ready to build, test, and install Ceres.

.. code-block:: bash

 tar zxf ceres-solver-2.0.0.tar.gz
 mkdir ceres-bin
 cd ceres-bin
 cmake ../ceres-solver-2.0.0
 make -j3
 make test
 # Optionally install Ceres, it can also be exported using CMake which
 # allows Ceres to be used without requiring installation, see the documentation
 # for the EXPORT_BUILD_DIR option for more information.
 make install

You can also try running the command line bundling application with one of the
included problems, which comes from the University of Washington's BAL
dataset [Agarwal]_.

.. code-block:: bash

 bin/simple_bundle_adjuster ../ceres-solver-2.0.0/data/problem-16-22106-pre.txt

This runs Ceres for a maximum of 10 iterations using the
``DENSE_SCHUR`` linear solver. The output should look something like
this.

.. code-block:: bash

    iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
       0  4.185660e+06    0.00e+00    1.09e+08   0.00e+00   0.00e+00  1.00e+04       0    7.59e-02    3.37e-01
       1  1.062590e+05    4.08e+06    8.99e+06   5.36e+02   9.82e-01  3.00e+04       1    1.65e-01    5.03e-01
       2  4.992817e+04    5.63e+04    8.32e+06   3.19e+02   6.52e-01  3.09e+04       1    1.45e-01    6.48e-01
       3  1.899774e+04    3.09e+04    1.60e+06   1.24e+02   9.77e-01  9.26e+04       1    1.43e-01    7.92e-01
       4  1.808729e+04    9.10e+02    3.97e+05   6.39e+01   9.51e-01  2.78e+05       1    1.45e-01    9.36e-01
       5  1.803399e+04    5.33e+01    1.48e+04   1.23e+01   9.99e-01  8.33e+05       1    1.45e-01    1.08e+00
       6  1.803390e+04    9.02e-02    6.35e+01   8.00e-01   1.00e+00  2.50e+06       1    1.50e-01    1.23e+00

    Ceres Solver v2.0.0 Solve Report
    ----------------------------------
                                         Original                  Reduced
    Parameter blocks                        22122                    22122
    Parameters                              66462                    66462
    Residual blocks                         83718                    83718
    Residual                               167436                   167436

    Minimizer                        TRUST_REGION

    Dense linear algebra library            EIGEN
    Trust region strategy     LEVENBERG_MARQUARDT

                                            Given                     Used
    Linear solver                     DENSE_SCHUR              DENSE_SCHUR
    Threads                                     1                        1
    Linear solver threads                       1                        1
    Linear solver ordering              AUTOMATIC                22106, 16

    Cost:
    Initial                          4.185660e+06
    Final                            1.803390e+04
    Change                           4.167626e+06

    Minimizer iterations                        6
    Successful steps                            6
    Unsuccessful steps                          0

    Time (in seconds):
    Preprocessor                            0.261

      Residual evaluation                   0.082
      Jacobian evaluation                   0.412
      Linear solver                         0.442
    Minimizer                               1.051

    Postprocessor                           0.002
    Total                                   1.357

    Termination:                      CONVERGENCE (Function tolerance reached. |cost_change|/cost: 1.769766e-09 <= 1.000000e-06)

.. section-osx:

Mac OS X
========
.. NOTE::

 Ceres will not compile using Xcode 4.5.x (Clang version 4.1) due to a
 bug in that version of Clang.  If you are running Xcode 4.5.x, please
 update to Xcode >= 4.6.x before attempting to build Ceres.


On OS X, you can either use `MacPorts <https://www.macports.org/>`_ or
`Homebrew <http://mxcl.github.com/homebrew/>`_ to install Ceres Solver.

If using `MacPorts <https://www.macports.org/>`_, then

.. code-block:: bash

   sudo port install ceres-solver

will install the latest version.

If using `Homebrew <http://mxcl.github.com/homebrew/>`_ and assuming
that you have the ``homebrew/science`` [#f1]_ tap enabled, then

.. code-block:: bash

      brew install ceres-solver

will install the latest stable version along with all the required
dependencies and

.. code-block:: bash

      brew install ceres-solver --HEAD

will install the latest version in the git repo.

You can also install each of the dependencies by hand using `Homebrew
<http://mxcl.github.com/homebrew/>`_. There is no need to install
``BLAS`` or ``LAPACK`` separately as OS X ships with optimized
``BLAS`` and ``LAPACK`` routines as part of the `vecLib
<https://developer.apple.com/library/mac/#documentation/Performance/Conceptual/vecLib/Reference/reference.html>`_
framework.

.. code-block:: bash

      # CMake
      brew install cmake
      # google-glog and gflags
      brew install glog
      # Eigen3
      brew install eigen
      # SuiteSparse and CXSparse
      brew install suite-sparse

We are now ready to build, test, and install Ceres.

.. code-block:: bash

   tar zxf ceres-solver-2.0.0.tar.gz
   mkdir ceres-bin
   cd ceres-bin
   cmake ../ceres-solver-2.0.0
   make -j3
   make test
   # Optionally install Ceres, it can also be exported using CMake which
   # allows Ceres to be used without requiring installation, see the
   # documentation for the EXPORT_BUILD_DIR option for more information.
   make install

Building with OpenMP on OS X
----------------------------

Up to at least Xcode 8, OpenMP support was disabled in Apple's version of
Clang.  However, you can install the latest version of the LLVM toolchain
from Homebrew which does support OpenMP, and thus build Ceres with OpenMP
support on OS X.  To do this, you must install llvm via Homebrew:

.. code-block:: bash

      # Install latest version of LLVM toolchain.
      brew install llvm

As the LLVM formula in Homebrew is keg-only, it will not be installed to
``/usr/local`` to avoid conflicts with the standard Apple LLVM toolchain.
To build Ceres with the Homebrew LLVM toolchain you should do the
following:

.. code-block:: bash

   tar zxf ceres-solver-2.0.0.tar.gz
   mkdir ceres-bin
   cd ceres-bin
   # Configure the local shell only (not persistent) to use the Homebrew LLVM
   # toolchain in favour of the default Apple version.  This is taken
   # verbatim from the instructions output by Homebrew when installing the
   # llvm formula.
   export LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib"
   export CPPFLAGS="-I/usr/local/opt/llvm/include"
   export PATH="/usr/local/opt/llvm/bin:$PATH"
   # Force CMake to use the Homebrew version of Clang.  OpenMP will be
   # automatically enabled if it is detected that the compiler supports it.
   cmake -DCMAKE_C_COMPILER=/usr/local/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/usr/local/opt/llvm/bin/clang++ ../ceres-solver-2.0.0
   make -j3
   make test
   # Optionally install Ceres.  It can also be exported using CMake which
   # allows Ceres to be used without requiring installation.  See the
   # documentation for the EXPORT_BUILD_DIR option for more information.
   make install

Like the Linux build, you should now be able to run
``bin/simple_bundle_adjuster``.


.. rubric:: Footnotes

.. [#f1] Ceres and many of its dependencies are in `homebrew/science
   <https://github.com/Homebrew/homebrew-science>`_ tap. So, if you
   don't have this tap enabled, then you will need to enable it as
   follows before executing any of the commands in this section.

   .. code-block:: bash

      brew tap homebrew/science


.. _section-windows:

Windows
=======

.. NOTE::

  If you find the following CMake difficult to set up, then you may
  be interested in a `Microsoft Visual Studio wrapper
  <https://github.com/tbennun/ceres-windows>`_ for Ceres Solver by Tal
  Ben-Nun.

On Windows, we support building with Visual Studio 2015.2 of newer. Note
that the Windows port is less featureful and less tested than the
Linux or Mac OS X versions due to the lack of an officially supported
way of building SuiteSparse and CXSparse.  There are however a number
of unofficial ways of building these libraries. Building on Windows
also a bit more involved since there is no automated way to install
dependencies.

.. NOTE:: Using ``google-glog`` & ``miniglog`` with windows.h.

 The windows.h header if used with GDI (Graphics Device Interface)
 defines ``ERROR``, which conflicts with the definition of ``ERROR``
 as a LogSeverity level in ``google-glog`` and ``miniglog``.  There
 are at least two possible fixes to this problem:

 #. Use ``google-glog`` and define ``GLOG_NO_ABBREVIATED_SEVERITIES``
    when building Ceres and your own project, as documented `here
    <http://google-glog.googlecode.com/svn/trunk/doc/glog.html>`__.
    Note that this fix will not work for ``miniglog``, but use of
    ``miniglog`` is strongly discouraged on any platform for which
    ``google-glog`` is available (which includes Windows).
 #. If you do not require GDI, then define ``NOGDI`` **before**
    including windows.h.  This solution should work for both
    ``google-glog`` and ``miniglog`` and is documented for
    ``google-glog`` `here
    <https://code.google.com/p/google-glog/issues/detail?id=33>`__.

#. Make a toplevel directory for deps & build & src somewhere: ``ceres/``
#. Get dependencies; unpack them as subdirectories in ``ceres/``
   (``ceres/eigen``, ``ceres/glog``, etc)

   #. ``Eigen`` 3.3 . There is no need to build anything; just unpack
      the source tarball.

   #. ``google-glog`` Open up the Visual Studio solution and build it.
   #. ``gflags`` Open up the Visual Studio solution and build it.

   #. (Experimental) ``SuiteSparse`` Previously SuiteSparse was not
      available on Windows, recently it has become possible to build
      it on Windows using the `suitesparse-metis-for-windows
      <https://github.com/jlblancoc/suitesparse-metis-for-windows>`_
      project.  If you wish to use ``SuiteSparse``, follow their
      instructions for obtaining and building it.

   #. (Experimental) ``CXSparse`` Previously CXSparse was not
      available on Windows, there are now several ports that enable it
      to be, including: `[1] <https://github.com/PetterS/CXSparse>`_
      and `[2] <https://github.com/TheFrenchLeaf/CXSparse>`_.  If you
      wish to use ``CXSparse``, follow their instructions for
      obtaining and building it.

#. Unpack the Ceres tarball into ``ceres``. For the tarball, you
   should get a directory inside ``ceres`` similar to
   ``ceres-solver-2.0.0``. Alternately, checkout Ceres via ``git`` to
   get ``ceres-solver.git`` inside ``ceres``.

#. Install ``CMake``,

#. Make a dir ``ceres/ceres-bin`` (for an out-of-tree build)

#. Run ``CMake``; select the ``ceres-solver-X.Y.Z`` or
   ``ceres-solver.git`` directory for the CMake file. Then select the
   ``ceres-bin`` for the build dir.

#. Try running ``Configure``. It won't work. It'll show a bunch of options.
   You'll need to set:

   #. ``GLOG_INCLUDE_DIR_HINTS``
   #. ``GLOG_LIBRARY_DIR_HINTS``
   #. (Optional) ``SUITESPARSE_INCLUDE_DIR_HINTS``
   #. (Optional) ``SUITESPARSE_LIBRARY_DIR_HINTS``
   #. (Optional) ``CXSPARSE_INCLUDE_DIR_HINTS``
   #. (Optional) ``CXSPARSE_LIBRARY_DIR_HINTS``

   to the appropriate directories where you unpacked/built them. If
   any of the variables are not visible in the ``CMake`` GUI, create a
   new entry for them.  We recommend using the
   ``<NAME>_(INCLUDE/LIBRARY)_DIR_HINTS`` variables rather than
   setting the ``<NAME>_INCLUDE_DIR`` & ``<NAME>_LIBRARY`` variables
   directly to keep all of the validity checking, and to avoid having
   to specify the library files manually.

#. You may have to tweak some more settings to generate a MSVC
   project.  After each adjustment, try pressing Configure & Generate
   until it generates successfully.

#. Open the solution and build it in MSVC


To run the tests, select the ``RUN_TESTS`` target and hit **Build
RUN_TESTS** from the build menu.

Like the Linux build, you should now be able to run
``bin/simple_bundle_adjuster``.

Notes:

#. The default build is Debug; consider switching it to release mode.
#. Currently ``system_test`` is not working properly.
#. CMake puts the resulting test binaries in ``ceres-bin/examples/Debug``
   by default.
#. The solvers supported on Windows are ``DENSE_QR``, ``DENSE_SCHUR``,
   ``CGNR``, and ``ITERATIVE_SCHUR``.
#. We're looking for someone to work with upstream ``SuiteSparse`` to
   port their build system to something sane like ``CMake``, and get a
   fully supported Windows port.


.. _section-android:

Android
=======

.. NOTE::

    You will need Android NDK r15 or higher to build Ceres solver.

To build Ceres for Android, we need to force ``CMake`` to find
the toolchains from the Android NDK instead of using the standard
ones. For example, assuming you have specified ``$NDK_DIR``:

.. code-block:: bash

    cmake \
    -DCMAKE_TOOLCHAIN_FILE=\
        $NDK_DIR/build/cmake/android.toolchain.cmake \
    -DEIGEN_INCLUDE_DIR=/path/to/eigen/header \
    -DANDROID_ABI=armeabi-v7a \
    -DANDROID_STL=c++_shared \
    -DANDROID_NATIVE_API_LEVEL=android-24 \
    -DBUILD_SHARED_LIBS=ON \
    -DMINIGLOG=ON \
    <PATH_TO_CERES_SOURCE>

You can build for any Android STL or ABI, but the c++_shared STL
and the armeabi-v7a or arm64-v8a ABI are recommended for 32bit
and 64bit architectures, respectively. Several API levels may
be supported, but it is recommended that you use the highest
level that is suitable for your Android project.

.. NOTE::

    You must always use the same API level and STL library for
    your Android project and the Ceres binaries.

After building, you get a ``libceres.so`` library, which you can
link in your Android build system by using a
``PREBUILT_SHARED_LIBRARY`` target in your build script.

If you are building any Ceres samples and would like to verify
your library, you will need to place them in an executable public
directory together with ``libceres.so`` on your Android device
(e.g. in /data/local/tmp) and ensure that the STL library from
your NDK is present in that same directory. You may then execute
the sample by running for example:

.. code-block:: bash

    adb shell
    cd /data/local/tmp
    LD_LIBRARY_PATH=/data/local/tmp ./helloworld

Note that any solvers or other shared dependencies you include in
your project must also be present in your android build config and
your test directory on Android.

.. _section-ios:

iOS
===

.. NOTE::

   You need iOS version 7.0 or higher to build Ceres Solver.

To build Ceres for iOS, we need to force ``CMake`` to find the
toolchains from the iOS SDK instead of using the standard ones. For
example:

.. code-block:: bash

   cmake \
   -DCMAKE_TOOLCHAIN_FILE=../ceres-solver/cmake/iOS.cmake \
   -DEIGEN_INCLUDE_DIR=/path/to/eigen/header \
   -DIOS_PLATFORM=<PLATFORM> \
   <PATH_TO_CERES_SOURCE>

``PLATFORM`` can be: ``OS``, ``SIMULATOR`` or ``SIMULATOR64``. You can
build for ``OS`` (``armv7``, ``armv7s``, ``arm64``), ``SIMULATOR``
(``i386``) or ``SIMULATOR64`` (``x86_64``) separately and use ``lipo``
to merge them into one static library.  See ``cmake/iOS.cmake`` for
more options.

.. NOTE::

   iOS version 11.0+ requires a 64-bit architecture, so you cannot
   build for armv7/armv7s with iOS 11.0+ (only arm64 is supported).

After building, you will get a ``libceres.a`` library, which you will
need to add to your Xcode project.

The default CMake configuration builds a bare bones version of Ceres
Solver that only depends on Eigen (``MINIGLOG`` is compiled into Ceres
if it is used), this should be sufficient for solving small to
moderate sized problems (No ``SPARSE_SCHUR``,
``SPARSE_NORMAL_CHOLESKY`` linear solvers and no ``CLUSTER_JACOBI``
and ``CLUSTER_TRIDIAGONAL`` preconditioners).

If you decide to use ``LAPACK`` and ``BLAS``, then you also need to
add ``Accelerate.framework`` to your Xcode project's linking
dependency.

.. _section-customizing:

Customizing the build
=====================

It is possible to reduce the libraries needed to build Ceres and
customize the build process by setting the appropriate options in
``CMake``.  These options can either be set in the ``CMake`` GUI, or
via ``-D<OPTION>=<ON/OFF>`` when running ``CMake`` from the command
line.  In general, you should only modify these options from their
defaults if you know what you are doing.

.. NOTE::

 If you are setting variables via ``-D<VARIABLE>=<VALUE>`` when
 calling ``CMake``, it is important to understand that this forcibly
 **overwrites** the variable ``<VARIABLE>`` in the ``CMake`` cache at
 the start of *every configure*.

 This can lead to confusion if you are invoking the ``CMake`` `curses
 <http://www.gnu.org/software/ncurses/ncurses.html>`_ terminal GUI
 (via ``ccmake``, e.g. ```ccmake -D<VARIABLE>=<VALUE>
 <PATH_TO_SRC>``).  In this case, even if you change the value of
 ``<VARIABLE>`` in the ``CMake`` GUI, your changes will be
 **overwritten** with the value passed via ``-D<VARIABLE>=<VALUE>``
 (if one exists) at the start of each configure.

 As such, it is generally easier not to pass values to ``CMake`` via
 ``-D`` and instead interactively experiment with their values in the
 ``CMake`` GUI.  If they are not present in the *Standard View*,
 toggle to the *Advanced View* with ``<t>``.


Modifying default compilation flags
-----------------------------------

The ``CMAKE_CXX_FLAGS`` variable can be used to define additional
default compilation flags for all build types.  Any flags specified
in ``CMAKE_CXX_FLAGS`` will be used in addition to the default
flags used by Ceres for the current build type.

For example, if you wished to build Ceres with `-march=native
<https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html>`_ which is not
enabled by default (even if ``CMAKE_BUILD_TYPE=Release``) you would invoke
CMake with:

.. code-block:: bash

       cmake -DCMAKE_CXX_FLAGS="-march=native" <PATH_TO_CERES_SOURCE>

.. NOTE ::

    The use of ``-march=native`` will limit portability, as it will tune the
    implementation to the specific CPU of the compiling machine (e.g. use of
    AVX if available).  Run-time segfaults may occur if you then tried to
    run the resulting binaries on a machine with a different processor, even
    if it is from the same family (e.g. x86) if the specific options available
    are different.  Note that the performance gains from the use of
    ``-march=native`` are not guaranteed to be significant.

.. _options-controlling-ceres-configuration:

Options controlling Ceres configuration
---------------------------------------

#. ``LAPACK [Default: ON]``: If this option is enabled, and the ``BLAS`` and
   ``LAPACK`` libraries are found, Ceres will enable **direct** use of
   ``LAPACK`` routines (i.e. Ceres itself will call them).  If this option is
   disabled, then Ceres will not require ``LAPACK`` or ``BLAS``.  It is
   however still possible that Ceres may call ``LAPACK`` routines indirectly
   via SuiteSparse if ``LAPACK=OFF`` and ``SUITESPARSE=ON``.  Finally
   note that if ``LAPACK=ON`` and ``SUITESPARSE=ON``, the ``LAPACK`` and
   ``BLAS`` libraries used by SuiteSparse and Ceres should be the same.

#. ``SUITESPARSE [Default: ON]``: By default, Ceres will link to
   ``SuiteSparse`` if it and all of its dependencies are present. Turn
   this ``OFF`` to build Ceres without ``SuiteSparse``.

   .. NOTE::

      SuiteSparse is licensed under a mixture of GPL/LGPL/Commercial
      terms.  Ceres requires some components that are only licensed under
      GPL/Commercial terms.

#. ``CXSPARSE [Default: ON]``: By default, Ceres will link to
   ``CXSparse`` if all its dependencies are present. Turn this ``OFF``
   to build Ceres without ``CXSparse``.

   .. NOTE::

      CXSparse is licensed under the LGPL.

#. ``ACCELERATESPARSE [Default: ON]``: By default, Ceres will link to
   Apple's Accelerate framework directly if a version of it is detected
   which supports solving sparse linear systems.  Note that on Apple OSs
   Accelerate usually also provides the BLAS/LAPACK implementations and
   so would be linked against irrespective of the value of ``ACCELERATESPARSE``.

#. ``EIGENSPARSE [Default: ON]``: By default, Ceres will not use
   Eigen's sparse Cholesky factorization.

#. ``GFLAGS [Default: ON]``: Turn this ``OFF`` to build Ceres without
   ``gflags``. This will also prevent some of the example code from
   building.

#. ``MINIGLOG [Default: OFF]``: Ceres includes a stripped-down,
   minimal implementation of ``glog`` which can optionally be used as
   a substitute for ``glog``, thus removing ``glog`` as a required
   dependency. Turn this ``ON`` to use this minimal ``glog``
   implementation.

#. ``SCHUR_SPECIALIZATIONS [Default: ON]``: If you are concerned about
   binary size/compilation time over some small (10-20%) performance
   gains in the ``SPARSE_SCHUR`` solver, you can disable some of the
   template specializations by turning this ``OFF``.

#. ``CERES_THREADING_MODEL [Default: CXX_THREADS > OPENMP > NO_THREADS]``:
   Multi-threading backend Ceres should be compiled with.  This will
   automatically be set to only accept the available subset of threading
   options in the CMake GUI.

#. ``BUILD_SHARED_LIBS [Default: OFF]``: By default Ceres is built as
   a static library, turn this ``ON`` to instead build Ceres as a
   shared library.

#. ``EXPORT_BUILD_DIR [Default: OFF]``: By default Ceres is configured
   solely for installation, and so must be installed in order for
   clients to use it.  Turn this ``ON`` to export Ceres' build
   directory location into the `user's local CMake package registry
   <http://www.cmake.org/cmake/help/v3.2/manual/cmake-packages.7.html#user-package-registry>`_
   where it will be detected **without requiring installation** in a
   client project using CMake when `find_package(Ceres)
   <http://www.cmake.org/cmake/help/v3.2/command/find_package.html>`_
   is invoked.

#. ``BUILD_DOCUMENTATION [Default: OFF]``: Use this to enable building
   the documentation, requires `Sphinx <http://sphinx-doc.org/>`_ and
   the `sphinx-better-theme
   <https://pypi.python.org/pypi/sphinx-better-theme>`_ package
   available from the Python package index. In addition, ``make
   ceres_docs`` can be used to build only the documentation.

#. ``MSVC_USE_STATIC_CRT [Default: OFF]`` *Windows Only*: By default
   Ceres will use the Visual Studio default, *shared* C-Run Time (CRT)
   library.  Turn this ``ON`` to use the *static* C-Run Time library
   instead.

#. ``LIB_SUFFIX [Default: "64" on non-Debian/Arch based 64-bit Linux,
   otherwise: ""]``: The suffix to append to the library install
   directory, built from:
   ``${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}``.

   The filesystem hierarchy standard recommends that 64-bit systems
   install native libraries to lib64 rather than lib.  Most Linux
   distributions follow this convention, but Debian and Arch based
   distros do not.  Note that the only generally sensible values for
   ``LIB_SUFFIX`` are "" and "64".

   Although by default Ceres will auto-detect non-Debian/Arch based
   64-bit Linux distributions and default ``LIB_SUFFIX`` to "64", this
   can always be overridden by manually specifying LIB_SUFFIX using:
   ``-DLIB_SUFFIX=<VALUE>`` when invoking CMake.


Options controlling Ceres dependency locations
----------------------------------------------

Ceres uses the ``CMake`` `find_package
<http://www.cmake.org/cmake/help/v3.2/command/find_package.html>`_
function to find all of its dependencies using
``Find<DEPENDENCY_NAME>.cmake`` scripts which are either included in
Ceres (for most dependencies) or are shipped as standard with
``CMake`` (for ``LAPACK`` & ``BLAS``).  These scripts will search all
of the "standard" install locations for various OSs for each
dependency.  However, particularly for Windows, they may fail to find
the library, in this case you will have to manually specify its
installed location.  The ``Find<DEPENDENCY_NAME>.cmake`` scripts
shipped with Ceres support two ways for you to do this:

#. Set the *hints* variables specifying the *directories* to search in
   preference, but in addition, to the search directories in the
   ``Find<DEPENDENCY_NAME>.cmake`` script:

   - ``<DEPENDENCY_NAME (CAPS)>_INCLUDE_DIR_HINTS``
   - ``<DEPENDENCY_NAME (CAPS)>_LIBRARY_DIR_HINTS``

   These variables should be set via ``-D<VAR>=<VALUE>``
   ``CMake`` arguments as they are not visible in the GUI.

#. Set the variables specifying the *explicit* include directory
   and library file to use:

   - ``<DEPENDENCY_NAME (CAPS)>_INCLUDE_DIR``
   - ``<DEPENDENCY_NAME (CAPS)>_LIBRARY``

   This bypasses *all* searching in the
   ``Find<DEPENDENCY_NAME>.cmake`` script, but validation is still
   performed.

   These variables are available to set in the ``CMake`` GUI. They are
   visible in the *Standard View* if the library has not been found
   (but the current Ceres configuration requires it), but are always
   visible in the *Advanced View*.  They can also be set directly via
   ``-D<VAR>=<VALUE>`` arguments to ``CMake``.

Building using custom BLAS & LAPACK installs
----------------------------------------------

If the standard find package scripts for ``BLAS`` & ``LAPACK`` which
ship with ``CMake`` fail to find the desired libraries on your system,
try setting ``CMAKE_LIBRARY_PATH`` to the path(s) to the directories
containing the ``BLAS`` & ``LAPACK`` libraries when invoking ``CMake``
to build Ceres via ``-D<VAR>=<VALUE>``.  This should result in the
libraries being found for any common variant of each.

Alternatively, you may also directly specify the ``BLAS_LIBRARIES`` and
``LAPACK_LIBRARIES`` variables via ``-D<VAR>=<VALUE>`` when invoking CMake
to configure Ceres.

.. _section-using-ceres:

Using Ceres with CMake
======================

In order to use Ceres in client code with CMake using `find_package()
<http://www.cmake.org/cmake/help/v3.2/command/find_package.html>`_
then either:

#. Ceres must have been installed with ``make install``.  If the
    install location is non-standard (i.e. is not in CMake's default
    search paths) then it will not be detected by default, see:
    :ref:`section-local-installations`.

    Note that if you are using a non-standard install location you
    should consider exporting Ceres instead, as this will not require
    any extra information to be provided in client code for Ceres to
    be detected.

#. Or Ceres' build directory must have been exported by enabling the
    ``EXPORT_BUILD_DIR`` option when Ceres was configured.


As an example of how to use Ceres, to compile `examples/helloworld.cc
<https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/helloworld.cc>`_
in a separate standalone project, the following CMakeList.txt can be
used:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.5)

    project(helloworld)

    find_package(Ceres REQUIRED)

    # helloworld
    add_executable(helloworld helloworld.cc)
    target_link_libraries(helloworld Ceres::ceres)

Irrespective of whether Ceres was installed or exported, if multiple
versions are detected, set: ``Ceres_DIR`` to control which is used.
If Ceres was installed ``Ceres_DIR`` should be the path to the
directory containing the installed ``CeresConfig.cmake`` file
(e.g. ``/usr/local/lib/cmake/Ceres``).  If Ceres was exported, then
``Ceres_DIR`` should be the path to the exported Ceres build
directory.

  .. NOTE ::

     You do not need to call include_directories(${CERES_INCLUDE_DIRS})
     as the exported Ceres CMake target already contains the definitions
     of its public include directories which will be automatically
     included by CMake when compiling a target that links against Ceres.

Specify Ceres components
-------------------------------------

You can specify particular Ceres components that you require (in order
for Ceres to be reported as found) when invoking
``find_package(Ceres)``.  This allows you to specify, for example,
that you require a version of Ceres built with SuiteSparse support.
By definition, if you do not specify any components when calling
``find_package(Ceres)`` (the default) any version of Ceres detected
will be reported as found, irrespective of which components it was
built with.

The Ceres components which can be specified are:

#. ``LAPACK``: Ceres built using LAPACK (``LAPACK=ON``).

#. ``SuiteSparse``: Ceres built with SuiteSparse (``SUITESPARSE=ON``).

#. ``CXSparse``: Ceres built with CXSparse (``CXSPARSE=ON``).

#. ``AccelerateSparse``: Ceres built with Apple's Accelerate sparse solvers (``ACCELERATESPARSE=ON``).

#. ``EigenSparse``: Ceres built with Eigen's sparse Cholesky factorization
   (``EIGENSPARSE=ON``).

#. ``SparseLinearAlgebraLibrary``: Ceres built with *at least one* sparse linear
   algebra library.  This is equivalent to ``SuiteSparse`` **OR** ``CXSparse``
   **OR** ``AccelerateSparse``  **OR** ``EigenSparse``.

#. ``SchurSpecializations``: Ceres built with Schur specializations
   (``SCHUR_SPECIALIZATIONS=ON``).

#. ``OpenMP``: Ceres built with OpenMP (``CERES_THREADING_MODEL=OPENMP``).

#. ``Multithreading``: Ceres built with *a* multithreading library.
   This is equivalent to (``CERES_THREAD != NO_THREADS``).

To specify one/multiple Ceres components use the ``COMPONENTS`` argument to
`find_package()
<http://www.cmake.org/cmake/help/v3.2/command/find_package.html>`_ like so:

.. code-block:: cmake

    # Find a version of Ceres compiled with SuiteSparse & EigenSparse support.
    #
    # NOTE: This will report Ceres as **not** found if the detected version of
    #            Ceres was not compiled with both SuiteSparse & EigenSparse.
    #            Remember, if you have multiple versions of Ceres installed, you
    #            can use Ceres_DIR to specify which should be used.
    find_package(Ceres REQUIRED COMPONENTS SuiteSparse EigenSparse)


Specify Ceres version
---------------------

Additionally, when CMake has found Ceres it can optionally check the package
version, if it has been specified in the `find_package()
<http://www.cmake.org/cmake/help/v3.2/command/find_package.html>`_
call.  For example:

.. code-block:: cmake

    find_package(Ceres 1.2.3 REQUIRED)

.. _section-local-installations:

Local installations
-------------------

If Ceres was installed in a non-standard path by specifying
``-DCMAKE_INSTALL_PREFIX="/some/where/local"``, then the user should
add the **PATHS** option to the ``find_package()`` command, e.g.,

.. code-block:: cmake

   find_package(Ceres REQUIRED PATHS "/some/where/local/")

Note that this can be used to have multiple versions of Ceres
installed.  However, particularly if you have only a single version of
Ceres which you want to use but do not wish to install to a system
location, you should consider exporting Ceres using the
``EXPORT_BUILD_DIR`` option instead of a local install, as exported
versions of Ceres will be automatically detected by CMake,
irrespective of their location.

Understanding the CMake Package System
----------------------------------------

Although a full tutorial on CMake is outside the scope of this guide,
here we cover some of the most common CMake misunderstandings that
crop up when using Ceres.  For more detailed CMake usage, the
following references are very useful:

- The `official CMake tutorial <http://www.cmake.org/cmake-tutorial/>`_

   Provides a tour of the core features of CMake.

- `ProjectConfig tutorial
  <http://www.cmake.org/Wiki/CMake/Tutorials/How_to_create_a_ProjectConfig.cmake_file>`_
  and the `cmake-packages documentation
  <http://www.cmake.org/cmake/help/git-master/manual/cmake-packages.7.html>`_

   Cover how to write a ``ProjectConfig.cmake`` file, discussed below,
   for your own project when installing or exporting it using CMake.
   It also covers how these processes in conjunction with
   ``find_package()`` are actually handled by CMake.  The
   `ProjectConfig tutorial
   <http://www.cmake.org/Wiki/CMake/Tutorials/How_to_create_a_ProjectConfig.cmake_file>`_
   is the older style, currently used by Ceres for compatibility with
   older versions of CMake.

  .. NOTE :: **Targets in CMake.**

    All libraries and executables built using CMake are represented as
    *targets* created using `add_library()
    <http://www.cmake.org/cmake/help/v3.2/command/add_library.html>`_
    and `add_executable()
    <http://www.cmake.org/cmake/help/v3.2/command/add_executable.html>`_.
    Targets encapsulate the rules and dependencies (which can be other
    targets) required to build or link against an object.  This allows
    CMake to implicitly manage dependency chains.  Thus it is
    sufficient to tell CMake that a library target: ``B`` depends on a
    previously declared library target ``A``, and CMake will
    understand that this means that ``B`` also depends on all of the
    public dependencies of ``A``.

When a project like Ceres is installed using CMake, or its build
directory is exported into the local CMake package registry (see
:ref:`section-install-vs-export`), in addition to the public headers
and compiled libraries, a set of CMake-specific project configuration
files are also installed to: ``<INSTALL_ROOT>/lib/cmake/Ceres`` (if Ceres
is installed), or created in the build directory (if Ceres' build
directory is exported).  When `find_package
<http://www.cmake.org/cmake/help/v3.2/command/find_package.html>`_ is
invoked, CMake checks various standard install locations (including
``/usr/local`` on Linux & UNIX systems), and the local CMake package
registry for CMake configuration files for the project to be found
(i.e. Ceres in the case of ``find_package(Ceres)``).  Specifically it
looks for:

- ``<PROJECT_NAME>Config.cmake`` (or
  ``<lower_case_project_name>-config.cmake``)

   Which is written by the developers of the project, and is
   configured with the selected options and installed locations when
   the project is built and imports the project targets and/or defines
   the legacy CMake variables: ``<PROJECT_NAME>_INCLUDE_DIRS`` &
   ``<PROJECT_NAME>_LIBRARIES`` which are used by the caller.

The ``<PROJECT_NAME>Config.cmake`` typically includes a second file
installed to the same location:

- ``<PROJECT_NAME>Targets.cmake``

   Which is autogenerated by CMake as part of the install process and defines
   **imported targets** for the project in the caller's CMake scope.

An **imported target** contains the same information about a library
as a CMake target that was declared locally in the current CMake
project using ``add_library()``.  However, imported targets refer to
objects that have already been built by a different CMake project.
Principally, an imported target contains the location of the compiled
object and all of its public dependencies required to link against it
as well as all required include directories.  Any locally declared target
can depend on an imported target, and CMake will manage the dependency
chain, just as if the imported target had been declared locally by the
current project.

Crucially, just like any locally declared CMake target, an imported target is
identified by its **name** when adding it as a dependency to another target.

Since v2.0, Ceres has used the target namespace feature of CMake to prefix
its export targets: ``Ceres::ceres``.  However, historically the Ceres target
did not have a namespace, and was just called ``ceres``.

Whilst an alias target called ``ceres`` is still provided in v2.0 for backwards
compatibility, it creates a potential drawback, if you failed to call
``find_package(Ceres)``, and Ceres is installed in a default search path for
your compiler, then instead of matching the imported Ceres target, it will
instead match the installed libceres.so/dylib/a library.  If this happens you
will get either compiler errors for missing include directories or linker errors
due to missing references to Ceres public dependencies.

Note that this description applies both to projects that are
**installed** using CMake, and to those whose **build directory is
exported** using `export()
<http://www.cmake.org/cmake/help/v3.2/command/export.html>`_ (instead
of `install()
<http://www.cmake.org/cmake/help/v3.2/command/install.html>`_).  Ceres
supports both installation and export of its build directory if the
``EXPORT_BUILD_DIR`` option is enabled, see
:ref:`section-customizing`.

.. _section-install-vs-export:

Installing a project with CMake vs Exporting its build directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a project is **installed**, the compiled libraries and headers
are copied from the source & build directory to the install location,
and it is these copied files that are used by any client code.  When a
project's build directory is **exported**, instead of copying the
compiled libraries and headers, CMake creates an entry for the project
in the `user's local CMake package registry
<http://www.cmake.org/cmake/help/v3.2/manual/cmake-packages.7.html#user-package-registry>`_,
``<USER_HOME>/.cmake/packages`` on Linux & OS X, which contains the
path to the project's build directory which will be checked by CMake
during a call to ``find_package()``.  The effect of which is that any
client code uses the compiled libraries and headers in the build
directory directly, **thus not requiring the project to be installed
to be used**.

Installing / Exporting a project that uses Ceres
--------------------------------------------------

As described in `Understanding the CMake Package System`_, the contents of
the ``CERES_LIBRARIES`` variable is the **name** of an imported target which
represents Ceres.  If you are installing / exporting your *own* project which
*uses* Ceres, it is important to understand that:

**Imported targets are not (re)exported when a project which imported them is
exported**.

Thus, when a project ``Foo`` which uses Ceres is exported, its list of
dependencies as seen by another project ``Bar`` which imports ``Foo``
via: ``find_package(Foo REQUIRED)`` will contain: ``ceres``.  However,
the definition of ``ceres`` as an imported target is **not
(re)exported** when Foo is exported.  Hence, without any additional
steps, when processing ``Bar``, ``ceres`` will not be defined as an
imported target.  Thus, when processing ``Bar``, CMake will assume
that ``ceres`` refers only to: ``libceres.a/so/dylib/lib`` (the
compiled Ceres library) directly if it is on the current list of
search paths.  In which case, no CMake errors will occur, but ``Bar``
will not link properly, as it does not have the required public link
dependencies of Ceres, which are stored in the imported target
definition.

The solution to this is for ``Foo`` (i.e., the project that uses
Ceres) to invoke ``find_package(Ceres)`` in ``FooConfig.cmake``, thus
``ceres`` will be defined as an imported target when CMake processes
``Bar``.  An example of the required modifications to
``FooConfig.cmake`` are show below:

.. code-block:: cmake

    # Importing Ceres in FooConfig.cmake using CMake 3.x style.
    #
    # In CMake v3.x, the find_dependency() macro exists to forward the REQUIRED
    # / QUIET parameters to find_package() when searching for dependencies.
    #
    # Note that find_dependency() does not take a path hint, so if Ceres was
    # installed in a non-standard location, that location must be added to
    # CMake's search list before this call.
    include(CMakeFindDependencyMacro)
    find_dependency(Ceres)
