.. _chapter-installation:

============
Installation
============

.. _section-dependencies:

Dependencies
============

 .. note ::

    Starting with version 2.2, Ceres Solver requires a **fully
    C++17-compliant** compiler.

Ceres relies on a number of open source libraries, some of which are
optional. For details on customizing the build process, see
:ref:`section-customizing` .

- `CMake <https://cmake.org>`_ (**Required**) 3.22 or later.

- `Eigen <https://libeigen.gitlab.io/>`_
  (**Required**) 3.3.4 or later.

  .. NOTE ::

    Ceres can also use Eigen as a sparse linear algebra
    library. Please see the documentation for ``WITH_EIGENSPARSE`` for
    more details.

- `Abseil <https://abseil.io/>`_ (**Required**) 20240116 or later.

- `GoogleTest <https://github.com/google/googletest>`_ (**Optional**;
  Required if you wish to build and run tests) 1.14.0 or later.

- `Google Benchmark <https://github.com/google/benchmark>`_ (**Optional**;
  used if ``BUILD_BENCHMARKS`` is ``ON``).

- `SuiteSparse <https://people.engr.tamu.edu/davis/suitesparse.html>`_
  (**Optional; strongly recommended for large problems**) 4.5.6 or
  later (7.4.0 or later for single-precision ``CHOLMOD`` support). Needed
  for solving large sparse linear systems. Because ``SuiteSparse``'s
  supernodal ``CHOLMOD`` and ``SPQR`` components are licensed under the
  GPL, ``WITH_SUITESPARSE`` defaults to ``OFF``; pass
  ``-DWITH_SUITESPARSE=ON`` to ``CMake`` to enable it (see
  ``WITH_SUITESPARSE`` in :ref:`options-controlling-ceres-configuration`).

  .. NOTE ::

     If SuiteSparseQR is found, Ceres attempts to find the Intel
     Thread Building Blocks (TBB) library. If found, Ceres assumes
     SuiteSparseQR was compiled with TBB support and will link to the
     found TBB version. You can customize the searched TBB location
     with the ``TBB_ROOT`` variable.

  If Ceres is built with oneMKL, SuiteSparse must use the same BLAS and
  LAPACK integer interface, LP64 or ILP64, as oneMKL. A mismatch can link
  successfully but fail at runtime, for example with ``Intel oneMKL ERROR:
  Parameter 4 was incorrect on entry to DPOTRF``. During configuration, Ceres
  compares ``SUITESPARSE_BLAS_INT`` with ``MKL_INTERFACE_FULL`` and runs a
  small supernodal CHOLMOD factorization, which calls LAPACK. It disables
  SuiteSparse if either check fails or if SuiteSparse does not define
  ``SUITESPARSE_BLAS_INT``. When cross
  compiling, the factorization runs only if ``CMAKE_CROSSCOMPILING_EMULATOR``
  is set. Projects that change the BLAS or LAPACK libraries after Ceres is
  configured must keep the interfaces consistent themselves. See the `oneMKL
  documentation on LP64 and ILP64
  <https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2026-0/using-the-ilp64-interface-vs-lp64-interface.html>`_.

- `Intel oneMKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`_
  (**Optional**). oneMKL provides the Sparse QR covariance estimation selected
  by ``MKL_SPARSE``. If enabled, oneMKL also replaces the BLAS and LAPACK
  libraries.

- `METIS <https://github.com/KarypisLab/METIS>`_ (**Optional**) Used by
  ``SuiteSparse``, optionally by ``Eigen``'s sparse solvers (when
  ``WITH_EIGENMETIS=ON``), and directly by Ceres for graph partitioning.

- `Apple's Accelerate sparse solvers
  <https://developer.apple.com/documentation/accelerate/sparse-solvers-library>`_. (**Optional**)

  As of Xcode 9.0, Apple's Accelerate framework includes support for
  solving sparse linear systems across macOS, iOS et al.

- `BLAS <https://www.netlib.org/blas/>`_ and `LAPACK
  <https://www.netlib.org/lapack/>`_ (**Optional**) ``LAPACK`` and
  ``BLAS`` routines are used by ``SuiteSparse``, and optionally used by
  Ceres directly for some operations (controlled by ``WITH_LAPACK``).

  For best performance on ``x86`` based Linux systems we recommend
  using `Intel oneAPI MKL
  <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`_.

  Another good option is `OpenBLAS
  <https://github.com/OpenMathLib/OpenBLAS>`_. However, one needs to be
  careful to `turn off the threading
  <https://www.openmathlib.org/OpenBLAS/docs/faq/#how-can-i-use-openblas-in-multi-threaded-applications>`_
  inside ``OpenBLAS`` as it conflicts with use of threads in Ceres.

  macOS ships with an optimized ``LAPACK`` and ``BLAS``
  implementation as part of the ``Accelerate`` framework. The Ceres
  build system will automatically detect and use it.

  On Windows, ``BLAS`` and ``LAPACK`` (such as ``OpenBLAS`` or ``Intel MKL``)
  can be installed via `vcpkg <https://github.com/microsoft/vcpkg>`_ or
  `MSYS2 <https://www.msys2.org/>`_, or obtained as part of the prebuilt
  ``SuiteSparse`` binary packages (see :ref:`section-windows`).


- `CUDA <https://developer.nvidia.com/cuda/toolkit>`_ and `cuDSS
  <https://developer.nvidia.com/cudss>`_ (**Optional**)

  If you have an NVIDIA GPU then Ceres Solver can use it to accelerate
  the solution of the Gauss-Newton linear systems using the CMake flag
  ``WITH_CUDA``.

  This support depends on two libraries from NVIDIA: ``CUDA`` and ``cuDSS``.

  If ``CUDA`` is available, Ceres Solver is able to use
  GPU acceleration to speed up ``DENSE_QR``, ``DENSE_NORMAL_CHOLESKY``,
  ``DENSE_SCHUR``, and ``CGNR``.  This also enables ``CUDA`` based
  mixed precision solves for ``DENSE_NORMAL_CHOLESKY`` and
  ``DENSE_SCHUR``.

  Additionally, if ``cuDSS`` is also available (controlled by
  ``WITH_CUDSS``), then GPU acceleration can be used for
  ``SPARSE_NORMAL_CHOLESKY`` and ``SPARSE_SCHUR`` via the
  ``CUDA_SPARSE`` sparse linear algebra library.

.. _section-source:

Getting the source code
=======================


You can start with the `latest stable release
<http://ceres-solver.org/ceres-solver-2.3.0.tar.gz>`_ . Or if you want
the latest version, you can clone the git repository

.. code-block:: bash

   git clone https://github.com/ceres-solver/ceres-solver

If your system does not have recent enough versions of `Abseil
<https://abseil.io/>`_ (>= 20240116) and/or `GoogleTest
<https://github.com/google/googletest>`_ (>= 1.14.0), then you can use
the versions included with Ceres Solver as git submodules by using the
command

.. code-block:: bash

   git clone --recurse-submodules https://github.com/ceres-solver/ceres-solver

The build instructions below use ``ceres-solver-2.3.0`` as the source
directory because they assume a release archive. If you build from a Git
checkout, replace it with the name of your checkout, usually ``ceres-solver``.


.. _section-linux:

Linux
=====

We will use `Ubuntu <https://ubuntu.com/>`_ as our example Linux
distribution.

.. NOTE::

   Ceres Solver always supports the previous and current Ubuntu LTS
   releases, currently 22.04 and 24.04 using the default Ubuntu
   repositories and compiler toolchain. Support for earlier versions
   is not guaranteed or maintained.

   Because the default ``apt`` repositories on Ubuntu 22.04 and 24.04
   ship versions of ``libabsl-dev`` older than ``20240116``, we
   recommend cloning Ceres with ``--recurse-submodules`` on those
   distributions so that CMake automatically builds the bundled
   ``Abseil`` and ``GoogleTest`` submodules in ``third_party/``. On
   newer distributions that ship ``Abseil >= 20240116`` and
   ``GoogleTest >= 1.14.0``, you can install ``libabsl-dev`` and
   ``libgtest-dev`` directly via ``apt``.

Start by installing the dependencies:

.. code-block:: bash

     # CMake
     sudo apt-get install cmake
     # BLAS & LAPACK
     sudo apt-get install libblas-dev liblapack-dev
     # Eigen3
     sudo apt-get install libeigen3-dev
     # SuiteSparse and METIS (optional)
     sudo apt-get install libsuitesparse-dev libmetis-dev
     # Abseil and GoogleTest (if your distribution ships >= 20240116 and >= 1.14.0;
     # otherwise clone Ceres with --recurse-submodules)
     sudo apt-get install libabsl-dev libgtest-dev

We are now ready to build, test, and install Ceres.

.. code-block:: bash

 tar zxf ceres-solver-2.3.0.tar.gz
 cmake -S ceres-solver-2.3.0 -B ceres-bin -DWITH_SUITESPARSE=ON
 cmake --build ceres-bin -j
 ctest --test-dir ceres-bin --output-on-failure
 # Optionally install Ceres system-wide (or use ceres-bin directly via Ceres_DIR;
 # see "Using Ceres with CMake" below).
 sudo cmake --install ceres-bin

You can also try running the command line bundling application with one of the
included problems, which comes from the University of Washington's BAL
dataset [Agarwal]_.

.. code-block:: bash

 ceres-bin/bin/simple_bundle_adjuster ceres-solver-2.3.0/data/problem-16-22106-pre.txt

This runs Ceres for a maximum of 10 iterations using the
``DENSE_SCHUR`` linear solver. The output should look something like
this.

.. code-block:: bash

    iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
       0  4.185660e+06    0.00e+00    1.09e+08   0.00e+00   0.00e+00  1.00e+04        0    2.18e-02    6.57e-02
       1  1.062590e+05    4.08e+06    8.99e+06   0.00e+00   9.82e-01  3.00e+04        1    5.07e-02    1.16e-01
       2  4.992817e+04    5.63e+04    8.32e+06   3.19e+02   6.52e-01  3.09e+04        1    4.75e-02    1.64e-01
       3  1.899774e+04    3.09e+04    1.60e+06   1.24e+02   9.77e-01  9.26e+04        1    4.74e-02    2.11e-01
       4  1.808729e+04    9.10e+02    3.97e+05   6.39e+01   9.51e-01  2.78e+05        1    4.75e-02    2.59e-01
       5  1.803399e+04    5.33e+01    1.48e+04   1.23e+01   9.99e-01  8.33e+05        1    4.74e-02    3.06e-01
       6  1.803390e+04    9.02e-02    6.35e+01   8.00e+01   1.00e+00  2.50e+06        1    4.76e-02    3.54e-01

    Solver Summary (v 2.3.0-eigen-(3.4.0)-lapack-suitesparse-(7.1.0)-metis-(5.1.0)-acceleratesparse-eigensparse)

                                         Original                  Reduced
    Parameter blocks                        22122                    22122
    Parameters                              66462                    66462
    Residual blocks                         83718                    83718
    Residuals                              167436                   167436

    Minimizer                        TRUST_REGION

    Dense linear algebra library            EIGEN
    Trust region strategy     LEVENBERG_MARQUARDT
                                            Given                     Used
    Linear solver                     DENSE_SCHUR              DENSE_SCHUR
    Threads                                     1                        1
    Linear solver ordering              AUTOMATIC                 22106,16
    Schur structure                         2,3,9                    2,3,9

    Cost:
    Initial                          4.185660e+06
    Final                            1.803390e+04
    Change                           4.167626e+06

    Minimizer iterations                        7
    Successful steps                            7
    Unsuccessful steps                          0

    Time (in seconds):
    Preprocessor                         0.043895

      Residual only evaluation           0.029855 (7)
      Jacobian & residual evaluation     0.120581 (7)
      Linear solver                      0.153665 (7)
    Minimizer                            0.339275

    Postprocessor                        0.000540
    Total                                0.383710

    Termination:                      CONVERGENCE (Function tolerance reached. |cost_change|/cost: 1.769759e-09 <= 1.000000e-06)


.. _section-macos:

macOS
=====

On macOS, you can either use `Homebrew <https://brew.sh/>`_
(recommended) or `MacPorts <https://www.macports.org/>`_ to install
Ceres Solver.

If using `Homebrew <https://brew.sh/>`_, then

.. code-block:: bash

      brew install ceres-solver

will install the latest stable version along with all the required
dependencies and

.. code-block:: bash

      brew install ceres-solver --HEAD

will install the latest version in the git repo.

If using `MacPorts <https://www.macports.org/>`_, then

.. code-block:: bash

   sudo port install ceres-solver

will install the latest version.

You can also install each of the dependencies by hand using `Homebrew
<https://brew.sh/>`_. There is no need to install
``BLAS`` or ``LAPACK`` separately as macOS ships with optimized
``BLAS`` and ``LAPACK`` routines as part of the `Accelerate
<https://developer.apple.com/documentation/accelerate>`_
framework.

.. code-block:: bash

      # CMake
      brew install cmake
      # Abseil and GoogleTest
      brew install abseil googletest
      # Eigen3
      brew install eigen
      # SuiteSparse and METIS (optional)
      brew install suite-sparse metis

We are now ready to build, test, and install Ceres.

.. code-block:: bash

   tar zxf ceres-solver-2.3.0.tar.gz
   cmake -S ceres-solver-2.3.0 -B ceres-bin -DWITH_SUITESPARSE=ON
   cmake --build ceres-bin -j
   ctest --test-dir ceres-bin --output-on-failure
   # Optionally install Ceres (or use ceres-bin directly via Ceres_DIR;
   # see "Using Ceres with CMake" below).
   cmake --install ceres-bin

.. _section-windows:

Windows
=======

Using a Library Manager
-----------------------

`vcpkg <https://github.com/microsoft/vcpkg>`_ is a library manager for Microsoft
Windows that can be used to install Ceres Solver and all its dependencies.

#. Install the library manager into a top-level directory ``vcpkg/`` on Windows
   following the `guide
   <https://github.com/microsoft/vcpkg#quick-start-windows>`_, e.g., using
   Visual Studio 2022 community edition, or simply run

    .. code:: bat

        git clone https://github.com/Microsoft/vcpkg.git
        cd vcpkg
        .\bootstrap-vcpkg.bat
        .\vcpkg integrate install

#. Use vcpkg to install and build Ceres and all its dependencies, e.g., for 64
   bit Windows

   .. code:: bat

      vcpkg\vcpkg.exe install ceres:x64-windows

   Or with optional components, e.g., SuiteSparse, using

   .. code:: bat

      vcpkg\vcpkg.exe install ceres[suitesparse]:x64-windows

#. Integrate vcpkg packages with Visual Studio to allow it to automatically
   find all the libraries installed by vcpkg.

   .. code:: bat

      vcpkg\vcpkg.exe integrate install

#. To use Ceres in a CMake project, follow our :ref:`instructions
   <section-using-ceres>`.


Building from Source
--------------------

Ceres Solver can also be built from source on Windows using Visual Studio 2019
or newer (or MSYS2 / MinGW-w64).

#. Create a top-level directory for dependencies, build, and sources somewhere,
   e.g., ``ceres/``

#. Obtain the Ceres source code and its dependencies:

   #. **Abseil** (>= 20240116) and **GoogleTest** (>= 1.14.0): The simplest way
      to provide ``Abseil`` and ``GoogleTest`` when building from source on
      Windows is to clone Ceres with ``--recurse-submodules``:

      .. code:: bat

         git clone --recurse-submodules https://github.com/ceres-solver/ceres-solver

      Alternatively, you can build and install ``Abseil`` and ``GoogleTest``
      separately using CMake and pass their install prefix via
      ``CMAKE_PREFIX_PATH`` (or set ``absl_DIR`` / ``absl_ROOT`` and
      ``GTest_DIR`` / ``GTest_ROOT``).

   #. **Eigen** (>= 3.3.4): Unpack and configure/install Eigen using CMake, and
      pass ``-DEigen3_ROOT=<path/to/eigen>`` (or
      ``-DEigen3_DIR=<path/to/Eigen3Config.cmake>``) when configuring Ceres.

   #. (Optional) **SuiteSparse**: You can build SuiteSparse on Windows using
      `suitesparse-metis-for-windows
      <https://github.com/jlblancoc/suitesparse-metis-for-windows>`_ or
      official ``SuiteSparse`` CMake releases, or use the prebuilt binary
      packages for Visual Studio 2019 and 2022 provided by the `CMake support
      for SuiteSparse <https://github.com/sergiud/SuiteSparse>`_ project that
      also include `reference LAPACK <https://www.netlib.org/blas/>`_ (and
      BLAS). When using the prebuilt ``SuiteSparse`` archive, add its directory
      to ``CMAKE_PREFIX_PATH``, enable ``-DWITH_SUITESPARSE=ON``, and set
      ``BLAS_blas_LIBRARY`` and ``LAPACK_lapack_LIBRARY`` to
      ``<suitesparse_path>/lib/libblas.lib`` and
      ``<suitesparse_path>/lib/liblapack.lib``.

#. Install ``CMake``.

#. Configure and build Ceres out-of-tree in ``ceres-bin`` (either from the
   command line or using ``cmake-gui`` and the Visual Studio IDE):

   .. code:: bat

      cmake -S ceres-solver -B ceres-bin -DEigen3_ROOT=C:\path\to\eigen
      cmake --build ceres-bin --config Release

   If you are using ``cmake-gui``, select ``ceres-solver`` as the source
   directory and ``ceres-bin`` as the build directory, click **Configure**, set
   any missing dependency locations (see
   :ref:`options-controlling-ceres-dependency-locations`), click **Generate**,
   and then open the generated Visual Studio solution in ``ceres-bin`` to build
   it.

#. To run the tests, run ``ctest`` from the command line:

   .. code:: bat

      ctest --test-dir ceres-bin -C Release --output-on-failure

   Or in Visual Studio, select the ``RUN_TESTS`` target and click **Build
   RUN_TESTS** from the build menu.

Like the Linux build, you should now be able to run
``ceres-bin\bin\Release\simple_bundle_adjuster.exe``.

.. note::

    #. The default multi-config build in Visual Studio is ``Debug``; always pass
       ``--config Release`` on the command line or switch the active
       configuration to ``Release`` in Visual Studio for optimal performance.
    #. CMake puts the resulting binaries in ``ceres-bin/bin/Debug`` or
       ``ceres-bin/bin/Release`` by default.
    #. Without a sparse linear algebra library, only a subset of
       solvers is usable, namely: ``DENSE_QR``, ``DENSE_NORMAL_CHOLESKY``,
       ``DENSE_SCHUR``, ``CGNR``, and ``ITERATIVE_SCHUR``.


.. _section-android:

Android
=======

.. NOTE::

    You will need Android NDK r20 or higher (current LTS such as r27 is
    recommended) to build Ceres Solver with full C++17 support.

To build Ceres for Android, clone Ceres with ``--recurse-submodules`` (so that
``Abseil`` is built alongside Ceres for the target Android ABI, or provide a
cross-compiled ``Abseil`` via ``-Dabsl_DIR=...``) and instruct ``CMake`` to use
the toolchain file from the Android NDK. For example, assuming you have
specified ``$NDK_DIR``:

.. code-block:: bash

    cmake -S <PATH_TO_CERES_SOURCE> -B ceres-bin \
      -DCMAKE_TOOLCHAIN_FILE=$NDK_DIR/build/cmake/android.toolchain.cmake \
      -DEigen3_DIR=/path/to/Eigen3Config.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_STL=c++_shared \
      -DANDROID_NATIVE_API_LEVEL=android-29 \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_TESTING=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_BENCHMARKS=OFF
    cmake --build ceres-bin -j

You can build for any supported Android STL or ABI (such as ``arm64-v8a``,
``armeabi-v7a``, ``x86_64``, or ``x86``). Several API levels are supported;
use an API level appropriate for your Android project.

.. NOTE::

    You must always use the same API level and STL library for
    your Android project and the Ceres binaries.

After building, you get a ``libceres.so`` library (and Abseil libraries if
built as shared libraries), which you can link into your Android build.

If you also build the Ceres sample binaries (by setting
``-DBUILD_EXAMPLES=ON``) and would like to verify your library on an Android
device, place the sample binary together with ``libceres.so`` (and the NDK STL
shared library ``libc++_shared.so`` and any other shared dependencies) in an
executable directory on the device such as ``/data/local/tmp``, and run:

.. code-block:: bash

    adb shell
    cd /data/local/tmp
    LD_LIBRARY_PATH=/data/local/tmp ./helloworld

.. _section-ios:

iOS
===

.. NOTE::

   You need at least iOS 7.0 or higher to build Ceres Solver (iOS 12.0 or
   higher is recommended for modern Xcode toolchains).

To build Ceres for iOS, clone Ceres with ``--recurse-submodules`` (or provide
``Abseil`` for iOS via ``-Dabsl_DIR=...``) and instruct ``CMake`` to use the
``cmake/iOS.cmake`` toolchain file:

.. code-block:: bash

   cmake -S <PATH_TO_CERES_SOURCE> -B ceres-bin \
     -DCMAKE_TOOLCHAIN_FILE=<PATH_TO_CERES_SOURCE>/cmake/iOS.cmake \
     -DEigen3_DIR=/path/to/Eigen3Config.cmake \
     -DIOS_PLATFORM=<PLATFORM> \
     -DBUILD_TESTING=OFF \
     -DBUILD_EXAMPLES=OFF \
     -DBUILD_BENCHMARKS=OFF
   cmake --build ceres-bin --config Release

``PLATFORM`` can be ``OS`` (``iphoneos``, ``arm64``), ``SIMULATOR64``
(``iphonesimulator``, ``x86_64``), or ``SIMULATOR`` (``iphonesimulator``,
``i386``). See ``cmake/iOS.cmake`` for additional options such as
``IOS_DEPLOYMENT_TARGET``.

After building, you will get ``libceres.a`` (and the required ``Abseil`` static
libraries when built from the submodule), which you will need to link into your
Xcode project.

The default iOS configuration builds Ceres Solver using ``Eigen`` and
``Abseil``, which is sufficient for solving small to moderate sized problems
(and can also use Apple's ``Accelerate`` sparse solvers when available).

If you decide to use ``Accelerate`` (including ``AccelerateSparse``), you also
need to add ``Accelerate.framework`` to your Xcode project's linked
frameworks.

.. _section-customizing:

Customizing the build
=====================

It is possible to reduce the libraries needed to build Ceres and
customize the build process by setting the appropriate options in
``CMake``.  These options can either be set in the ``CMake`` GUI, or
via ``-D<OPTION>=<ON/OFF>`` when running ``CMake`` from the command
line.  In general, you should only modify these options from their
defaults if you know what you are doing.

For additional configure diagnostics, pass ``--log-level=VERBOSE``,
``--log-level=DEBUG`` or ``--log-level=TRACE`` to ``CMake``.  To keep a log
level for subsequent configure runs in a build directory, set
``CMAKE_MESSAGE_LOG_LEVEL``, for example with
``-DCMAKE_MESSAGE_LOG_LEVEL=DEBUG``.  The command-line option takes precedence
over the variable.

``STATUS`` messages describe normal configuration progress.  ``VERBOSE``
messages add detailed toolchain and configuration choices.  ``DEBUG`` messages
expose internal setup details.  ``TRACE`` is reserved for temporary low-level
diagnostics.

.. NOTE::

   Passing ``-D<VARIABLE>=<VALUE>`` on the command line forcibly overwrites
   ``<VARIABLE>`` in the ``CMake`` cache at the start of every configure. If you
   are using the interactive ``ccmake`` terminal GUI, avoid passing ``-D`` on
   the ``ccmake`` command line so that interactive changes in the GUI (use
   ``<t>`` to toggle *Advanced View*) are not overwritten on reconfigure.


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

Ceres-specific feature options use the ``WITH_`` prefix. The configure step
prints a summary of enabled features and discovered dependencies at the end of
the output. The summary includes the Ceres version, package versions when they
are reported by the package finders, and optional packages that were not found.
An unavailable optional package does not change the corresponding cache option.

#. ``BUILD_BENCHMARKS [Default: ON]``: Enable the Ceres benchmarking suite
   when the benchmark dependency is available.

#. ``BUILD_DOCUMENTATION [Default: OFF]``: Use this to enable building
   the documentation. This requires `Sphinx <https://www.sphinx-doc.org/>`_ and
   the `sphinx-rtd-theme
   <https://pypi.org/project/sphinx-rtd-theme/>`_ package
   available from the Python package index. In addition, ``make ceres_docs``
   can be used to build only the documentation.

#. ``BUILD_EXAMPLES [Default: ON]``: Build the Ceres example programs.

#. ``BUILD_SHARED_LIBS [Default: OFF]``: By default Ceres is built as
   a static library. Turn this ``ON`` to build Ceres as a shared library.

#. ``BUILD_TESTING [Default: ON]``: Enable the Ceres unit and integration tests.

#. ``CMAKE_EXPORT_PACKAGE_REGISTRY [Default: OFF]``: Ceres always generates a
   build-tree package export, so clients can use the build directory without
   installing Ceres. Set this standard CMake variable to ``ON`` to also have
   ``export(PACKAGE)`` register the build directory in the `user's local
   CMake package registry
   <https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#user-package-registry>`_,
   so that it is found automatically by ``find_package(Ceres)`` without
   setting ``Ceres_DIR``. It is left disabled by default, matching CMake's
   own default for ``export(PACKAGE)``.

#. ``CMAKE_INSTALL_LIBDIR [Default: platform-dependent]``: Set this standard
   CMake variable to choose the directory below ``CMAKE_INSTALL_PREFIX`` where
   Ceres libraries are installed. It defaults according to CMake's
   ``GNUInstallDirs`` module.

#. ``CMAKE_MSVC_RUNTIME_LIBRARY [Default: compiler default]`` *Windows Only*:
   Set this standard CMake variable to select the MSVC runtime library used to
   build Ceres. When it is not set, CMake uses the compiler default.

#. ``WITH_ACCELERATESPARSE [Default: ON]``: By default, Ceres will link to
   Apple's Accelerate framework directly if a version of it is detected
   which supports solving sparse linear systems. On Apple operating systems,
   Accelerate usually also provides the BLAS and LAPACK implementations and
   is linked irrespective of this option.

#. ``WITH_BITCODE [Default: OFF]`` *Apple platforms*: Enable bitcode for iOS
   builds. This disables Eigen's additional Clang inlining optimization.

#. ``WITH_CUDA [Default: default]``: Enable CUDA linear algebra solvers. The
   value can be ``OFF``, ``default``, or ``static``. The latter links against
   static CUDA runtime libraries when supported.

#. ``WITH_CUDSS [Default: ON]``: Enable NVIDIA cuDSS support for sparse CUDA
   linear solvers (``CUDA_SPARSE``). This option is available when ``WITH_CUDA``
   is enabled and the ``cudss`` package is found.

#. ``WITH_CUSTOM_BLAS [Default: ON]``: Use Ceres' custom BLAS routines instead
   of Eigen's implementations where available.

#. ``WITH_EIGENMETIS [Default: ON]``: Enable METIS support for Eigen's sparse
   solvers when METIS is available. This option is available when
   ``WITH_EIGENSPARSE`` is enabled.

#. ``WITH_EIGENSPARSE [Default: ON]``: By default, Ceres will use Eigen's
   sparse Cholesky factorization.

#. ``WITH_LAPACK [Default: ON]``: If this option is enabled, and the ``BLAS`` and
   ``LAPACK`` libraries are found, Ceres will enable **direct** use of
   ``LAPACK`` routines. If this option is disabled, Ceres does not use
   ``LAPACK`` directly. SuiteSparse determines its own ``BLAS`` and ``LAPACK``
   link dependencies independently. Direct LAPACK use is always disabled for
   iOS builds because Apple treats ``dsyrk_`` as a private API.

#. ``WITH_MKL [Default: ON]``: Use Intel oneMKL if its CMake package
   configuration ``MKLConfig.cmake`` is found. oneMKL provides it since
   version 2021.3. This enables ``MKL_SPARSE`` for covariance estimation, and
   Ceres then also uses oneMKL for BLAS and LAPACK. If oneMKL is not found,
   set ``MKL_DIR`` to the directory containing ``MKLConfig.cmake`` or add the
   oneMKL installation prefix to ``CMAKE_PREFIX_PATH``.

   ``MKL_INTERFACE_FULL`` selects the integer interface. If it is not set,
   oneMKL chooses its own default, which is ILP64 in oneMKL 2026.0. Set
   ``MKL_INTERFACE_FULL=intel_lp64`` to keep using a SuiteSparse installation
   built for LP64. Otherwise, Ceres disables SuiteSparse, which also changes
   the default sparse linear algebra library.

#. ``WITH_SANITIZERS [Default: empty]``: A semicolon-separated list of
   sanitizers to enable, such as ``address`` or ``thread``.

#. ``WITH_SCHUR_SPECIALIZATIONS [Default: ON]``: If you are concerned about
   binary size or compilation time over some small performance gains in the
   ``SPARSE_SCHUR`` solver, you can disable some of the template
   specializations by turning this ``OFF``.

#. ``WITH_STRIPPED_DEBUG_SYMBOLS [Default: ON]`` *Android platforms*: Strip
   debug symbols from Android builds to reduce file sizes.

#. ``WITH_SUITESPARSE [Default: OFF]``: SuiteSparse support is opt-in. Turn this
   ``ON`` to link Ceres against ``SuiteSparse``, provided it and all of its
   dependencies are present.

   .. WARNING::

      SuiteSparse is licensed under a mixture of GPL/LGPL/Commercial terms.
      Ceres requires the CHOLMOD supernodal factorization and SPQR components,
      which are only available under GPL/Commercial terms. Consequently, unless
      you hold a commercial SuiteSparse license, a Ceres build with
      ``WITH_SUITESPARSE=ON`` is GPL licensed. This is why SuiteSparse support is
      opt-in rather than enabled by default. Obtaining a commercial SuiteSparse
      license removes this restriction.

#. ``WITH_UNINSTALL_TARGET [Default: ON]``: Add an ``uninstall`` target to
   remove files installed by Ceres.


.. _options-controlling-ceres-dependency-locations:

Options controlling Ceres dependency locations
----------------------------------------------

Ceres uses the ``CMake`` `find_package()
<https://cmake.org/cmake/help/latest/command/find_package.html>`_
command to find all of its dependencies.

#. **Config-mode packages** (``Eigen3``, ``absl``, ``GTest``, ``benchmark``,
   ``cudss``, ``TBB``, and ``SuiteSparse`` >= 7.0):

   These dependencies provide CMake package configuration files
   (``<PackageName>Config.cmake`` or ``<lowercase>-config.cmake``). You can
   customize where CMake searches for them using standard CMake variables:

   - ``CMAKE_PREFIX_PATH``: A semicolon-separated list of installation prefixes
     to search (e.g., ``-DCMAKE_PREFIX_PATH="/opt/local;/custom/prefix"``).
   - ``<PackageName>_ROOT``: The installation prefix for a specific package
     (e.g., ``-DEigen3_ROOT=/path/to/eigen``, ``-Dabsl_ROOT=/path/to/abseil``,
     ``-DTBB_ROOT=/path/to/tbb``).
   - ``<PackageName>_DIR``: The exact directory containing the package's
     ``<PackageName>Config.cmake`` file (e.g., ``-DEigen3_DIR=...``,
     ``-Dabsl_DIR=...``, ``-DGTest_DIR=...``, ``-Dcudss_DIR=...``).

   Note that if ``third_party/abseil-cpp`` and ``third_party/googletest`` are
   present (e.g., when Ceres is cloned with ``--recurse-submodules``), Ceres
   uses the bundled submodules directly instead of searching for installed
   ``absl`` and ``GTest`` packages.

#. **CUDA Toolkit** (``CUDAToolkit``):

   When ``WITH_CUDA`` is enabled, Ceres uses CMake's standard
   `FindCUDAToolkit
   <https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html>`_ module.
   You can customize the CUDA installation location using ``CUDAToolkit_ROOT``
   or ``CUDACXX``.

#. **Find-module dependencies** (``METIS``, ``SuiteSparse`` < 7.0 without
   CMake config files, ``AccelerateSparse``, and ``Sphinx``):

   When a dependency does not provide a CMake package configuration file, Ceres
   uses its bundled ``Find<Package>.cmake`` modules. You can guide these modules
   using ``CMAKE_PREFIX_PATH``, ``CMAKE_INCLUDE_PATH``, ``CMAKE_LIBRARY_PATH``,
   ``<Package>_ROOT``, or by setting explicit cache variables:

   - **METIS**: ``METIS_INCLUDE_DIR`` and ``METIS_LIBRARY``.
   - **SuiteSparse** (fallback when ``SuiteSparseConfig.cmake`` is not present):
     ``SuiteSparse_<COMPONENT>_INCLUDE_DIR`` and
     ``SuiteSparse_<COMPONENT>_LIBRARY`` (for components ``AMD``, ``CAMD``,
     ``CCOLAMD``, ``CHOLMOD``, ``COLAMD``, ``SPQR``, ``Config``).
   - **Sphinx**: ``Sphinx_EXECUTABLE``.

.. NOTE::

   The legacy Ceres-specific ``<DEPENDENCY_NAME>_INCLUDE_DIR_HINTS`` and
   ``<DEPENDENCY_NAME>_LIBRARY_DIR_HINTS`` variables from earlier versions of
   Ceres have been removed. Use ``CMAKE_PREFIX_PATH`` or ``<PackageName>_ROOT``
   instead.

A Ceres package built with oneMKL finds oneMKL again for downstream projects
using the integer interface Ceres was built with.

Building using custom BLAS & LAPACK installs
----------------------------------------------

If the standard find package scripts for ``BLAS`` & ``LAPACK`` which
ship with ``CMake`` fail to find the desired libraries on your system,
try setting ``CMAKE_LIBRARY_PATH`` (or ``CMAKE_PREFIX_PATH``) to the path(s) to
the directories containing the ``BLAS`` & ``LAPACK`` libraries when invoking
``CMake`` to build Ceres via ``-D<VAR>=<VALUE>``.  This should result in the
libraries being found for any common variant of each.

Alternatively, you may also directly specify the ``BLAS_LIBRARIES`` and
``LAPACK_LIBRARIES`` variables via ``-D<VAR>=<VALUE>`` when invoking CMake
to configure Ceres.

.. _section-bazel:

Building with Bazel
===================

Ceres Solver also provides a `Bazel <https://bazel.build/>`_ build configuration
using Bzlmod (``MODULE.bazel``), which builds Ceres with ``Eigen`` and
``Abseil`` (and ``GoogleTest`` / ``Google Benchmark`` for tests and benchmarks):

.. code-block:: bash

   # Build the Ceres library
   bazel build //:ceres

   # Build and run the unit tests
   bazel test //...

   # Build the example binaries
   bazel build //examples/...

.. _section-using-ceres:

Using Ceres with CMake
======================

In order to use Ceres in client code with CMake using `find_package()
<https://cmake.org/cmake/help/latest/command/find_package.html>`_
then either:

#. Ceres must have been installed with ``cmake --install`` (or
   ``make install``). If the install location is non-standard (i.e. is not in
   CMake's default search paths) then it will not be detected by default; see
   :ref:`section-local-installations`.

#. Or Ceres' build directory can be used directly without installing (see
   :ref:`section-install-vs-export`) by setting ``Ceres_DIR`` to the Ceres
   build directory (or by enabling ``CMAKE_EXPORT_PACKAGE_REGISTRY=ON`` when
   configuring Ceres).


As an example of how to use Ceres, to compile `examples/helloworld.cc
<https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/helloworld.cc>`_
in a separate standalone project, the following CMakeList.txt can be
used:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.22)

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
     In fact, since v2.0 ``CERES_INCLUDE_DIRS`` is not even set.

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

#. ``LAPACK``: Ceres built with direct LAPACK support (``WITH_LAPACK=ON``).

#. ``MKL``: Ceres built with oneMKL.

#. ``SuiteSparse``: Ceres built with SuiteSparse support (``WITH_SUITESPARSE=ON``).

#. ``AccelerateSparse``: Ceres built with Apple's Accelerate sparse solver
   support (``WITH_ACCELERATESPARSE=ON``).

#. ``EigenSparse``: Ceres built with Eigen's sparse Cholesky factorization
   support (``WITH_EIGENSPARSE=ON``).

#. ``cuDSS``: Ceres built with NVIDIA cuDSS sparse solver support
   (``WITH_CUDA`` and ``WITH_CUDSS=ON``).

#. ``SparseLinearAlgebraLibrary``: Ceres built with *at least one*
   sparse linear algebra library.  This is equivalent to
   ``SuiteSparse`` **OR** ``AccelerateSparse`` **OR** ``EigenSparse`` **OR**
   ``cuDSS``.

#. ``SchurSpecializations``: Ceres built with Schur specializations
   (``WITH_SCHUR_SPECIALIZATIONS=ON``).

#. ``Multithreading``: Ceres built with multithreading support.

To specify one/multiple Ceres components use the ``COMPONENTS`` argument to
`find_package()
<https://cmake.org/cmake/help/latest/command/find_package.html>`_ like so:

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
<https://cmake.org/cmake/help/latest/command/find_package.html>`_
call.  For example:

.. code-block:: cmake

    find_package(Ceres 2.3.0 REQUIRED)

.. _section-local-installations:

Local installations
-------------------

If Ceres was installed in a non-standard path by specifying
``-DCMAKE_INSTALL_PREFIX="/some/where/local"``, you can either pass
``-DCMAKE_PREFIX_PATH="/some/where/local"`` (or
``-DCeres_DIR="/some/where/local/lib/cmake/Ceres"``) when running CMake to
configure your project, or add the **PATHS** option to the ``find_package()``
command in your ``CMakeLists.txt``:

.. code-block:: cmake

   find_package(Ceres REQUIRED PATHS "/some/where/local/")

If you do not wish to install Ceres to a system location, you can also use
Ceres directly from its build directory (see :ref:`section-install-vs-export`).

Understanding the CMake Package System
----------------------------------------

Although a full tutorial on CMake is outside the scope of this guide,
here we cover some of the most common CMake misunderstandings that
crop up when using Ceres.  For more detailed CMake usage, the
following references are very useful:

- The `official CMake tutorial <https://cmake.org/cmake/help/latest/guide/tutorial/index.html>`_

   Provides a tour of the core features of CMake.

- `cmake-packages documentation
  <https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html>`_

   Covers how to write a ``ProjectConfig.cmake`` file, discussed below,
   for your own project when installing or exporting it using CMake,
   and how these processes in conjunction with ``find_package()`` are
   handled by CMake.

  .. NOTE :: **Targets in CMake.**

    All libraries and executables built using CMake are represented as
    *targets* created using `add_library()
    <https://cmake.org/cmake/help/latest/command/add_library.html>`_
    and `add_executable()
    <https://cmake.org/cmake/help/latest/command/add_executable.html>`_.
    Targets encapsulate the rules and dependencies (which can be other
    targets) required to build or link against an object.  This allows
    CMake to implicitly manage dependency chains.  Thus it is
    sufficient to tell CMake that a library target: ``B`` depends on a
    previously declared library target ``A``, and CMake will
    understand that this means that ``B`` also depends on all of the
    public dependencies of ``A``.

When a project like Ceres is installed using CMake, or its build
directory is exported (see :ref:`section-install-vs-export`), in addition to
the public headers and compiled libraries, a set of CMake-specific project
configuration files are also installed to: ``<INSTALL_ROOT>/lib/cmake/Ceres``
(if Ceres is installed), or created in the build directory. When `find_package
<https://cmake.org/cmake/help/latest/command/find_package.html>`_ is
invoked, CMake checks various standard install locations (including
``/usr/local`` on Linux & UNIX systems), any paths specified via
``CMAKE_PREFIX_PATH`` or ``Ceres_DIR``, and the local CMake package
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

Whilst a deprecated target called ``ceres`` is still provided for backwards
compatibility (and emits a CMake deprecation warning when linked against), it
creates a potential drawback: if you failed to call
``find_package(Ceres)``, and Ceres is installed in a default search path for
your compiler, then instead of matching the imported Ceres target, it will
instead match the installed ``libceres.so``/``dylib``/``a`` library.  If this
happens you will get either compiler errors for missing include directories or
linker errors due to missing references to Ceres public dependencies. Always
link against ``Ceres::ceres``.

.. _section-install-vs-export:

Installing a project with CMake vs Exporting its build directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a project is **installed**, the compiled libraries and headers
are copied from the source & build directory to the install location,
and it is these copied files that are used by any client code.

Ceres also generates ``CeresConfig.cmake`` and ``CeresTargets.cmake`` directly
in its **build directory** using `export()
<https://cmake.org/cmake/help/latest/command/export.html>`_, so client code can
use the compiled libraries and headers in the build directory directly
**without requiring Ceres to be installed**:

- By default, you can point a client project at the Ceres build directory by
  passing ``-DCeres_DIR=/path/to/ceres-bin`` when configuring the client
  project.
- Alternatively, if you pass ``-DCMAKE_EXPORT_PACKAGE_REGISTRY=ON`` when
  configuring Ceres, CMake also records the path to the Ceres build directory
  in the `user's local CMake package registry
  <https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#user-package-registry>`_
  (``<USER_HOME>/.cmake/packages`` on Linux & macOS), which is checked
  automatically during ``find_package(Ceres)``.

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
via: ``find_package(Foo REQUIRED)`` will contain: ``Ceres::ceres``.  However,
the definition of ``Ceres::ceres`` as an imported target is **not
(re)exported** when ``Foo`` is exported.  Hence, without any additional
steps, when processing ``Bar``, ``Ceres::ceres`` will not be defined as an
imported target.

The solution to this is for ``Foo`` (i.e., the project that uses
Ceres) to invoke ``find_dependency(Ceres)`` in ``FooConfig.cmake``, thus
``Ceres::ceres`` will be defined as an imported target when CMake processes
``Bar``.  An example of the required modifications to
``FooConfig.cmake`` is shown below:

.. code-block:: cmake

    # Importing Ceres in FooConfig.cmake.
    #
    # The find_dependency() macro forwards the REQUIRED / QUIET parameters to
    # find_package() when searching for dependencies.
    #
    # Note that find_dependency() does not take a path hint, so if Ceres was
    # installed in a non-standard location, that location must be added to
    # CMake's search list before this call.
    include(CMakeFindDependencyMacro)
    find_dependency(Ceres)

.. _section-migration:

Migration
=========

The following includes some hints for migrating from previous versions.

Version 2.3
-----------

- **Abseil replaces glog, gflags, and miniglog**: ``google-glog``, ``gflags``,
  and ``MINIGLOG`` are no longer used by Ceres. Instead, Ceres requires
  `Abseil <https://abseil.io/>`_ (version ``20240116`` or later) and uses
  `GoogleTest <https://github.com/google/googletest>`_ (version ``1.14.0`` or
  later) when ``BUILD_TESTING=ON``. If your system does not provide recent
  enough versions of ``Abseil`` or ``GoogleTest``, clone Ceres with
  ``git clone --recurse-submodules`` so that CMake automatically builds the
  bundled submodules in ``third_party/``.
- **CMake minimum version**: Building Ceres now requires CMake 3.22 or later.
- **CMake feature options use the ``WITH_`` prefix**:
  The Ceres-specific CMake options have been standardized to use the ``WITH_``
  prefix:

  - ``SUITESPARSE`` :math:`\rightarrow` ``WITH_SUITESPARSE``
  - ``EIGENSPARSE`` :math:`\rightarrow` ``WITH_EIGENSPARSE``
  - ``ACCELERATESPARSE`` :math:`\rightarrow` ``WITH_ACCELERATESPARSE``
  - ``USE_CUDA`` :math:`\rightarrow` ``WITH_CUDA`` (accepts ``OFF``,
    ``default``, or ``static``)
  - ``LAPACK`` :math:`\rightarrow` ``WITH_LAPACK``
  - ``CUSTOM_BLAS`` :math:`\rightarrow` ``WITH_CUSTOM_BLAS``
  - ``EIGENMETIS`` :math:`\rightarrow` ``WITH_EIGENMETIS``
  - ``SCHUR_SPECIALIZATIONS`` :math:`\rightarrow` ``WITH_SCHUR_SPECIALIZATIONS``
  - ``IOS_BITCODE`` :math:`\rightarrow` ``WITH_BITCODE``
  - ``ANDROID_STRIP_DEBUG_SYMBOLS`` :math:`\rightarrow`
    ``WITH_STRIPPED_DEBUG_SYMBOLS``
  - ``PROVIDE_UNINSTALL_TARGET`` :math:`\rightarrow` ``WITH_UNINSTALL_TARGET``

- **SuiteSparse is now opt-in (``WITH_SUITESPARSE=OFF`` by default)**:
  Because ``SuiteSparse``'s supernodal ``CHOLMOD`` and ``SPQR`` components are
  licensed under the GPL, ``WITH_SUITESPARSE`` now defaults to ``OFF``. Pass
  ``-DWITH_SUITESPARSE=ON`` to CMake to build Ceres with ``SuiteSparse``
  support.
- **Standard CMake variables replace custom Ceres variables**:

  - ``EXPORT_BUILD_DIR`` has been removed; use the standard
    ``CMAKE_EXPORT_PACKAGE_REGISTRY`` variable instead.
  - ``LIB_SUFFIX`` has been removed; use the standard ``CMAKE_INSTALL_LIBDIR``
    variable instead.
  - ``MSVC_USE_STATIC_CRT`` has been removed; use the standard
    ``CMAKE_MSVC_RUNTIME_LIBRARY`` variable instead.
  - Custom ``<DEPENDENCY>_INCLUDE_DIR_HINTS`` and
    ``<DEPENDENCY>_LIBRARY_DIR_HINTS`` variables have been removed; use
    ``CMAKE_PREFIX_PATH``, ``<PackageName>_ROOT``, or ``<PackageName>_DIR``
    instead.

Version 2.0
-----------

- When using Ceres with CMake, the target name in v2.0 is
  ``Ceres::ceres`` following modern naming conventions. The legacy
  target ``ceres`` exists for backwards compatibility, but is
  deprecated. ``CERES_INCLUDE_DIRS`` is not set any more, as the
  exported Ceres CMake target already contains the definitions of its
  public include directories which will be automatically included by
  CMake when compiling a target that links against Ceres.
- While TBB is not used any more directly by Ceres, it might still try
  to link against it, if SuiteSparseQR was found. The variable (environment
  or CMake) to customize this is ``TBB_ROOT`` (used to be ``TBBROOT``).
  For example, use ``cmake -DTBB_ROOT=/opt/intel/tbb ...`` if you want to
  link against TBB installed from Intel's binary packages on Linux.
