# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2018 Google Inc. All rights reserved.
# http://ceres-solver.org/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Google Inc. nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Authors: mierle@gmail.com (Keir Mierle)
#
# Bazel workspace file to enable building Ceres with Bazel.

workspace(name = 'com_google_ceres_solver')

# External dependency: Google Flags; has Bazel build already.
http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "6e16c8bc91b1310a44f3965e616383dbda48f83e8c1eaa2370a215057b00cabe",
    strip_prefix = "gflags-77592648e3f3be87d6c7123eb81cbad75f9aef5a",
    urls = [
        "https://mirror.bazel.build/github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
        "https://github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
    ],
)

# External dependency: Google Log; has Bazel build already.
http_archive(
    name = "com_github_google_glog",
    sha256 = "1ee310e5d0a19b9d584a855000434bb724aa744745d5b8ab1855c85bff8a8e21",
    strip_prefix = "glog-028d37889a1e80e8a07da1b8945ac706259e5fd8",
    urls = [
        "https://github.com/google/glog/archive/028d37889a1e80e8a07da1b8945ac706259e5fd8.tar.gz",
    ],
)

# External dependency: Eigen; has no Bazel build.
new_http_archive(
    name   = 'com_github_eigen_eigen',
    sha256 = "dd254beb0bafc695d0f62ae1a222ff85b52dbaa3a16f76e781dce22d0d20a4a6",
    strip_prefix = "eigen-eigen-5a0156e40feb",
    urls = [
        "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2"
    ],
    build_file = "bazel/eigen.BUILD"
)

# External dependency: SuiteSparse; has no Bazel build.
# SuiteSparse has many subcomponents, including CXSparse, CHOLMOD, etc.
new_http_archive(
    name   = "edu_texasanm_suitesparse",
    sha256 = "4ec8d344bd8e95b898132ddffd7ee93bfbb2c1224925d11bab844b08f9b4c3b7",
    strip_prefix = "SuiteSparse",
    urls = [
        "http://faculty.cse.tamu.edu/davis/SuiteSparse/SuiteSparse-5.1.2.tar.gz"
    ],
    build_file = "bazel/suitesparse.BUILD"
)

# External dependency: CLAPACK; has no Bazel build.
new_http_archive(
    name   = "org_netlib_clapack",
    sha256 = "6dc4c382164beec8aaed8fd2acc36ad24232c406eda6db462bd4c41d5e455fac",
    strip_prefix = "CLAPACK-3.2.1",
    urls = [
        "http://www.netlib.org/clapack/clapack.tgz"
    ],
    build_file = "bazel/lapack.BUILD"
)

new_http_archive(
    name   = "org_netlib_libf2c",
    sha256 = "ca404070e9ce0a9aaa6a71fc7d5489d014ade952c5d6de7efb88de8e24f2e8e0",
    urls = [
        "http://www.netlib.org/f2c/libf2c.zip"
    ],
    build_file = "bazel/libf2c.BUILD"
)

new_http_archive(
    name   = "org_netlib_2c",
    sha256 = "7054ff0f6b3718586911521a368fa89976c33d18add2403d811446ec73a16d2e",
    strip_prefix = "src",
    urls = [
        "http://www.netlib.org/f2c/src.tgz",
    ],
    build_file = "bazel/f2c.BUILD"
)

