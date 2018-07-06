#!/bin/bash

# Stop processing on any error.
set -e

# Install default versions of standard dependencies that are new enough in 14.04.
sudo apt-get install -y cmake
sudo apt-get install -y libatlas-base-dev libsuitesparse-dev
sudo apt-get install -y libgoogle-glog-dev libgflags-dev

# Install Eigen 3.3.4 as the default 14.04 version is 3.2.0 in which the sparse solvers
# have known poor performance.
wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz -O /tmp/eigen-3.3.4.tar.gz
tar -C /tmp -xzf /tmp/eigen-3.3.4.tar.gz
rm /tmp/eigen-3.3.4.tar.gz
# Fix detection of BLAS/LAPACK broken in 3.3.4 release.
wget https://bitbucket.org/eigen/eigen/commits/dbab66d00651bf050d1426334a39b627abe7216e/raw -O /tmp/eigen-3.3.4.fortran.patch
cd /tmp/eigen-eigen-5a0156e40feb && patch -p1 < /tmp/eigen-3.3.4.fortran.patch
mkdir /tmp/eigen-3.3.4-build
cd /tmp/eigen-3.3.4-build
cmake /tmp/eigen-eigen-5a0156e40feb && make && sudo make install
