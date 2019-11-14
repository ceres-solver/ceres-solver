#!/bin/bash

# Stop processing on any error.
set -e

# Install default versions of standard dependencies that are new enough in 18.04
sudo apt-get install -y cmake
sudo apt-get install -y libatlas-base-dev libsuitesparse-dev
sudo apt-get install -y libgoogle-glog-dev libgflags-dev
sudo apt-get install -y libeigen3-dev
