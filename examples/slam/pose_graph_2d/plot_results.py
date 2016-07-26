#!/usr/bin/python
#
# Plots the results from the 2D pose graph optimization. It will draw a line
# between consecutive vertices.  The commandline expects two optional filenames:
#
#   ./plot_results.py --poses_original_filename optional \
#       --poses_optimized_filename optional
#
# The files have the following format:
#   ID x y yaw_radians

import matplotlib.pyplot as plot
import numpy
import sys
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--poses_original_filename", dest="poses_original_filename",
                  default="", help="The filename that contains the original poses.")
parser.add_option("--poses_optimized_filename", dest="poses_optimized_filename",
                  default="", help="The filename that contains the optimized poses.")
(options, args) = parser.parse_args()

# Read the original and optimized poses files.
poses_original = None
if options.poses_original_filename != '':
  poses_original = numpy.genfromtxt(options.poses_original_filename,
                                    usecols = (1, 2))

poses_optimized = None
if options.poses_optimized_filename != '':
  poses_optimized = numpy.genfromtxt(options.poses_optimized_filename,
                                     usecols = (1, 2))

# Plots the results for the specified poses.
plot.figure()
if poses_original is not None:
  plot.plot(poses_original[:, 0], poses_original[:, 1], '-', label="Original",
            alpha=0.5, color="green")

if poses_optimized is not None:
  plot.plot(poses_optimized[:, 0], poses_optimized[:, 1], '-', label="Optimized",
            alpha=0.5, color="blue")

plot.axis('equal')
plot.legend()
# Show the plot and wait for the user to close.
plot.show()
