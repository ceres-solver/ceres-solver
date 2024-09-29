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
# Author: mierle@gmail.com (Keir Mierle)



EXAMPLE_DEPS = [
    "//:ceres",
    "@eigen//:eigen",
    "@abseil-cpp//absl/log",
    "@abseil-cpp//absl/flags:parse",
    "@abseil-cpp//absl/flags:flag",
    "@abseil-cpp//absl/time",
    "@abseil-cpp//absl/log:check",
    "@abseil-cpp//absl/log:initialize",
]

cc_binary(
    name = "bundle_adjuster",
    srcs = [
        "bal_problem.cc",
        "bal_problem.h",
        "bundle_adjuster.cc",
        "snavely_reprojection_error.h",
    ],
    deps = EXAMPLE_DEPS,
)

cc_binary(
    name = "denoising",
    srcs = [
        "denoising.cc",
        "fields_of_experts.cc",
        "fields_of_experts.h",
        "pgm_image.h",
    ],
    deps = EXAMPLE_DEPS,
)

cc_binary(
    name = "robot_pose_mle",
    srcs = [
        "robot_pose_mle.cc",
    ],
    deps = EXAMPLE_DEPS,
)

cc_binary(
    name = "pose_graph_2d",
    srcs = [
        "slam/common/read_g2o.h",
        "slam/pose_graph_2d/angle_manifold.h",
        "slam/pose_graph_2d/normalize_angle.h",
        "slam/pose_graph_2d/pose_graph_2d.cc",
        "slam/pose_graph_2d/pose_graph_2d_error_term.h",
        "slam/pose_graph_2d/types.h",
    ],
    includes = ["slam"],
    deps = EXAMPLE_DEPS,
)

cc_binary(
    name = "pose_graph_3d",
    srcs = [
        "slam/common/read_g2o.h",
        "slam/pose_graph_3d/pose_graph_3d.cc",
        "slam/pose_graph_3d/pose_graph_3d_error_term.h",
        "slam/pose_graph_3d/types.h",
    ],
    includes = ["slam"],
    deps = EXAMPLE_DEPS,
)

[cc_binary(
    name = example,
    srcs = [example + ".cc"],
    deps = EXAMPLE_DEPS,
) for example in [
    "circle_fit",
    "curve_fitting",
    "ellipse_approximation",
    "helloworld",
    "helloworld_analytic_diff",
    "helloworld_numeric_diff",
    "libmv_bundle_adjuster",
    "libmv_homography",
    "more_garbow_hillstrom",
    "nist",
    "powell",
    "robust_curve_fitting",
    "rosenbrock",
    "sampled_function/sampled_function",
    "simple_bundle_adjuster",
]]
