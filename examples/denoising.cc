// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: strandmark@google.com (Petter Strandmark)
//
// Denoising using Fields of Experts and the Ceres minimizer.
//
// Note that for good denoising results the weighting between the data term
// and the Fields of Experts term needs to be adjusted. This is discussed
// in [1]. This program assumes Gaussian noise. The noise model can be changed
// by substituing another function for QuadraticCostFunction.
//
// [1] S. Roth and M.J. Black. "Fields of Experts." International Journal of
//     Computer Vision, 82(2):205--229, 2009.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>

#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "fields_of_experts.h"
#include "pgm_image.h"

DEFINE_string(foe_file, "data/2x2.foe", "FoE file to use.");
DEFINE_string(output_filename, "denoising_output.pgm",
              "File to which the output image should be written.");
DEFINE_double(sigma, 20.0, "Standard deviation of noise.");

// This cost function is used to build the data term.
//
//   a*(x_i - b)^2
//
class QuadraticCostFunction : public ceres::SizedCostFunction<1, 1> {
 public:
  QuadraticCostFunction(double a, double b)
    : sqrta_(std::sqrt(a)), b_(b) {}
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    const double x = parameters[0][0];
    residuals[0] = sqrta_ * (x - b_);
    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = sqrta_;
    }
    return true;
  }
 private:
  double sqrta_, b_;
};

int main(int argc, char** argv) {
  using namespace ceres::examples;
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // Load the Fields of Experts filters from file.
  FieldsOfExperts foe;
  if (!foe.LoadFromFile(FLAGS_foe_file)) {
    std::cout << "Loading of \"" << FLAGS_foe_file << "\" failed.\n";
    exit(2);
  }

  // Print information about the FoE file loaded.
  std::cout << "Using a " << foe.Size() << "x" << foe.Size() << " FoE (arity "
            << foe.NumVariables() << ") with " << foe.NumFilters()
            << " filters.\n";

  // Read the images
  std::string image_filename;
  if (argc <= 1) {
    // Default image file name.
    image_filename = "data/test001_020.pgm";
  } else {
    image_filename = argv[1];
  }
  std::cout << "Reading " << image_filename << "...\n";
  PGMImage<double> image(image_filename);
  if (image.width() == 0) {
    std::cout << "Reading \"" << image_filename << "\" failed.\n";
    exit(3);
  }
  std::cout << "Image is " << image.width() << 'x' << image.height() << '\n';
  PGMImage<double> ceres_image(image.width(), image.height());

  ceres::Problem problem;

  std::cout << "Building data term...\n";
  for (unsigned ind = 0; ind < image.NumPixels(); ++ind) {
    double c = 1 / (2.0 * FLAGS_sigma * FLAGS_sigma);
    problem.AddResidualBlock(new QuadraticCostFunction(c, image.Pixel(ind)),
                             NULL,
                             &ceres_image.Pixel(ind));
  }

  // Create Ceres cost and loss functions for regularization. One is needed for
  // each filter.
  std::vector<ceres::LossFunction*> loss_function(foe.NumFilters());
  std::vector<ceres::CostFunction*> cost_function(foe.NumFilters());
  for (int alpha_index = 0; alpha_index < foe.NumFilters(); ++alpha_index) {
    loss_function[alpha_index] = foe.NewLossFunction(alpha_index);
    cost_function[alpha_index] = foe.NewCostFunction(alpha_index);
  }

  std::cout << "Building FoE term..\n";
  // Add FoE regularization for each patch in the image.
  for (unsigned x = 0; x < image.width() - (foe.Size() - 1); ++x) {
    for (unsigned y = 0; y < image.height() - (foe.Size() - 1); ++y) {
      // Build a vector with the pixel indices of this patch.
      std::vector<double*> xs;
      for (int i = 0; i < foe.NumVariables(); ++i) {
        int index = image.LinearIndex(x + foe.GetXDeltaIndices()[i],
                                      y + foe.GetYDeltaIndices()[i]);
        xs.push_back(&ceres_image.Pixel(index));
      }
      // For this patch with coordinates (x, y), we will add foe.NumFilters()
      // terms to the objective function.
      for (int alpha_index = 0;
           alpha_index < foe.NumFilters();
           ++alpha_index) {
        problem.AddResidualBlock(cost_function[alpha_index],
                                 loss_function[alpha_index],
                                 xs);
      }
    }
  }

  // These parameters may be experimented with. For example, ceres::DOGLEG tends
  // to be faster for 2x2 filters, but give solutions with slightly higher
  // objective function value.
  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = true;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.function_tolerance = 1e-3;  // Enough for denoising.

  std::cout << "Starting Ceres...\n";
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  // Make the solution stay in [0, 255].
  for (unsigned x = 0; x < image.width(); ++x) {
    for (unsigned y = 0; y < image.height(); ++y) {
      ceres_image.Pixel(x, y) =
        std::min(255.0, std::max(0.0, ceres_image.Pixel(x, y)));
    }
  }

  std::cout << "Saving image to \"" << FLAGS_output_filename << "\"...\n";
  CHECK(ceres_image.WriteToFile(FLAGS_output_filename));

  return 0;
}
