// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/residual_block_utils.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include "ceres/array_utils.h"
#include "ceres/fpclassify.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/port.h"
#include "ceres/parameter_block.h"
#include "ceres/residual_block.h"
#include "ceres/stringprintf.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

using std::string;

void InvalidateEvaluation(const ResidualBlock& block,
                          double* cost,
                          double* residuals,
                          double** jacobians) {
  const int num_parameter_blocks = block.NumParameterBlocks();
  const int num_residuals = block.NumResiduals();

  InvalidateArray(1, cost);
  InvalidateArray(num_residuals, residuals);
  if (jacobians != NULL) {
    for (int i = 0; i < num_parameter_blocks; ++i) {
      const int parameter_block_size = block.parameter_blocks()[i]->Size();
      InvalidateArray(num_residuals * parameter_block_size, jacobians[i]);
    }
  }
}

static const char* UserSuppliedNumberCommentary(double x) {
  if (!IsFinite(x)) {
    return "ERROR: Value is not finite";
  }
  if (x == kImpossibleValue) {
    return "ERROR: Value was not set by cost function";
  }
  return "OK";
}

#define MAX_DISPLAY_VALUES 30
#define MAX_DISPLAY_JACOBIANS 30

string EvaluationErrorReportString(const ResidualBlock& block,
                                   double const* const* parameters,
                                   double* cost,
                                   double* residuals,
                                   double** jacobians) {
  CHECK_NOTNULL(cost);
  CHECK_NOTNULL(residuals);
  const int num_parameter_blocks = block.NumParameterBlocks();
  const int num_residuals = block.NumResiduals();

  // Determine if there are unfilled or non-finite residuals.
  bool unfilled_residuals;
  bool nonfinite_residuals;
  IsArrayValid(num_residuals,
               residuals,
               &unfilled_residuals,
               &nonfinite_residuals);

  // Determine if there are unfilled or non-finite jacobians.
  bool unfilled_jacobians = false;
  bool nonfinite_jacobians = false;
  for (int i = 0; i < num_parameter_blocks; ++i) {
    if (jacobians[i] == NULL) {
      continue;
    }
    bool block_has_unfilled_jacobians = false;
    bool block_has_nonfinite_jacobians = false;
    IsArrayValid(num_residuals * block.parameter_blocks()[i]->Size(),
                 jacobians[i],
                 &block_has_unfilled_jacobians,
                 &block_has_nonfinite_jacobians);
    unfilled_jacobians = unfilled_jacobians || block_has_unfilled_jacobians;
    nonfinite_jacobians = nonfinite_jacobians || block_has_nonfinite_jacobians;
    if (unfilled_jacobians && nonfinite_jacobians) {
      break;
    }
  }

  // Explain which classes of errors were detected (unfilled or non-finite).
  string result =
      "\nERROR: Ceres found a bug in your user-supplied cost function:\n"
      "\n";
  if (unfilled_residuals) {
    result += "- You did not fill in all the residuals returned from your cost function\n";  // NOLINT
  }
  if (unfilled_jacobians) {
    result += "- You did not fill in all the jacobians returned from your cost function\n";  // NOLINT
  }
  if (nonfinite_residuals) {
    result += "- You produced inf/nan values in the residuals returned from your cost function\n";  // NOLINT
  }
  if (nonfinite_jacobians) {
    result += "- You produced inf/nan values in the jacobians returned from your cost function\n";  // NOLINT
  }

  result +=
      "\n"
      "User-supplied cost funtions must do the following:\n"
      "\n"
      "  (1) Fill in all residual elements with finite values\n"
      "  (2) If jacobians == NULL, you are done (Ceres is not asking for jacobians)\n"      // NOLINT
      "  (3) Otherwise, Ceres is asking for the jacobians of at least some parameters\n"    // NOLINT
      "  (4) For each jacobians[i] != NULL (if jacobians[i] == NULL, it's constant)\n"      // NOLINT
      "  (5)   Fill in all jacobian values for parameter i with finite values\n"            // NOLINT
      "\n"
      "If you are seeing this error, your cost function is either producing non-finite\n"   // NOLINT
      "values (infs or NaNs) or is not filling in all the values. Ceres pre-fills\n"        // NOLINT
      "arrays with a sentinel value (kImpossibleValue in the Ceres source) to detect\n"     // NOLINT
      "when you have not filled in all the values in either the residuals or jacobians.\n"  // NOLINT
      "\n"                                                                                  // NOLINT
      "If you are using Ceres' autodiff implementation, then it is likely either (a)\n"     // NOLINT
      "residual values are causing the problems or (b) some part of the autodiff\n"         // NOLINT
      "evaluation has bad numeric behaviour. Take a look at ceres/rotation.h for\n"         // NOLINT
      "example code showing special case handling of functions in autodiff.\n"              // NOLINT
      "\n"                                                                                  // NOLINT
      "Which residual block is this? For architecture reasons at this point Ceres\n"        // NOLINT
      "cannot easily identify the block but here is the block's size information:\n"        // NOLINT
      "\n";                                                                                 // NOLINT

  // (2) Show the residual block sizing details; this is needed since at the
  // point that this is evaluated the information needed to pinpoint which
  // residual this is in the overall program is not available, so the user will
  // have to figure that out based on the sizes.
  StringAppendF(&result,
                "  Input: %d parameter blocks; sizes: (",
                num_parameter_blocks);
  for (int i = 0; i < num_parameter_blocks; ++i) {
    StringAppendF(&result, "%d", block.parameter_blocks()[i]->Size());
    if (i != num_parameter_blocks - 1) {
      StringAppendF(&result, ", ");
    }
  }
  result += ")\n";
  StringAppendF(&result, "  Output: %d residuals\n", num_residuals);
  result += "\n";

  // Display residual problems (if any).
  int num_bad_residuals = 0;
  if (!IsArrayValid(num_residuals, residuals)) {
    result += "Problem exists in: User-returned residual values (r[N])\n"
              "\n";
    for (int i = 0; i < num_residuals; ++i) {
      if (!IsValueValid(residuals[i]) ||
          // Only print out the full residuals if there aren't too many values.
          num_residuals < MAX_DISPLAY_VALUES) {
        StringAppendF(&result,
                      "  r[%2d] = %-15.4e     %s\n",
                      i,
                      residuals[i],
                      UserSuppliedNumberCommentary(residuals[i]));
        num_bad_residuals++;
      }
      if (num_bad_residuals > MAX_DISPLAY_VALUES) {
        StringAppendF(&result,
                      "  ... too many bad residuals; skipping the rest\n");
        break;
      }
    }
    result += "\n";
  }

  // Display jacobian problems (if any).
  if (unfilled_jacobians || nonfinite_jacobians) {
    int num_bad_jacobians = 0;
    result +=
        "Problem exists in: User-returned jacobian values d r[N] / d p[M][Q]\n"   // NOLINT
        "where  r[N] is residual element N\n"
        "       p[M][Q] is parameter M, element Q\n"
        "       d r[N] / d p[M][Q] is the derivative of the residual w.r.t. the parameter"  // NOLINT
        "\n";
    for (int i = 0; i < num_parameter_blocks; ++i) {
      // Skip over jacobians that are OK.
      const int parameter_block_size = block.parameter_blocks()[i]->Size();
      if (jacobians[i] != NULL &&
          IsArrayValid(parameter_block_size * num_residuals, residuals)) {
        continue;
      }
      StringAppendF(
          &result,
          "\n  Jacobian values for parameter block %d (p[%d][...]):\n", i, i);

      int num_jacobian_values = num_residuals * parameter_block_size;
      int num_bad_jacobian_values = 0;
      for (int n = 0; n < num_residuals; ++n) {
        for (int q = 0; q < parameter_block_size; ++q) {
          double drdp = jacobians[i][n * parameter_block_size + q];
          if (!IsValueValid(drdp) ||
              // Print out the full jacobians if there aren't too many values.
              num_jacobian_values < MAX_DISPLAY_VALUES) {
            StringAppendF(&result,
                          "  d r[%2d] / d p[%2d][%2d] = %-15.4e     %s\n",
                          n,
                          i,
                          q,
                          drdp,
                          UserSuppliedNumberCommentary(drdp));
            num_bad_jacobian_values++;
          }
          if (num_bad_jacobian_values > MAX_DISPLAY_VALUES) {
            StringAppendF(&result,
                "  ... too many bad jacobian values in this "
                "block; skipping the rest\n");
            break;
          }
        }
      }
      num_bad_jacobians++;
      if (num_bad_jacobians > MAX_DISPLAY_JACOBIANS) {
        StringAppendF(&result,
                      "  ... too many bad jacobians; skipping the rest\n");
      }
    }
  }
  return result + "\n";
}

bool IsEvaluationValid(const ResidualBlock& block,
                       double const* const* parameters,
                       double* cost,
                       double* residuals,
                       double** jacobians) {
  const int num_parameter_blocks = block.NumParameterBlocks();
  const int num_residuals = block.NumResiduals();

  if (!IsArrayValid(num_residuals, residuals)) {
    return false;
  }

  if (jacobians != NULL) {
    for (int i = 0; i < num_parameter_blocks; ++i) {
      const int parameter_block_size = block.parameter_blocks()[i]->Size();
      if (!IsArrayValid(num_residuals * parameter_block_size, jacobians[i])) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace internal
}  // namespace ceres
