// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2022 Google Inc. All rights reserved.
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
// Author: joydeepb@cs.utexas.edu (Joydeep Biswas)

#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA

#include "cuda_runtime.h"

namespace ceres::internal {

void CudaFP64ToFP32(const double* input,
                    float* output,
                    const int size,
                    cudaStream_t stream);

void CudaFP32ToFP64(const float* input,
                    double* output,
                    const int size,
                    cudaStream_t stream);

// Set all elements of the array to the FP32 value 0.
void CudaSetZeroFP32(float* output, const int size, cudaStream_t stream);

// Set all elements of the array to the FP64 value 0.
void CudaSetZeroFP64(double* output, const int size, cudaStream_t stream);

// Compute x = x + double(y).
void CudaXPlusEqualsY(double* x, float* y, const int size, cudaStream_t stream);
}  // namespace ceres::internal

#endif  // CERES_NO_CUDA