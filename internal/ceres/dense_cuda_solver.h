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
//
// A C++ interface to dense CUDA solvers.

#ifndef INTERNAL_CERES_DENSE_CUDA_SOLVER_H_
#define INTERNAL_CERES_DENSE_CUDA_SOLVER_H_

#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA

#include <string>

#include "ceres/cuda_buffer.h"
#include "ceres/execution_summary.h"
#include "ceres/linear_solver.h"

#include "cusolverDn.h"

namespace ceres {
namespace internal {

// This class abstracts away low-level details of initialization, memory
// management, and error handling of CUDA solvers, providing a simple API for
// calling cuSOLVER's Cholesky and QR implementations on dense matrices.
class DenseCudaSolver {
 public:
  DenseCudaSolver();
  ~DenseCudaSolver();

  // Perform Cholesky factorization of a symmetric matrix A.
  LinearSolverTerminationType CholeskyFactorize(int num_cols,
                                                double* A,
                                                std::string* message);

  // Solve the linear system A * X = B, using the previously computed Cholesky
  // factorization of A. The user must ensure that CholeskyFactorize() has been
  // called before calling this method. The number of rows in B and X must be
  // equal to the number of columns (and rows) in A.
  LinearSolverTerminationType CholeskySolve(const double* B,
                                            double* X,
                                            std::string* message);

 private:
  // Handle to the cuSOLVER context.
  cusolverDnHandle_t cusolver_handle_;
  // CUDA device stream.
  cudaStream_t stream_;
  // Number of rows in the A matrix, to be cached between calls to *Factorize
  // and *Solve.
  size_t num_rows_;
  // Number of columns in the A matrix, to be cached between calls to *Factorize
  // and *Solve.
  size_t num_cols_;
  // GPU memory allocated for the A matrix (lhs matrix).
  CudaBuffer<double> gpu_a_;
  // GPU memory allocated for the B matrix (rhs vector).
  CudaBuffer<double> gpu_b_;
  // GPU memory allocated for the X matrix (solution vector).
  CudaBuffer<double> gpu_tau_;
  // Scratch space for cuSOLVER on the GPU.
  CudaBuffer<uint8_t> gpu_scratch_;
  // Scratch space for cuSOLVER on the host.
  double* host_scratch_;
  // Size of the host scratch space.
  size_t host_scratch_size_;
  // Required for error handling with cuSOLVER.
  int* gpu_error_;
};


}  // namespace internal
}  // namespace ceres

#endif  // CERES_NO_CUDA

#endif  // INTERNAL_CERES_DENSE_CUDA_SOLVER_H_
