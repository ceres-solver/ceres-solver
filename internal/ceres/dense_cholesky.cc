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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/dense_cholesky.h"

#include <algorithm>
#include <memory>
#include <string>

#ifndef CERES_NO_LAPACK

// C interface to the LAPACK Cholesky factorization and triangular solve.
extern "C" void dpotrf_(
    const char* uplo, const int* n, double* a, const int* lda, int* info);

extern "C" void dpotrs_(const char* uplo,
                        const int* n,
                        const int* nrhs,
                        const double* a,
                        const int* lda,
                        double* b,
                        const int* ldb,
                        int* info);
#endif

namespace ceres {
namespace internal {

std::unique_ptr<DenseCholesky> DenseCholesky::Create(
    const LinearSolver::Options& options) {
  std::unique_ptr<DenseCholesky> dense_cholesky;

  switch (options.dense_linear_algebra_library_type) {
    case EIGEN:
      dense_cholesky = std::make_unique<EigenDenseCholesky>();
      break;

    case LAPACK:
#ifndef CERES_NO_LAPACK
      dense_cholesky = std::make_unique<LAPACKDenseCholesky>();
      break;
#else
      LOG(FATAL) << "Ceres was compiled without support for LAPACK.";
#endif

    case CUDA:
#ifndef CERES_NO_CUDA
    dense_cholesky = std::make_unique<CUDADenseCholesky>();
    break;
#else
    LOG(FATAL) << "Ceres was compiled without support for CUDA.";
#endif

    default:
      LOG(FATAL) << "Unknown dense linear algebra library type : "
                 << DenseLinearAlgebraLibraryTypeToString(
                        options.dense_linear_algebra_library_type);
  }
  return dense_cholesky;
}

LinearSolverTerminationType DenseCholesky::FactorAndSolve(
    int num_cols,
    double* lhs,
    const double* rhs,
    double* solution,
    std::string* message) {
  LinearSolverTerminationType termination_type =
      Factorize(num_cols, lhs, message);
  if (termination_type == LINEAR_SOLVER_SUCCESS) {
    termination_type = Solve(rhs, solution, message);
  }
  return termination_type;
}

LinearSolverTerminationType EigenDenseCholesky::Factorize(
    int num_cols, double* lhs, std::string* message) {
  Eigen::Map<Eigen::MatrixXd> m(lhs, num_cols, num_cols);
  llt_ = std::make_unique<LLTType>(m);
  if (llt_->info() != Eigen::Success) {
    *message = "Eigen failure. Unable to perform dense Cholesky factorization.";
    return LINEAR_SOLVER_FAILURE;
  }

  *message = "Success.";
  return LINEAR_SOLVER_SUCCESS;
}

LinearSolverTerminationType EigenDenseCholesky::Solve(const double* rhs,
                                                      double* solution,
                                                      std::string* message) {
  if (llt_->info() != Eigen::Success) {
    *message = "Eigen failure. Unable to perform dense Cholesky factorization.";
    return LINEAR_SOLVER_FAILURE;
  }

  VectorRef(solution, llt_->cols()) =
      llt_->solve(ConstVectorRef(rhs, llt_->cols()));
  *message = "Success.";
  return LINEAR_SOLVER_SUCCESS;
}

#ifndef CERES_NO_LAPACK
LinearSolverTerminationType LAPACKDenseCholesky::Factorize(
    int num_cols, double* lhs, std::string* message) {
  lhs_ = lhs;
  num_cols_ = num_cols;

  const char uplo = 'L';
  int info = 0;
  dpotrf_(&uplo, &num_cols_, lhs_, &num_cols_, &info);

  if (info < 0) {
    termination_type_ = LINEAR_SOLVER_FATAL_ERROR;
    LOG(FATAL) << "Congratulations, you found a bug in Ceres."
               << "Please report it."
               << "LAPACK::dpotrf fatal error."
               << "Argument: " << -info << " is invalid.";
  } else if (info > 0) {
    termination_type_ = LINEAR_SOLVER_FAILURE;
    *message = StringPrintf(
        "LAPACK::dpotrf numerical failure. "
        "The leading minor of order %d is not positive definite.",
        info);
  } else {
    termination_type_ = LINEAR_SOLVER_SUCCESS;
    *message = "Success.";
  }
  return termination_type_;
}

LinearSolverTerminationType LAPACKDenseCholesky::Solve(const double* rhs,
                                                       double* solution,
                                                       std::string* message) {
  const char uplo = 'L';
  const int nrhs = 1;
  int info = 0;

  std::copy_n(rhs, num_cols_, solution);
  dpotrs_(
      &uplo, &num_cols_, &nrhs, lhs_, &num_cols_, solution, &num_cols_, &info);

  if (info < 0) {
    termination_type_ = LINEAR_SOLVER_FATAL_ERROR;
    LOG(FATAL) << "Congratulations, you found a bug in Ceres."
               << "Please report it."
               << "LAPACK::dpotrs fatal error."
               << "Argument: " << -info << " is invalid.";
  }

  *message = "Success";
  termination_type_ = LINEAR_SOLVER_SUCCESS;

  return termination_type_;
}

#endif  // CERES_NO_LAPACK

#ifndef CERES_NO_CUDA

CUDADenseCholeskyOld::CUDADenseCholeskyOld() :
    cusolver_handle_(nullptr),
    stream_(nullptr),
    num_cols_(0),
    gpu_error_(nullptr) {
  CHECK_EQ(cusolverDnCreate(&cusolver_handle_), CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  CHECK_EQ(cusolverDnSetStream(cusolver_handle_, stream_),
      CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaMalloc(&gpu_error_, sizeof(int)), cudaSuccess);
}

CUDADenseCholeskyOld::~CUDADenseCholeskyOld() {
  CHECK_EQ(cudaFree(gpu_error_), cudaSuccess);
  CHECK_EQ(cusolverDnDestroy(cusolver_handle_), CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaStreamDestroy(stream_), cudaSuccess);
}

LinearSolverTerminationType CUDADenseCholeskyOld::Factorize(
    int num_cols,
    double* lhs,
    std::string* message) {
  // Allocate GPU memory if necessary.
  gpu_a_.Reserve(num_cols * num_cols);
  num_cols_ = num_cols;
  // Copy A to GPU.
  gpu_a_.CopyToGpu(lhs, num_cols * num_cols);
  // Allocate scratch space on GPU.
  int device_scratch_size = 0;
  CHECK_EQ(cusolverDnDpotrf_bufferSize(cusolver_handle_,
                                      CUBLAS_FILL_MODE_LOWER,
                                      num_cols,
                                      gpu_a_.data(),
                                      num_cols,
                                      &device_scratch_size),
          CUSOLVER_STATUS_SUCCESS);
  // ALlocate GPU scratch memory.
  gpu_scratch_.Reserve(device_scratch_size);
  // Run the actual factorization (potrf)
  CHECK_EQ(cusolverDnDpotrf(cusolver_handle_,
                            CUBLAS_FILL_MODE_LOWER,
                            num_cols,
                            gpu_a_.data(),
                            num_cols,
                            reinterpret_cast<double*>(gpu_scratch_.data()),
                            gpu_scratch_.size(),
                            gpu_error_),
          CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  int error = 0;
  // Check for errors.
  CHECK_EQ(cudaMemcpy(&error,
                      gpu_error_,
                      sizeof(int),
                      cudaMemcpyDeviceToHost),
          cudaSuccess);
  if (error != 0) {
    LOG(FATAL) << "Congratulations, you found a bug in Ceres."
               << "Please report it."
               << "cuSolverDN::cusolverDnDpotrf fatal error."
               << "Argument: " << -error << " is invalid.";
  }
  *message = "Success";
  return LinearSolverTerminationType::LINEAR_SOLVER_SUCCESS;
}

LinearSolverTerminationType CUDADenseCholeskyOld::Solve(
    const double* rhs,
    double* solution,
    std::string* message) {
  // Copy RHS to GPU.
  gpu_b_.CopyToGpu(rhs, num_cols_);
  // Solve the system.
  CHECK_EQ(cusolverDnDpotrs(cusolver_handle_,
                            CUBLAS_FILL_MODE_LOWER,
                            num_cols_,
                            1,
                            gpu_a_.data(),
                            num_cols_,
                            gpu_b_.data(),
                            num_cols_,
                            gpu_error_),
          CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  // Check for errors.
  int error = 0;
  // Copy error variable from GPU to host.
  CHECK_EQ(cudaMemcpy(&error,
                      gpu_error_,
                      sizeof(int),
                      cudaMemcpyDeviceToHost),
          cudaSuccess);
  // Copy X from GPU to host.
  gpu_b_.CopyToHost(solution, num_cols_);
  if (error != 0) {
    LOG(FATAL) << "Congratulations, you found a bug in Ceres."
               << "Please report it."
               << "cuSolverDN::cusolverDnDpotrs fatal error."
               << "Argument: " << -error << " is invalid.";
  }
  *message = "Success";
  return LinearSolverTerminationType::LINEAR_SOLVER_SUCCESS;
}

#ifdef CERES_CUDA_VERSION_LT_11_1
// CUDA < 11.1 did not have the 64-bit APIs, so the implementation of the new
// interface will just generate a fatal error.

CUDADenseCholeskyNew::CUDADenseCholeskyNew() {
  LOG(FATAL) << "Cannot use CUDADenseCholeskyNew with CUDA < 11.1.";
}

CUDADenseCholeskyNew::~CUDADenseCholeskyNew() {}

LinearSolverTerminationType CUDADenseCholeskyNew::Factorize(
    int,
    double*,
    std::string*) {
  // This will never run, since the constructor will always generate a fatal
  // error. Just including a return statement to avoid strict compiler errors.
  return LinearSolverTerminationType::LINEAR_SOLVER_FATAL_ERROR;
}

LinearSolverTerminationType CUDADenseCholeskyNew::Solve(
    const double*,
    double*,
    std::string*) {
  // This will never run, since the constructor will always generate a fatal
  // error. Just including a return statement to avoid strict compiler errors.
  return LinearSolverTerminationType::LINEAR_SOLVER_FATAL_ERROR;
}

#else  // CERES_CUDA_VERSION_LT_11_1

CUDADenseCholeskyNew::CUDADenseCholeskyNew() :
    cusolver_handle_(nullptr),
    stream_(nullptr),
    num_cols_(0),
    host_scratch_(nullptr),
    host_scratch_size_(0),
    gpu_error_(nullptr) {
  CHECK_EQ(cusolverDnCreate(&cusolver_handle_), CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  CHECK_EQ(cusolverDnSetStream(cusolver_handle_, stream_),
      CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaMalloc(&gpu_error_, sizeof(int)), cudaSuccess);
}

CUDADenseCholeskyNew::~CUDADenseCholeskyNew() {
  CHECK_EQ(cudaFree(gpu_error_), cudaSuccess);
  if (host_scratch_) {
    free(host_scratch_);
  }
  CHECK_EQ(cusolverDnDestroy(cusolver_handle_), CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaStreamDestroy(stream_), cudaSuccess);
}

LinearSolverTerminationType CUDADenseCholeskyNew::Factorize(
    int num_cols,
    double* lhs,
    std::string* message) {
  // Allocate GPU memory if necessary.
  gpu_a_.Reserve(num_cols * num_cols);
  num_cols_ = num_cols;
  // Copy A to GPU.
  gpu_a_.CopyToGpu(lhs, num_cols * num_cols);
  // Allocate scratch space on GPU.
  size_t host_scratch_size = 0;
  size_t device_scratch_size = 0;
  CHECK_EQ(cusolverDnXpotrf_bufferSize(cusolver_handle_,
                                      nullptr,
                                      CUBLAS_FILL_MODE_LOWER,
                                      num_cols,
                                      CUDA_R_64F,
                                      gpu_a_.data(),
                                      num_cols,
                                      CUDA_R_64F,
                                      &device_scratch_size,
                                      &host_scratch_size),
          CUSOLVER_STATUS_SUCCESS);
  // Allocate host scratch memory.
  if (host_scratch_size > host_scratch_size_) {
    CHECK(realloc(reinterpret_cast<void**>(&host_scratch_),
                  host_scratch_size) != nullptr);
    host_scratch_size_ = host_scratch_size;
  }
  // ALlocate GPU scratch memory.
  gpu_scratch_.Reserve(device_scratch_size);
  // Run the actual factorization (potrf)
  CHECK_EQ(cusolverDnXpotrf(cusolver_handle_,
                            nullptr,
                            CUBLAS_FILL_MODE_LOWER,
                            num_cols,
                            CUDA_R_64F,
                            gpu_a_.data(),
                            num_cols,
                            CUDA_R_64F,
                            gpu_scratch_.data(),
                            gpu_scratch_.size(),
                            host_scratch_,
                            host_scratch_size_,
                            gpu_error_),
          CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  int error = 0;
  // Check for errors.
  CHECK_EQ(cudaMemcpy(&error,
                      gpu_error_,
                      sizeof(int),
                      cudaMemcpyDeviceToHost),
          cudaSuccess);
  if (error != 0) {
    LOG(FATAL) << "Congratulations, you found a bug in Ceres."
               << "Please report it."
               << "cuSolverDN::cusolverDnXpotrf fatal error."
               << "Argument: " << -error << " is invalid.";
  }
  *message = "Success";
  return LinearSolverTerminationType::LINEAR_SOLVER_SUCCESS;
}

LinearSolverTerminationType CUDADenseCholeskyNew::Solve(
    const double* rhs,
    double* solution,
    std::string* message) {
  // Copy RHS to GPU.
  gpu_b_.CopyToGpu(rhs, num_cols_);
  // Solve the system.
  CHECK_EQ(cusolverDnXpotrs(cusolver_handle_,
                            nullptr,
                            CUBLAS_FILL_MODE_LOWER,
                            num_cols_,
                            1,
                            CUDA_R_64F,
                            gpu_a_.data(),
                            num_cols_,
                            CUDA_R_64F,
                            gpu_b_.data(),
                            num_cols_,
                            gpu_error_),
          CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  // Check for errors.
  int error = 0;
  // Copy error variable from GPU to host.
  CHECK_EQ(cudaMemcpy(&error,
                      gpu_error_,
                      sizeof(int),
                      cudaMemcpyDeviceToHost),
          cudaSuccess);
  // Copy X from GPU to host.
  gpu_b_.CopyToHost(solution, num_cols_);
  if (error != 0) {
    LOG(FATAL) << "Congratulations, you found a bug in Ceres."
               << "Please report it."
               << "cuSolverDN::cusolverDnXpotrs fatal error."
               << "Argument: " << -error << " is invalid.";
  }
  *message = "Success";
  return LinearSolverTerminationType::LINEAR_SOLVER_SUCCESS;
}

#endif // CERES_CUDA_VERSION_LT_11_1

#endif  // CERES_NO_CUDA

}  // namespace internal
}  // namespace ceres
