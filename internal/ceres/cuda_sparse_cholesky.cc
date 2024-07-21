// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2024 Google Inc. All rights reserved.
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
// Author: markshachkov@gmail.com (Mark Shachkov)

#include "ceres/cuda_sparse_cholesky.h"

#ifndef CERES_NO_CUDSS

#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

#include "Eigen/Core"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/cuda_buffer.h"
#include "ceres/linear_solver.h"
#include "cudss.h"

namespace ceres::internal {

inline std::string cuDSSStatusToString(cudssStatus_t status) {
  switch (status) {
    case CUDSS_STATUS_SUCCESS:
      return "CUDSS_STATUS_SUCCESS";
    case CUDSS_STATUS_NOT_INITIALIZED:
      return "CUDSS_STATUS_NOT_INITIALIZED";
    case CUDSS_STATUS_ALLOC_FAILED:
      return "CUDSS_STATUS_ALLOC_FAILED";
    case CUDSS_STATUS_INVALID_VALUE:
      return "CUDSS_STATUS_INVALID_VALUE";
    case CUDSS_STATUS_NOT_SUPPORTED:
      return "CUDSS_STATUS_NOT_SUPPORTED";
    case CUDSS_STATUS_EXECUTION_FAILED:
      return "CUDSS_STATUS_EXECUTION_FAILED";
    case CUDSS_STATUS_INTERNAL_ERROR:
      return "CUDSS_STATUS_INTERNAL_ERROR";
    default:
      return "unknown cuDSS status: " + std::to_string(status);
  }
}

#define CUDSS_STATUS_CHECK(IN)                                     \
  if (cudssStatus_t status = IN; status != CUDSS_STATUS_SUCCESS) { \
    CHECK(false) << "Got error: " << cuDSSStatusToString(status);  \
  }

#define CUDSS_STATUS_OK_OR_RETURN_FATAL_ERROR(IN, additional_message)    \
  if (cudssStatus_t status = IN; status != CUDSS_STATUS_SUCCESS) {       \
    *message = std::string(additional_message) +                         \
               " Got error: " + cuDSSStatusToString(status);             \
    return factorize_result_ = LinearSolverTerminationType::FATAL_ERROR; \
  }

#define CUDSS_STATUS_OK_OR_RETURN_CUDSS_STATUS(IN)                 \
  if (cudssStatus_t status = IN; status != CUDSS_STATUS_SUCCESS) { \
    return status;                                                 \
  }

class CERES_NO_EXPORT CuDSSMatrixBase {
 public:
  CuDSSMatrixBase() = default;
  CuDSSMatrixBase(const CuDSSMatrixBase&) = delete;
  CuDSSMatrixBase(CuDSSMatrixBase&&) = delete;
  CuDSSMatrixBase& operator=(const CuDSSMatrixBase&) = delete;
  CuDSSMatrixBase& operator=(CuDSSMatrixBase&&) = delete;
  ~CuDSSMatrixBase() { CUDSS_STATUS_CHECK(Free()); }

  cudssStatus_t Free() noexcept {
    if (matrix_) {
      const auto status = cudssMatrixDestroy(matrix_);
      matrix_ = nullptr;
      return status;
    }

    return CUDSS_STATUS_SUCCESS;
  }

  cudssMatrix_t Get() const noexcept { return matrix_; }

 protected:
  cudssMatrix_t matrix_{nullptr};
};

class CERES_NO_EXPORT CuDSSMatrixCSR : public CuDSSMatrixBase {
 public:
  cudssStatus_t Reset(int64_t num_rows,
                      int64_t num_cols,
                      int64_t num_nonzeros,
                      void* rows_start,
                      void* rows_end,
                      void* cols,
                      void* values,
                      cudaDataType_t index_type,
                      cudaDataType_t value_type,
                      cudssMatrixType_t matrix_type,
                      cudssMatrixViewType_t matrix_storage_type,
                      cudssIndexBase_t index_base) {
    CUDSS_STATUS_OK_OR_RETURN_CUDSS_STATUS(Free());

    return cudssMatrixCreateCsr(&matrix_,
                                num_rows,
                                num_cols,
                                num_nonzeros,
                                rows_start,
                                rows_end,
                                cols,
                                values,
                                index_type,
                                value_type,
                                matrix_type,
                                matrix_storage_type,
                                index_base);
  }
};

class CERES_NO_EXPORT CuDSSMatrixDense : public CuDSSMatrixBase {
 public:
  cudssStatus_t Reset(int64_t num_rows,
                      int64_t num_cols,
                      int64_t leading_dimension_size,
                      void* values,
                      cudaDataType_t value_type,
                      cudssLayout_t layout) {
    CUDSS_STATUS_OK_OR_RETURN_CUDSS_STATUS(Free());

    return cudssMatrixCreateDn(&matrix_,
                               num_rows,
                               num_cols,
                               leading_dimension_size,
                               values,
                               value_type,
                               layout);
  }
};

struct CudssContext {
  CudssContext(cudssHandle_t cudss_handle) : cudss_handle_(cudss_handle) {
    CUDSS_STATUS_CHECK(cudssConfigCreate(&solver_config_));
    CUDSS_STATUS_CHECK(cudssDataCreate(cudss_handle_, &solver_data_));
  }
  CudssContext(const CudssContext&) = delete;
  CudssContext(CudssContext&&) = delete;
  CudssContext& operator=(const CudssContext&) = delete;
  CudssContext& operator=(CudssContext&&) = delete;
  ~CudssContext() {
    CUDSS_STATUS_CHECK(cudssDataDestroy(cudss_handle_, solver_data_));
    CUDSS_STATUS_CHECK(cudssConfigDestroy(solver_config_));
  }

  cudssHandle_t cudss_handle_{nullptr};
  cudssConfig_t solver_config_{nullptr};
  cudssData_t solver_data_{nullptr};
};

template <typename Scalar>
class CERES_NO_EXPORT CudaSparseCholeskyImpl final : public SparseCholesky {
 public:
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "Scalar type is unsupported by cuDSS");
  static constexpr cudaDataType_t kCuDSSScalar =
      std::is_same_v<Scalar, float> ? CUDA_R_32F : CUDA_R_64F;

  CudaSparseCholeskyImpl(ContextImpl* context)
      : context_(context),
        lhs_cols_d_(context_),
        lhs_rows_d_(context_),
        lhs_values_d_(context_),
        rhs_d_(context_),
        x_d_(context_) {}
  CudaSparseCholeskyImpl(const CudaSparseCholeskyImpl&) = delete;
  CudaSparseCholeskyImpl(CudaSparseCholeskyImpl&&) = delete;
  CudaSparseCholeskyImpl& operator=(const CudaSparseCholeskyImpl&) = delete;
  CudaSparseCholeskyImpl& operator=(CudaSparseCholeskyImpl&&) = delete;
  ~CudaSparseCholeskyImpl() = default;

  CompressedRowSparseMatrix::StorageType StorageType() const {
    return CompressedRowSparseMatrix::StorageType::LOWER_TRIANGULAR;
  }

  LinearSolverTerminationType Factorize(CompressedRowSparseMatrix* lhs,
                                        std::string* message) {
    if (lhs->num_rows() != lhs->num_cols()) {
      *message = "lhs matrix must be square";
      return factorize_result_ = LinearSolverTerminationType::FATAL_ERROR;
    }
    if (lhs->storage_type() != StorageType()) {
      *message = "lhs matrix must be lower triangular";
      return factorize_result_ = LinearSolverTerminationType::FATAL_ERROR;
    }

    // If, after previous attempt to factorize, cudssDataGet(CUDSS_DATA_INFO)
    // returned a numerical error, such error will be preserved by cuDSS 0.3.0
    // and returned by cudssDataGet(CUDSS_DATA_INFO) even after correctly
    // factorizeable matrix is provided. Such behaviour forces us to reset a
    // cudssData_t object (managed by CudssContext) and to loose a result of
    // anylyze stage, thus we have to perform analyze one more time.
    // TODO: do not re-perform analyze in case of failed factorization numerics
    if (analyze_result_ != LinearSolverTerminationType::SUCCESS ||
        factorize_result_ != LinearSolverTerminationType::SUCCESS) {
      analyze_result_ = Analyze(lhs, message);
      if (analyze_result_ != LinearSolverTerminationType::SUCCESS) {
        return analyze_result_;
      }
    }
    CHECK_NE(cudss_context_.get(), nullptr);

    ConvertAndCopyToDevice(lhs->values(), lhs_values_h_.data(), lhs_values_d_);

    CUDSS_STATUS_OK_OR_RETURN_FATAL_ERROR(
        cudssExecute(context_->cudss_handle_,
                     CUDSS_PHASE_FACTORIZATION,
                     cudss_context_->solver_config_,
                     cudss_context_->solver_data_,
                     cudss_lhs_.Get(),
                     cudss_x_.Get(),
                     cudss_rhs_.Get()),
        "cudssExecute with CUDSS_PHASE_FACTORIZATION failed");

    int cudss_data_info;
    CUDSS_STATUS_OK_OR_RETURN_FATAL_ERROR(
        GetCudssDataInfo(cudss_data_info),
        "cudssDataGet with CUDSS_DATA_INFO failed");
    const auto factorization_status =
        static_cast<cudssStatus_t>(cudss_data_info);

    if (factorization_status == CUDSS_STATUS_SUCCESS) {
      return factorize_result_ = LinearSolverTerminationType::SUCCESS;
    }

    if (cudss_data_info > 0) {
      return factorize_result_ = LinearSolverTerminationType::FAILURE;
    }

    return factorize_result_ = LinearSolverTerminationType::FATAL_ERROR;
  }

  LinearSolverTerminationType Solve(const double* rhs,
                                    double* solution,
                                    std::string* message) {
    CHECK_NE(cudss_context_.get(), nullptr);

    if (factorize_result_ != LinearSolverTerminationType::SUCCESS) {
      *message = "Factorize did not complete successfully previously.";
      return factorize_result_;
    }

    ConvertAndCopyToDevice(rhs, rhs_h_.data(), rhs_d_);

    CUDSS_STATUS_OK_OR_RETURN_FATAL_ERROR(
        cudssExecute(context_->cudss_handle_,
                     CUDSS_PHASE_SOLVE,
                     cudss_context_->solver_config_,
                     cudss_context_->solver_data_,
                     cudss_lhs_.Get(),
                     cudss_x_.Get(),
                     cudss_rhs_.Get()),
        "cudssExecute with CUDSS_PHASE_SOLVE failed");

    ConvertAndCopyToHost(x_d_, x_h_.data(), solution);

    int cudss_data_info;
    CUDSS_STATUS_OK_OR_RETURN_FATAL_ERROR(
        GetCudssDataInfo(cudss_data_info),
        "cudssDataGet with CUDSS_DATA_INFO failed");
    const auto solve_status = static_cast<cudssStatus_t>(cudss_data_info);

    if (solve_status != CUDSS_STATUS_SUCCESS) {
      return LinearSolverTerminationType::FAILURE;
    }

    return LinearSolverTerminationType::SUCCESS;
  }

 private:
  cudssStatus_t GetCudssDataInfo(int& cudss_data_info) {
    CHECK_NE(cudss_context_.get(), nullptr);
    std::size_t size_written = 0;
    CUDSS_STATUS_OK_OR_RETURN_CUDSS_STATUS(
        cudssDataGet(context_->cudss_handle_,
                     cudss_context_->solver_data_,
                     CUDSS_DATA_INFO,
                     &cudss_data_info,
                     sizeof(cudss_data_info),
                     &size_written));
    // TODO: enable following check after cudssDataGet will be fixed
    // CHECK_EQ(size_written, sizeof(cudss_data_info));

    return CUDSS_STATUS_SUCCESS;
  }

  LinearSolverTerminationType Analyze(const CompressedRowSparseMatrix* lhs,
                                      std::string* message) {
    if (auto status = SetupCudssMatrices(lhs, message);
        status != LinearSolverTerminationType::SUCCESS) {
      return status;
    }

    lhs_rows_d_.CopyFromCpu(lhs->rows(), lhs->num_rows() + 1);
    lhs_cols_d_.CopyFromCpu(lhs->cols(), lhs->num_nonzeros());

    // Analyze and factorization results are stored in cudssData_t (managed by
    // CudssContext). Given that cuDSS 0.3.0 does not reset it's error state in
    // case of failed numerics at factorization stage, we have to reset
    // cudssData_t and to recompute an analyze stage while trying to factorize a
    // rescaled matrix with the same structure.
    // TODO: move creation of CudssContext to ctor of CudaSparseCholeskyImpl
    cudss_context_ = std::make_unique<CudssContext>(context_->cudss_handle_);

    CUDSS_STATUS_OK_OR_RETURN_FATAL_ERROR(
        cudssExecute(context_->cudss_handle_,
                     CUDSS_PHASE_ANALYSIS,
                     cudss_context_->solver_config_,
                     cudss_context_->solver_data_,
                     cudss_lhs_.Get(),
                     cudss_x_.Get(),
                     cudss_rhs_.Get()),
        "cudssExecute with CUDSS_PHASE_ANALYSIS failed");

    return LinearSolverTerminationType::SUCCESS;
  }

  // Resize buffers and setup cuDSS structs that describe the type and storage
  // configuration of linear system operands.
  LinearSolverTerminationType SetupCudssMatrices(
      const CompressedRowSparseMatrix* lhs, std::string* message) {
    const auto num_rows = lhs->num_rows();
    const auto num_nonzeros = lhs->num_nonzeros();

    if constexpr (std::is_same_v<Scalar, float>) {
      lhs_values_h_.Reserve(num_nonzeros);
      rhs_h_.Reserve(num_rows);
      x_h_.Reserve(num_rows);
    }

    lhs_rows_d_.Reserve(num_rows + 1);
    lhs_cols_d_.Reserve(num_nonzeros);
    lhs_values_d_.Reserve(num_nonzeros);
    rhs_d_.Reserve(num_rows);
    x_d_.Reserve(num_rows);

    static constexpr auto kFailedToCreateCuDSSMatrix =
        "cudssMatrixCreate() call failed";
    CUDSS_STATUS_OK_OR_RETURN_FATAL_ERROR(cudss_lhs_.Reset(num_rows,
                                                           num_rows,
                                                           num_nonzeros,
                                                           lhs_rows_d_.data(),
                                                           nullptr,
                                                           lhs_cols_d_.data(),
                                                           lhs_values_d_.data(),
                                                           CUDA_R_32I,
                                                           kCuDSSScalar,
                                                           CUDSS_MTYPE_SPD,
                                                           CUDSS_MVIEW_LOWER,
                                                           CUDSS_BASE_ZERO),
                                          kFailedToCreateCuDSSMatrix);

    CUDSS_STATUS_OK_OR_RETURN_FATAL_ERROR(
        cudss_rhs_.Reset(num_rows,
                         1,
                         num_rows,
                         rhs_d_.data(),
                         kCuDSSScalar,
                         CUDSS_LAYOUT_COL_MAJOR),
        kFailedToCreateCuDSSMatrix);

    CUDSS_STATUS_OK_OR_RETURN_FATAL_ERROR(
        cudss_x_.Reset(num_rows,
                       1,
                       num_rows,
                       x_d_.data(),
                       kCuDSSScalar,
                       CUDSS_LAYOUT_COL_MAJOR),
        kFailedToCreateCuDSSMatrix);

    return LinearSolverTerminationType::SUCCESS;
  }

  template <typename S, typename D>
  void Convert(const S* source, D* destination, size_t size) {
    Eigen::Map<Eigen::Matrix<D, Eigen::Dynamic, 1>>(destination, size) =
        Eigen::Map<const Eigen::Matrix<S, Eigen::Dynamic, 1>>(source, size)
            .template cast<D>();
  }

  void ConvertAndCopyToDevice(const double* source,
                              Scalar* intermediate,
                              CudaBuffer<Scalar>& destination) {
    const auto size = destination.size();
    if constexpr (std::is_same_v<Scalar, double>) {
      destination.CopyFromCpu(source, size);
    } else {
      Convert(source, intermediate, size);
      destination.CopyFromCpu(intermediate, size);
    }
  }

  void ConvertAndCopyToHost(const CudaBuffer<Scalar>& source,
                            Scalar* intermediate,
                            double* destination) {
    const auto size = source.size();
    if constexpr (std::is_same_v<Scalar, double>) {
      source.CopyToCpu(destination, source.size());
    } else {
      source.CopyToCpu(intermediate, source.size());
      Convert(intermediate, destination, size);
    }
  }

  ContextImpl* context_{nullptr};
  std::unique_ptr<CudssContext> cudss_context_;
  CuDSSMatrixCSR cudss_lhs_;
  CuDSSMatrixDense cudss_rhs_;
  CuDSSMatrixDense cudss_x_;

  CudaPinnedHostBuffer<Scalar> lhs_values_h_;
  CudaPinnedHostBuffer<Scalar> rhs_h_;
  CudaPinnedHostBuffer<Scalar> x_h_;
  CudaBuffer<int> lhs_rows_d_;
  CudaBuffer<int> lhs_cols_d_;
  CudaBuffer<Scalar> lhs_values_d_;
  CudaBuffer<Scalar> rhs_d_;
  CudaBuffer<Scalar> x_d_;

  LinearSolverTerminationType analyze_result_ =
      LinearSolverTerminationType::FATAL_ERROR;
  LinearSolverTerminationType factorize_result_ =
      LinearSolverTerminationType::FATAL_ERROR;
};

template <typename Scalar>
std::unique_ptr<SparseCholesky> CudaSparseCholesky<Scalar>::Create(
    ContextImpl* context, const OrderingType ordering_type) {
  if (ordering_type == OrderingType::NESDIS) {
    LOG(FATAL)
        << "Congratulations you have found a bug in Ceres Solver. Please "
           "report it to the Ceres Solver developers.";
    return nullptr;
  }

  if (context == nullptr || !context->IsCudaInitialized()) {
    LOG(FATAL) << "CudaSparseCholesky requires CUDA context to be initialized";
    return nullptr;
  }

  return std::make_unique<CudaSparseCholeskyImpl<Scalar>>(context);
}

template class CudaSparseCholesky<float>;
template class CudaSparseCholesky<double>;

}  // namespace ceres::internal

#undef CUDSS_STATUS_CHECK
#undef CUDSS_STATUS_OK_OR_RETURN_FATAL_ERROR
#undef CUDSS_STATUS_OK_OR_RETURN_CUDSS_STATUS

#endif  // CERES_NO_CUDSS
