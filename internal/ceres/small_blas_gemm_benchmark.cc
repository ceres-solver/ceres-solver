// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2018 Google Inc. All rights reserved.
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
// Authors: sameeragarwal@google.com (Sameer Agarwal)

#include <iostream>
#include "Eigen/Dense"
#include "benchmark/benchmark.h"
#include "ceres/small_blas.h"

namespace ceres {

// Benchmarking matrix-matrix multiply routines and optimizing memory
// access requires that we make sure that they are not just sitting in
// the cache. So, as the benchmarking routine iterates, we need to
// multiply new/different matrice. Allocating/creating these objects
// in the benchmarking loop is too heavy duty, so we create them
// before hand and cycle through them in the benchmark. This class,
// given the size of the matrices creates such objects for use in the
// benchmark.
class MatrixMatrixMultiplyData {
 public:
  MatrixMatrixMultiplyData(
      int a_rows, int a_cols, int b_rows, int b_cols, int c_rows, int c_cols) {
    num_elements_ = 1000;
    a_size_ = a_rows * a_cols;
    b_size_ = b_rows * b_cols;
    c_size_ = c_cols * c_cols;
    a_.resize(num_elements_ * a_size_, 1.00001);
    b_.resize(num_elements_ * b_size_, 1.00002);
    c_.resize(num_elements_ * c_size_, 1.00003);
  }

  int num_elements() const { return num_elements_; }
  double* GetA(int i) { return &a_[i * a_size_]; };
  double* GetB(int i) { return &b_[i * b_size_]; };
  double* GetC(int i) { return &c_[i * c_size_]; };

 private:
  int num_elements_;
  int a_size_;
  int b_size_;
  int c_size_;
  std::vector<double> a_;
  std::vector<double> b_;
  std::vector<double> c_;
};

static void MatrixMatrixMultiplySizeArguments(
    benchmark::internal::Benchmark* benchmark) {
  std::vector<int> b_rows = {2, 4, 6, 8};
  std::vector<int> b_cols = {2, 4, 6, 8, 10, 12, 15};
  std::vector<int> c_cols = {2, 4, 6, 8, 10, 12, 15};
  for (int i : b_rows) {
    for (int j : b_cols) {
      for (int k : c_cols) {
        benchmark->Args({i, j, k});
      }
    }
  }
}

void BM_MatrixMatrixMultiplyDynamic(benchmark::State& state) {
  const int b_rows = state.range(0);
  const int b_cols = state.range(1);
  const int c_cols = state.range(2);
  MatrixMatrixMultiplyData data(b_rows, c_cols, b_rows, b_cols, b_cols, c_cols);

  const int num_elements = data.num_elements();
  int i = 0;
  for (auto _ : state) {
    i = (i + 1) % num_elements;
    // a += b * c
    internal::MatrixMatrixMultiply<Eigen::Dynamic,
                                   Eigen::Dynamic,
                                   Eigen::Dynamic,
                                   Eigen::Dynamic,
                                   1>(data.GetB(i), b_rows, b_cols,
                                      data.GetC(i), b_cols, c_cols,
                                      data.GetA(i), 0, 0, b_rows, c_cols);
    i = (i + 1) % num_elements;
  }
}

BENCHMARK(BM_MatrixMatrixMultiplyDynamic)
    ->Apply(MatrixMatrixMultiplySizeArguments);

static void MatrixTransposeMatrixMultiplySizeArguments(
    benchmark::internal::Benchmark* benchmark) {
  std::vector<int> b_rows = {2, 4, 6, 8};
  std::vector<int> b_cols = {2, 4, 5, 8, 10, 12, 15};
  std::vector<int> c_cols = {2, 4, 6, 8};
  for (int i : b_rows) {
    for (int j : b_cols) {
      for (int k : c_cols) {
        benchmark->Args({i, j, k});
      }
    }
  }
}

void BM_MatrixTransposeMatrixMultiplyDynamic(benchmark::State& state) {
  const int b_rows = state.range(0);
  const int b_cols = state.range(1);
  const int c_cols = state.range(2);
  MatrixMatrixMultiplyData data(b_cols, c_cols, b_rows, b_cols, b_cols, c_cols);

  const int num_elements = data.num_elements();
  int i = 0;
  for (auto _ : state) {
    i = (i + 1) % num_elements;
    // a += b * c
    internal::MatrixTransposeMatrixMultiply<Eigen::Dynamic,
                                            Eigen::Dynamic,
                                            Eigen::Dynamic,
                                            Eigen::Dynamic,
                                            1>(data.GetB(i), b_rows, b_cols,
                                               data.GetC(i), b_cols, c_cols,
                                               data.GetA(i), 0, 0, b_cols, c_cols);
    i = (i + 1) % num_elements;
  }
}

BENCHMARK(BM_MatrixTransposeMatrixMultiplyDynamic)
    ->Apply(MatrixTransposeMatrixMultiplySizeArguments);

}  // namespace ceres

BENCHMARK_MAIN();
