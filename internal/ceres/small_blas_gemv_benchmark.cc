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

// Benchmarking matrix-vector multiply routines and optimizing memory
// access requires that we make sure that they are not just sitting in
// the cache. So, as the benchmarking routine iterates, we need to
// multiply new/different matrice and vectors. Allocating/creating
// these objects in the benchmarking loop is too heavy duty, so we
// create them before hand and cycle through them in the
// benchmark. This class, given the size of the matrix creates such
// matrix and vector objects for use in the benchmark.
class MatrixVectorMultiplyData {
 public:
  MatrixVectorMultiplyData(int rows, int cols) {
    rows_ = rows;
    cols_ = cols;
    num_elements_ = 1000;
    a_.resize(num_elements_ * rows, 1.00001);
    b_.resize(num_elements_ * rows * cols, 1.00002);
    c_.resize(num_elements_ * cols, 1.00003);
  }

  int num_elements() const { return num_elements_; }
  double* GetA(int i) { return &a_[i * rows_]; };
  double* GetB(int i) { return &b_[i * rows_ * cols_]; };
  double* GetC(int i) { return &c_[i * cols_]; };

 private:
  int num_elements_;
  int rows_;
  int cols_;
  std::vector<double> a_;
  std::vector<double> b_;
  std::vector<double> c_;
};

// Run on (8 X 2200 MHz CPU s)
// 2018-02-24 15:51:34
// ------------------------------------------------------------------------------------
// Benchmark                                             Time           CPU Iterations
// ------------------------------------------------------------------------------------
// BM_MatrixVectorMultiplyDynamic/1/1                    9 ns          9 ns   74207569
// BM_MatrixVectorMultiplyDynamic/1/2                    9 ns          9 ns   75074270
// BM_MatrixVectorMultiplyDynamic/1/3                    8 ns          8 ns   85795879
// BM_MatrixVectorMultiplyDynamic/1/4                    9 ns          9 ns   75314168
// BM_MatrixVectorMultiplyDynamic/1/6                   10 ns         10 ns   64418759
// BM_MatrixVectorMultiplyDynamic/1/7                   11 ns         11 ns   60762307
// BM_MatrixVectorMultiplyDynamic/1/12                  16 ns         16 ns   43118848
// BM_MatrixVectorMultiplyDynamic/1/16                  17 ns         17 ns   42608881
// BM_MatrixVectorMultiplyDynamic/1/20                  18 ns         18 ns   36463081
// BM_MatrixVectorMultiplyDynamic/2/1                    8 ns          8 ns   81921168
// BM_MatrixVectorMultiplyDynamic/2/2                   10 ns         10 ns   68008705
// BM_MatrixVectorMultiplyDynamic/2/3                   11 ns         11 ns   57629295
// BM_MatrixVectorMultiplyDynamic/2/4                   13 ns         13 ns   51549808
// BM_MatrixVectorMultiplyDynamic/2/6                   15 ns         15 ns   47373158
// BM_MatrixVectorMultiplyDynamic/2/7                   15 ns         15 ns   41420609
// BM_MatrixVectorMultiplyDynamic/2/12                  20 ns         20 ns   34076194
// BM_MatrixVectorMultiplyDynamic/2/16                  24 ns         24 ns   27355397
// BM_MatrixVectorMultiplyDynamic/2/20                  28 ns         28 ns   23775236
// BM_MatrixVectorMultiplyDynamic/3/1                   10 ns         10 ns   64430024
// BM_MatrixVectorMultiplyDynamic/3/2                   14 ns         14 ns   48081877
// BM_MatrixVectorMultiplyDynamic/3/3                   15 ns         15 ns   43476641
// BM_MatrixVectorMultiplyDynamic/3/4                   17 ns         17 ns   41499188
// BM_MatrixVectorMultiplyDynamic/3/6                   19 ns         19 ns   34956479
// BM_MatrixVectorMultiplyDynamic/3/7                   20 ns         20 ns   34430858
// BM_MatrixVectorMultiplyDynamic/3/12                  25 ns         25 ns   25351297
// BM_MatrixVectorMultiplyDynamic/3/16                  34 ns         34 ns   22710903
// BM_MatrixVectorMultiplyDynamic/3/20                  42 ns         42 ns   16135018
// BM_MatrixVectorMultiplyDynamic/4/1                   13 ns         13 ns   56316725
// BM_MatrixVectorMultiplyDynamic/4/2                   16 ns         16 ns   39346179
// BM_MatrixVectorMultiplyDynamic/4/3                   17 ns         17 ns   37380383
// BM_MatrixVectorMultiplyDynamic/4/4                   21 ns         20 ns   34415961
// BM_MatrixVectorMultiplyDynamic/4/6                   22 ns         22 ns   29604192
// BM_MatrixVectorMultiplyDynamic/4/7                   25 ns         25 ns   29451735
// BM_MatrixVectorMultiplyDynamic/4/12                  36 ns         36 ns   18568230
// BM_MatrixVectorMultiplyDynamic/4/16                  47 ns         47 ns   13575889
// BM_MatrixVectorMultiplyDynamic/4/20                  55 ns         55 ns   13016717
void BM_MatrixVectorMultiplyDynamic(benchmark::State& state) {
  const int rows = state.range(0);
  const int cols = state.range(1);
  MatrixVectorMultiplyData data(rows, cols);
  const int num_elements = data.num_elements();
  int i = 0;
  for (auto _ : state) {
    // A += B * C;
    internal::MatrixVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
        data.GetB(i), rows, cols, data.GetC(i), data.GetA(i));
    i = (i + 1) % num_elements;
  }
}

static void MatrixSizeArguments(benchmark::internal::Benchmark* benchmark) {
  std::vector<int> rows = {1, 2, 3, 4};
  std::vector<int> cols = {1, 2, 3, 4, 6, 7, 12, 16, 20};
  for (int r : rows) {
    for (int c : cols) {
      benchmark->Args({r, c});
    }
  }
}

BENCHMARK(BM_MatrixVectorMultiplyDynamic)->Apply(MatrixSizeArguments);

// Run on (8 X 2200 MHz CPU s)
// 2018-02-24 15:51:34
// ------------------------------------------------------------------------------------
// Benchmark                                             Time           CPU Iterations
// ------------------------------------------------------------------------------------
// BM_MatrixTransposeVectorMultiplyDynamic/1/1           9 ns          9 ns   86150665
// BM_MatrixTransposeVectorMultiplyDynamic/1/2           9 ns          9 ns   82534517
// BM_MatrixTransposeVectorMultiplyDynamic/1/3          10 ns         10 ns   73243209
// BM_MatrixTransposeVectorMultiplyDynamic/1/4          12 ns         12 ns   56850022
// BM_MatrixTransposeVectorMultiplyDynamic/1/6          16 ns         16 ns   42462072
// BM_MatrixTransposeVectorMultiplyDynamic/1/7          17 ns         17 ns   42546208
// BM_MatrixTransposeVectorMultiplyDynamic/1/12         23 ns         23 ns   31237031
// BM_MatrixTransposeVectorMultiplyDynamic/1/16         34 ns         34 ns   20534363
// BM_MatrixTransposeVectorMultiplyDynamic/1/20         37 ns         37 ns   18656368
// BM_MatrixTransposeVectorMultiplyDynamic/2/1           9 ns          9 ns   81061676
// BM_MatrixTransposeVectorMultiplyDynamic/2/2          11 ns         11 ns   62989859
// BM_MatrixTransposeVectorMultiplyDynamic/2/3          15 ns         15 ns   51009255
// BM_MatrixTransposeVectorMultiplyDynamic/2/4          17 ns         17 ns   38093165
// BM_MatrixTransposeVectorMultiplyDynamic/2/6          22 ns         22 ns   33472003
// BM_MatrixTransposeVectorMultiplyDynamic/2/7          24 ns         24 ns   29251863
// BM_MatrixTransposeVectorMultiplyDynamic/2/12         36 ns         36 ns   19115185
// BM_MatrixTransposeVectorMultiplyDynamic/2/16         47 ns         46 ns   15031588
// BM_MatrixTransposeVectorMultiplyDynamic/2/20         56 ns         56 ns   12340891
// BM_MatrixTransposeVectorMultiplyDynamic/3/1           9 ns          9 ns   79240200
// BM_MatrixTransposeVectorMultiplyDynamic/3/2          13 ns         13 ns   53128510
// BM_MatrixTransposeVectorMultiplyDynamic/3/3          16 ns         16 ns   42053913
// BM_MatrixTransposeVectorMultiplyDynamic/3/4          18 ns         18 ns   34659814
// BM_MatrixTransposeVectorMultiplyDynamic/3/6          24 ns         24 ns   29154155
// BM_MatrixTransposeVectorMultiplyDynamic/3/7          28 ns         28 ns   25546979
// BM_MatrixTransposeVectorMultiplyDynamic/3/12         40 ns         40 ns   16952559
// BM_MatrixTransposeVectorMultiplyDynamic/3/16         50 ns         50 ns   12813472
// BM_MatrixTransposeVectorMultiplyDynamic/3/20         62 ns         62 ns   10624251
// BM_MatrixTransposeVectorMultiplyDynamic/4/1           9 ns          9 ns   75487976
// BM_MatrixTransposeVectorMultiplyDynamic/4/2          15 ns         15 ns   46294459
// BM_MatrixTransposeVectorMultiplyDynamic/4/3          17 ns         17 ns   40505040
// BM_MatrixTransposeVectorMultiplyDynamic/4/4          20 ns         20 ns   33256527
// BM_MatrixTransposeVectorMultiplyDynamic/4/6          26 ns         26 ns   25882309
// BM_MatrixTransposeVectorMultiplyDynamic/4/7          30 ns         30 ns   22910857
// BM_MatrixTransposeVectorMultiplyDynamic/4/12         46 ns         46 ns   14966763
// BM_MatrixTransposeVectorMultiplyDynamic/4/16         64 ns         64 ns    9234463
// BM_MatrixTransposeVectorMultiplyDynamic/4/20         79 ns         79 ns    8412147
void BM_MatrixTransposeVectorMultiplyDynamic(benchmark::State& state) {
  const int rows = state.range(0);
  const int cols = state.range(1);
  MatrixVectorMultiplyData data(cols, rows);
  const int num_elements = data.num_elements();
  int i = 0;
  for (auto _ : state) {
    internal::MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
        data.GetB(i), rows, cols, data.GetC(i), data.GetA(i));
    i = (i + 1) % num_elements;
  }
}

BENCHMARK(BM_MatrixTransposeVectorMultiplyDynamic)->Apply(MatrixSizeArguments);

}  // namespace ceres

BENCHMARK_MAIN();
