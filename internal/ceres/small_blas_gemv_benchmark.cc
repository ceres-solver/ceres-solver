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

// Helper function to generate the various matrix sizes for which we
// run the benchmark.
static void MatrixSizeArguments(benchmark::internal::Benchmark* benchmark) {
  std::vector<int> rows = {1, 2, 3, 4};
  std::vector<int> cols = {1, 2, 3, 4, 6, 7, 12, 16, 20};
  for (int r : rows) {
    for (int c : cols) {
      benchmark->Args({r, c});
    }
  }
}

// Run on (8 X 2200 MHz CPU s)
// 2018-02-24 15:51:34
// ------------------------------------------------------------------------------------
// Benchmark                                             Time           CPU Iterations
// ------------------------------------------------------------------------------------
// BM_MatrixVectorMultiply/1/1                    9 ns          9 ns   74207569
// BM_MatrixVectorMultiply/1/2                    9 ns          9 ns   75074270
// BM_MatrixVectorMultiply/1/3                    8 ns          8 ns   85795879
// BM_MatrixVectorMultiply/1/4                    9 ns          9 ns   75314168
// BM_MatrixVectorMultiply/1/6                   10 ns         10 ns   64418759
// BM_MatrixVectorMultiply/1/7                   11 ns         11 ns   60762307
// BM_MatrixVectorMultiply/1/12                  16 ns         16 ns   43118848
// BM_MatrixVectorMultiply/1/16                  17 ns         17 ns   42608881
// BM_MatrixVectorMultiply/1/20                  18 ns         18 ns   36463081
// BM_MatrixVectorMultiply/2/1                    8 ns          8 ns   81921168
// BM_MatrixVectorMultiply/2/2                   10 ns         10 ns   68008705
// BM_MatrixVectorMultiply/2/3                   11 ns         11 ns   57629295
// BM_MatrixVectorMultiply/2/4                   13 ns         13 ns   51549808
// BM_MatrixVectorMultiply/2/6                   15 ns         15 ns   47373158
// BM_MatrixVectorMultiply/2/7                   15 ns         15 ns   41420609
// BM_MatrixVectorMultiply/2/12                  20 ns         20 ns   34076194
// BM_MatrixVectorMultiply/2/16                  24 ns         24 ns   27355397
// BM_MatrixVectorMultiply/2/20                  28 ns         28 ns   23775236
// BM_MatrixVectorMultiply/3/1                   10 ns         10 ns   64430024
// BM_MatrixVectorMultiply/3/2                   14 ns         14 ns   48081877
// BM_MatrixVectorMultiply/3/3                   15 ns         15 ns   43476641
// BM_MatrixVectorMultiply/3/4                   17 ns         17 ns   41499188
// BM_MatrixVectorMultiply/3/6                   19 ns         19 ns   34956479
// BM_MatrixVectorMultiply/3/7                   20 ns         20 ns   34430858
// BM_MatrixVectorMultiply/3/12                  25 ns         25 ns   25351297
// BM_MatrixVectorMultiply/3/16                  34 ns         34 ns   22710903
// BM_MatrixVectorMultiply/3/20                  42 ns         42 ns   16135018
// BM_MatrixVectorMultiply/4/1                   13 ns         13 ns   56316725
// BM_MatrixVectorMultiply/4/2                   16 ns         16 ns   39346179
// BM_MatrixVectorMultiply/4/3                   17 ns         17 ns   37380383
// BM_MatrixVectorMultiply/4/4                   21 ns         20 ns   34415961
// BM_MatrixVectorMultiply/4/6                   22 ns         22 ns   29604192
// BM_MatrixVectorMultiply/4/7                   25 ns         25 ns   29451735
// BM_MatrixVectorMultiply/4/12                  36 ns         36 ns   18568230
// BM_MatrixVectorMultiply/4/16                  47 ns         47 ns   13575889
// BM_MatrixVectorMultiply/4/20                  55 ns         55 ns   13016717
void BM_MatrixVectorMultiply(benchmark::State& state) {
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

BENCHMARK(BM_MatrixVectorMultiply)->Apply(MatrixSizeArguments);

Run on (8 X 2200 MHz CPU s)
2018-02-25 12:39:03
-----------------------------------------------------------------------------
Benchmark                                      Time           CPU Iterations
-----------------------------------------------------------------------------
BM_MatrixTransposeVectorMultiply/1/1          11 ns         11 ns   65731403
BM_MatrixTransposeVectorMultiply/1/2          11 ns         11 ns   61544954
BM_MatrixTransposeVectorMultiply/1/3          11 ns         11 ns   55358725
BM_MatrixTransposeVectorMultiply/1/4          10 ns         10 ns   68624087
BM_MatrixTransposeVectorMultiply/1/6          13 ns         13 ns   54298502
BM_MatrixTransposeVectorMultiply/1/7          15 ns         15 ns   45811818
BM_MatrixTransposeVectorMultiply/1/12         16 ns         16 ns   40978088
BM_MatrixTransposeVectorMultiply/1/16         19 ns         19 ns   40724903
BM_MatrixTransposeVectorMultiply/1/20         20 ns         20 ns   34253279
BM_MatrixTransposeVectorMultiply/2/1          12 ns         12 ns   62459847
BM_MatrixTransposeVectorMultiply/2/2          13 ns         13 ns   51665092
BM_MatrixTransposeVectorMultiply/2/3          16 ns         16 ns   45675508
BM_MatrixTransposeVectorMultiply/2/4          16 ns         16 ns   42943468
BM_MatrixTransposeVectorMultiply/2/6          19 ns         19 ns   36854907
BM_MatrixTransposeVectorMultiply/2/7          21 ns         21 ns   33189512
BM_MatrixTransposeVectorMultiply/2/12         23 ns         23 ns   29639917
BM_MatrixTransposeVectorMultiply/2/16         26 ns         26 ns   27554716
BM_MatrixTransposeVectorMultiply/2/20         27 ns         27 ns   24042259
BM_MatrixTransposeVectorMultiply/3/1          13 ns         13 ns   54472165
BM_MatrixTransposeVectorMultiply/3/2          17 ns         17 ns   43412737
BM_MatrixTransposeVectorMultiply/3/3          19 ns         19 ns   38757329
BM_MatrixTransposeVectorMultiply/3/4          17 ns         17 ns   40583004
BM_MatrixTransposeVectorMultiply/3/6          22 ns         22 ns   30218262
BM_MatrixTransposeVectorMultiply/3/7          23 ns         23 ns   29814765
BM_MatrixTransposeVectorMultiply/3/12         24 ns         24 ns   27344284
BM_MatrixTransposeVectorMultiply/3/16         31 ns         31 ns   24471417
BM_MatrixTransposeVectorMultiply/3/20         33 ns         33 ns   18655225
BM_MatrixTransposeVectorMultiply/4/1          14 ns         14 ns   44872370
BM_MatrixTransposeVectorMultiply/4/2          17 ns         17 ns   39525915
BM_MatrixTransposeVectorMultiply/4/3          21 ns         21 ns   32630990
BM_MatrixTransposeVectorMultiply/4/4          19 ns         19 ns   37524860
BM_MatrixTransposeVectorMultiply/4/6          28 ns         28 ns   26768847
BM_MatrixTransposeVectorMultiply/4/7          31 ns         31 ns   22582831
BM_MatrixTransposeVectorMultiply/4/12         32 ns         32 ns   22881799
BM_MatrixTransposeVectorMultiply/4/16         34 ns         34 ns   19839527
BM_MatrixTransposeVectorMultiply/4/20         39 ns         39 ns   17312077
void BM_MatrixTransposeVectorMultiply(benchmark::State& state) {
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

BENCHMARK(BM_MatrixTransposeVectorMultiply)->Apply(MatrixSizeArguments);

}  // namespace ceres

BENCHMARK_MAIN();
