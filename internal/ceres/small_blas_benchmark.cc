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

class MatrixVectorMultiplyData {
 public:
  MatrixVectorMultiplyData(int rows, int cols) {
    num_elements_ = 10000000;
    data_.resize(num_elements_, 1.00000000000001);
    num_buffers_ = 1000;
    ptrs_.resize(3 * num_buffers_, NULL);
    double* p = &data_[0];
    for (int i = 0; i < num_buffers_; ++i) {
      ptrs_[3 * i] = p;
      p += rows * cols;
      ptrs_[3 * i + 1] = p;
      p += cols;
      ptrs_[3 * i + 2] = p;
      p += rows;
    }
  }

  int num_elements() const { return num_elements_; }
  int num_buffers() const { return num_buffers_; }
  double* data() { return &data_[0]; }
  const std::vector<double*>& ptrs() const { return ptrs_; }

 private:
  size_t num_elements_;
  int num_buffers_;
  std::vector<double> data_;
  std::vector<double*> ptrs_;
};

// Run on (8 X 2200 MHz CPU s)
// 2018-02-06 21:23:59
// ---------------------------------------------------------------------------
// Benchmark                                    Time           CPU Iterations
// ---------------------------------------------------------------------------
// BM_MatrixVectorMultiplyDynamic/1/1           4 ns          4 ns  165611093
// BM_MatrixVectorMultiplyDynamic/1/2           5 ns          5 ns  140648672
// BM_MatrixVectorMultiplyDynamic/1/3           5 ns          5 ns  139414459
// BM_MatrixVectorMultiplyDynamic/1/4           5 ns          5 ns  144247512
// BM_MatrixVectorMultiplyDynamic/1/6           6 ns          6 ns  106639042
// BM_MatrixVectorMultiplyDynamic/1/8           7 ns          7 ns  102367617
// BM_MatrixVectorMultiplyDynamic/1/10          9 ns          9 ns   82419847
// BM_MatrixVectorMultiplyDynamic/1/12         10 ns         10 ns   65129002
// BM_MatrixVectorMultiplyDynamic/1/16         12 ns         12 ns   53500867
// BM_MatrixVectorMultiplyDynamic/1/20         16 ns         16 ns   46067179
// BM_MatrixVectorMultiplyDynamic/2/1           5 ns          5 ns  128880215
// BM_MatrixVectorMultiplyDynamic/2/2           8 ns          8 ns   81938429
// BM_MatrixVectorMultiplyDynamic/2/3          10 ns         10 ns   68807565
// BM_MatrixVectorMultiplyDynamic/2/4           8 ns          8 ns   91833388
// BM_MatrixVectorMultiplyDynamic/2/6          10 ns         10 ns   64031028
// BM_MatrixVectorMultiplyDynamic/2/8          12 ns         12 ns   59788179
// BM_MatrixVectorMultiplyDynamic/2/10         15 ns         15 ns   44737868
// BM_MatrixVectorMultiplyDynamic/2/12         17 ns         17 ns   37423949
// BM_MatrixVectorMultiplyDynamic/2/16         22 ns         22 ns   33470723
// BM_MatrixVectorMultiplyDynamic/2/20         26 ns         26 ns   27076057
// BM_MatrixVectorMultiplyDynamic/3/1           6 ns          6 ns  100932908
// BM_MatrixVectorMultiplyDynamic/3/2          12 ns         12 ns   65591589
// BM_MatrixVectorMultiplyDynamic/3/3          14 ns         14 ns   48182819
// BM_MatrixVectorMultiplyDynamic/3/4          11 ns         11 ns   61770338
// BM_MatrixVectorMultiplyDynamic/3/6          15 ns         15 ns   44712435
// BM_MatrixVectorMultiplyDynamic/3/8          18 ns         18 ns   35177294
// BM_MatrixVectorMultiplyDynamic/3/10         21 ns         21 ns   32164683
// BM_MatrixVectorMultiplyDynamic/3/12         24 ns         24 ns   28222279
// BM_MatrixVectorMultiplyDynamic/3/16         30 ns         30 ns   23050731
// BM_MatrixVectorMultiplyDynamic/3/20         38 ns         38 ns   17832714
// BM_MatrixVectorMultiplyDynamic/4/1           8 ns          8 ns   85763293
// BM_MatrixVectorMultiplyDynamic/4/2          16 ns         16 ns   41959886
// BM_MatrixVectorMultiplyDynamic/4/3          19 ns         19 ns   36674176
// BM_MatrixVectorMultiplyDynamic/4/4          15 ns         15 ns   43561867
// BM_MatrixVectorMultiplyDynamic/4/6          21 ns         21 ns   34278607
// BM_MatrixVectorMultiplyDynamic/4/8          22 ns         22 ns   31484163
// BM_MatrixVectorMultiplyDynamic/4/10         26 ns         26 ns   25605197
// BM_MatrixVectorMultiplyDynamic/4/12         31 ns         31 ns   23380172
// BM_MatrixVectorMultiplyDynamic/4/16         38 ns         38 ns   18054638
// BM_MatrixVectorMultiplyDynamic/4/20         49 ns         49 ns   14771703
void BM_MatrixVectorMultiplyDynamic(benchmark::State& state) {
  const int rows = state.range(0);
  const int cols = state.range(1);
  MatrixVectorMultiplyData data(rows, cols);
  const std::vector<double*> ptrs = data.ptrs();
  const int num_buffers = data.num_buffers();

  int i = 0;
  for (auto _ : state) {
    double* a_ptr = ptrs[3 * i];
    double* b_ptr = ptrs[3 * i + 1];
    double* c_ptr = ptrs[3 * i + 2];
    internal::MatrixVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
        a_ptr, rows, cols, b_ptr, c_ptr);
    i = (i + 1) % num_buffers;
  }
}

BENCHMARK(BM_MatrixVectorMultiplyDynamic)
->ArgPair(1, 1)
->ArgPair(1, 2)
->ArgPair(1, 3)
->ArgPair(1, 4)
->ArgPair(1, 6)
->ArgPair(1, 8)
->ArgPair(1, 10)
->ArgPair(1, 12)
->ArgPair(1, 16)
->ArgPair(1, 20)
->ArgPair(2, 1)
->ArgPair(2, 2)
->ArgPair(2, 3)
->ArgPair(2, 4)
->ArgPair(2, 6)
->ArgPair(2, 8)
->ArgPair(2, 10)
->ArgPair(2, 12)
->ArgPair(2, 16)
->ArgPair(2, 20)
->ArgPair(3, 1)
->ArgPair(3, 2)
->ArgPair(3, 3)
->ArgPair(3, 4)
->ArgPair(3, 6)
->ArgPair(3, 8)
->ArgPair(3, 10)
->ArgPair(3, 12)
->ArgPair(3, 16)
->ArgPair(3, 20)
->ArgPair(4, 1)
->ArgPair(4, 2)
->ArgPair(4, 3)
->ArgPair(4, 4)
->ArgPair(4, 6)
->ArgPair(4, 8)
->ArgPair(4, 10)
->ArgPair(4, 12)
->ArgPair(4, 16)
->ArgPair(4, 20);

// Run on (8 X 2200 MHz CPU s)
// 2018-02-06 21:18:17
// ------------------------------------------------------------------------------------
// Benchmark                                             Time           CPU Iterations
// ------------------------------------------------------------------------------------
// BM_MatrixTransposeVectorMultiplyDynamic/1/1           5 ns          5 ns  139356174
// BM_MatrixTransposeVectorMultiplyDynamic/1/2           6 ns          6 ns  120800041
// BM_MatrixTransposeVectorMultiplyDynamic/1/3           7 ns          7 ns  100267858
// BM_MatrixTransposeVectorMultiplyDynamic/1/4           9 ns          9 ns   70778564
// BM_MatrixTransposeVectorMultiplyDynamic/1/6          14 ns         14 ns   47748651
// BM_MatrixTransposeVectorMultiplyDynamic/1/8          16 ns         16 ns   43903663
// BM_MatrixTransposeVectorMultiplyDynamic/1/10         18 ns         18 ns   34838177
// BM_MatrixTransposeVectorMultiplyDynamic/1/12         20 ns         20 ns   36138731
// BM_MatrixTransposeVectorMultiplyDynamic/1/16         23 ns         23 ns   27063704
// BM_MatrixTransposeVectorMultiplyDynamic/1/20         29 ns         29 ns   23400336
// BM_MatrixTransposeVectorMultiplyDynamic/2/1           6 ns          6 ns  121572101
// BM_MatrixTransposeVectorMultiplyDynamic/2/2           8 ns          8 ns   82896155
// BM_MatrixTransposeVectorMultiplyDynamic/2/3          12 ns         12 ns   56705415
// BM_MatrixTransposeVectorMultiplyDynamic/2/4          14 ns         14 ns   51241509
// BM_MatrixTransposeVectorMultiplyDynamic/2/6          18 ns         18 ns   38377403
// BM_MatrixTransposeVectorMultiplyDynamic/2/8          25 ns         25 ns   28560121
// BM_MatrixTransposeVectorMultiplyDynamic/2/10         29 ns         29 ns   23608052
// BM_MatrixTransposeVectorMultiplyDynamic/2/12         33 ns         33 ns   20668478
// BM_MatrixTransposeVectorMultiplyDynamic/2/16         44 ns         44 ns   16335446
// BM_MatrixTransposeVectorMultiplyDynamic/2/20         53 ns         53 ns   13462315
// BM_MatrixTransposeVectorMultiplyDynamic/3/1           6 ns          6 ns  117031415
// BM_MatrixTransposeVectorMultiplyDynamic/3/2          10 ns         10 ns   71040747
// BM_MatrixTransposeVectorMultiplyDynamic/3/3          14 ns         14 ns   49453538
// BM_MatrixTransposeVectorMultiplyDynamic/3/4          17 ns         17 ns   39161935
// BM_MatrixTransposeVectorMultiplyDynamic/3/6          22 ns         22 ns   32118490
// BM_MatrixTransposeVectorMultiplyDynamic/3/8          28 ns         28 ns   25295689
// BM_MatrixTransposeVectorMultiplyDynamic/3/10         34 ns         34 ns   20900389
// BM_MatrixTransposeVectorMultiplyDynamic/3/12         39 ns         39 ns   17934922
// BM_MatrixTransposeVectorMultiplyDynamic/3/16         51 ns         51 ns   10000000
// BM_MatrixTransposeVectorMultiplyDynamic/3/20         64 ns         64 ns   10594824
// BM_MatrixTransposeVectorMultiplyDynamic/4/1           7 ns          7 ns   98903583
// BM_MatrixTransposeVectorMultiplyDynamic/4/2          13 ns         13 ns   57301899
// BM_MatrixTransposeVectorMultiplyDynamic/4/3          16 ns         16 ns   44622083
// BM_MatrixTransposeVectorMultiplyDynamic/4/4          18 ns         18 ns   39645007
// BM_MatrixTransposeVectorMultiplyDynamic/4/6          26 ns         26 ns   27239262
// BM_MatrixTransposeVectorMultiplyDynamic/4/8          33 ns         33 ns   20869171
// BM_MatrixTransposeVectorMultiplyDynamic/4/10         39 ns         39 ns   17169614
// BM_MatrixTransposeVectorMultiplyDynamic/4/12         47 ns         47 ns   15045286
// BM_MatrixTransposeVectorMultiplyDynamic/4/16         62 ns         62 ns   11437535
// BM_MatrixTransposeVectorMultiplyDynamic/4/20         77 ns         77 ns    8351428
void BM_MatrixTransposeVectorMultiplyDynamic(benchmark::State& state) {
  const int rows = state.range(0);
  const int cols = state.range(1);
  MatrixVectorMultiplyData data(rows, cols);
  const std::vector<double*> ptrs = data.ptrs();
  const int num_buffers = data.num_buffers();

  int i = 0;
  for (auto _ : state) {
    double* a_ptr = ptrs[3 * i];
    double* b_ptr = ptrs[3 * i + 1];
    double* c_ptr = ptrs[3 * i + 2];
    internal::MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
        a_ptr, rows, cols, c_ptr, b_ptr);
    i = (i + 1) % num_buffers;
  }
}

BENCHMARK(BM_MatrixTransposeVectorMultiplyDynamic)
->ArgPair(1, 1)
->ArgPair(1, 2)
->ArgPair(1, 3)
->ArgPair(1, 4)
->ArgPair(1, 6)
->ArgPair(1, 8)
->ArgPair(1, 10)
->ArgPair(1, 12)
->ArgPair(1, 16)
->ArgPair(1, 20)
->ArgPair(2, 1)
->ArgPair(2, 2)
->ArgPair(2, 3)
->ArgPair(2, 4)
->ArgPair(2, 6)
->ArgPair(2, 8)
->ArgPair(2, 10)
->ArgPair(2, 12)
->ArgPair(2, 16)
->ArgPair(2, 20)
->ArgPair(3, 1)
->ArgPair(3, 2)
->ArgPair(3, 3)
->ArgPair(3, 4)
->ArgPair(3, 6)
->ArgPair(3, 8)
->ArgPair(3, 10)
->ArgPair(3, 12)
->ArgPair(3, 16)
->ArgPair(3, 20)
->ArgPair(4, 1)
->ArgPair(4, 2)
->ArgPair(4, 3)
->ArgPair(4, 4)
->ArgPair(4, 6)
->ArgPair(4, 8)
->ArgPair(4, 10)
->ArgPair(4, 12)
->ArgPair(4, 16)
->ArgPair(4, 20);

}  // namespace ceres

BENCHMARK_MAIN();
