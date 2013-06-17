#include "ceres/incomplete_qr_factorization.h"
#include "ceres/internal/scoped_ptr.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

void ExpectMatricesAreEqual(const CRSMatrix& expected, const CRSMatrix& actual,
                            const double tolerance) {
  EXPECT_EQ(expected.num_rows, actual.num_rows);
  EXPECT_EQ(expected.num_cols, actual.num_cols);
  EXPECT_EQ(expected.rows, actual.rows);
  EXPECT_EQ(expected.cols, actual.cols);
  for (int i = 0; i < actual.values.size(); ++i) {
    EXPECT_NEAR(expected.values[i], actual.values[i], tolerance);
  }
}

TEST(IncompleteQRFactorization, OneByOneMatrix) {
  CRSMatrix matrix;
  matrix.rows.resize(2);
  matrix.rows[0] = 0;
  matrix.rows[1] = 1;
  matrix.cols.resize(1);
  matrix.cols[0] = 0;
  matrix.values.resize(1);
  matrix.values[0] = 2;
  matrix.num_rows = 1;
  matrix.num_cols = 1;

  scoped_ptr<CRSMatrix> l(IncompleteQRFactorization(matrix, 1, 0.0, 1, 0.0));
  ExpectMatricesAreEqual(matrix, *l, 1e-16);
}

TEST(IncompleteQRFactorization, CompleteFactorization) {
  CRSMatrix matrix;
  matrix.rows.resize(4);
  matrix.rows[0] = 0;
  matrix.rows[1] = 4;
  matrix.rows[2] = 8;
  matrix.rows[3] = 12;

  matrix.cols.resize(12);
  matrix.values.resize(12);

  matrix.cols[0] = 0;
  matrix.cols[1] = 1;
  matrix.cols[2] = 2;
  matrix.cols[3] = 3;
  matrix.cols[4] = 0;
  matrix.cols[5] = 1;
  matrix.cols[6] = 2;
  matrix.cols[7] = 3;
  matrix.cols[8] = 0;
  matrix.cols[9] = 1;
  matrix.cols[10] = 2;
  matrix.cols[11] = 3;

  matrix.values[0] = 1;
  matrix.values[1] = 4;
  matrix.values[2] = 3;
  matrix.values[3] = 10;
  matrix.values[4] = 2;
  matrix.values[5] = 8;
  matrix.values[6] = 9;
  matrix.values[7] = 12;
  matrix.values[8] = 3;
  matrix.values[9] = 16;
  matrix.values[10] = 27;
  matrix.values[11] = 16;


  matrix.num_rows = 3;
  matrix.num_cols = 4;

  scoped_ptr<CRSMatrix> l(IncompleteQRFactorization(matrix, 3, 0.0, 4, 0.0));
  CRSMatrix expected;
  expected.num_rows = 3;
  expected.num_cols = 3;

  expected.rows.resize(4);
  expected.cols.resize(6);
  expected.values.resize(6);

  expected.rows[0] = 0;
  expected.rows[1] = 1;
  expected.rows[2] = 3;
  expected.rows[3] = 6;
  expected.cols[0] = 0;
  expected.values[0] = 11.22497;
  expected.cols[1] = 0;
  expected.values[1] = 16.12476;
  expected.cols[2] = 1;
  expected.values[2] = 5.74387;
  expected.cols[3] = 0;
  expected.values[3] = 27.43882;
  expected.cols[4] = 1;
  expected.values[4] = 22.03314;
  expected.cols[5] = 2;
  expected.values[5] = 3.41345;
  ExpectMatricesAreEqual(expected, *l, 1e-4);
}

}  // namespace internal
}  // namespace ceres
