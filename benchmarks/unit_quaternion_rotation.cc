#include <cstdint>
#include <cmath>
#include <benchmark/benchmark.h>

class nr3_rng
{
  uint64_t u;
  uint64_t v;
  uint64_t w;

public:

  nr3_rng() = delete;

  nr3_rng(uint64_t j) :
    v(4101842887655102017ull),
    w(1)
  {
    u = j ^ v;
    uint64();
    v = u;
    uint64();
    w = v;
    uint64();
  }

  uint64_t uint64()
  {
    u = u * 2862933555777941757ull + 7046029254386353087ull;
    v ^= v >> 17;
    v ^= v << 31;
    v ^= v >> 8;

    w = 4294957665u * (w & 0xffffffff) + (w >> 32);

    uint64_t x = u ^ (u << 21);
    x ^= x >> 35;
    x ^= x << 4;
    return (x + v) ^ w;
  }

  double uniformd()
  {
    return 5.42101086242752217E-20 * uint64();
  }

  uint32_t uint32()
  {
    return static_cast<uint32_t>(uint64());
  }

  float uniformf()
  {
    return 2.3283064365e-10f * uint32();
  }

  void quaternion_and_vecd(double q[4], double pt[3])
  {
    double x,y,z, u,v,w, s;
    do { x = 2.0 * uniformd() - 1.0; y = 2.0 * uniformd() - 1.0; z = x * x + y * y; } while ( z > 1 );
    do { u = 2.0 * uniformd() - 1.0; v = 2.0 * uniformd() - 1.0; w = x * x + y * y; } while ( w > 1 );
    s = std::sqrt((1.0 - z) / w);
    q[0] = x; q[1] = y; q[2] = s * u; q[3] = s * w;

    s = 2.0 * std::sqrt(1.0 - w);
    pt[0] = s * u;
    pt[1] = s * w;
    pt[2] = 1.0 - s - s;
  }

  void quaternion_and_vecf(float q[4], float pt[3])
  {
    float x,y,z, u,v,w, s;
    do { x = 2.0f * uniformf() - 1.0f; y = 2.0f * uniformf() - 1.0f; z = x * x + y * y; } while ( z > 1 );
    do { u = 2.0f * uniformf() - 1.0f; v = 2.0f * uniformf() - 1.0f; w = x * x + y * y; } while ( w > 1 );
    s = std::sqrt((1.0f - z) / w);
    q[0] = x; q[1] = y; q[2] = s * u; q[3] = s * w;

    s = 2.0 * std::sqrt(1.0f - w);
    pt[0] = s * u;
    pt[1] = s * w;
    pt[2] = 1.0f - s - s;
  }
};

template <typename T> inline
void UnitQuaternionRotatePoint(const T q[4], const T pt[3], T result[3]) {
  const T t2 =  q[0] * q[1];
  const T t3 =  q[0] * q[2];
  const T t4 =  q[0] * q[3];
  const T t5 = -q[1] * q[1];
  const T t6 =  q[1] * q[2];
  const T t7 =  q[1] * q[3];
  const T t8 = -q[2] * q[2];
  const T t9 =  q[2] * q[3];
  const T t1 = -q[3] * q[3];
  result[0] = T(2) * ((t8 + t1) * pt[0] + (t6 - t4) * pt[1] + (t3 + t7) * pt[2]) + pt[0];  // NOLINT
  result[1] = T(2) * ((t4 + t6) * pt[0] + (t5 + t1) * pt[1] + (t9 - t2) * pt[2]) + pt[1];  // NOLINT
  result[2] = T(2) * ((t7 - t3) * pt[0] + (t2 + t9) * pt[1] + (t5 + t8) * pt[2]) + pt[2];  // NOLINT
}

template <typename T> inline
void UnitQuaternionRotatePointEx(const T q[4], const T pt[3], T result[3])
{
  T uv0 = q[2] * pt[2] - q[3] * pt[1];
  T uv1 = q[3] * pt[0] - q[1] * pt[2];
  T uv2 = q[1] * pt[1] - q[2] * pt[0];
  uv0 += uv0;
  uv1 += uv1;
  uv2 += uv2;
  result[0] = pt[0] + q[0] * uv0;
  result[1] = pt[1] + q[0] * uv1;
  result[2] = pt[2] + q[0] * uv2;
  result[0] += q[2] * uv2 - q[3] * uv1;
  result[1] += q[3] * uv0 - q[1] * uv2;
  result[2] += q[1] * uv1 - q[2] * uv0;
}

static void DummyRNGDouble(benchmark::State& state) {
  // Code inside this loop is measured repeatedly
  uint64_t seed = 42ull;
  nr3_rng rng_state(seed);
  for (auto _ : state) {
    double q[4];
    double v[3];
    rng_state.quaternion_and_vecd(q, v);
    // Make sure the variable is not optimized away by compiler
    benchmark::DoNotOptimize(q);
    benchmark::DoNotOptimize(v);
  }
}
// Register the function as a benchmark
BENCHMARK(DummyRNGDouble);

static void CeresOrigUnitQRotDouble(benchmark::State& state) {
  // Code inside this loop is measured repeatedly
  uint64_t seed = 42ull;
  nr3_rng rng_state(seed);
  for (auto _ : state) {
    double q[4];
    double v[3];
    double r[3];
    rng_state.quaternion_and_vecd(q, v);
    UnitQuaternionRotatePoint<double>(q, v, r);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK(CeresOrigUnitQRotDouble);

static void CeresProposedUnitQRotDouble(benchmark::State& state) {
  // Code inside this loop is measured repeatedly
  uint64_t seed = 42ull;
  nr3_rng rng_state(seed);
  for (auto _ : state) {
    double q[4];
    double v[3];
    double r[3];
    rng_state.quaternion_and_vecd(q, v);
    UnitQuaternionRotatePointEx<double>(q, v, r);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK(CeresProposedUnitQRotDouble);

static void DummyRNGFloat(benchmark::State& state) {
  // Code inside this loop is measured repeatedly
  uint64_t seed = 42ull;
  nr3_rng rng_state(seed);
  for (auto _ : state) {
    float q[4];
    float v[3];
    rng_state.quaternion_and_vecf(q, v);
    // Make sure the variable is not optimized away by compiler
    benchmark::DoNotOptimize(q);
    benchmark::DoNotOptimize(v);
  }
}
// Register the function as a benchmark
BENCHMARK(DummyRNGFloat);

static void CeresOrigUnitQRotFloat(benchmark::State& state) {
  // Code inside this loop is measured repeatedly
  uint64_t seed = 42ull;
  nr3_rng rng_state(seed);
  for (auto _ : state) {
    float q[4];
    float v[3];
    float r[3];
    rng_state.quaternion_and_vecf(q, v);
    UnitQuaternionRotatePoint<float>(q, v, r);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK(CeresOrigUnitQRotFloat);

static void CeresProposedUnitQRotFloat(benchmark::State& state) {
  // Code inside this loop is measured repeatedly
  uint64_t seed = 42ull;
  nr3_rng rng_state(seed);
  for (auto _ : state) {
    float q[4];
    float v[3];
    float r[3];
    rng_state.quaternion_and_vecf(q, v);
    UnitQuaternionRotatePointEx<float>(q, v, r);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK(CeresProposedUnitQRotFloat);

BENCHMARK_MAIN();

