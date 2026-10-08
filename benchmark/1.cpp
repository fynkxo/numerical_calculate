// benchmark_plus_sse_array.cpp
#include <benchmark/benchmark.h>

#include <cstdint>
#include <random>
#include <vector>

#include <xsimd/xsimd.hpp>

#include <numerical_calculate.h>

// ---------- 运行期数据，防止常量折叠 ----------
template <typename T>
static std::vector<T> make_data(size_t n, uint64_t seed = 42) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<T> dist(T(-1), T(1));
  std::vector<T> v(n);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

// ---------- 消费 batch 结果 ----------
template <typename T>
static T consume_batches(const std::vector<xsimd::batch<T>> &bs) {
  using batch_t = xsimd::batch<T>;
  alignas(batch_t) T buf[batch_t::size];
  T sum = T(0);
  for (auto const &b : bs) {
    b.store_aligned(buf);
    for (size_t i = 0; i < batch_t::size; ++i)
      sum += buf[i];
  }
  return sum;
}

// ---------- 标量对照：同接口、内部标量 ----------
template <typename T>
static std::vector<T> scalar_dynamic(T const *data, size_t count, size_t span) {
  const size_t n = count / span;
  std::vector<T> r(span, T(0));
  for (size_t i = 0; i < span; ++i) {
    T s = T(0);
    for (size_t j = 0; j < n; ++j)
      s += data[j * span + i];
    r[i] = s;
  }
  return r;
}

// ---------- naive：运行期双循环、返回 vector<T> ----------
template <typename T>
static std::vector<T> naive_sum(T const *data, size_t count, size_t span) {
  const size_t n = count / span;
  std::vector<T> r(span, T(0));
  for (size_t i = 0; i < span; ++i) {
    T s = T(0);
    for (size_t j = 0; j < n; ++j)
      s += data[j * span + i];
    r[i] = s;
  }
  return r;
}

// ---------- 三个 benchmark ----------
template <typename T> static void BM_plus_sse_array(benchmark::State &state) {
  const size_t count = static_cast<size_t>(state.range(0));
  const size_t span = static_cast<size_t>(state.range(1));

  if (span == 0 || count % span != 0) {
    state.SkipWithError("count % span != 0");
    return;
  }

  auto data = make_data<T>(count);
  benchmark::DoNotOptimize(data);

  for (auto _ : state) {
    auto r = nc::numerical_calculate::plus_sse_array(data.data(), count, span);
    T s = consume_batches(r);
    benchmark::DoNotOptimize(s);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * count);
  state.SetBytesProcessed(state.iterations() * count * sizeof(T));
}

template <typename T> static void BM_scalar_dynamic(benchmark::State &state) {
  const size_t count = static_cast<size_t>(state.range(0));
  const size_t span = static_cast<size_t>(state.range(1));

  if (span == 0 || count % span != 0) {
    state.SkipWithError("count % span != 0");
    return;
  }

  auto data = make_data<T>(count);
  benchmark::DoNotOptimize(data);

  for (auto _ : state) {
    auto r = scalar_dynamic(data.data(), count, span);
    T s = T(0);
    for (auto x : r)
      s += x;
    benchmark::DoNotOptimize(s);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * count);
  state.SetBytesProcessed(state.iterations() * count * sizeof(T));
}

template <typename T> static void BM_naive_dynamic(benchmark::State &state) {
  const size_t count = static_cast<size_t>(state.range(0));
  const size_t span = static_cast<size_t>(state.range(1));

  if (span == 0 || count % span != 0) {
    state.SkipWithError("count % span != 0");
    return;
  }

  auto data = make_data<T>(count);
  benchmark::DoNotOptimize(data);

  for (auto _ : state) {
    auto r = naive_sum(data.data(), count, span);
    T s = T(0);
    for (auto x : r)
      s += x;
    benchmark::DoNotOptimize(s);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * count);
  state.SetBytesProcessed(state.iterations() * count * sizeof(T));
}

// ---------- 注册：显式列举合法 (count, span) ----------
// float + SSE: batch<float>::size = 4
//   span = 4 / 8 → 整除，走完整 batch 路径
//   span = 3      → 不整除，走 patch 路径
//
// double + SSE: batch<double>::size = 2
//   span = 2 / 4 → 整除
//   span = 3      → 不整除，走 patch 路径
//
// count 覆盖 L1(<32KB) / L2(<1MB) / L3 三种规模

#define REGISTER_FLOAT(NAME)                                                   \
  BENCHMARK_TEMPLATE(NAME, float)                                              \
      ->Args({64, 4})                                                          \
      ->Args({64, 8})                                                          \
      ->Args({256, 4})                                                         \
      ->Args({256, 8})                                                         \
      ->Args({4096, 4})                                                        \
      ->Args({4096, 8})                                                        \
      ->Args({48, 3})                                                          \
      ->Args({96, 3})                                                          \
      ->Args({192, 3})                                                         \
      ->Args({768, 3})                                                         \
      ->Unit(benchmark::kNanosecond);

#define REGISTER_DOUBLE(NAME)                                                  \
  BENCHMARK_TEMPLATE(NAME, double)                                             \
      ->Args({64, 2})                                                          \
      ->Args({64, 4})                                                          \
      ->Args({256, 2})                                                         \
      ->Args({256, 4})                                                         \
      ->Args({4096, 2})                                                        \
      ->Args({4096, 4})                                                        \
      ->Args({48, 3})                                                          \
      ->Args({96, 3})                                                          \
      ->Args({192, 3})                                                         \
      ->Args({768, 3})                                                         \
      ->Unit(benchmark::kNanosecond);

REGISTER_FLOAT(BM_plus_sse_array)
REGISTER_FLOAT(BM_scalar_dynamic)
REGISTER_FLOAT(BM_naive_dynamic)

REGISTER_DOUBLE(BM_plus_sse_array)
REGISTER_DOUBLE(BM_scalar_dynamic)
REGISTER_DOUBLE(BM_naive_dynamic)

BENCHMARK_MAIN();