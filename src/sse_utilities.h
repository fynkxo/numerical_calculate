#pragma once

#include <algorithm>
#include <array>
#include <xsimd/types/xsimd_batch.hpp>
namespace nc {
struct sse_utilities {
  template <typename T>
  static auto make_batch(T const *p, size_t count, T padding = {}) {
    using batch_t = xsimd::batch<T>;
    using Buf = std::array<T, batch_t::size>;

    if (count >= batch_t::size) {
      return batch_t::load_unaligned(p);
    }

    Buf buf{};
    std::copy_n(p, count, buf.begin());
    std::fill_n(std::next(buf.begin(), count), batch_t::size - count, padding);
    return batch_t::load_unaligned(buf.data());
  }
};
} // namespace nc
