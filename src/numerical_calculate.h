#pragma once

#include "sse_utilities.h"
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <xsimd/types/xsimd_batch.hpp>
#include <xsimd/xsimd.hpp>

namespace nc {
struct numerical_calculate {
  template <typename T>
  static auto plus_sse_array(T const *data, size_t count, size_t span,
                             T padding = T{}) {
    if (count % span != 0) {
      throw std::logic_error("count % patch != 0");
    }

    using batch_type = xsimd::batch<T>;
    auto const div = std::lldiv(span, batch_type::size);
    auto const n = count / span;
    auto const component = div.quot + (div.rem == 0 ? 0 : 1);

    std::vector<batch_type> result{};
    result.reserve(component);

    for (size_t i = 0; i < div.quot; ++i) {
      auto p = data + batch_type::size * i;
      batch_type batch = batch_type::load_unaligned(p);
      for (size_t j = 1; j < n; ++j) {
        batch += batch_type::load_unaligned(p + j * span);
      }
      result.push_back(batch);
    }

    if (div.rem) {
      auto p = data + div.quot * batch_type::size;
      auto batch = sse_utilities::make_batch(p, div.rem, padding);
      for (size_t j = 1; j < n; ++j) {
        batch += sse_utilities::make_batch(p + j * span, div.rem, padding);
      }
      result.push_back(batch);
    }

    return result;
  }
};
} // namespace nc
