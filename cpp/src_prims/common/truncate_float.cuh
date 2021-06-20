/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cmath>
#include <raft/cuda_utils.cuh>

namespace MLCommon {
/**
 * \brief Constructs a rounding factor used to truncate elements in a sum such that the
 * sum of the truncated elements is the same no matter what the order of the sum is.
 *
 * Algorithm 5: Reproducible Sequential Sum in 'Fast Reproducible Floating-Point
 * Summation' by Demmel and Nguyen
 *
 * In algorithm 5 the bound is calculated as $max(|v_i|) * n$.
 *
 * The calculation trick is borrowed from fbcuda, which is BSD-licensed.
 */
template <typename T>
T create_rounding_factor(T max_abs, int n) {
  T delta =
    max_abs / (static_cast<T>(1.0) -
               static_cast<T>(2.0) * n * std::numeric_limits<T>::epsilon());

  // Calculate ceil(log_2(delta)).
  // frexpf() calculates exp and returns `x` such that
  // delta = x * 2^exp, where `x` in (-1.0, -0.5] U [0.5, 1).
  // Because |x| < 1, exp is exactly ceil(log_2(delta)).
  int exp;
  std::frexp(delta, &exp);

  // return M = 2 ^ ceil(log_2(delta))
  return std::ldexp(static_cast<T>(1.0), exp);
}

template <typename T>
DI T truncate_float(T const rounding_factor, T const x) {
  return (rounding_factor + x) - rounding_factor;
}
}  // namespace MLCommon
