/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace ML {
namespace DT {

template <typename DataT>
struct Quantiles {
  /** quantiles computed for each feature of dataset in col-major */
  DataT* quantiles_array;
  /** The number of distinct thresholds per feature; zero when all values are NaN.*/
  int* n_bins_array;
};

}  // namespace DT
}  // namespace ML
