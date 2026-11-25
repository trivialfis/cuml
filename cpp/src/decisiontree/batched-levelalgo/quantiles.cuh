/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "quantiles.h"

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/common/export.hpp>
#include <cuml/common/utils.hpp>

#include <raft/core/device_mdspan.hpp>  // for device_matrix_view
#include <raft/core/error.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/per_device_resource.hpp>  // for get_current_device_resource_ref

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda/memory_resource>  // for get_memory_resource
#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda/std/execution>  // for env, prop
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstddef>  // for size_t
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace ML {
namespace DT {

namespace detail {

template <typename T>
struct PrioritySample {
  std::uint64_t priority;
  T value;

  __host__ __device__ bool operator<(const PrioritySample& other) const
  { return priority < other.priority; }
};

// SplitMix64's bijective finalizer gives each global row a distinct priority within a feature.
__host__ __device__ inline std::uint64_t mixPriority(std::uint64_t x)
{
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__host__ __device__ inline std::uint64_t samplePriority(std::uint64_t seed,
                                                        unsigned col,
                                                        std::uint64_t global_row)
{
  const auto salt = mixPriority(seed + (std::uint64_t(col) + 1) * 0x9e3779b97f4a7c15ULL);
  return mixPriority(global_row ^ salt);
}

// One block per feature. Threads read the input once and reject candidates above the current
// cutoff in parallel. Only qualifying entries visit the bounded max-heap.
template <int Threads, typename T, typename ReadSample>
CUML_KERNEL void retainLowestPriorities(
  PrioritySample<T>* samples, int* counts, int capacity, int n_rows, ReadSample read_sample)
{
  auto col   = blockIdx.x;
  auto* heap = samples + std::size_t(col) * capacity;
  __shared__ PrioritySample<T> candidates[Threads];
  __shared__ int size;
  __shared__ std::uint64_t cutoff;
  if (threadIdx.x == 0) {
    size   = counts[col];
    cutoff = size == capacity ? heap[0].priority : UINT64_MAX;
  }
  __syncthreads();
  for (std::int64_t start = 0; start < n_rows; start += Threads) {
    auto row = start + threadIdx.x;
    PrioritySample<T> candidate{0, std::numeric_limits<T>::quiet_NaN()};
    if (row < n_rows) { candidate = read_sample(static_cast<int>(row), col); }
    bool eligible = !::cuda::std::isnan(candidate.value) && candidate.priority <= cutoff;
    if (!eligible) { candidate.value = std::numeric_limits<T>::quiet_NaN(); }
    candidates[threadIdx.x] = candidate;
    if (__syncthreads_or(eligible)) {
      if (threadIdx.x == 0) {
        for (const auto& entry : candidates) {
          if (::cuda::std::isnan(entry.value)) { continue; }
          if (size < capacity) {
            heap[size++] = entry;
            if (size == capacity) { ::cuda::std::make_heap(heap, heap + size); }
          } else if (entry.priority < heap[0].priority) {
            ::cuda::std::pop_heap(heap, heap + size);
            heap[size - 1] = entry;
            ::cuda::std::push_heap(heap, heap + size);
          }
        }
        if (size == capacity) { cutoff = heap[0].priority; }
      }
      __syncthreads();
    }
  }
  if (threadIdx.x == 0) { counts[col] = size; }
  // A peer can merge this fixed-size buffer directly, including underfull and empty samples.
  for (std::int64_t i = std::int64_t(size) + threadIdx.x; i < capacity; i += Threads) {
    heap[i] = {0, std::numeric_limits<T>::quiet_NaN()};
  }
}

}  // namespace detail

// Convert sorted per-column samples into quantile candidates and compact duplicate candidates.
template <typename T>
CUML_KERNEL void computeQuantilesBatchedKernel(
  T* quantiles, int* n_bins, const T* sorted_data, const int max_n_bins, int sample_stride)
{
  int col           = blockIdx.x;
  T* col_quantiles  = quantiles + static_cast<int64_t>(col) * max_n_bins;
  auto sample_count = n_bins[col];
  if (sample_count == 0) {
    for (int bin = threadIdx.x; bin < max_n_bins; bin += blockDim.x) {
      col_quantiles[bin] = T{0};
    }
    return;
  }
  const T* col_data = sorted_data + std::size_t(col) * sample_stride;
  double bin_width  = static_cast<double>(sample_count) / max_n_bins;

  for (int bin = threadIdx.x; bin < max_n_bins; bin += blockDim.x) {
    // get index by interpolation
    int idx            = int(round((bin + 1) * bin_width)) - 1;
    idx                = min(max(0, idx), sample_count - 1);
    col_quantiles[bin] = col_data[idx];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    // make quantiles unique, in-place
    // thrust::seq to explicitly disable cuda dynamic parallelism here
    auto new_last = thrust::unique(thrust::seq, col_quantiles, col_quantiles + max_n_bins);
    // get the unique count
    n_bins[col] = new_last - col_quantiles;
  }

  __syncthreads();
  return;
}

template <typename T>
struct QuantileResult {
  rmm::device_uvector<T> quantiles_array;
  rmm::device_uvector<int> n_bins_array;

  Quantiles<T> view() & { return {quantiles_array.data(), n_bins_array.data()}; }
  Quantiles<T> view() && = delete;
};

/**
 * @brief Compute per-feature quantile split candidates from uniformly sampled rows.
 *
 * Each feature retains the lowest random priorities among its non-NaN rows across all ranks.
 * Global row IDs make the sample independent of rank boundaries. Sampling is without replacement.
 * If the budget covers all observed values, each is used once. An all-NaN feature has zero bins.
 *
 * @tparam T Floating-point input type.
 * @param handle RAFT handle used for stream and resource access.
 * @param data Input matrix of shape `[n_rows, n_cols]` in the storage order given by `row_major`.
 * @param max_n_bins Maximum number of quantile candidates to retain per feature.
 * @param n_rows Number of local rows in `data` for this rank.
 * @param n_cols Number of columns in `data`.
 * @param oversampling_factor Multiplier applied to `max_n_bins` to choose the
 * sampled row budget per feature before sorting and quantile extraction. The
 * default of 4 is a conservative choice while still bounding memory; for fixed
 * `max_n_bins`, rank error decreases like O(1 / sqrt(oversampling_factor)), so
 * returns from increasing this are strongly diminishing.
 * @param seed User seed for deterministic sampling.
 * @param row_major Whether `data` is row-major instead of column-major.
 * @return Quantile metadata and owning device buffers for quantile values and bin counts.
 */
template <typename T>
CUML_EXPORT QuantileResult<T> computeQuantiles(const raft::handle_t& handle,
                                               const T* data,
                                               int max_n_bins,
                                               int n_rows,
                                               int n_cols,
                                               int oversampling_factor = 4,
                                               uint64_t seed           = uint64_t{0},
                                               bool row_major          = false)
{
  raft::common::nvtx::push_range("computeQuantiles");
  auto stream      = handle.get_stream().get();
  bool distributed = raft::resource::comms_initialized(handle) && handle.get_comms().get_size() > 1;

  RAFT_EXPECTS(max_n_bins > 0, "max_n_bins must be positive");
  RAFT_EXPECTS(distributed ? n_rows >= 0 : n_rows > 0, "n_rows must be positive");
  RAFT_EXPECTS(n_rows == 0 || data != nullptr, "data pointer must not be null");
  RAFT_EXPECTS(n_cols > 0, "n_cols must be positive");
  RAFT_EXPECTS(oversampling_factor > 0, "oversampling_factor must be positive");
  int rank      = distributed ? handle.get_comms().get_rank() : 0;
  int comm_size = distributed ? handle.get_comms().get_size() : 1;

  std::uint64_t first_row   = 0;
  std::uint64_t global_rows = n_rows;
  if (distributed) {
    rmm::device_uvector<int> local_rows(1, stream);
    rmm::device_uvector<int> gathered(comm_size, stream);
    raft::update_device(local_rows.data(), &n_rows, 1, stream);
    handle.get_comms().allgather(local_rows.data(), gathered.data(), 1, stream);
    std::vector<int> rows(comm_size);
    raft::update_host(rows.data(), gathered.data(), rows.size(), stream);
    ASSERT(handle.get_comms().sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred gathering RF quantile row counts.");
    first_row   = std::accumulate(rows.begin(), rows.begin() + rank, std::uint64_t{0});
    global_rows = std::accumulate(rows.begin() + rank, rows.end(), first_row);
  }
  RAFT_EXPECTS(global_rows > 0, "global row count must be positive");
  auto budget       = ML::checked_mul<std::uint64_t>(max_n_bins, oversampling_factor);
  int capacity      = ML::narrow_cast<int>(std::min(global_rows, budget));
  auto sample_slots = ML::checked_mul<std::size_t>(n_cols, capacity);
  using Sample      = detail::PrioritySample<T>;
  rmm::device_uvector<Sample> samples(sample_slots, stream);
  rmm::device_uvector<int> n_bins_array(n_cols, stream);
  rmm::device_uvector<T> quantiles_array(ML::checked_mul<std::size_t>(n_cols, max_n_bins), stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(n_bins_array.data(), 0, sizeof(int) * std::size_t(n_cols), stream));

  auto input = row_major
                 ? raft::make_device_strided_matrix_view<const T, std::size_t, raft::row_major>(
                     data, n_rows, n_cols, n_cols)
                 : raft::make_device_strided_matrix_view<const T, std::size_t, raft::col_major>(
                     data, n_rows, n_cols, std::max(n_rows, 1));
  auto from_input = [=] __device__(int row, unsigned col) {
    return Sample{detail::samplePriority(seed, col, first_row + row), input(row, col)};
  };
  constexpr int threads = 128;
  detail::retainLowestPriorities<threads><<<n_cols, threads, 0, stream>>>(
    samples.data(), n_bins_array.data(), capacity, n_rows, from_input);
  RAFT_CUDA_TRY(cudaGetLastError());

  if (distributed) {
    // Reduce up a binary tree. A receiver merges one peer's K candidates per feature at a time.
    const auto& comms = handle.get_comms();
    rmm::device_uvector<Sample> peer(rank % 2 == 0 && rank + 1 < comm_size ? sample_slots : 0,
                                     stream);
    for (std::int64_t stride = 1; stride < comm_size; stride *= 2) {
      if (rank & stride) {
        comms.device_send(samples.data(), samples.size(), int(rank - stride), stream);
        break;
      }
      if (rank + stride < comm_size) {
        // An unpaired last rank only sends at a later level; all receivers have scratch.
        comms.device_recv(peer.data(), peer.size(), int(rank + stride), stream);
        auto from_peer = [ptr = peer.data(), capacity] __device__(int row, unsigned col) {
          return ptr[std::size_t(col) * capacity + row];
        };
        detail::retainLowestPriorities<threads><<<n_cols, threads, 0, stream>>>(
          samples.data(), n_bins_array.data(), capacity, capacity, from_peer);
        RAFT_CUDA_TRY(cudaGetLastError());
      }
    }
  }

  if (rank == 0) {
    // Sort only the occupied prefix of each feature's sample.
    rmm::device_uvector<T> values(sample_slots, stream);
    thrust::transform(rmm::exec_policy_nosync(stream),
                      samples.begin(),
                      samples.end(),
                      values.begin(),
                      [] __host__ __device__(Sample sample) { return sample.value; });
    samples.release();  // Priorities can be freed before allocating sorting workspace.
    rmm::device_uvector<T> sorted(sample_slots, stream);
    auto columns = thrust::make_counting_iterator<std::int64_t>(0);
    auto begins  = thrust::make_transform_iterator(
      columns, [capacity] __host__ __device__(std::int64_t col) { return col * capacity; });
    const int* sizes = n_bins_array.data();
    auto ends        = thrust::make_transform_iterator(
      columns, [capacity, sizes] __host__ __device__(std::int64_t col) {
        return col * capacity + sizes[col];
      });
    auto env = cuda::std::execution::env{
      handle.get_stream(),
      cuda::std::execution::prop{cuda::mr::get_memory_resource,
                                 rmm::mr::get_current_device_resource_ref()}};
    RAFT_CUDA_TRY(
      cub::DeviceSegmentedRadixSort::SortKeys(values.data(),
                                              sorted.data(),
                                              ML::narrow_cast<std::int64_t>(sample_slots),
                                              static_cast<std::int64_t>(n_cols),
                                              begins,
                                              ends,
                                              0,
                                              8 * sizeof(T),
                                              env));
    computeQuantilesBatchedKernel<<<n_cols, std::min(1024, max_n_bins), 0, stream>>>(
      quantiles_array.data(), n_bins_array.data(), sorted.data(), max_n_bins, capacity);
    RAFT_CUDA_TRY(cudaGetLastError());
  }
  if (distributed) {
    const auto& comms = handle.get_comms();
    comms.bcast(quantiles_array.data(), quantiles_array.size(), 0, stream);
    comms.bcast(n_bins_array.data(), n_bins_array.size(), 0, stream);
    ASSERT(comms.sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred distributing RF quantiles.");
  }
  handle.sync_stream(stream);

  raft::common::nvtx::pop_range();
  return {std::move(quantiles_array), std::move(n_bins_array)};
}

}  // namespace DT
}  // namespace ML
