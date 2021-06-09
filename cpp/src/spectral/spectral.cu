/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cuml/manifold/common.hpp>
#include <cuml/manifold/spectral.hpp>

#include <raft/sparse/coo.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/linalg/spectral.cuh>
#include <raft/spatial/knn/knn.hpp>

#include <rmm/exec_policy.hpp>

namespace raft {
class handle_t;
}

namespace ML {

namespace Spectral {

/**
   * Given a COO formatted (symmetric) knn graph, this function
   * computes the spectral embeddings (lowest n_components
   * eigenvectors), using Lanczos min cut algorithm.
   * @param rows source vertices of knn graph (size nnz)
   * @param cols destination vertices of knn graph (size nnz)
   * @param vals edge weights connecting vertices of knn graph (size nnz)
   * @param nnz size of rows/cols/vals
   * @param n number of samples in X
   * @param n_neighbors the number of neighbors to query for knn graph construction
   * @param n_components the number of components to project the X into
   * @param out output array for embedding (size n*n_comonents)
   */
void fit_embedding(const raft::handle_t &handle, int *rows, int *cols,
                   float *vals, int nnz, int n, int n_components, float *out,
                   unsigned long long seed) {
  raft::sparse::spectral::fit_embedding(handle, rows, cols, vals, nnz, n,
                                        n_components, out, seed);
}

void fit_embedding_with_knn(raft::handle_t const &handle, int n,
                            int32_t *knn_indices, float *knn_dists,
                            int n_components, int n_neighbors, float *out,
                            uint64_t seed) {
  using value_t = float;
  using value_idx = int32_t;
  raft::sparse::COO<value_t, value_idx> knn_coo(handle.get_device_allocator(),
                                                handle.get_stream());
  raft::sparse::linalg::from_knn_symmetrize_matrix<value_idx, value_t>(
    knn_indices, knn_dists, n, n_neighbors, &knn_coo, handle.get_stream(),
    handle.get_device_allocator());

  raft::sparse::spectral::fit_embedding(
    handle, /*rows=*/knn_coo.rows(), /*cols=*/knn_coo.cols(),
    /*vals=*/knn_coo.vals(),
    /*nnz=*/knn_coo.nnz, /*n=*/(value_idx)n, n_components, out, seed);
}

void fit_embedding(raft::handle_t const &handle, float *X, int n_samples,
                   int n_features, int n_neighbors, int n_components,
                   float *out, uint64_t seed) {
  manifold_dense_inputs_t<float> inputs(X, nullptr, n_samples, n_features);

  rmm::device_uvector<knn_indices_dense_t> knn_indices(
    n_samples * n_neighbors, handle.get_stream_view());
  rmm::device_uvector<float> knn_dists(n_samples * n_neighbors,
                                       handle.get_stream_view());
  knn_graph<knn_indices_dense_t, float> knn(n_samples, n_neighbors, knn_indices.data(), knn_dists.data());

  std::vector<float *> ptrs(1);
  std::vector<int> sizes(1);
  ptrs[0] = inputs.X;
  sizes[0] = inputs.n;

  raft::spatial::knn::brute_force_knn(handle, ptrs, sizes, n_features, inputs.X,
                                      inputs.n, knn.knn_indices, knn.knn_dists,
                                      n_neighbors);

  rmm::device_uvector<int> knn_indices_i32(n_samples * n_neighbors,
                                           handle.get_stream());
  thrust::copy_n(rmm::exec_policy(handle.get_stream_view()), knn.knn_dists,
                 n_samples * n_neighbors, knn_indices.begin());
  fit_embedding_with_knn(handle, n_samples, knn_indices_i32.data(),
                         knn.knn_dists, n_components, n_neighbors, out, seed);
}
}  // namespace Spectral
}  // namespace ML
