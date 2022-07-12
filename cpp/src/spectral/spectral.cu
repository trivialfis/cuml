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

#include <raft/core/mdarray.hpp>
#include <raft/linalg/transpose.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/spectral.hpp>
#include <raft/sparse/linalg/symmetrize.hpp>
#include <raft/sparse/op/detail/filter.cuh>
#include <raft/spatial/knn/knn.cuh>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

namespace ML {
namespace Spectral {

/**
 * \brief Get diagonal values from a COO matrix
 */
template <typename index_type, typename value_type>
auto Diagonal(raft::handle_t const& handle,
              index_type const* rows,
              index_type const* cols,
              value_type const* values,
              size_t nnz,
              size_t n) -> rmm::device_uvector<value_type>
{
  auto policy  = handle.get_thrust_policy();
  auto it      = thrust::make_zip_iterator(rows, cols, values);
  using Triple = thrust::tuple<index_type, index_type, value_type>;
  rmm::device_uvector<value_type> results(n, handle.get_stream());
  auto d_results = results.data();
  thrust::for_each(policy, it, it + nnz, [=] __host__ __device__(Triple const& triple) {
    if (thrust::get<0>(triple) == thrust::get<1>(triple)) {
      d_results[thrust::get<0>(triple)] = thrust::get<2>(triple);
    }
  });
  return results;
}

template <typename index_type, typename value_type>
cusparseStatus_t cuSparseCreateCOO(cusparseSpMatDescr_t* desc,
                                   int64_t n_rows,
                                   int64_t n_cols,
                                   int64_t nnz,
                                   index_type* rows,
                                   index_type* cols,
                                   value_type* vals);

template <>
cusparseStatus_t cuSparseCreateCOO(cusparseSpMatDescr_t* desc,
                                   int64_t n_rows,
                                   int64_t n_cols,
                                   int64_t nnz,
                                   int32_t* rows,
                                   int32_t* cols,
                                   float* vals)
{
  return cusparseCreateCoo(desc,
                           n_rows,
                           n_cols,
                           nnz,
                           rows,
                           cols,
                           vals,
                           CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_32F);
}

template <typename offset_t, typename index_t, typename value_t>
struct CSR {
  rmm::device_uvector<offset_t> indptr;
  rmm::device_uvector<index_t> cols;
  rmm::device_uvector<value_t> vals;
  raft::handle_t const& handle_;
  index_t n_cols;

  CSR(raft::handle_t const& handle, size_t n_rows, size_t _n_cols, size_t nnz)
    : indptr{n_rows + 1, handle.get_stream()},
      cols{nnz, handle.get_stream()},
      vals{nnz, handle.get_stream()},
      handle_{handle},
      n_cols{index_t(_n_cols)}
  {
  }

  explicit operator raft::spectral::matrix::sparse_matrix_t<index_t, value_t>()
  {
    raft::spectral::matrix::sparse_matrix_t<index_t, value_t> m{
      handle_,
      indptr.data(),
      cols.data(),
      vals.data(),
      index_t(indptr.is_empty() ? 0ul : indptr.size() - 1),
      n_cols,
      index_t(vals.size())};
    return m;
  }
};

template <typename index_t, typename value_t>
struct coo_view_t {
  index_t* rows;
  index_t* cols;
  value_t* vals;
  size_t nnz;
  size_t n_rows;
  size_t n_cols;

  coo_view_t(raft::sparse::COO<value_t, index_t>& coo)
  {
    rows   = coo.rows();
    cols   = coo.cols();
    vals   = coo.vals();
    nnz    = coo.nnz;
    n_rows = coo.n_rows;
    n_cols = coo.n_cols;
  }

  void toCSR(raft::handle_t const& handle, CSR<index_t, index_t, value_t>* out)
  {
    out->indptr.resize(n_rows + 1, handle.get_stream());
    out->cols.resize(nnz, handle.get_stream());
    out->vals.resize(nnz, handle.get_stream());
    raft::sparse::convert::coo_to_csr(handle,
                                      rows,
                                      cols,
                                      vals,
                                      nnz,
                                      n_rows,
                                      out->indptr.data(),
                                      out->cols.data(),
                                      out->vals.data());
  }
};

template <typename index_type, typename value_type>
struct laplacian_matrix_t : public raft::sparse::COO<value_type, index_type> {
  template <typename ThrustExePolicy>
  laplacian_matrix_t(raft::handle_t const& handle,
                     ThrustExePolicy policy,
                     index_type const* _rows,
                     index_type const* _cols,
                     value_type const* _vals,
                     size_t n_rows,
                     size_t nnz)
    : raft::sparse::COO<value_type, index_type>{handle.get_stream(),
                                                index_type(nnz),
                                                index_type(n_rows),
                                                index_type(n_rows),
                                                false},
      handle_{handle},
      diagonal_(n_rows, handle.get_stream())
  {
    rmm::device_uvector<value_type> ones{n_rows, handle.get_stream()};
    ASSERT(this->n_rows != 0, "Invalid shape");
    thrust::copy(policy, _rows, _rows + nnz, this->rows());
    thrust::copy(policy, _cols, _cols + nnz, this->cols());
    thrust::copy(policy, _vals, _vals + nnz, this->vals());

    thrust::fill(policy, ones.begin(), ones.end(), 1.0f);
    diagonal_ = ::ML::Spectral::Diagonal(handle, _rows, _cols, _vals, nnz, n_rows);
    // calcuate the degree matrix
    this->SpMV(1, ones.data(), 0, diagonal_.data());
    // normalize it.
    this->Normalize(policy, handle, diagonal_);

    ASSERT(this->rows_arr.size() == this->cols_arr.size(), "Check");
    ASSERT(this->rows_arr.size() == this->vals_arr.size(), "Check");
  }

  void SpMV(value_type alpha,
            value_type* __restrict__ x,
            value_type beta,
            value_type* __restrict__ y)
  {
    auto cusparse_h = this->handle_.get_cusparse_handle();
    auto stream     = this->handle_.get_stream();

    cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpMatDescr_t matA;
    ASSERT(this->n_rows != 0, "Invalid shape");
    CUSPARSE_CHECK(cuSparseCreateCOO(
      &matA, this->n_rows, this->n_rows, NNZ(), this->rows(), this->cols(), this->vals()));

    cusparseDnVecDescr_t vec_X;
    CUSPARSE_CHECK(raft::sparse::detail::cusparsecreatednvec(&vec_X, this->n_rows, x));

    cusparseDnVecDescr_t vec_y;
    CUSPARSE_CHECK(raft::sparse::detail::cusparsecreatednvec(&vec_y, this->n_rows, y));

    size_t buffer_size;
    CUSPARSE_CHECK(raft::sparse::detail::cusparsespmv_buffersize(cusparse_h,
                                                                 trans,
                                                                 &alpha,
                                                                 matA,
                                                                 vec_X,
                                                                 &beta,
                                                                 vec_y,
                                                                 CUSPARSE_COOMV_ALG,
                                                                 &buffer_size,
                                                                 stream));
    rmm::device_uvector<value_type> external_buffer(buffer_size, stream);
    CUSPARSE_CHECK(raft::sparse::detail::cusparsespmv(cusparse_h,
                                                      trans,
                                                      &alpha,
                                                      matA,
                                                      vec_X,
                                                      &beta,
                                                      vec_y,
                                                      CUSPARSE_COOMV_ALG,
                                                      external_buffer.data(),
                                                      stream));

    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_X));
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
  }

  template <typename ThrustExePolicy>
  void Normalize(ThrustExePolicy policy,
                 raft::handle_t const& handle,
                 rmm::device_uvector<value_type> const& diag)
  {
    auto it         = thrust::make_counting_iterator(0ul);
    auto d_diagonal = diagonal_.data();
    auto d_orig     = diag.data();
    thrust::for_each(
      policy, it, it + this->n_rows, [=] HD(size_t i) { d_diagonal[i] -= d_orig[i]; });

    rmm::device_uvector<int32_t> isolated_node_mask(diagonal_.size(), handle.get_stream());
    thrust::transform(
      policy, diagonal_.begin(), diagonal_.end(), isolated_node_mask.begin(), [] HD(float w) {
        return int32_t(w == 0);
      });
    auto d_mask = isolated_node_mask.data();
    thrust::for_each(policy, it, it + this->n_rows, [=] HD(size_t i) {
      if (d_diagonal[i] == 0) {
        d_diagonal[i] = 1;
      } else {
        d_diagonal[i] = sqrt(d_diagonal[i]);
      }
    });

    auto d_rows = this->rows();
    auto d_cols = this->cols();
    auto d_vals = this->vals();
    thrust::for_each(policy, it, it + this->NNZ(), [=] HD(size_t i) {
      d_vals[i] /= d_diagonal[d_rows[i]];
      d_vals[i] /= d_diagonal[d_cols[i]];
      d_vals[i] *= -1;

      if (d_rows[i] == d_cols[i]) { d_vals[i] = 1.0 - value_type(d_mask[d_rows[i]]); }
    });
  }

  size_t NNZ() const { return this->vals_arr.size(); }
  auto Diagonal() const
  {
    return raft::make_device_vector_view(diagonal_.data(), diagonal_.size());
  }

 private:
  raft::handle_t const& handle_;
  rmm::device_uvector<value_type> diagonal_;
};

template <typename vertex_t, typename weight_t, typename EigenSolver>
void Partition(raft::handle_t const& handle,
               vertex_t const* rows,
               vertex_t const* cols,
               weight_t const* vals,
               size_t n_samples,
               size_t nnz,
               size_t n_components,
               EigenSolver const& eigen_solver,
               weight_t* eigVals,
               weight_t* eig_vecs)
{
  auto cublas_h = handle.get_cublas_handle();
  auto stream   = handle.get_stream();
  auto policy   = handle.get_thrust_policy();

  laplacian_matrix_t<vertex_t, weight_t> laplacian{
    handle, policy, rows, cols, vals, n_samples, nnz};
  thrust::transform(policy,
                    laplacian.vals(),
                    laplacian.vals() + laplacian.NNZ(),
                    laplacian.vals(),
                    [] HD(weight_t v) { return -v; });
  auto it = thrust::make_counting_iterator(0ul);

  CSR<vertex_t, vertex_t, weight_t> csr(handle, n_samples, n_samples, nnz);
  coo_view_t<vertex_t, weight_t>{laplacian}.toCSR(handle, &csr);

  auto r_csr_m = raft::spectral::matrix::sparse_matrix_t<vertex_t, weight_t>(csr);
  eigen_solver.solve_largest_eigenvectors(handle, r_csr_m, eigVals, eig_vecs);

  raft::device_matrix_view<float> eig_vectors(eig_vecs, n_components, n_samples);
  auto permutation = raft::make_device_matrix<float>(handle, n_components, n_samples);

  it          = thrust::make_counting_iterator(0ul);
  auto result = permutation.view();
  thrust::for_each(policy, it, it + n_samples * n_components, [=] __device__(size_t i) {
    size_t ridx        = i / n_samples;
    size_t src_ridx    = n_components - 1 - ridx;
    size_t cidx        = i % n_samples;
    result(ridx, cidx) = eig_vectors(src_ridx, cidx);
  });
  CUDA_CHECK(cudaMemcpyAsync(eig_vectors.data(),
                             permutation.data(),
                             permutation.size() * sizeof(float),
                             cudaMemcpyDefault,
                             stream));

  auto d_diagonal = laplacian.Diagonal();
  it              = thrust::make_counting_iterator(0ul);
  thrust::for_each(policy, it, it + n_samples * n_components, [=] HD(size_t i) {
    size_t cidx = i % n_samples;
    assert(d_diagonal[cidx] != 0);
    eig_vecs[i] /= d_diagonal[cidx];
  });

  {
    // deterministic vector sign flip
    auto abs_it =
      thrust::make_transform_iterator(eig_vecs, [=] HD(float v) -> float { return std::abs(v); });
    auto key_it =
      thrust::make_transform_iterator(it, [=] HD(size_t i) -> size_t { return i % n_samples; });
    auto signs = raft::make_device_vector<float>(n_components, stream);

    thrust::reduce_by_key(policy,
                          key_it,
                          key_it + n_components * n_samples,
                          abs_it,
                          thrust::make_discard_iterator(),
                          signs.data(),
                          thrust::equal_to<size_t>{},
                          thrust::maximum<float>{});
    thrust::transform(
      policy, signs.data(), signs.data() + signs.size(), signs.data(), [] __device__(float v) {
        return (v >= 0) ? 1.0f : -1.0f;
      });
    auto signs_v = signs.view();
    thrust::for_each(policy, it, it + n_components * n_samples, [=] __device__(size_t i) {
      auto ridx = i / n_samples;
      eig_vecs[i] *= signs_v(ridx);
    });
  }
}

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
void fit_embedding(const raft::handle_t& handle,
                   int* rows,
                   int* cols,
                   float* vals,
                   int nnz,
                   int n,
                   int n_components,
                   float* out,
                   unsigned long long seed)
{
  ASSERT(n > 0, "Expect number of samples to be greater than 0.");
  ASSERT(n_components > 0, "Expect number of components to be greater than 0.");
  ASSERT(out, "Invalid pointer for embedding output.");

  auto stream   = handle.get_stream();
  auto policy   = handle.get_thrust_policy();

  using index_type = std::int32_t;
  using value_type = float;

  index_type neigvs       = n_components + 1;
  index_type maxiter      = 5000;  // default reset value (when set to 0);
  value_type tol          = 0.001;
  index_type restart_iter = 15 + neigvs;  // what cugraph is using

  raft::spectral::eigen_solver_config_t<index_type, value_type> cfg{
    neigvs, maxiter, restart_iter, tol, true, 0};
  cfg.reorthogonalize = true;
  raft::spectral::lanczos_solver_t<index_type, value_type> eig_solver{cfg};

  rmm::device_uvector<value_type> eigen_values(neigvs, stream);
  rmm::device_uvector<value_type> eigen_vectors(neigvs * n, stream);

  Partition<index_type, value_type>(handle,
                                    rows,
                                    cols,
                                    vals,
                                    size_t(n),
                                    size_t(nnz),
                                    size_t(neigvs),
                                    eig_solver,
                                    eigen_values.data(),
                                    eigen_vectors.data());

  auto map_v      = raft::make_device_matrix_view(eigen_vectors.data(), neigvs, n);
  auto drop_first = raft::detail::stdex::submdspan(
    map_v, std::make_tuple(1, neigvs), raft::detail::stdex::full_extent);
  ASSERT(drop_first.size() == static_cast<size_t>(n_components * n), "Invalid shape of eigen map.");

  auto tran   = raft::make_device_matrix<float>(n_components, n, stream);
  auto tran_v = tran.view();

  auto it = thrust::make_counting_iterator(0ul);
  thrust::for_each_n(policy, it, n_components * n, [=] HD(size_t i) {
    auto coord = raft::unravel_index(i, drop_first.extents(), raft::detail::stdex::layout_right{});
    std::apply(tran_v, coord) = std::apply(drop_first, coord);
  });

  {
    float one  = 1;
    float zero = 0;
    CUBLAS_CHECK(cublasSgeam(handle.get_cublas_handle(),
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             n_components,
                             n,
                             &one,
                             tran.data(),
                             n,
                             &zero,
                             (float*)nullptr,
                             n_components,
                             out,
                             n_components));
  }
}
}  // namespace Spectral
}  // namespace ML
