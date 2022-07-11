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

#include <raft/linalg/unary_op.cuh>
#include <raft/linalg/transpose.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/spectral.hpp>
#include <raft/sparse/linalg/symmetrize.hpp>
#include <raft/sparse/op/detail/filter.cuh>
#include <raft/spatial/knn/knn.cuh>
#include <raft/core/mdarray.hpp>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

namespace ML {

namespace Spectral {

template <typename T>
MDSPAN_INLINE_FUNCTION auto native_popc(T v) -> int32_t
{
  int c = 0;
  for (; v != 0; v &= v - 1) {
    c++;
  }
  return c;
}

MDSPAN_INLINE_FUNCTION auto popc(uint32_t v) -> int32_t
{
#if defined(__CUDA_ARCH__)
  return __popc(v);
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_popcount(v);
#else
  return native_popc(v);
#endif  // compiler
}

MDSPAN_INLINE_FUNCTION auto popc(uint64_t v) -> int32_t
{
#if defined(__CUDA_ARCH__)
  return __popcll(v);
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(v);
#else
  return native_popc(v);
#endif  // compiler
}

template <class T, std::size_t N, std::size_t... Idx>
MDSPAN_INLINE_FUNCTION constexpr auto arr_to_tup(T (&arr)[N], std::index_sequence<Idx...>)
{
  return std::make_tuple(arr[Idx]...);
}

template <class T, std::size_t N>
MDSPAN_INLINE_FUNCTION constexpr auto arr_to_tup(T (&arr)[N])
{
  return arr_to_tup(arr, std::make_index_sequence<N>{});
}

// uint division optimization inspired by the CIndexer in cupy.  Division operation is
// slow on both CPU and GPU, especially 64 bit integer.  So here we first try to avoid 64
// bit when the index is smaller, then try to avoid division when it's exp of 2.
template <typename I, size_t... Extents>
MDSPAN_INLINE_FUNCTION auto unravel_index_impl(I idx, raft::detail::stdex::extents<Extents...> shape)
{
  constexpr auto kRank = static_cast<int32_t>(shape.rank());
  size_t index[shape.rank()]{0};  // NOLINT
  static_assert(std::is_signed<decltype(kRank)>::value,
                "Don't change the type without changing the for loop.");
  for (int32_t dim = kRank; --dim > 0;) {
    auto s = static_cast<std::remove_const_t<std::remove_reference_t<I>>>(shape.extent(dim));
    if (s & (s - 1)) {
      auto t     = idx / s;
      index[dim] = idx - t * s;
      idx        = t;
    } else {  // exp of 2
      index[dim] = idx & (s - 1);
      idx >>= popc(s - 1);
    }
  }
  index[0] = idx;
  return arr_to_tup(index);
}

/**
 * \brief Turns linear index into coordinate.  Similar to numpy unravel_index. This is not
 *        exposed to public as it's not part of the mdspan proposal, the returned tuple
 *        can not be directly used for indexing into mdspan and we might change the return
 *        type in the future.
 *
 * \code
 *   auto m = make_host_matrix<float>(7, 6);
 *   auto m_v = m.view();
 *   auto coord = detail::unravel_index(2, m.extents(), typename decltype(m)::layout_type{});
 *   detail::apply(m_v, coord) = 2;
 * \endcode
 *
 * \param idx    The linear index.
 * \param shape  The shape of the array to use.
 * \param layout Must be `layout_right` (row-major) in current implementation.
 *
 * \return A thrust::tuple that represents the coordinate.
 */
template <typename LayoutPolicy, std::size_t... Exts>
MDSPAN_INLINE_FUNCTION auto unravel_index(size_t idx,
                                          raft::detail::stdex::extents<Exts...> shape,
                                          LayoutPolicy const&)
{
  static_assert(std::is_same<LayoutPolicy, raft::detail::stdex::layout_right>::value,
                "Only C layout is supported.");
  if (idx > std::numeric_limits<uint32_t>::max()) {
    return unravel_index_impl<uint64_t, Exts...>(static_cast<uint64_t>(idx), shape);
  } else {
    return unravel_index_impl<uint32_t, Exts...>(static_cast<uint32_t>(idx), shape);
  }
}

template <typename Fn, typename Tup, size_t... I>
MDSPAN_INLINE_FUNCTION auto constexpr apply_impl(Fn&& f, Tup&& t, std::index_sequence<I...>)
  -> decltype(auto)
{
  return f(thrust::get<I>(t)...);
}

/**
 * C++ 17 style apply for thrust tuple.
 *
 * \param f function to apply
 * \param t tuple of arguments
 */
template <typename Fn,
          typename Tup,
          std::size_t kTupSize = std::tuple_size<std::remove_reference_t<Tup>>::value>
MDSPAN_INLINE_FUNCTION auto constexpr apply(Fn&& f, Tup&& t) -> decltype(auto)
{
  return apply_impl(
    std::forward<Fn>(f), std::forward<Tup>(t), std::make_index_sequence<kTupSize>{});
}


template <typename index_type, typename value_type, typename ThrustExecPolicy>
auto Diagonal(index_type const* rows,
              index_type const* cols,
              value_type const* values,
              size_t nnz,
              size_t n,
              raft::handle_t const& handle,
              ThrustExecPolicy policy) -> rmm::device_uvector<value_type>
{
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
  std::cout << "r:" << n_rows << ", c:" << n_cols << ", nnz" << nnz << std::endl;
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

template <typename index_t, typename value_t>
std::string coo_to_str(index_t const* rows,
                       index_t const* cols,
                       value_t const* vals,
                       size_t nnz,
                       std::string name,
                       rmm::cuda_stream_view stream)
{
  std::vector<index_t> h_rows(nnz);
  std::vector<index_t> h_cols(nnz);
  std::vector<value_t> h_vals(nnz);
  CUDA_CHECK(cudaMemcpyAsync(
    h_rows.data(), rows, nnz * sizeof(index_t), cudaMemcpyDeviceToHost, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(
    h_cols.data(), cols, nnz * sizeof(index_t), cudaMemcpyDeviceToHost, stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(
    h_vals.data(), vals, nnz * sizeof(value_t), cudaMemcpyDeviceToHost, stream.value()));

  std::stringstream ss;
  for (size_t i = 0; i < h_rows.size(); ++i) {
    ss << "(" << h_rows[i] << ", " << h_cols[i] << ")\t" << h_vals[i] << "\n";
  }
  return ss.str();
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

  std::string toStr(raft::handle_t const& handle, std::string name) const
  {
    std::vector<index_t> h_rows(nnz);
    std::vector<index_t> h_cols(nnz);
    std::vector<value_t> h_vals(nnz);
    CUDA_CHECK(cudaMemcpyAsync(
      h_rows.data(), rows, nnz * sizeof(index_t), cudaMemcpyDeviceToHost, handle.get_stream()));
    CUDA_CHECK(cudaMemcpyAsync(
      h_cols.data(), cols, nnz * sizeof(index_t), cudaMemcpyDeviceToHost, handle.get_stream()));
    CUDA_CHECK(cudaMemcpyAsync(
      h_vals.data(), vals, nnz * sizeof(value_t), cudaMemcpyDeviceToHost, handle.get_stream()));

    std::stringstream ss;
    ss << name << ":\n";
    for (size_t i = 0; i < h_rows.size(); ++i) {
      ss << "(" << h_rows[i] << ", " << h_cols[i] << ")\t" << h_vals[i] << "\n";
    }
    return ss.str();
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
                     size_t nnz,
                     rmm::device_uvector<value_type> const& diag)
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
    std::cout << "o.size(): " << ones.size() << ", d.size():" << diagonal_.size() << std::endl;
    // calcuate the degree matrix
    this->SpMV(1, ones.data(), 0, diagonal_.data());
    // normalize it.
    this->Normalize(policy, handle, diag);

    ASSERT(this->rows_arr.size() == this->cols_arr.size(), "Check");
    ASSERT(this->rows_arr.size() == this->vals_arr.size(), "Check");
  }

  void SpMV(value_type alpha,
            value_type* __restrict__ x,
            value_type beta,
            value_type* __restrict__ y)
  {
    CUDA_CHECK(cudaDeviceSynchronize());
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

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Finished spmv" << std::endl;
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
               weight_t* eigVecs)
{
  auto cublas_h = handle.get_cublas_handle();
  auto stream   = handle.get_stream();
  auto policy   = handle.get_thrust_policy();

  auto diagonal = Diagonal(rows, cols, vals, nnz, n_samples, handle, policy);

  laplacian_matrix_t<vertex_t, weight_t> laplacian{
    handle, policy, rows, cols, vals, n_samples, nnz, diagonal};
  thrust::transform(policy,
                    laplacian.vals(),
                    laplacian.vals() + laplacian.NNZ(),
                    laplacian.vals(),
                    [] HD(weight_t v) { return -v; });
  auto it = thrust::make_counting_iterator(0ul);
  CUDA_CHECK(cudaDeviceSynchronize());

  // std::cout << coo_view_t<vertex_t, weight_t>{laplacian}.toStr(handle, "Laplacian") << std::endl;
  // std::cout << raft::arr2Str(laplacian.rows(), laplacian.nnz, "rows", stream) << std::endl;
  // std::cout << raft::arr2Str(laplacian.cols(), laplacian.nnz, "cols", stream) << std::endl;
  // std::cout << raft::arr2Str(laplacian.vals(), laplacian.nnz, "vals", stream) << std::endl;

  CSR<vertex_t, vertex_t, weight_t> csr(handle, n_samples, n_samples, nnz);
  coo_view_t<vertex_t, weight_t>{laplacian}.toCSR(handle, &csr);

  auto r_csr_m = raft::spectral::matrix::sparse_matrix_t<vertex_t, weight_t>(csr);
  eigen_solver.solve_largest_eigenvectors(handle, r_csr_m, eigVals, eigVecs);
  // thrust::transform(
  //     policy, eigVecs, eigVecs + n_components * n_samples, eigVecs, [] HD(weight_t v) { return -v; });
  // raft::spectral::transform_eigen_matrix(handle, policy, n_samples, n_components, eigVecs);
  // std::cout << raft::arr2Str(eigVals, n_components, "cuml eigen values", stream) << std::endl;

  raft::device_matrix_view<float> eig_vectors(eigVecs, n_components, n_samples);
  auto permutation = raft::make_device_matrix<float>(handle, n_components, n_samples);

  it              = thrust::make_counting_iterator(0ul);
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
                             handle.get_stream()));

  // std::cout << raft::arr2Str(eigVecs, n_components * n_samples, "embeddings::", stream)
  //           << std::endl;


  // std::cout << raft::arr2Str(
  //                laplacian.diagonal_.data(), laplacian.diagonal_.size(), "diagonal", stream)
  //           << std::endl;

  auto d_diagonal = laplacian.diagonal_.data();
  it              = thrust::make_counting_iterator(0ul);
  std::cout << "n_diagonal:" << laplacian.diagonal_.size() << std::endl;
  thrust::for_each(policy, it, it + n_samples * n_components, [=] HD(size_t i) {
    size_t cidx = i % n_samples;
    assert(d_diagonal[cidx] != 0);
    eigVecs[i] /= d_diagonal[cidx];
  });

  // std::cout << raft::arr2Str(eigVecs, n_components * n_samples, "normalization::", stream)
  //           << std::endl;

  CUDA_CHECK(cudaDeviceSynchronize());
  {
    // deterministic vector sign flip
    auto abs_it =
      thrust::make_transform_iterator(eigVecs, [=] HD(float v) -> float { return std::abs(v); });
    auto key_it =
      thrust::make_transform_iterator(it, [=] HD(size_t i) -> size_t { return i % n_samples; });
    auto signs = raft::make_device_vector<float>(n_components, stream);
    CUDA_CHECK(cudaDeviceSynchronize());
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
      eigVecs[i] *= signs_v(ridx);
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
  auto policy   = rmm::exec_policy(handle.get_stream());
  auto diagonal = Diagonal(rows, cols, vals, nnz, n, handle, policy);

  using index_type = int32_t;
  using value_type = float;

  handle.get_stream().synchronize();

  std::cout << "n_comp:" << n_components << std::endl;
  index_type neigvs       = n_components + 1;
  index_type maxiter      = 5000;  // default reset value (when set to 0);
  value_type tol          = 0.001;
  index_type restart_iter = 15 + neigvs;  // what cugraph is using

  raft::spectral::eigen_solver_config_t<index_type, value_type> cfg{
    neigvs, maxiter, restart_iter, tol, true, 0};
  cfg.reorthogonalize = true;
  raft::spectral::lanczos_solver_t<index_type, value_type> eig_solver{cfg};

  rmm::device_uvector<value_type> eigen_values(neigvs, handle.get_stream());
  rmm::device_uvector<value_type> eigen_vectors(neigvs * n, handle.get_stream());

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

  // std::cout << raft::arr2Str(eigen_vectors.data(), eigen_vectors.size(), "eig vectors::", stream)
  //           << std::endl;

  // drop first
  // raft::linalg::transpose(
  //   handle, eigen_vectors.data(), tran.data(), neigvs, n, handle.get_stream());
  // std::cout << raft::arr2Str(tran.data(), tran.size(), "trans::", stream)
  //           << std::endl;

  // auto tran_v = tran.view();
  auto map_v = raft::make_device_matrix_view(eigen_vectors.data(), neigvs, n);
  auto drop_first = raft::detail::stdex::submdspan(
    map_v, std::make_tuple(1, neigvs), raft::detail::stdex::full_extent);
  std::cout << "extent:" << drop_first.extent(0) << ", " << drop_first.extent(1) << std::endl;
  ASSERT(drop_first.size() == static_cast<size_t>(n_components * n), "Invalid shape of eigen map.");

  auto tran   = raft::make_device_matrix<float>(n_components, n, handle.get_stream());
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

void fit_embedding_with_knn(raft::handle_t const& handle,
                            int n,
                            int32_t* knn_indices,
                            float* knn_dists,
                            int n_components,
                            int n_neighbors,
                            float* out,
                            uint64_t seed)
{
  using value_t   = float;
  using value_idx = int32_t;
  raft::sparse::COO<value_t, value_idx> knn_coo(handle.get_stream());

  // fixme: is this necessary?
  raft::sparse::linalg::from_knn_symmetrize_matrix<value_idx, value_t>(
    knn_indices, knn_dists, n, n_neighbors, &knn_coo, handle.get_stream());
  raft::sparse::COO<value_t, value_idx> out_coo(handle.get_stream());
  raft::sparse::op::detail::coo_remove_zeros<256, value_t>(&knn_coo, &out_coo, handle.get_stream());
  raft::linalg::unaryOp(
    out_coo.vals(),
    out_coo.vals(),
    out_coo.nnz,
    [] __device__(float v) { return v * 0.5; },
    handle.get_stream());
  // std::cout << raft::arr2Str(out_coo.vals(), out_coo.nnz, "symmetrized valus",
  //                            handle.get_stream())
  //           << std::endl;

  raft::sparse::op::coo_sort<value_t>(&out_coo, handle.get_stream());

  raft::sparse::spectral::fit_embedding(handle,
                                        /*rows=*/out_coo.rows(),
                                        /*cols=*/out_coo.cols(),
                                        /*vals=*/out_coo.vals(),
                                        /*nnz=*/out_coo.nnz,
                                        /*n=*/(value_idx)n,
                                        n_components,
                                        out,
                                        seed);
}

void fit_embedding(raft::handle_t const& handle,
                   float* X,
                   int n_samples,
                   int n_features,
                   int n_neighbors,
                   int n_components,
                   float* out,
                   uint64_t seed)
{
  manifold_dense_inputs_t<float> inputs(X, nullptr, n_samples, n_features);

  rmm::device_uvector<knn_indices_dense_t> knn_indices(n_samples * n_neighbors,
                                                       handle.get_stream());
  rmm::device_uvector<float> knn_dists(n_samples * n_neighbors, handle.get_stream());
  knn_graph<knn_indices_dense_t, float> knn(
    n_samples, n_neighbors, knn_indices.data(), knn_dists.data());

  std::vector<float*> ptrs(1, inputs.X);
  std::vector<int> sizes(1, inputs.n);

  raft::spatial::knn::brute_force_knn(handle,
                                      ptrs,
                                      sizes,
                                      n_features,
                                      inputs.X,
                                      inputs.n,
                                      knn.knn_indices,
                                      knn.knn_dists,
                                      n_neighbors);

  std::string indices_str =
    raft::arr2Str(knn_indices.data(), knn_indices.size(), "knn_indices", handle.get_stream());
  std::string values_str =
    raft::arr2Str(knn_dists.data(), knn_dists.size(), "knn_distances", handle.get_stream());

  // std::pair<int32_t, int32_t> shape;
  // shape.first  = n_samples;
  // shape.second = n_samples;

  rmm::device_uvector<int> knn_indices_i32(n_samples * n_neighbors, handle.get_stream());
  thrust::copy_n(rmm::exec_policy(handle.get_stream()),
                 knn.knn_indices,
                 n_samples * n_neighbors,
                 knn_indices_i32.begin());

  // std::cout << raft::arr2Str(knn_indices_i32.data(), knn_indices_i32.size(),
  //                            "knn_indices", handle.get_stream())
  //           << std::endl;

  fit_embedding_with_knn(
    handle, n_samples, knn_indices_i32.data(), knn.knn_dists, n_components, n_neighbors, out, seed);
}
}  // namespace Spectral
}  // namespace ML
