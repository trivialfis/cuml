import numpy as np

from sklearn.utils import check_random_state

import cupy
from raft.common.handle cimport handle_t

import cuml.internals
from cuml.common.sparsefuncs import extract_knn_graph
from cuml.common.base import Base
from cuml.common.mixins import CMajorInputTagMixin
from cuml.common.input_utils import input_to_cuml_array
from cuml.common.array import CumlArray
from cuml.common.array_sparse import SparseCumlArray
from cuml.metrics.pairwise_kernels import rbf_kernel
from cuml.neighbors import NearestNeighbors, kneighbors_graph

from libc.stdint cimport uintptr_t
from libc.stdint cimport uint64_t


cdef extern from "cuml/manifold/spectral.hpp" namespace "ML::Spectral":
    void fit_embedding(handle_t &handle, int *rows, int *cols,
                       float *vals, int nnz, int n, int n_components, float *out,
                       uint64_t seed) except +

def get_rs(random_state):
    if isinstance(random_state, np.uint64):
        return random_state
    else:
        # Otherwise create a RandomState instance to generate a new
        # np.uint64
        if isinstance(random_state, np.random.RandomState):
            rs = random_state
        else:
            rs = np.random.RandomState(random_state)

        return rs.randint(
            low=0, high=np.iinfo(np.uint64).max, dtype=np.uint64
        )


class SpectralEmbedding(Base, CMajorInputTagMixin):
    """Spectral embedding for non-linear dimensionality reduction."""
    def __init__(
        self,
        *,
        n_components=2,
        affinity="nearest_neighbors",
        random_state=None,
        n_neighbors=None,
        gamma=None,
        verbose=False,
        output_type=None,
        handle=None,
    ):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)

        self.n_components = n_components
        self.affinity = affinity

        if self.affinity not in {
            "nearest_neighbors",
            "precomputed_nearest_neighbors",
            "precomputed",
            "rbf",
        }:
            raise ValueError("Unsupported affinity type: %s".format(self.affinity))

        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.random_state = random_state

    def _get_affinity_matrix(self, X):
        if self.affinity == "precomputed":
            self.affinity_matrix_ = X
        elif self.affinity == "precomputed_nearest_neighbors":
            from sklearn.neighbors import NearestNeighbors as sckl_nn
            estimator = sckl_nn(
                n_neighbors=self.n_neighbors, metric="precomputed",
            ).fit(X)
            connectivity = estimator.kneighbors_graph(X=X, mode='connectivity')
            self.affinity_matrix_ = cupy.sparse.csr_matrix(0.5 * (connectivity + connectivity.T))
        elif self.affinity == "nearest_neighbors":
            connectivity = X
            self.affinity_matrix_ = cupy.sparse.csr_matrix(0.5 * (connectivity + connectivity.T))
        else:
            assert self.affinity == "rbf"
            X = cupy.asarray(X, dtype=cupy.float32)
            self.gamma_ = (self.gamma if self.gamma is not None else 1.0 / X.shape[1])
            self.affinity_matrix_ = rbf_kernel(X, X, gamma=self.gamma)
        return self.affinity_matrix_

    def _fit_precomputed_nn(self, X):
        if isinstance(X, SparseCumlArray):
            n_neighbors = X.data.shape[0] // self.n_rows
        else:
            n_neighbors = X.shape[1]
        cdef handle_t * handle = <handle_t*> < size_t > self.handle.getHandle()
        cdef uintptr_t embed_raw = self.embedding_.ptr

        if self.n_rows <= 1:
            raise ValueError("There needs to be more than 1 sample.")

        X = X.tocoo()
        rows_m, _, _, _ = \
            input_to_cuml_array(X.row, order='C',
                                deepcopy=True,
                                check_dtype=(np.int32, np.int64),
                                convert_to_dtype=np.int32)
        cols_m, _, _, _ = \
            input_to_cuml_array(X.col, order='C',
                                deepcopy=True,
                                check_dtype=(np.int32, np.int64),
                                convert_to_dtype=np.int32)

        val_m, _, _, _ = \
            input_to_cuml_array(X.data, order='C',
                                deepcopy=True,
                                check_dtype=(np.float32, np.float64),
                                convert_to_dtype=np.float32)

        rs = check_random_state(self.random_state)
        seed = rs.randint(low=0, high=np.iinfo(np.uint64).max, dtype=np.uint64)
        fit_embedding(
            handle[0],
            <int*><uintptr_t> rows_m.ptr,
            <int*><uintptr_t> cols_m.ptr,
            <float*><uintptr_t> val_m.ptr,
            X.nnz,
            X.shape[0],
            self.n_components,
            <float*> embed_raw,
            seed,
        )


    def fit(self, X, y=None, convert_dtype=True) -> "SpectralEmbedding":
        if y is not None:
            raise ValueError("y is not used for `SpectralEmbedding`.")
        cdef handle_t * handle = <handle_t*> < size_t > self.handle.getHandle()
        self.n_rows = X.shape[0]
        self.embedding_ = CumlArray.zeros(
            (self.n_rows, self.n_components), order="C", dtype=np.float32
        )
        cdef uintptr_t embed_raw = self.embedding_.ptr
        if self.n_neighbors is None:
            n_neighbors = 15
        else:
            n_neighbors = self.n_neighbors

        if self.affinity == "nearest_neighbors":
            self.n_neighbors_ = (
                self.n_neighbors if self.n_neighbors is not None else max(int(X.shape[0] / 10), 1)
            )
            neigh = NearestNeighbors(
                n_neighbors=self.n_neighbors_, handle=self.handle, output_type="cupy"
            )
            neigh.fit(X)
            knn_graph = neigh.kneighbors_graph(
                X, n_neighbors=self.n_neighbors_, mode="connectivity"
            ).to_output(output_format="coo").tocsr()
            affinity = self._get_affinity_matrix(knn_graph)
            self._fit_precomputed_nn(affinity)
        elif self.affinity == "precomputed_nearest_neighbors":
            affinity = self._get_affinity_matrix(X)
            import cupy
            affinity = cupy.sparse.coo_matrix(affinity)
            self._fit_precomputed_nn(affinity)
        elif self.affinity == "rbf":
            affinity = self._get_affinity_matrix(X)
            import cupy
            affinity = cupy.sparse.coo_matrix(affinity)
            self._fit_precomputed_nn(affinity)
        else:
            raise ValueError("Unknown affinity.")

        return self

    @cuml.internals.api_base_fit_transform()
    def fit_transform(self, X, y=None, convert_dtype=True):
        self.fit(X, y, convert_dtype)
        return self.embedding_


def spectral_embedding(adjacency, *, n_components=8, random_state=None, handle=None):
    estimator = SpectralEmbedding(handle=handle, n_components=n_components)
    return estimator.fit_transform(adjacency)
