import numpy as np

import cuml.internals
from cuml.common.sparsefuncs import extract_knn_graph
from cuml.common.base import Base
from cuml.common.mixins import CMajorInputTagMixin
from cuml.common.input_utils import input_to_cuml_array
from cuml.common.array import CumlArray
from cuml.raft.common.handle cimport handle_t
from cuml.neighbors import NearestNeighbors

from libc.stdint cimport uintptr_t
from libc.stdint cimport uint64_t


cdef extern from "cuml/manifold/spectral.hpp" namespace "ML::Spectral":

    # pre-computed knn
    void fit_embedding_with_knn(
        handle_t &handle,
        int n,
        int *knn_indices,
        float *knn_dists,
        int n_components,
        int n_neighbors,
        float *out, uint64_t seed
    ) except +

    # dense input
    void fit_embedding(
        handle_t& handle,
        float* X,
        int n_samples,
        int n_features,
        int n_neighbors,
        int n_components,
        float* out,
        uint64_t seed
    ) except +


def spectral_embedding(adjacency, *, n_components=8, random_state=None):
    pass


class SpectralEmbedding(Base, CMajorInputTagMixin):
    """Spectral embedding for non-linear dimensionality reduction."""
    def __init__(
        self,
        *,
        n_components=2,
        affinity="nearest_neighbors",
        random_state=None,
        n_neighbors=None,
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
        }:
            raise ValueError("Unsupported affinity type: %s".format(self.affinity))

        self.n_neighbors = n_neighbors

        if isinstance(random_state, np.uint64):
            self.random_state = random_state
        else:
            # Otherwise create a RandomState instance to generate a new
            # np.uint64
            if isinstance(random_state, np.random.RandomState):
                rs = random_state
            else:
                rs = np.random.RandomState(random_state)

            self.random_state = rs.randint(
                low=0, high=np.iinfo(np.uint64).max, dtype=np.uint64
            )

    def _fit_precomputed_nn(self, X):
        n_neighbors = X.data.reshape((self.n_rows, -1)).shape[1]
        cdef handle_t * handle = <handle_t*> < size_t > self.handle.getHandle()
        cdef uintptr_t embed_raw = self.embedding_.ptr

        if self.n_rows <= 1:
            raise ValueError("There needs to be more than 1 sample.")

        (knn_indices_m, knn_indices_ctype), (knn_dists_m, knn_dists_ctype) =\
            extract_knn_graph(X, True, True)

        cdef uintptr_t knn_indices_raw = knn_indices_ctype or 0
        cdef uintptr_t knn_dists_raw = knn_dists_ctype or 0

        fit_embedding_with_knn(
            handle[0],
            self.n_rows,
            <int*> knn_indices_raw,
            <float*> knn_dists_raw,
            self.n_components,
            n_neighbors,
            <float*> embed_raw,
            self.random_state
        )


    def fit(self, X, y=None, convert_dtype=True) -> "SpectralEmbedding":
        assert y is None
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
            neigh = NearestNeighbors(n_neighbors=self.n_neighbors)
            neigh.fit(X, convert_dtype=convert_dtype)
            knn_graph = neigh.kneighbors_graph(X, mode="distance")
            self._fit_precomputed_nn(knn_graph)
        elif self.affinity == "precomputed_nearest_neighbors":
            self._fit_precomputed_nn(X)
        else:
            raise ValueError("Unknown affinity.")

        return self

    @cuml.internals.api_base_fit_transform()
    def fit_transform(self, X, y=None, convert_dtype=True):
        self.fit(X, y, convert_dtype)
        return self.embedding_
