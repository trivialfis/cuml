import numpy as np

from cuml.common.sparsefuncs import extract_knn_graph
from cuml.common.base import Base
from cuml.common.mixins import CMajorInputTagMixin
from cuml.common.input_utils import input_to_cuml_array
from cuml.common.array import CumlArray
from cuml.raft.common.handle cimport handle_t

from libc.stdint cimport uintptr_t
from libc.stdint cimport uint64_t


cdef extern from "cuml/manifold/spectral.hpp" namespace "ML::Spectral":

    void fit_embedding(
        handle_t & handle,
        int * rows,
        int * cols,
        float * vals,
        int nnz,
        int n,
        int n_components,
        float * out,
        uint64_t seed
    ) except +

    void fit_embedding(
        handle_t &handle,
        int n,
        int *knn_indices,
        float *knn_dists,
        int n_components,
        int n_neighbors,
        float *out, uint64_t seed
    ) except +

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
    def __init__(
        self, *,
        n_components,
        affinity="nearest_neighbors",
        random_state=None,
        verbose=False,
        output_type=None,
        handle=None,
    ):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)

        self.n_components = n_components
        self.affinity = affinity
        if isinstance(random_state, np.uint64):
            self.random_state = random_state
        else:
            # Otherwise create a RandomState instance to generate a new
            # np.uint64
            if isinstance(random_state, np.random.RandomState):
                rs = random_state
            else:
                rs = np.random.RandomState(random_state)

            self.random_state = rs.randint(low=0,
                                           high=np.iinfo(np.uint64).max,
                                           dtype=np.uint64)

        super()

    def fit(self, X, y=None):
        assert y is None
        self.n_rows = X.shape[0]
        n_neighbors = X.data.reshape((self.n_rows, -1)).shape[1]

        if self.n_rows <= 1:
            raise ValueError("There needs to be more than 1 sample.")

        (knn_indices_m, knn_indices_ctype), (knn_dists_m, knn_dists_ctype) =\
            extract_knn_graph(X, True, True)

        cdef uintptr_t knn_indices_raw = knn_indices_ctype or 0
        cdef uintptr_t knn_dists_raw = knn_dists_ctype or 0
        cdef handle_t * handle = <handle_t*> < size_t > self.handle.getHandle()

        self.embedding_ = CumlArray.zeros(
            (self.n_rows, self.n_components), order="C", dtype=np.float32
        )

        cdef uintptr_t embed_raw = self.embedding_.ptr
        fit_embedding(
            handle[0],
            self.n_rows,
            <int*> knn_indices_raw,
            <float*> knn_dists_raw,
            self.n_components,
            n_neighbors,
            <float*> embed_raw,
            self.random_state
        )
