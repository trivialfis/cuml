import numpy as np

from cuml.common.sparsefuncs import extract_knn_graph
from cuml.common.base import Base
from cuml.common.mixins import CMajorInputTagMixin
from cuml.common.input_utils import input_to_cuml_array
from cuml.raft.common.handle cimport handle_t

from libc.stdint cimport uint64_t


cdef extern from "cuml/manifold/spectral.hpp" namespace "ML::Spectral":

    void fit_embedding(
        handle_t & handle,
        int* rows,
        int* cols,
        float* vals,
        int nnz,
        int n,
        int n_components,
        float* out,
        uint64_t seed
    ) except +


def spectral_embedding(adjacency, *, n_components=8, random_state=None):
    pass


class SpectralEmbedding(Base, CMajorInputTagMixin):
    def __init__(self, n_components, affinity):
        self.n_components = n_components
        self.affinity = affinity

    def fit(self, X, y=None):
        assert y is None

        X_m, n_rows, n_dims = input_to_cuml_array(
            X, order='C', check_dtype=np.float32, convert_to_dtype=np.float32
        )
        if n_rows <= 1:
            raise ValueError("There needs to be more than 1 sample.")

        (knn_indices_m, knn_indices_ctype), (knn_dists_m, knn_dists_ctype) =\
            extract_knn_graph(X, True)
