from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from cuml.neighbors import NearestNeighbors
from cuml.manifold import SpectralEmbedding
import numpy as np


def test_spectral_knn():
    n_neighbors = 2
    data, labels = datasets.make_blobs(
        n_samples=2000, n_features=10, centers=5, random_state=0)

    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(data)
    knn_graph = neigh.kneighbors_graph(data, mode="distance")

    spectral = SpectralEmbedding(n_components=2, affinity="precomputed_nearest_neighbors")
    spectral.fit(knn_graph)


def test_spectral_embedding():
    data, labels = datasets.make_blobs(
        n_samples=2000, n_features=10, centers=5, random_state=0)
    spectral = SpectralEmbedding(n_components=2)
    assert spectral.affinity == "nearest_neighbors"
    spectral.fit(data)
