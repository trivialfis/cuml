from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from cuml.manifold import SpectralEmbedding
import numpy as np


def test_spectral_knn():
    n_neighbors = 2
    data, labels = datasets.make_blobs(
        n_samples=2000, n_features=10, centers=5, random_state=0)

    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(data)
    knn_graph = neigh.kneighbors_graph(data, mode="distance")
    print(type(knn_graph), knn_graph.shape)
    print(np.allclose(knn_graph.todense(), knn_graph.todense().T))
    print("equal:", (knn_graph.T != knn_graph).nnz)
