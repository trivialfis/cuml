#pragma once
#include <cinttypes>

namespace raft {
class handle_t;
}

namespace ML {
namespace Spectral {
void fit_embedding(const raft::handle_t &handle, int *rows, int *cols,
                   float *vals, int nnz, int n, int n_components, float *out,
                   unsigned long long seed);

void fit_embedding(const raft::handle_t &handle, int n, int *knn_indices,
                   float *knn_dists, int n_components, int n_neighbors,
                   float *out, uint64_t seed);
}  // namespace Spectral
}  // namespace ML
