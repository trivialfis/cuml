#pragma once
#include <cinttypes>

namespace raft {
class handle_t;
}

namespace ML {
namespace Spectral {
void fit_embedding(const raft::handle_t& handle,
                   int* rows,
                   int* cols,
                   float* vals,
                   int nnz,
                   int n,
                   int n_components,
                   float* out,
                   unsigned long long seed);

/**
 * \brief Compute the spectral embeddings given dense input X.
 */
void fit_embedding(raft::handle_t const& handle,
                   float* X,
                   int n_samples,
                   int n_features,
                   int n_neighbors,
                   int n_components,
                   float* out,
                   uint64_t seed);
}  // namespace Spectral
}  // namespace ML
