#pragma once
namespace raft {
class handle_t;
}

namespace ML {
namespace Spectral {
void fit_embedding(const raft::handle_t &handle, int *rows, int *cols,
                   float *vals, int nnz, int n, int n_components, float *out,
                   unsigned long long seed);
}  // namespace Spectral
}  // namespace ML
