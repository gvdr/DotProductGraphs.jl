# DotProductGraphs

[![Build Status](https://github.com/gvdr/DotProductGraphs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gvdr/DotProductGraphs.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/github/gvdr/DotProductGraphs.jl/branch/main/graph/badge.svg?token=2A4OWAQIFF)](https://codecov.io/github/gvdr/DotProductGraphs.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://gvdr.github.io/DotProductGraphs.jl/dev)

A Julia package for Random Dot Product Graphs (RDPG).

It covers SVD-based embeddings of networks, multiple multi-graph embedding methods for temporal networks, Procrustes alignment, and automatic dimensionality selection.

**Requires Julia 1.6+**

## Installation

```julia
using Pkg
Pkg.add("DotProductGraphs")
```

## Basic Usage

### Single Graph Embedding

```julia
using DotProductGraphs

# Create a random adjacency matrix
A = rand(Bool, 100, 100)

# Embed with specified dimension
L, R = svd_embedding(A, 4)

# Or let automatic dimensionality selection choose
L, R = svd_embedding(A)

# Reconstruct probability matrix via dot product
P = dot_product(L, R)
```

## Multi-Graph Embedding Methods

DotProductGraphs provides five methods for embedding temporal/multi-layer networks:

| Method | Function | Reference |
|--------|----------|-----------|
| **Omnibus** | `omni_embedding` | Levin et al. (2017) |
| **UASE** | `uase_embedding` | Gallagher et al. (2021) |
| **DUASE** | `duase_embedding` | Baum et al. (2024) |
| **MASE** | `mase_embedding` | Arroyo et al. (2021) |
| **GB-DASE** | `gbdase` | Loyal (2025) |

### Quick Example

```julia
using DotProductGraphs

# Create temporal network (list of adjacency matrices)
# Here we simulate a network with slowly evolving latent positions
n, d, T = 50, 4, 10
X = [0.3 * randn(n, d) for _ in 1:T]
for t in 2:T
    X[t] = X[t-1] + 0.05 * randn(n, d)  # smooth evolution
end
A = [Float64.(rand(n, n) .< clamp.(X[t] * X[t]', 0, 1)) for t in 1:T]
A = [(a + transpose(a)) / 2 for a in A]  # symmetrize

# Omnibus embedding - block matrix with averaged off-diagonals
result = omni_embedding(A, 4)
# result.L is (n x d x T), result.Sigma contains singular values

# UASE - Unfolded ASE using right singular vectors
result = uase_embedding(A, 4)

# DUASE - Returns shared basis + per-time embeddings
result = duase_embedding(A, 4)
# result.X_shared is (n x d), result.Y is (n x d x T)

# MASE - Two-stage SVD for common latent positions
result = mase_embedding(A, 4; return_scores=true)
# result.V is common embedding (n x d), result.scores are per-graph score matrices

# GB-DASE - Bayesian Gibbs sampler with random walk prior
result = gbdase(A, 4; rw_order=2, n_burnin=500, n_samples=1000)
# result.X is (T x n x d), smoothed by penalizing acceleration (r=2)
# result.samples contains posterior samples for uncertainty quantification

# GB-DASE MAP estimation (no uncertainty quantification)
result = gbdase_MAP(A, 4; rw_order=2)
# result.X is (T x n x d)
```

### TemporalNetworkEmbedding

For a unified interface with interpolation support:

```julia
# Default: Sequential SVD with Procrustes alignment
T = TemporalNetworkEmbedding(A, 4)

# Or specify method explicitly
T = TemporalNetworkEmbedding(A, 4, :omni)      # Omnibus
T = TemporalNetworkEmbedding(A, 4, :uase)      # UASE
T = TemporalNetworkEmbedding(A, 4, :mase)      # MASE
T = TemporalNetworkEmbedding(A, 4, :procrustes) # Default

# Access embeddings at integer or fractional time points
T[1]        # Embedding at t=1
T[1.5]      # Linear interpolation between t=1 and t=2

# Reconstruct adjacency matrices
A_reconstructed = constructRDPG(T)
```

## References

- **Omnibus**: Levin et al. (2017) "A central limit theorem for an omnibus embedding" IEEE ICDMW
- **UASE**: Gallagher et al. (2021) "Spectral embedding for dynamic networks with stability guarantees" NeurIPS
- **DUASE**: Baum, Sanna Passino & Gandy (2024) "Doubly unfolded adjacency spectral embedding" arXiv:2410.09810
- **MASE**: Arroyo et al. (2021) "Inference for multiple heterogeneous networks" JMLR
- **GB-DASE**: Loyal (2025) "Generalized Bayesian Dynamic Adjacency Spectral Embedding" [arXiv:2509.19748](https://arxiv.org/abs/2509.19748)

## TODO

- [ ] Document all the things!
- [ ] Unit Tests All The Things!
- [x] Include basic embedding and dot product capability
- [x] Allow user to choose the SVD engine
- [x] Implement Omniembedding functions
    - [x] Automatic block matrix building
    - [x] Embedding extraction
- [x] Implement multi-graph embedding methods (UASE, DUASE, MASE, GB-DASE)
- [x] Implement Procrustes Alignment
    - [x] Orthogonal rotation matrix
    - [ ] Allowing translations
    - [ ] Seedless alignment
- [x] Elbow method for dimensionality selection
- [x] Register package
- [ ] Integration with Graphs.jl and EcologicalNetworks.jl

## Acknowledgement

DotProductGraphs is inspired by [graspologic](https://microsoft.github.io/graspologic), focusing on being lightweight and performant in Julia while complementing ecosystems like Graphs.jl and EcologicalNetworks.jl.
