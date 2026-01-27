"""
    uase_embedding(A_list, d=nothing; svd_engine=truncated_svd)

Compute the Unfolded Adjacency Spectral Embedding (UASE) for a list of adjacency matrices.

UASE horizontally stacks adjacency matrices [A_1 | A_2 | ... | A_T] and extracts
per-time embeddings from the RIGHT singular vectors V, partitioned by time.

# Arguments
* `A_list`: Vector of adjacency matrices (all must be same size n × n)
* `d`: Embedding dimension (or nothing for automatic selection)
* `svd_engine`: Function to perform SVD (default: truncated_svd)

# Returns
Named tuple with:
* `L`: Per-time embeddings from V (n × d × T)
* `R`: Same as L
* `Sigma`: Singular values

# Algorithm
1. Unfold: A = [A_1 | A_2 | ... | A_T] of shape (n, n*T)
2. SVD: A = U * S * V'
3. Extract from V (RIGHT singular vectors), partitioned by time:
   X̂(t) = V[(t-1)*n+1:t*n, :] * sqrt(diag(S))

# Example
```julia
A = [rand(10,10) for _ in 1:5]
A = [a + a' for a in A]  # symmetrize
result = uase_embedding(A, 3)
```

# Reference
Gallagher et al. (2021) "Spectral embedding for dynamic networks with stability guarantees" NeurIPS
Source: https://github.com/iggallagher/Dynamic-Network-Embedding
"""
function uase_embedding(A_list::V, d=nothing; svd_engine=truncated_svd) where V <: Vector{<:AbstractMatrix}
    n = size(A_list[1], 1)
    T = length(A_list)

    A_unfolded = hcat(A_list...)  # n × nT
    U, Sigma, V_mat = svd_engine(A_unfolded, d)
    sqrt_Sigma = sqrt.(Sigma)
    actual_d = length(Sigma)

    # Extract from V (RIGHT singular vectors), partitioned by time
    # V is (nT × d), partition into T blocks of n rows
    L = zeros(eltype(V_mat), n, actual_d, T)
    for t in 1:T
        idx = ((t-1)*n+1):(t*n)
        L[:, :, t] = V_mat[idx, :] .* sqrt_Sigma'
    end
    return (L=L, R=L, Sigma=Sigma)
end

"""
    duase_embedding(A_list, d=nothing; svd_engine=truncated_svd)

Compute the Doubly Unfolded Adjacency Spectral Embedding (DUASE) for a list of adjacency matrices.

DUASE returns both a shared basis (from left singular vectors U) and per-time
embeddings (from right singular vectors V).

# Arguments
* `A_list`: Vector of adjacency matrices (all must be same size n × n)
* `d`: Embedding dimension (or nothing for automatic selection)
* `svd_engine`: Function to perform SVD (default: truncated_svd)

# Returns
Named tuple with:
* `X_shared`: Shared basis embedding (n × d), computed as U * sqrt(Σ)
* `Y`: Per-time embeddings (n × d × T), computed from V partitioned by time
* `Sigma`: Singular values

# Algorithm (adapted for single-layer temporal networks)
1. Unfold: A = [A_1 | A_2 | ... | A_T] of shape (n × nT)
2. SVD: A = U * S * V'
3. Shared basis: X̂ = U * sqrt(S)
4. Time-specific: Ŷ_t = V[(t-1)*n+1:t*n, :] * sqrt(S)

# Example
```julia
A = [rand(10,10) for _ in 1:5]
A = [a + a' for a in A]  # symmetrize
result = duase_embedding(A, 3)
# result.X_shared is the shared basis (n × d)
# result.Y[:,:,t] is the time-specific embedding at time t
```

# Reference
Baum, Sanna Passino & Gandy (2024) "Doubly unfolded adjacency spectral embedding
of dynamic multiplex graphs" arXiv:2410.09810
Source: https://github.com/mjbaum/dmprdpg
"""
function duase_embedding(A_list::V, d=nothing; svd_engine=truncated_svd) where V <: Vector{<:AbstractMatrix}
    n = size(A_list[1], 1)
    T = length(A_list)

    A_unfolded = hcat(A_list...)  # n × nT
    U, Sigma, V_mat = svd_engine(A_unfolded, d)
    sqrt_Sigma = sqrt.(Sigma)
    actual_d = length(Sigma)

    # Shared basis: X̂ = U * sqrt(D)
    X_shared = U .* sqrt_Sigma'

    # Time-specific: Ŷ_t = V[block_t, :] * sqrt(D)
    Y = zeros(eltype(V_mat), n, actual_d, T)
    for t in 1:T
        idx = ((t-1)*n+1):(t*n)
        Y[:, :, t] = V_mat[idx, :] .* sqrt_Sigma'
    end

    return (X_shared=X_shared, Y=Y, Sigma=Sigma)
end

"""
    mase_embedding(A_list, d=nothing; svd_engine=truncated_svd, scaled=true, return_scores=false)

Compute the Multiple Adjacency Spectral Embedding (MASE) for a list of adjacency matrices.

MASE is a two-stage method: first embed each graph individually, then perform
a second SVD on the horizontally stacked embeddings to extract a common subspace.

# Arguments
* `A_list`: Vector of adjacency matrices (all must be same size n × n)
* `d`: Embedding dimension (or nothing for automatic selection)
* `svd_engine`: Function to perform SVD (default: truncated_svd)
* `scaled`: If true, use √Σ-scaled embeddings in stage 1 (default: true)
* `return_scores`: If true, also return per-graph score matrices R(i) = V' A(i) V

# Returns
Named tuple with:
* `V`: Common latent positions (n × d)
* `Sigma`: Singular values from second SVD
* `scores`: (only if return_scores=true) Vector of score matrices R(i)

# Algorithm
1. Stage 1: Embed each graph individually: L̂_i = svd_embedding(A_i)
2. Stage 2: Stack horizontally [L̂_1 | ... | L̂_T] and perform second SVD
3. Common embedding V = U from second SVD (NOT scaled by √Σ)
4. Optional: Compute score matrices R(i) = V' A(i) V

# Example
```julia
A = [rand(10,10) for _ in 1:5]
A = [a + a' for a in A]  # symmetrize
result = mase_embedding(A, 3)
# result.V is the common embedding (n × d)

# With score matrices:
result = mase_embedding(A, 3; return_scores=true)
# result.scores[i] is the score matrix for graph i
```

# Reference
Arroyo et al. (2021) "Inference for multiple heterogeneous networks with a
common invariant subspace" JMLR
"""
function mase_embedding(A_list::V, d=nothing; svd_engine=truncated_svd, scaled=true, return_scores=false) where V <: Vector{<:AbstractMatrix}
    T = length(A_list)

    # Stage 1: Embed each graph individually
    if scaled
        L_list = [svd_embedding(A, svd_engine, d).L̂ for A in A_list]
    else
        # Re-extract unscaled U if needed
        L_list = [svd_engine(A, d)[1] for A in A_list]
    end

    # Stage 2: Stack and second SVD
    L_stacked = hcat(L_list...)
    U_hat, Sigma_hat, _ = svd_engine(L_stacked, d)

    # Common embedding (NOT scaled - that's the MASE convention)
    V_common = U_hat

    if !return_scores
        return (V=V_common, Sigma=Sigma_hat)
    end

    # Compute score matrices R(i) = V' A(i) V
    scores = [V_common' * A * V_common for A in A_list]
    return (V=V_common, Sigma=Sigma_hat, scores=scores)
end
