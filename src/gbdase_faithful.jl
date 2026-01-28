#=
GB-DASE: Generalized Bayesian Dynamic Adjacency Spectral Embedding

Gibbs sampler for temporal network embedding with random walk priors.
This implementation follows the Python DynamicRDPG package by Josh Loyal
(https://github.com/joshloyal/DynamicRDPG).

Key features matching Python:
- Exact conditional Gaussian sampling via banded Cholesky
- Half-Cauchy prior for σ via auxiliary variable ν
- Kronecker structure: K ⊗ I_d
- Sequential node-by-node updates
- Post-processing: Procrustes align all samples to last

Reference: arXiv:2509.19748
=#

using BandedMatrices
using LinearAlgebra: cholesky, Symmetric, I, norm, mul!
using Statistics: var

"""
    rw_precision_banded(T::Int, r::Int)

Create banded precision matrix K = D'D for r-th order random walk prior.
Returns a BandedMatrix for efficient Cholesky factorization.

The bandwidth is 2r (r upper and r lower diagonals).
"""
function rw_precision_banded(T::Int, r::Int)
    D = difference_matrix(T, r)
    K = D' * D
    return BandedMatrix(K, (r, r))
end

"""
    sample_half_cauchy_sigma!(sigma, nu, quad_forms, df, rng)

Sample σ from Half-Cauchy(0,1) prior using auxiliary variable representation.
Matches Python DynamicRDPG exactly:
    sigma ~ InverseGamma(shape, scale)
    nu ~ InverseGamma(1, 1 + 1/sigma)

Note: In Python, sigma is used directly as 1/sigma in precision,
so sigma here represents variance-like quantity, not std dev.

# Arguments
- `sigma`: Vector of σ values to update in-place (variance-like)
- `nu`: Vector of auxiliary variables ν to update in-place
- `quad_forms`: Sum of squared r-th order differences for each node
- `df`: Degrees of freedom (d * (T - r))
- `rng`: Random number generator
"""
function sample_half_cauchy_sigma!(sigma::Vector{Float64}, nu::Vector{Float64},
                                    quad_forms::Vector{Float64}, df::Int, rng)
    n = length(sigma)
    for i in 1:n
        # Sample σ from Inverse-Gamma (matching Python exactly)
        # Python: shape_sigma = 0.5 * ((n_time_points - self.rw_order) * self.n_features + 1)
        # Python: scale_sigma = 0.5 * sum(diff^2) + 1/nu
        shape = 0.5 * (df + 1)
        scale = 0.5 * quad_forms[i] + 1.0 / nu[i]
        sigma[i] = rand(rng, InverseGamma(shape, scale))

        # Sample ν from Inverse-Gamma
        # Python: nu = invgamma.rvs(1, 1 + 1/sigma)
        nu[i] = rand(rng, InverseGamma(1.0, 1.0 + 1.0 / sigma[i]))
    end
end

"""
    build_node_precision_banded(K_banded, XtX_list, sigma_i, scale, d, T, rw_order, prior_std)

Build the banded precision matrix for node i, matching Python exactly:
    P_i = (1/σ_i) * K ⊗ I_d + 0.5*scale*I + K_init + scale * block_diag(XtX)

Where:
- K ⊗ I_d has bandwidth 2r*d
- 0.5*scale*I is diagonal regularization
- K_init adds prior_std regularization to first rw_order blocks

Returns a BandedMatrix of size (T*d) × (T*d).
"""
function build_node_precision_banded(K_banded::BandedMatrix, XtX_list::Vector{Matrix{Float64}},
                                      sigma_i::Float64, scale::Float64, d::Int, T::Int,
                                      rw_order::Int, prior_std::Float64)
    r = bandwidths(K_banded)[1]
    total_bw = r * d

    # Initialize precision matrix
    P = BandedMatrix(Zeros(T * d, T * d), (total_bw, total_bw))

    # Add K ⊗ I_d term (prior precision) - Python uses 1/sigma, not 1/sigma^2
    inv_sigma = 1.0 / sigma_i
    for t1 in 1:T
        for t2 in max(1, t1 - r):min(T, t1 + r)
            K_val = K_banded[t1, t2]
            if abs(K_val) > 1e-15
                for dim in 1:d
                    row = (t1 - 1) * d + dim
                    col = (t2 - 1) * d + dim
                    P[row, col] += inv_sigma * K_val
                end
            end
        end
    end

    # Add 0.5 * scale to diagonal (Python line 185)
    for k in 1:(T * d)
        P[k, k] += 0.5 * scale
    end

    # Add K_init prior regularization (Python lines 157-160)
    # Python: for r in range(self.rw_order): K_init[r, r] += 1/prior_std^2
    # This adds to time indices 0, 1, ..., rw_order-1 (0-indexed)
    # In Julia 1-indexed: time indices 1, 2, ..., rw_order
    if prior_std > 0
        inv_prior_var = 1.0 / (prior_std^2)
        for t in 1:rw_order
            for dim in 1:d
                k = (t - 1) * d + dim
                P[k, k] += inv_prior_var
            end
        end
    end

    # Add block_diag(XtX) term (likelihood precision)
    for t in 1:T
        for i in 1:d
            for j in 1:d
                row = (t - 1) * d + i
                col = (t - 1) * d + j
                if abs(row - col) <= total_bw
                    P[row, col] += scale * XtX_list[t][i, j]
                end
            end
        end
    end

    return P
end

"""
    compute_XtX_list(X_list, d)

Compute X[t]' * X[t] for each time point t, excluding diagonal contributions.
This is used in the precision matrix for node updates.

Returns a vector of d × d matrices.
"""
function compute_XtX_list(X_list::Vector{Matrix{Float64}}, d::Int)
    T = length(X_list)
    XtX_list = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        XtX_list[t] = X_list[t]' * X_list[t]
    end
    return XtX_list
end

"""
    compute_XtY_node(X_list, A_list, node_idx, d)

Compute the right-hand side for the node update:
    XtY[t] = Σ_{j≠i} A[t][i,j] * X[t][j,:]

Returns a T*d vector (flattened across time and dimensions).
"""
function compute_XtY_node(X_list::Vector{Matrix{Float64}}, A_list::Vector{<:AbstractMatrix},
                           node_idx::Int, d::Int)
    T = length(X_list)
    n = size(X_list[1], 1)
    XtY = zeros(T * d)

    for t in 1:T
        for j in 1:n
            if j != node_idx
                a_ij = A_list[t][node_idx, j]
                if abs(a_ij) > 1e-15
                    for dim in 1:d
                        XtY[(t-1)*d + dim] += a_ij * X_list[t][j, dim]
                    end
                end
            end
        end
    end

    return XtY
end

"""
    remove_node_from_XtX!(XtX_list, X_list, node_idx)

Remove node i's contribution from XtX matrices (in-place).
"""
function remove_node_from_XtX!(XtX_list::Vector{Matrix{Float64}},
                                X_list::Vector{Matrix{Float64}}, node_idx::Int)
    T = length(X_list)
    for t in 1:T
        xi = X_list[t][node_idx, :]
        XtX_list[t] .-= xi * xi'
    end
end

"""
    add_node_to_XtX!(XtX_list, X_list, node_idx)

Add node i's contribution back to XtX matrices (in-place).
"""
function add_node_to_XtX!(XtX_list::Vector{Matrix{Float64}},
                           X_list::Vector{Matrix{Float64}}, node_idx::Int)
    T = length(X_list)
    for t in 1:T
        xi = X_list[t][node_idx, :]
        XtX_list[t] .+= xi * xi'
    end
end

"""
    sample_node_conditional!(X_list, A_list, K_banded, XtX_list, sigma_i, scale, node_idx, d, rng, rw_order, prior_std)

Sample node i's trajectory from the exact conditional Gaussian distribution.

This uses banded Cholesky factorization for efficiency:
1. Build precision: P = (1/σ)K⊗I + 0.5*scale*I + K_init + scale * blkdiag(XtX)
2. Compute mean: μ = P⁻¹ * (scale * XtY)
3. Sample: X = μ + L⁻ᵀ * z, where z ~ N(0,I), L = chol(P)
"""
function sample_node_conditional!(X_list::Vector{Matrix{Float64}},
                                   A_list::Vector{<:AbstractMatrix},
                                   K_banded::BandedMatrix,
                                   XtX_list::Vector{Matrix{Float64}},
                                   sigma_i::Float64, scale::Float64,
                                   node_idx::Int, d::Int, rng,
                                   rw_order::Int, prior_std::Float64)
    T = length(X_list)

    # Build precision matrix
    P = build_node_precision_banded(K_banded, XtX_list, sigma_i, scale, d, T, rw_order, prior_std)

    # Compute right-hand side
    XtY = compute_XtY_node(X_list, A_list, node_idx, d)
    b = scale * XtY

    # Add small regularization for numerical stability
    for k in 1:(T * d)
        P[k, k] += 1e-8
    end

    # Cholesky factorization
    try
        C = cholesky(Symmetric(Matrix(P)))

        # Compute mean: μ = P⁻¹ * b
        mu = C \ b

        # Sample: X = μ + L⁻ᵀ * z
        z = randn(rng, T * d)
        x_sample = mu + C.U \ z

        # Update X_list
        for t in 1:T
            for dim in 1:d
                X_list[t][node_idx, dim] = x_sample[(t-1)*d + dim]
            end
        end
    catch e
        # If Cholesky fails, keep current values (shouldn't happen with regularization)
        @warn "Cholesky factorization failed for node " * string(node_idx) * ", keeping current values"
    end
end

"""
    compute_node_quad_form(X_list, node_idx, d, rw_order)

Compute sum of squared r-th order differences for node i.
This matches Python: np.sum(np.diff(X, rw_order, axis=2) ** 2, axis=(1, 2))

Mathematically equivalent to x_i' (K ⊗ I_d) x_i where K = D'D.
"""
function compute_node_quad_form(X_list::Vector{Matrix{Float64}},
                                 node_idx::Int, d::Int, rw_order::Int)
    T = length(X_list)

    # Extract node trajectory: shape (d, T)
    traj = zeros(d, T)
    for t in 1:T
        traj[:, t] = X_list[t][node_idx, :]
    end

    # Compute r-th order differences along time axis
    diff_traj = copy(traj)
    for _ in 1:rw_order
        diff_traj = diff(diff_traj, dims=2)
    end

    # Sum of squared differences
    return sum(diff_traj .^ 2)
end

"""
    procrustes_align_samples!(samples, reference)

Align all samples to the reference embedding using orthogonal Procrustes.
This matches Python DynamicRDPG's post-processing.
"""
function procrustes_align_samples!(samples::Vector{Vector{Matrix{Float64}}},
                                    reference::Vector{Matrix{Float64}})
    for sample in samples
        for t in eachindex(sample)
            Omega = ortho_procrustes_RM(sample[t]', reference[t]')
            sample[t] = sample[t] * Omega
        end
    end
end

"""
    gbdase(A_list, d::Int; rw_order=2, n_burnin=500, n_samples=2000, ...)

Faithful reproduction of Python DynamicRDPG Gibbs sampler.

This implementation exactly matches the Python package for validation and
reproducibility. For optimized performance, use `gbdase_gibbs` instead.

# Key differences from `gbdase_gibbs`:
- Uses banded Cholesky for exact conditional Gaussian sampling
- Half-Cauchy prior for σ (instead of Inverse-Gamma)
- Sequential node updates (matches Python exactly)
- Procrustes aligns all samples to the last sample

# Arguments
* `A_list`: Vector of T adjacency matrices (n × n each)
* `d`: Embedding dimension
* `rw_order`: Order of random walk prior (default: 2)
* `n_burnin`: Number of burn-in iterations (default: 500)
* `n_samples`: Number of posterior samples to collect (default: 2000)
* `prior_std`: Prior standard deviation for initialization (default: 10.0)
* `sample_scale`: Whether to sample the scale parameter (default: true)
* `scale_init`: Initial scale value (default: 1.0)
* `use_procrustes_init`: Initialize with SVD + Procrustes alignment (default: true)
* `align_samples`: Procrustes align all samples to last (default: true)
* `seed`: Random seed for reproducibility (default: 42)

# Returns
Named tuple with:
* `X`: Posterior mean embedding (T × n × d) - note dimension ordering!
* `sigma`: Posterior mean σ values
* `samples`: Vector of all posterior samples
* `sigma_samples`: Vector of all σ samples
* `scale_samples`: Vector of scale parameter samples (if sample_scale=true)
* `K`: Precision matrix used

# Example
```julia
A = [rand(20, 20) for _ in 1:10]
A = [a + a' for a in A]  # symmetrize
result = gbdase(A, 3; n_burnin=100, n_samples=500, seed=42)
```

# Reference
Python DynamicRDPG: https://github.com/joshloyal/DynamicRDPG
arXiv:2509.19748
"""
function gbdase(A_list::V, d::Int;
                          rw_order::Int=2,
                          n_burnin::Int=500,
                          n_samples::Int=2000,
                          prior_std::Float64=10.0,
                          sample_scale::Bool=true,
                          scale_init::Float64=1.0,
                          use_procrustes_init::Bool=true,
                          align_samples::Bool=true,
                          seed::Int=42) where V <: Vector{<:AbstractMatrix}

    rng = Random.Xoshiro(seed)

    T = length(A_list)
    n = size(A_list[1], 1)

    # Build banded precision matrix for random walk prior
    K_banded = rw_precision_banded(T, rw_order)
    K_dense = Matrix(K_banded)  # For quadratic form computation

    # Initialize embeddings via SVD + Procrustes
    X_current = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        emb = svd_embedding(A_list[t], truncated_svd, d)
        X_current[t] = emb.L̂
    end

    if use_procrustes_init
        for t in 2:T
            Omega = ortho_procrustes_RM(X_current[t]', X_current[t-1]')
            X_current[t] = X_current[t] * Omega
        end
    end

    # Initialize σ (per-node) - matching Python's initialization from FIRST differences
    # Python line 141: sigma = np.mean(np.diff(X, axis=2) ** 2, axis=(1, 2))
    # This uses first differences (not rw_order), computing variance of first diffs
    sigma_current = zeros(n)
    for i in 1:n
        sum_sq = 0.0
        for t in 2:T
            for dim in 1:d
                diff_val = X_current[t][i, dim] - X_current[t-1][i, dim]
                sum_sq += diff_val^2
            end
        end
        sigma_current[i] = sum_sq / (d * (T - 1))
        if sigma_current[i] < 1e-6
            sigma_current[i] = 1.0  # fallback
        end
    end
    nu_current = fill(1.0, n)  # Auxiliary variable for Half-Cauchy

    # Initialize scale parameter like Python: 1/var(Y)
    # Python uses lower triangular only (n_dyads)
    subdiag_indices = [(i, j) for i in 1:n for j in 1:i-1]
    y_vec = Float64[]
    for t in 1:T
        for (i, j) in subdiag_indices
            push!(y_vec, A_list[t][i, j])
        end
    end
    y_var = var(y_vec)
    scale_current = y_var > 1e-10 ? 1.0 / y_var : scale_init

    # Precompute XtX matrices
    XtX_list = compute_XtX_list(X_current, d)

    # Degrees of freedom for σ posterior
    df = d * (T - rw_order)

    # Storage for samples
    X_samples = Vector{Vector{Matrix{Float64}}}()
    sigma_samples = Vector{Vector{Float64}}()
    scale_samples = Vector{Float64}()

    total_iter = n_burnin + n_samples

    for iter in 1:total_iter
        # Step 1: Sample each node's trajectory (sequential, matching Python)
        for i in 1:n
            # Remove node i's contribution from XtX
            remove_node_from_XtX!(XtX_list, X_current, i)

            # Sample node i from exact conditional
            sample_node_conditional!(X_current, A_list, K_banded, XtX_list,
                                     sigma_current[i], scale_current, i, d, rng,
                                     rw_order, prior_std)

            # Add node i's contribution back
            add_node_to_XtX!(XtX_list, X_current, i)
        end

        # Step 2: Sample σ from Half-Cauchy posterior
        quad_forms = [compute_node_quad_form(X_current, i, d, rw_order) for i in 1:n]
        sample_half_cauchy_sigma!(sigma_current, nu_current, quad_forms, df, rng)

        # Step 3: Optionally sample scale parameter (matching Python exactly)
        if sample_scale
            # Python uses lower triangular only
            # a = 1e-3 + 0.25 * n_nodes * (n_nodes + 1) * n_time_points
            # b = 1e-3 + 0.5 * sum((y_vec - XXt)^2) + 0.25 * sum(X^2)
            n_dyads = div(n * (n - 1), 2)

            total_sq_error = 0.0
            total_x_sq = 0.0
            for t in 1:T
                P_hat = X_current[t] * X_current[t]'
                for (i, j) in subdiag_indices
                    total_sq_error += (A_list[t][i, j] - P_hat[i, j])^2
                end
                total_x_sq += sum(X_current[t] .^ 2)
            end

            a = 1e-3 + 0.25 * n * (n + 1) * T
            b = 1e-3 + 0.5 * total_sq_error + 0.25 * total_x_sq
            # Python uses Gamma with scale=1/b, which is Gamma(a, 1/b)
            scale_current = rand(rng, Gamma(a, 1.0 / b))
        end

        # Store sample after burn-in
        if iter > n_burnin
            push!(X_samples, [copy(X_current[t]) for t in 1:T])
            push!(sigma_samples, copy(sigma_current))
            push!(scale_samples, scale_current)
        end
    end

    # Post-processing: Procrustes align all samples to the last sample
    # Python first smooths the reference (lines 249-250):
    #   self.samples_['X'][-1] = smooth_positions_procrustes(self.samples_['X'][-1])
    if align_samples && length(X_samples) > 0
        # First smooth the last sample by aligning consecutive time points
        reference = X_samples[end]
        for t in 2:T
            Omega = ortho_procrustes_RM(reference[t]', reference[t-1]')
            reference[t] = reference[t] * Omega
        end
        # Then align all samples to this smoothed reference
        procrustes_align_samples!(X_samples, reference)
    end

    # Compute posterior mean
    X_mean = [zeros(n, d) for t in 1:T]
    for sample in X_samples
        for t in 1:T
            X_mean[t] .+= sample[t]
        end
    end
    for t in 1:T
        X_mean[t] ./= length(X_samples)
    end

    sigma_mean = mean(hcat(sigma_samples...), dims=2)[:]

    # Convert to T × n × d format (matching Python convention)
    X = zeros(Float64, T, n, d)
    for t in 1:T
        X[t, :, :] = X_mean[t]
    end

    # Also convert samples to T × n × d format
    samples_formatted = Vector{Array{Float64,3}}()
    for sample in X_samples
        X_s = zeros(Float64, T, n, d)
        for t in 1:T
            X_s[t, :, :] = sample[t]
        end
        push!(samples_formatted, X_s)
    end

    return (
        X = X,
        sigma = sigma_mean,
        samples = samples_formatted,
        sigma_samples = sigma_samples,
        scale_samples = scale_samples,
        K = K_dense
    )
end
