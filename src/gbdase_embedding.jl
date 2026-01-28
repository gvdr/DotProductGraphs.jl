#=
GB-DASE: Generalized Bayesian Dynamic Adjacency Spectral Embedding

Uses r-th order random walk priors to smooth temporal embeddings:
- r=1: penalizes velocity (first differences)
- r=2: penalizes acceleration (second differences) - allows smooth curves
- r=3: penalizes jerk (third differences) - allows smooth acceleration

Reference: arXiv:2509.19748
Implementation based on: https://github.com/joshloyal/DynamicRDPG
=#

using Optim

"""
    difference_matrix(T::Int, r::Int)

Create r-th order difference matrix D such that D*x gives r-th differences.
- r=1: D[t,:] gives x[t+1] - x[t]
- r=2: D[t,:] gives x[t+2] - 2x[t+1] + x[t]

Returns a (T-r) × T matrix.
"""
function difference_matrix(T::Int, r::Int)
    D = Matrix{Float64}(I, T, T)
    for _ in 1:r
        D = diff(D, dims=1)
    end
    return D
end

"""
    rw_precision_matrix(T::Int, r::Int)

Precision matrix K = D'D for r-th order random walk prior.
This is a banded matrix with bandwidth 2r.

Returns a T × T matrix.
"""
function rw_precision_matrix(T::Int, r::Int)
    D = difference_matrix(T, r)
    return D' * D
end

"""
    gbdase_MAP(A_list, d=nothing; rw_order=2, λ_P=1.0, max_iter=50, ...)

GB-DASE embedding using r-th order random walk prior with MAP estimation.

This method smooths temporal embeddings by penalizing the r-th derivative of
node trajectories. Uses alternating optimization: optimize X given σ, then
update σ given X.

# Arguments
* `A_list`: Vector of T adjacency/probability matrices (n × n each)
* `d`: Embedding dimension (or nothing for automatic selection)
* `rw_order`: Order of random walk prior (default: 2, penalizes acceleration)
* `λ_P`: Weight on P reconstruction loss (default: 1.0)
* `node_specific_sigma`: If true, learn σ_i per node (default: true)
* `sigma_init`: Initial σ value (default: 0.1)
* `max_iter`: Maximum alternating iterations (default: 10)
* `tol`: Convergence tolerance for σ (default: 1e-4)
* `use_procrustes_init`: Initialize with SVD + Procrustes alignment (default: true)
* `svd_engine`: SVD function to use (default: truncated_svd)

# Returns
Named tuple with:
* `X`: Per-time embeddings (T × n × d) - consistent with gbdase_faithful
* `sigma`: Final σ values (per-node or single)
* `K`: Precision matrix used

# Algorithm
The objective being optimized (MAP):
  log π(X|Y) ∝ -λ_P * Σ_t ||X(t)X(t)' - A(t)||²_F  [likelihood]
              - Σ_i (1/σᵢ²) * x_i' K x_i            [prior]

where K = D'D is the r-th order difference precision matrix.

# Example
```julia
A = [rand(20, 20) for _ in 1:10]
A = [a + a' for a in A]  # symmetrize
result = gbdase_MAP(A, 3; rw_order=2)
```

# Reference
arXiv:2509.19748 "Generalized Bayesian Dynamic Spectral Embedding"
"""
function gbdase_MAP(A_list::V, d=nothing;
                          rw_order::Int=2,
                          λ_P::Float64=1.0,
                          node_specific_sigma::Bool=true,
                          sigma_init::Float64=0.1,
                          max_iter::Int=10,
                          tol::Float64=1e-4,
                          use_procrustes_init::Bool=true,
                          svd_engine=truncated_svd) where V <: Vector{<:AbstractMatrix}

    T = length(A_list)
    n = size(A_list[1], 1)

    # Build precision matrix for random walk prior
    K = rw_precision_matrix(T, rw_order)

    # Initialize embeddings via SVD
    if d === nothing
        # Use automatic dimension selection on first matrix
        _, Sigma_init, _ = svd_engine(A_list[1], nothing)
        d = length(Sigma_init)
    end

    X_list = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        emb = svd_embedding(A_list[t], svd_engine, d)
        X_list[t] = emb.L̂
    end

    # Procrustes alignment for initialization
    if use_procrustes_init
        for t in 2:T
            Omega = ortho_procrustes_RM(X_list[t]', X_list[t-1]')
            X_list[t] = X_list[t] * Omega
        end
    end

    # Initialize sigma (per-node or global)
    n_sigma = node_specific_sigma ? n : 1
    sigma = fill(sigma_init, n_sigma)

    # Pack/unpack helpers for optimization
    function pack_X(X_lst)
        return vcat([vec(X) for X in X_lst]...)
    end

    function unpack_X(x_vec)
        X_lst = Vector{Matrix{Float64}}(undef, T)
        offset = 0
        for t in 1:T
            X_lst[t] = reshape(x_vec[offset+1:offset+n*d], n, d)
            offset += n * d
        end
        return X_lst
    end

    # Objective function
    function objective(x_vec, sig)
        X_lst = unpack_X(x_vec)

        # P reconstruction loss
        loss_P = 0.0
        for t in 1:T
            P_hat = X_lst[t] * X_lst[t]'
            loss_P += norm(P_hat - A_list[t])^2
        end

        # Random walk prior loss
        loss_rw = 0.0
        for i in 1:n
            traj_i = vcat([X_lst[t][i:i, :] for t in 1:T]...)  # T × d
            sigma_i = node_specific_sigma ? sig[i] : sig[1]

            for dim in 1:d
                x_dim = traj_i[:, dim]
                loss_rw += (x_dim' * K * x_dim) / (sigma_i^2)
            end
        end

        return λ_P * loss_P + loss_rw
    end

    # Gradient of objective w.r.t. X
    function gradient_X(x_vec, sig)
        X_lst = unpack_X(x_vec)
        G = zeros(length(x_vec))

        offset = 0
        for t in 1:T
            # Gradient from P reconstruction: 4 * (XX' - A) * X
            P_hat = X_lst[t] * X_lst[t]'
            dL_dX_P = 4.0 * λ_P * (P_hat - A_list[t]) * X_lst[t]
            G[offset+1:offset+n*d] .= vec(dL_dX_P)
            offset += n * d
        end

        # Gradient from random walk prior
        for i in 1:n
            sigma_i = node_specific_sigma ? sig[i] : sig[1]

            for dim in 1:d
                traj = [X_lst[t][i, dim] for t in 1:T]
                grad_traj = 2.0 * K * traj / (sigma_i^2)

                for t in 1:T
                    idx = (t-1) * n * d + (i-1) * d + dim
                    G[idx] += grad_traj[t]
                end
            end
        end

        return G
    end

    # Update sigma from current X (closed form MLE)
    function update_sigma(X_lst)
        new_sigma = similar(sigma)

        for i in 1:n
            traj_i = vcat([X_lst[t][i:i, :] for t in 1:T]...)  # T × d

            sum_quad = 0.0
            for dim in 1:d
                x_dim = traj_i[:, dim]
                sum_quad += x_dim' * K * x_dim
            end

            df = d * (T - rw_order)
            sigma_sq = sum_quad / df

            if node_specific_sigma
                new_sigma[i] = sqrt(max(sigma_sq, 1e-8))
            else
                new_sigma[1] = get(new_sigma, 1, 0.0) + sum_quad
            end
        end

        if !node_specific_sigma
            df_total = n * d * (T - rw_order)
            new_sigma[1] = sqrt(max(new_sigma[1] / df_total, 1e-8))
        end

        return new_sigma
    end

    # Estimate sigma from initialization (like Python does)
    sigma = update_sigma(X_list)
    for i in eachindex(sigma)
        if sigma[i] < 1e-6
            sigma[i] = sigma_init
        end
    end

    # Single LBFGS optimization with fixed sigma
    x_vec = pack_X(X_list)

    f(x) = objective(x, sigma)
    g!(G, x) = G .= gradient_X(x, sigma)

    opt_result = Optim.optimize(
        f, g!, x_vec,
        LBFGS(m=10),
        Optim.Options(iterations=max_iter * 50, g_tol=1e-6, show_trace=false)
    )
    x_vec = Optim.minimizer(opt_result)

    X_final = unpack_X(x_vec)

    # Convert to 3D array format (T, n, d) - consistent with gbdase_faithful
    X = zeros(eltype(X_final[1]), T, n, d)
    for t in 1:T
        X[t, :, :] = X_final[t]
    end

    return (X=X, sigma=sigma, K=K)
end

"""
    gbdase_gibbs(A_list, d; rw_order=2, n_burnin=500, n_samples=1000, ...)

GB-DASE embedding using Gibbs sampling for full posterior inference.

This provides uncertainty quantification by sampling from the posterior
distribution rather than just finding the MAP estimate.

# Arguments
* `A_list`: Vector of T adjacency matrices
* `d`: Embedding dimension
* `rw_order`: Order of random walk prior (default: 2)
* `n_burnin`: Number of burn-in iterations (default: 500)
* `n_samples`: Number of posterior samples (default: 1000)
* `prior_alpha`: Inverse-Gamma shape parameter for σ² prior (default: 2.0)
* `prior_beta`: Inverse-Gamma scale parameter for σ² prior (default: 0.01)
* `proposal_std`: Proposal standard deviation for MH updates (default: 0.01)
* `use_procrustes_init`: Initialize with SVD + Procrustes (default: true)
* `seed`: Random seed (default: 42)

# Returns
Named tuple with:
* `X`: Posterior mean embedding (T × n × d) - consistent with gbdase_faithful
* `sigma`: Posterior mean σ values
* `samples`: All posterior samples (each T × n × d)
* `sigma_samples`: All σ samples

# Reference
arXiv:2509.19748
"""
function gbdase_gibbs(A_list::V, d::Int;
                      rw_order::Int=2,
                      n_burnin::Int=500,
                      n_samples::Int=1000,
                      prior_alpha::Float64=2.0,
                      prior_beta::Float64=0.01,
                      proposal_std::Float64=0.01,
                      use_procrustes_init::Bool=true,
                      seed::Int=42) where V <: Vector{<:AbstractMatrix}

    rng = Random.Xoshiro(seed)

    T = length(A_list)
    n = size(A_list[1], 1)

    # Build precision matrix
    K = rw_precision_matrix(T, rw_order)

    # Initialize via SVD + Procrustes
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

    # Initialize sigma (per-node)
    sigma_current = fill(0.1, n)

    # Log-posterior (up to constant)
    function log_posterior(X_lst, sig)
        log_p = 0.0

        # P reconstruction term
        for t in 1:T
            P_hat = X_lst[t] * X_lst[t]'
            log_p -= norm(P_hat - A_list[t])^2
        end

        # Random walk prior term
        for i in 1:n
            traj_i = vcat([X_lst[t][i:i, :] for t in 1:T]...)
            for dim in 1:d
                x_dim = traj_i[:, dim]
                log_p -= (x_dim' * K * x_dim) / (2 * sig[i]^2)
                log_p -= T * log(sig[i])
            end
        end

        return log_p
    end

    # Compute quadratic form for sigma update
    function compute_quad_form(X_lst, i)
        traj_i = vcat([X_lst[t][i:i, :] for t in 1:T]...)
        sum_quad = 0.0
        for dim in 1:d
            x_dim = traj_i[:, dim]
            sum_quad += x_dim' * K * x_dim
        end
        return sum_quad
    end

    # Storage
    X_samples = Vector{Vector{Matrix{Float64}}}()
    sigma_samples = Vector{Vector{Float64}}()

    n_accept = 0
    n_total = 0

    for iter in 1:(n_burnin + n_samples)
        # Step 1: Metropolis-Hastings for X
        X_proposed = [copy(X_current[t]) for t in 1:T]

        for i in 1:n
            for t in 1:T
                X_proposed[t][i, :] .+= proposal_std * randn(rng, d)
            end
        end

        log_p_current = log_posterior(X_current, sigma_current)
        log_p_proposed = log_posterior(X_proposed, sigma_current)

        log_alpha = log_p_proposed - log_p_current
        n_total += 1

        if log(rand(rng)) < log_alpha
            X_current = X_proposed
            n_accept += 1
        end

        # Step 2: Gibbs update for σ (conjugate Inverse-Gamma)
        df = d * (T - rw_order)

        for i in 1:n
            quad_form = compute_quad_form(X_current, i)
            alpha_post = prior_alpha + df / 2
            beta_post = prior_beta + quad_form / 2

            sigma_sq = rand(rng, InverseGamma(alpha_post, beta_post))
            sigma_current[i] = sqrt(sigma_sq)
        end

        # Store sample after burn-in
        if iter > n_burnin
            push!(X_samples, [copy(X_current[t]) for t in 1:T])
            push!(sigma_samples, copy(sigma_current))
        end
    end

    # Posterior mean
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

    # Convert to 3D array format (T, n, d) - consistent with gbdase_faithful
    X = zeros(Float64, T, n, d)
    for t in 1:T
        X[t, :, :] = X_mean[t]
    end

    # Also store samples in (T, n, d) format
    samples_formatted = [zeros(Float64, T, n, d) for _ in 1:n_samples]
    for (s, sample) in enumerate(X_samples)
        for t in 1:T
            samples_formatted[s][t, :, :] = sample[t]
        end
    end

    return (X=X, sigma=sigma_mean, samples=samples_formatted, sigma_samples=sigma_samples, K=K)
end

#=
Forecasting and Diagnostics for GB-DASE
=#

"""
    gbdase_forecast_positions(result, k_steps::Int; method=:linear)

Forecast future node positions based on GB-DASE embedding result.

Uses the temporal smoothness encoded in the random walk prior to extrapolate
node trajectories into the future.

# Arguments
* `result`: Output from `gbdase_MAP`, `gbdase_gibbs`, or `gbdase_faithful`
* `k_steps`: Number of future time steps to forecast
* `method`: Extrapolation method (:linear, :quadratic, or :ar)
    - `:linear` - Linear extrapolation from last two time points
    - `:quadratic` - Quadratic extrapolation from last three time points
    - `:ar` - Autoregressive model fitted to trajectory

# Returns
Array of size (k_steps, n, d) containing forecasted positions (consistent with X format).

# Example
```julia
result = gbdase_MAP(A, 3)
X_future = gbdase_forecast_positions(result, 5; method=:linear)
```
"""
function gbdase_forecast_positions(result, k_steps::Int; method::Symbol=:linear)
    # All methods now use X with (T, n, d) format
    if !haskey(result, :X)
        error("Result must contain :X key")
    end

    X = result.X
    T, n, d = size(X)
    X_future = zeros(k_steps, n, d)

    for i in 1:n
        for dim in 1:d
            traj = X[:, i, dim]
            forecasted = _extrapolate_trajectory(traj, k_steps, method)
            for k in 1:k_steps
                X_future[k, i, dim] = forecasted[k]
            end
        end
    end

    return X_future
end

"""
    _extrapolate_trajectory(traj, k_steps, method)

Internal function to extrapolate a single trajectory.
"""
function _extrapolate_trajectory(traj::Vector{<:Real}, k_steps::Int, method::Symbol)
    T = length(traj)
    future = zeros(k_steps)

    if method == :linear
        # Linear extrapolation from last two points
        if T >= 2
            slope = traj[T] - traj[T-1]
            for k in 1:k_steps
                future[k] = traj[T] + k * slope
            end
        else
            fill!(future, traj[T])
        end

    elseif method == :quadratic
        # Quadratic extrapolation from last three points
        if T >= 3
            # Fit quadratic through last 3 points
            t1, t2, t3 = T-2, T-1, T
            y1, y2, y3 = traj[t1], traj[t2], traj[t3]

            # Solve for coefficients: y = a*t^2 + b*t + c
            # Using Lagrange interpolation centered at t=T
            for k in 1:k_steps
                t = T + k
                # Quadratic extrapolation
                L1 = ((t - t2) * (t - t3)) / ((t1 - t2) * (t1 - t3))
                L2 = ((t - t1) * (t - t3)) / ((t2 - t1) * (t2 - t3))
                L3 = ((t - t1) * (t - t2)) / ((t3 - t1) * (t3 - t2))
                future[k] = L1 * y1 + L2 * y2 + L3 * y3
            end
        else
            # Fall back to linear
            future = _extrapolate_trajectory(traj, k_steps, :linear)
        end

    elseif method == :ar
        # Simple AR(1) model
        if T >= 3
            # Estimate AR(1) coefficient from trajectory
            x_prev = traj[1:T-1]
            x_curr = traj[2:T]
            phi = sum(x_prev .* x_curr) / sum(x_prev .* x_prev)
            phi = clamp(phi, -0.99, 0.99)  # Ensure stability

            last_val = traj[T]
            for k in 1:k_steps
                future[k] = phi * last_val
                last_val = future[k]
            end
        else
            # Fall back to constant
            fill!(future, traj[T])
        end

    else
        error("Unknown extrapolation method: " * string(method))
    end

    return future
end

"""
    gbdase_forecast(result, k_steps::Int; method=:linear)

Forecast future adjacency/probability matrices based on GB-DASE embedding.

Uses forecasted positions to reconstruct predicted P matrices.

# Arguments
* `result`: Output from `gbdase_MAP`, `gbdase_gibbs`, or `gbdase_faithful`
* `k_steps`: Number of future time steps to forecast
* `method`: Extrapolation method (passed to `gbdase_forecast_positions`)

# Returns
Vector of k_steps predicted probability matrices (n × n each).

# Example
```julia
result = gbdase_MAP(A, 3)
P_future = gbdase_forecast(result, 5)
```
"""
function gbdase_forecast(result, k_steps::Int; method::Symbol=:linear)
    X_future = gbdase_forecast_positions(result, k_steps; method=method)
    # X_future is now (k_steps, n, d) format
    n = size(X_future, 2)

    P_list = Vector{Matrix{Float64}}(undef, k_steps)
    for k in 1:k_steps
        X_k = X_future[k, :, :]  # (n, d)
        P_list[k] = X_k * X_k'
    end

    return P_list
end

"""
    gbdase_diagnostics(result)

Compute MCMC diagnostics for GB-DASE Gibbs sampling results.

Computes Effective Sample Size (ESS) and R-hat (potential scale reduction factor)
for the posterior samples.

# Arguments
* `result`: Output from `gbdase_gibbs` or `gbdase_faithful` (must contain samples)

# Returns
Named tuple with:
* `ess`: Effective sample size for each node's trajectory
* `rhat`: R-hat statistic (for split-chain diagnostic)
* `acceptance_rate`: Overall acceptance rate (if available)
* `summary`: Summary statistics (mean ESS, min ESS, max R-hat)

# Reference
Gelman et al. (2013) "Bayesian Data Analysis" for ESS and R-hat definitions.

# Example
```julia
result = gbdase_gibbs(A, 3; n_burnin=500, n_samples=1000)
diag = gbdase_diagnostics(result)
println("Mean ESS: ", diag.summary.mean_ess)
println("Max R-hat: ", diag.summary.max_rhat)
```
"""
function gbdase_diagnostics(result)
    # All methods now use :samples with (T, n, d) format
    if !haskey(result, :samples)
        error("Result does not contain posterior samples. Use gbdase_gibbs or gbdase_faithful.")
    end

    samples = result.samples
    first_sample = samples[1]
    T, n, d = size(first_sample)
    get_val = (s, i, dim, t) -> s[t, i, dim]

    n_samples = length(samples)

    if n_samples < 10
        error("Too few samples for reliable diagnostics. Need at least 10 samples.")
    end

    # Compute ESS for each (node, dimension, time) combination
    ess_array = zeros(n, d, T)
    rhat_array = zeros(n, d, T)

    for i in 1:n
        for dim in 1:d
            for t in 1:T
                # Extract chain for this (node, dim, time)
                chain = [get_val(samples[s], i, dim, t) for s in 1:n_samples]

                # Compute ESS using autocorrelation method
                ess_array[i, dim, t] = _compute_ess(chain)

                # Compute R-hat using split-chain method
                rhat_array[i, dim, t] = _compute_rhat_split(chain)
            end
        end
    end

    # Summary statistics
    summary = (
        mean_ess = mean(ess_array),
        min_ess = minimum(ess_array),
        max_ess = maximum(ess_array),
        mean_rhat = mean(rhat_array),
        max_rhat = maximum(rhat_array),
        n_samples = n_samples
    )

    return (
        ess = ess_array,
        rhat = rhat_array,
        summary = summary
    )
end

"""
    _compute_ess(chain)

Compute Effective Sample Size using autocorrelation method.
"""
function _compute_ess(chain::Vector{<:Real})
    n = length(chain)
    if n < 4
        return Float64(n)
    end

    # Center the chain
    chain_centered = chain .- mean(chain)
    var_chain = var(chain)

    if var_chain < 1e-15
        return Float64(n)  # Constant chain
    end

    # Compute autocorrelations
    max_lag = min(n - 1, 100)
    rho = zeros(max_lag + 1)

    for k in 0:max_lag
        if n - k > 0
            rho[k+1] = sum(chain_centered[1:n-k] .* chain_centered[k+1:n]) / ((n - k) * var_chain)
        end
    end

    # Sum autocorrelations using Geyer's initial positive sequence
    # Stop when sum of consecutive pairs becomes negative
    tau = 1.0
    for k in 1:2:max_lag
        if k + 1 <= max_lag
            pair_sum = rho[k+1] + rho[k+2]
            if pair_sum < 0
                break
            end
            tau += 2 * pair_sum
        else
            if rho[k+1] > 0
                tau += 2 * rho[k+1]
            end
        end
    end

    ess = n / max(tau, 1.0)
    return max(1.0, ess)
end

"""
    _compute_rhat_split(chain)

Compute R-hat using split-chain method (single chain split in half).
"""
function _compute_rhat_split(chain::Vector{<:Real})
    n = length(chain)
    if n < 4
        return 1.0
    end

    # Split chain in half
    mid = n ÷ 2
    chain1 = chain[1:mid]
    chain2 = chain[mid+1:2*mid]
    m = 2  # number of chains

    n_half = length(chain1)

    # Within-chain variance
    W = (var(chain1) + var(chain2)) / 2

    # Between-chain variance
    mean1 = mean(chain1)
    mean2 = mean(chain2)
    overall_mean = (mean1 + mean2) / 2
    B = n_half * ((mean1 - overall_mean)^2 + (mean2 - overall_mean)^2) / (m - 1)

    # Pooled variance estimate
    var_plus = ((n_half - 1) * W + B) / n_half

    if W < 1e-15
        return 1.0
    end

    # R-hat
    rhat = sqrt(var_plus / W)
    return rhat
end
