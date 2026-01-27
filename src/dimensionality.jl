
"""
    `d_elbow(Σ)`

Estimates the optimal embedding dimension using the Elbow method.

Used by default in `truncated_svd` is `d` is left to `nothing`.

See M. Zhu, and A. Ghodsi (2006). Automatic dimensionality selection from the scree plot via the use of profile likelihood. _Computational Statistics and Data Analysis_, Vol. 51, 918--930.
"""
function d_elbow(Σ::V) where V <: Vector{<:Real}

    N = length(Σ)

    ll = Vector{Float64}(undef, N-1)
    @inbounds for d in 1:(N-1)
        # Use views to avoid allocations
        Σ_left = @view Σ[1:d]
        Σ_right = @view Σ[d+1:end]

        μ₁ = mean(Σ_left)
        μ₂ = mean(Σ_right)

        # Compute variance terms without intermediate allocations
        ss_left = sum(x -> abs2(x - μ₁), Σ_left)
        ss_right = sum(x -> abs2(x - μ₂), Σ_right)
        σ = sqrt((ss_left + ss_right) / (N-2))

        # Compute log-likelihood directly
        dist₁ = Normal(μ₁, σ)
        dist₂ = Normal(μ₂, σ)
        ll[d] = sum(x -> logpdf(dist₁, x), Σ_left) + sum(x -> logpdf(dist₂, x), Σ_right)
    end

    return argmax(ll)
end


