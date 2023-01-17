
"""
    `d_elbow(Σ)`

Estimates the optimal embedding dimension using the Elbow method.

Used by default in `truncated_svd` is `d` is left to `nothing`.

See M. Zhu, and A. Ghodsi (2006). Automatic dimensionality selection from the scree plot via the use of profile likelihood. _Computational Statistics and Data Analysis_, Vol. 51, 918--930.
"""
function d_elbow(Σ::V) where V <: Vector{<:Real}

    N = length(Σ)

    ll = Array{Float64, 1}(undef, N-1)
    for d in 1:(N-1)
        μ₁ = mean(Σ[1:d])
        μ₂ = mean(Σ[d+1:end])
        σ = √(( sum(abs2,Σ[1:d] .- μ₁) + sum(abs2,Σ[d+1:end] .- μ₂) ) / (N-2))
        ll[d] = sum(logpdf.(Normal(μ₁,σ),Σ[1:d])) + sum(logpdf.(Normal(μ₂,σ),Σ[d+1:end]))
    end

    return argmax(ll)
end


