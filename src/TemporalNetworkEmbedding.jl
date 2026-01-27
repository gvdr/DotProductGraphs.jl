import Base: getindex, lastindex, length, iterate

@inline function Base.getindex(X::TemporalNetworkEmbedding, t::T, side::Symbol=:AL) where {T<:AbstractFloat} 
    X[Int(floor(t))][side]*(1-(t-floor(t))).+X[Int(ceil(t))][side]*(t-floor(t)) # uses linear interpolation between indices
end
# Base.getindex(X::TemporalNetworkEmbedding, t::T) where {T<:AbstractFloat} = reshape(X.A[:,Int(floor(t))], (X.n,X.d)) # this uses the floor of a float index
Base.getindex(X::TemporalNetworkEmbedding, t::T) where {T<:Int} = Dict(:AL => X.AL[:,:,t], :AR => X.AR[:,:,t])

Base.getindex(X::TemporalNetworkEmbedding, t::T, side::Symbol) where {T<:Int} = X[t][side]



Base.getindex(X::TemporalNetworkEmbedding, t::UnitRange{Int64})=TemporalNetworkEmbedding(X.AL[:,:,t],X.AR[:,:,t], X.n, X.d)
Base.lastindex(X::TemporalNetworkEmbedding) = size(X.AL)[3]
not(t::Bool)=!t
withoutNode(X::TemporalNetworkEmbedding, t::Int) = TemporalNetworkEmbedding(X.AL[not.(in.(1:X.n, [t])),:,:],X.AR[not.(in.(1:X.n, [t])),:,:], X.n-1, X.d)

targetNode(X::TemporalNetworkEmbedding, t::Int) = TemporalNetworkEmbedding(X.AL[t:t,:,:],X.AR[t:t,:,:], 1, X.d)

Base.length(X::TemporalNetworkEmbedding) = size(X.AL)[3]


Base.iterate(X::TemporalNetworkEmbedding)= [X[i] for i in 1:length(X)]

# This is a constructor for the class TemporalNetworkEmbedding
function TemporalNetworkEmbedding(TempNet::AbstractVector{T}, d::Int) where T<:AbstractArray
    n = size(TempNet[1], 1)
    num_timesteps = length(TempNet)

    # Pre-compute v0 once for Arpack (deterministic initialization)
    min_dim = minimum(size(TempNet[1]))
    v0 = [Float64(i % 7) for i in 1:min_dim]
    svd_engine(A, d) = Arpack.svds(A, nsv=d, v0=v0)[1]

    # Use Float64 to match SVD output type
    true_dataL = zeros(Float64, n, d, num_timesteps)
    true_dataR = zeros(Float64, n, d, num_timesteps)

    # First embedding (reference for alignment)
    emb = svd_embedding(TempNet[1], svd_engine, d)
    tempL = emb.L̂
    true_dataL[:, :, 1] = tempL
    true_dataR[:, :, 1] = emb.R̂

    # Subsequent embeddings with Procrustes alignment
    for i in 2:num_timesteps
        emb = svd_embedding(TempNet[i], svd_engine, d)
        L, R = emb.L̂, emb.R̂

        # Align to previous embedding
        TL = ortho_procrustes_RM(L', tempL')
        L = L * TL
        R = R * TL

        true_dataL[:, :, i] = L
        true_dataR[:, :, i] = R
        tempL = L
    end

    return TemporalNetworkEmbedding(true_dataL, true_dataR, n, d)
end

"""
    TemporalNetworkEmbedding(TempNet::AbstractVector{T}, d::Int, method::Symbol) where T<:AbstractArray

Construct a TemporalNetworkEmbedding using a specified embedding method.

# Arguments
* `TempNet`: Vector of adjacency matrices representing the temporal network
* `d`: Embedding dimension
* `method`: Embedding method to use. One of:
  - `:procrustes` - Sequential SVD with Procrustes alignment (default behavior)
  - `:omni` - Omnibus embedding (Levin et al. 2017)
  - `:uase` - Unfolded ASE (Gallagher et al. 2021)
  - `:mase` - Multiple ASE (Arroyo et al. 2021)

Note: For `:duase` (Doubly Unfolded ASE), use `duase_embedding()` directly as it
returns a different structure (shared basis + per-time embeddings).

# Example
```julia
A = [rand(10,10) for _ in 1:5]
A = [a + a' for a in A]  # symmetrize

T_procrustes = TemporalNetworkEmbedding(A, 3, :procrustes)
T_omni = TemporalNetworkEmbedding(A, 3, :omni)
T_uase = TemporalNetworkEmbedding(A, 3, :uase)
T_mase = TemporalNetworkEmbedding(A, 3, :mase)
```
"""
function TemporalNetworkEmbedding(TempNet::AbstractVector{T}, d::Int, method::Symbol) where T<:AbstractArray
    n = size(TempNet[1], 1)

    if method == :procrustes
        # Use the default constructor
        return TemporalNetworkEmbedding(TempNet, d)
    elseif method == :omni
        result = omni_embedding(TempNet, d)
        return TemporalNetworkEmbedding(result.L, result.R, n, size(result.L, 2))
    elseif method == :uase
        result = uase_embedding(TempNet, d)
        return TemporalNetworkEmbedding(result.L, result.R, n, size(result.L, 2))
    elseif method == :mase
        # MASE returns common latent positions V (n × d)
        # Replicate across time for TemporalNetworkEmbedding interface
        result = mase_embedding(TempNet, d)
        T_len = length(TempNet)
        actual_d = size(result.V, 2)
        L = zeros(eltype(result.V), n, actual_d, T_len)
        for t in 1:T_len
            L[:, :, t] = result.V
        end
        return TemporalNetworkEmbedding(L, L, n, actual_d)
    else
        error("Unknown embedding method: " * string(method) * ". Use :procrustes, :omni, :uase, or :mase")
    end
end


Base.println(X::TemporalNetworkEmbedding) = println("Time Steps: $(length(X))\nNodes: $(X.n)\nDimension: $(X.d)")
