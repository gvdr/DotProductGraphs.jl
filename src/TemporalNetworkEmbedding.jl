using Arpack
import Base: getindex, lastindex, length, iterate

"""
    TemporalNetworkEmbedding
    A: The raw embedding array dims = [d*n, :]
    n: The number of nodes in the network
    d: The dimension of the embedding

"""

struct TemporalNetworkEmbedding
    AL::AbstractArray
    AR::AbstractArray
    n::Int
    d::Int
end

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


    n = size(TempNet[1])[1]
    svd_engine(A,d) = Arpack.svds(A,nsv=d, v0=[Float64(i%7) for i in 1:minimum(size(A))])[1]
    

    true_dataL = zeros(Float32, n,d, length(TempNet))
    true_dataR = zeros(Float32, n,d, length(TempNet))

    tempL, tempR = svd_embedding(TempNet[1],svd_engine, convert(Int, d))
    for i in 1:length(TempNet)
        if i != 1
            filled(M) = [zeros(Float32, (1,d));M]
            L, R = svd_embedding(TempNet[i], svd_engine, convert(Int, d))
            TL = ortho_procrustes_RM(L',tempL')
            # TR = ortho_procrustes_RM(R',tempR')
            L .= L*TL
            R .= R*TL
            tempL, tempR = L, R
        else
            L, R = svd_embedding(TempNet[i], svd_engine, convert(Int, d))
        end
        true_dataL[:,:,i] = L
        true_dataR[:,:,i] = R

    end
    return TemporalNetworkEmbedding(true_dataL, true_dataR,n,d)

end


Base.println(X::TemporalNetworkEmbedding) = println("Time Steps: $(length(X))\nNodes: $(X.n)\nDimension: $(X.d)")
