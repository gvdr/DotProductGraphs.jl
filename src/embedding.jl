using LinearAlgebra

"""
  truncated_svd(Mat, dim)

Computes a truncated Singular Value Decomposition

Just some rice over the default LinearAlgebra.svd to return a truncated Singular Value Decomposition of dimension `d`.

# Arguments
* `Mat`: The matrix we want to factorize
* `d`: The dimension of the factorization

# Notes
* This is in general not efficient, as we are computing first a full factorization and then truncating it. We should consider using more optimized methods.

# Examples
```julia
julia> # we first build a 2 blocks matrix:
julia> block_matrix = reshape([ones(5,5) zeros(5,5); zeros(5,5) ones(5,5)],10,10)
julia> # and then decompose it:
julia> L, Σ, Rt = truncated_svd(block_matrix,2)
```
"""
function truncated_svd(Mat::T, d::Int) where T <: AbstractMatrix

    L,Σ,Rt = svd(Mat)
    L = L[:,1:d]
    Σ = Σ[1:d]
    Rt = Rt[:,1:d]
    return L, Σ, Rt
  
end

"""
  svd_embedding(A,d; svd_engine = nothing)

Computes an SVD embedding of an adjacency matrix `A` of dimension `d`, using `svd_engine` to perform the SVD factorization

Given an adjacency matrix A, the function returns its node embedding using the Singular Value Decomposition, as it is done in Random Dot Product Graphs.
The functions accepts a user defined `svd_engine` that can be different from the detaul `svd()` coming from `LinearAlgebra`.
In that case, `svd_engine` must be a function of the same form of `truncated_svd`.

# Arguments
* `A`: Adjacency matrix of the graph to embed.
* `d`: Dimension of the embedding
* `svd_engine`: The function used to perform the SVD factorization, by default `svd()` from `LinearAlgebra`

# Notes
* The matrix `A` can be asymmetric (that is, the graph can be directed) and rectangular (e.g., for bipartite graphs).

# Examples
```julia
julia> # we first build a 2 blocks matrix:
julia> block_matrix = reshape([ones(5,5) zeros(5,5); zeros(5,5) ones(5,5)],10,10)
julia> # and then decompose it:
julia> L,R = svd_embedding(block_matrix,2)
```
"""
function svd_embedding(A::T,d::Int,svd_engine::F) where {T <: AbstractMatrix, F<:Function}

    # decompose A in d dimensions
    L,Σ,R = svd_engine(A,d)

    # Obtain embeddings
    L̂ = L * diagm(.√Σ)
    R̂ = R * diagm(.√Σ)
    
    return (L̂ = L̂, R̂ = R̂)
end

# if no function is specified, use `truncated_svd`
function svd_embedding(A::T,d::Int) where T <: AbstractMatrix
    return svd_embedding(A,d,truncated_svd) 
end



# clamps a value x to be in the intervall [0,1]
clamp_to_prob(x::Number) = x > one(x) ? one(x) :
                   x < zero(x) ? zero(x) : x


"""
  dot_product(L,R;to_prob = true)

Computes the dot product between two embeddings and gives back a matrix of interaction probabilities

The left and right embeddings must be of the same dimension (e.g., `size(L)[2] == size(R)[2]`) but can be of different number of nodes (e.g. for bipartite networks).

# Arguments
* `L`: Left embedding
* `R`: Right embedding
* `to_prob`: (optional) whether to clamp the outcomes to the interval [0,1]

# Examples
```julia
julia> # we first build a 2 blocks matrix:
julia> block_matrix = reshape([ones(5,5) zeros(5,5); zeros(5,5) ones(5,5)],10,10)
julia> # and then decompose it:
julia> L,R = svd_embedding(block_matrix,2)
julia> dot_product(L,R) ≈ block_matrix
```
"""
function dot_product(L::M,R::M; to_prob::Bool = true) where M <: AbstractMatrix

  (size(L)[2] == size(R)[2]) || error("Embeddings must be of compatible size")

  Mat = to_prob ? clamp_to_prob.(L * R') : L * R'

  return Mat

end
