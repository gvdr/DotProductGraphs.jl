function testMatSize(Tₜ::V) where V <: Vector{<:AbstractMatrix}
  allequal(size.(Tₜ)) ||
  error("All the matrices in the temporal array MUST be of the same size")
end

# creates an unitialised block matrix of the right size and type
function initialize_omni(Tₜ::V) where V <: Vector{<:AbstractSparseMatrix}
  testMatSize(Tₜ)

  T = length(Tₜ)
  size_of_block = fill(size(Tₜ[1])[1], T)

  blockmatr = BlockArray(undef_blocks,
    SparseMatrixCSC{Float64, Int64},
    size_of_block, size_of_block)
  return blockmatr
end

# creates an unitialised block matrix of the right size and type (BlockArrays 1.x compatible)
function initialize_omni(Tₜ::V) where V <: Vector{<:Matrix}
  testMatSize(Tₜ)

  T = length(Tₜ)
  n = size(Tₜ[1], 1)
  elem_type = eltype(Tₜ[1])

  # Use mortar to create a blocked view of a matrix of matrices
  # First create the matrix of block matrices
  blocks = Matrix{Matrix{elem_type}}(undef, T, T)
  for i in 1:T, j in 1:T
    blocks[i, j] = Matrix{elem_type}(undef, n, n)
  end
  return mortar(blocks)
end

"""
  # fills an unitialised block matrix with the (averaged) matrices from the temporal sequence
"""
#function fill_bm!(blockmatrix, Tₜ)
#
#	T = length(Tₜ)
#
#	# TODO we want to eventually allow users to define their own weighting schemes for the omni embedding
#	for i in 1:T, j in 1:T
#		@inbounds setblock!(blockmatrix, (Tₜ[i] + Tₜ[j]) / 2, i, j)
#	end
#
#end

"""
  entrywise average of two matrices `weight(Tᵢ,Tⱼ,i,j) = (Tᵢ + Tⱼ)/2`).
"""
function aveMat_unsafe(Tᵢ,Tⱼ,i,j)
  return @inbounds (Tᵢ .+ Tⱼ) ./ 2
end

"""
  fills an unitialised block matrix with the matrices from the temporal sequence
  `weight(Tᵢ::M,T::Mⱼ,i::Int,j::Int) where M <: AbstractMatrix` defines how to aggregate the matrices of indeces `i` and `j`
  (for example, in the default function `weight(Tᵢ,Tⱼ,i,j) = (Tᵢ + Tⱼ)/2`).
"""
function fill_bm!(blockmatrix, Tₜ, weight::Function = aveMat_unsafe)

	T = length(Tₜ)

	# TODO we want to eventually allow users to define their own weighting schemes for the omni embedding
	for i in 1:T, j in 1:T
		@inbounds blockmatrix[Block(i, j)] = weight(Tₜ[i],Tₜ[j],i,j)
	end

end

"""
	transforms a sequence of adjacency matrix into an omni embedding
"""
function temp_net_to_bm(Tₜ::M) where M Vector{<:AbstractMatrix}

	bm = initialize_omni(Tₜ)

	fill_bm!(bm, Tₜ)

	return bm

end

"""
	transforms a sequence of adjacency matrix into an omni embedding and transforms it into an object of type `as_type`
"""
function temp_net_to_bm(Tₜ::M, as_type) where M <: Vector{<:AbstractMatrix}
	return as_type(temp_net_to_bm(Tₜ))
end

"""
    omni_extract(M::AbstractMatrix, n::Int, T::Int, d; svd_engine=truncated_svd)

Extract per-time embeddings from an omnibus block matrix.

Given an omnibus matrix M of size (nT × nT), performs SVD and extracts
the LEFT singular vectors U, partitioned by time and scaled by √Σ.

# Arguments
* `M`: The omnibus block matrix (nT × nT)
* `n`: Number of nodes per time point
* `T`: Number of time points
* `d`: Embedding dimension (or nothing for automatic selection)
* `svd_engine`: Function to perform SVD (default: truncated_svd)

# Returns
Named tuple with:
* `L`: Left embeddings (n × d × T)
* `R`: Right embeddings (same as L for symmetric matrices)
* `Sigma`: Singular values

# Reference
Levin et al. (2017) "A central limit theorem for an omnibus embedding" IEEE ICDMW
"""
function omni_extract(M::AbstractMatrix, n::Int, T::Int, d=nothing; svd_engine=truncated_svd)
    U, Sigma, V = svd_engine(M, d)
    sqrt_Sigma = sqrt.(Sigma)
    actual_d = length(Sigma)

    L = zeros(eltype(U), n, actual_d, T)
    for t in 1:T
        idx = ((t-1)*n+1):(t*n)
        L[:, :, t] = U[idx, :] .* sqrt_Sigma'
    end
    return (L=L, R=L, Sigma=Sigma)
end

"""
    omni_embedding(A_list, d=nothing; svd_engine=truncated_svd, weight=aveMat_unsafe)

Compute the omnibus embedding for a list of adjacency matrices.

The omnibus embedding constructs a block matrix where M[i,j] = weight(A_i, A_j)
(by default the average (A_i + A_j)/2), then extracts per-time embeddings from
the LEFT singular vectors.

# Arguments
* `A_list`: Vector of adjacency matrices (all must be same size)
* `d`: Embedding dimension (or nothing for automatic selection)
* `svd_engine`: Function to perform SVD (default: truncated_svd)
* `weight`: Function to compute off-diagonal blocks (default: aveMat_unsafe)

# Returns
Named tuple with:
* `L`: Left embeddings (n × d × T)
* `R`: Right embeddings (same as L)
* `Sigma`: Singular values

# Example
```julia
A = [rand(10,10) for _ in 1:5]
A = [a + a' for a in A]  # symmetrize
result = omni_embedding(A, 3)
```

# Reference
Levin et al. (2017) "A central limit theorem for an omnibus embedding" IEEE ICDMW
"""
function omni_embedding(A_list::V, d=nothing; svd_engine=truncated_svd, weight=aveMat_unsafe) where V <: Vector{<:AbstractMatrix}
    n = size(A_list[1], 1)
    T = length(A_list)
    M = temp_net_to_bm(A_list)
    M_mat = M isa AbstractSparseMatrix ? M : Matrix(M)
    return omni_extract(M_mat, n, T, d; svd_engine=svd_engine)
end
