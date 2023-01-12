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

# creates an unitialised block matrix of the right size and type
function initialize_omni(Tₜ::V) where V <: Vector{<:Matrix}

  testMatSize(Tₜ)

	T = length(Tₜ)

	size_of_block = fill(size(Tₜ[1])[1], T)

	blockmatr = PseudoBlockArray{eltype(Tₜ[1])}(undef,
		size_of_block, size_of_block)
	return blockmatr
end

"""
  # fills an unitialised block matrix with the (averaged) matrices from the temporal sequence
"""
function fill_bm!(blockmatrix, Tₜ)

	T = length(Tₜ)

	# TODO we want to eventually allow users to define their own weighting schemes for the omni embedding
	for i in 1:T, j in 1:T
		@inbounds setblock!(blockmatrix, (Tₜ[i] + Tₜ[j]) / 2, i, j)
	end

end

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
function fill_bm!(blockmatrix, Tₜ; weight::Function = aveMat_unsafe)

	T = length(Tₜ)

	# TODO we want to eventually allow users to define their own weighting schemes for the omni embedding
	for i in 1:T, j in 1:T
		@inbounds setblock!(blockmatrix, weight(Tₜ[i],Tₜ[j],i,j), i, j)
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
