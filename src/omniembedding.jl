  # creates an unitialised block matrix of the right size and type
function initialize_omni(Tₜ::V) where V <: Vector{<:AbstractSparseMatrix}
    T = length(Tₜ)
  
    size_of_block = fill(size(Tₜ[1])[1],T)
  
    blockmatr = BlockArray(undef_blocks,
                           SparseMatrixCSC{Float64, Int64},
                           size_of_block, size_of_block)
    return blockmatr
  end
  
  # creates an unitialised block matrix of the right size and type
  function initialize_omni(Tₜ::V) where V <: Vector{<:Matrix}
    T = length(Tₜ)
  
    size_of_block = fill(size(Tₜ[1])[1],T)
  
    blockmatr = PseudoBlockArray{eltype(Tₜ[1])}(undef,
                           size_of_block, size_of_block)
    return blockmatr
  end
  

  # fills an unitialised block matrix with the (averaged) matrices from the temporal sequence
  function fill_bm!(blockmatrix, Tₜ)
  
    T = length(Tₜ)
  
    # TODO we want to eventually allow users to define their own weighting schemes for the omni embedding
    for i in 1:T, j in 1:T
     @inbounds setblock!(blockmatrix, (Tₜ[i] + Tₜ[j])/2, i, j)
    end
  
  end
  

  # transforms a sequence of adjacency matrix into an omni embedding
  function temp_net_to_bm(Tₜ)
  
    bm = initialize_omni(Tₜ)
  
    fill_bm!(bm, Tₜ)
    
    return bm
  
  end