function initialize_omni(Tₜ::V) where V <: Vector{<:AbstractSparseMatrix}
    T = length(Tₜ)
  
    size_of_block = fill(size(Tₜ[1])[1],T)
  
    blockmatr = BlockArray(undef_blocks,
                           SparseMatrixCSC{Float64, Int64},
                           size_of_block, size_of_block)
    return blockmatr
  end
  
  function initialize_omni(Tₜ::V) where V <: Vector{<:Matrix}
    T = length(Tₜ)
  
    size_of_block = fill(size(Tₜ[1])[1],T)
  
    blockmatr = PseudoBlockArray{eltype(Tₜ[1])}(undef,
                           size_of_block, size_of_block)
    return blockmatr
  end
  
  function fill_bm!(blockmatrix, Tₜ)
  
    T = length(Tₜ)
  
    for i in 1:T, j in 1:T
     @inbounds setblock!(blockmatrix, (Tₜ[i] + Tₜ[j])/2, i, j)
    end
  
  end
  
  function temp_net_to_bm(Tₜ)
  
    bm = initialize_omni(Tₜ)
  
    fill_bm!(bm, Tₜ)
    
    return bm
  
  end