module DotProductGraphs

using BlockArrays
using SparseArrays
using LinearAlgebra

export svd_embedding
export dot_product
include("embedding.jl")

export temp_net_to_bm
include("omniembedding.jl")

export ortho_procrustes_RM
export rotated_ortho_procrustes!
export distance_ortho_procrustes
include("alignment.jl")


end
