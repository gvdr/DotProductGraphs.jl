"""
    DotProductGraphs

A Julia package for Random Dot Product Graphs.
"""
module DotProductGraphs

using BlockArrays
using SparseArrays
using LinearAlgebra
using Distributions
using StatsBase
using Arpack
export svd_embedding
export dot_product
include("embedding.jl")

export temp_net_to_bm
export aveMat_unsafe
include("omniembedding.jl")

export d_elbow
include("dimensionality.jl")

export ortho_procrustes_RM
export rotated_ortho_procrustes
export rotated_ortho_procrustes!
export distance_ortho_procrustes
include("alignment.jl")

export constructRDPG
include("constructRDPG.jl")

export TemporalNetworkEmbedding
export embed_temporalnetwork
include("TemporalNetworkEmbedding.jl")

export nearestNeighbours
include("nearestNeighbours.jl")
end
