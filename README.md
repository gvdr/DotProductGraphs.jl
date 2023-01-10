# DotProductGraphs

[![Build Status](https://github.com/gvdr/DotProductGraphs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gvdr/DotProductGraphs.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia Package to work with Random Dot Product Graphs.

It should cover all things RDPG, from SVD embeddings of networks, to more advanced functionalities such as omniembeddings, procrustes alignments, and so on.

## Functioning

For now, you can retrieve an embedding of a given size:

```julia
# we build a random, silly 1,0 matrix A
# and think of it as a Graph adjacency matrix
A = rand(Bool,100,100)

# we build a pair of 4 dimensional embeddings
Left_embedding, Right_embedding = svd_embedding(A,4)
```

And produce a matrix of interaction probability by dot product:

```julia
P = dot_product(Left_embedding, Right_embedding)
```

## TODO

- [ ] Document all the things!
- [ ] Unit Tests All The Things!
- [x] Include basic embedding and dot product capability relying on LinearAlgebra
- [ ] Allow user to choose the svd "engine" (e.g., KrylovKit.JL, Arpack.JL, LowRankApprox.jl, ...) by defining a unified interface
- [ ] Implement Omniembedding functions
    - [x] Automatic block matrix building
    - [ ] Embedding extraction
- [ ] Implement Procrustes Allignment
    - [ ] orthogonal
        - [x] Partial: Just getting the min distances (cheating, as we compute the full rotation matrix)
        - [x] Full: Obtaining rotation matrix
    - [ ] allowing translations
    - [ ] seedless
- [ ] Elbow and principled method to choose dimensionality
- [ ] Register package.
- [ ] Think how to integrate with Graphs.jl and EcologicalNetworks.jl

## Acknowledgement

DotProductGraphs is heavily inspired by [graspologic](https://microsoft.github.io/graspologic), but it focus on being light (focussing on a narrower set of techniques, as we complement wider ecosystems as Graphs.jl and EcologicalNetworks.jl) and squeezing out performance from being written in Julia.
