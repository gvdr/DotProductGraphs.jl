# DotProductGraphs

[![Build Status](https://github.com/gvdr/DotProductGraphs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gvdr/DotProductGraphs.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia Package to work with Random Dot Product Graphs.

It should cover all things RDPG, from SVD embeddings of networks, to more advanced functionalities such as omniembeddings, procrustes alignments, and so on.

## Functioning

For now, you can retrieve an embedding of a given size:

```julia
Left_embedding, Right_embedding = svd_embedding(A,4) # where A is an adjacency matrix
```

And produce a matrix of interaction probability by dot product:

```julia
P = dot_product(Left_embedding, Right_embedding)
```

## TODO
