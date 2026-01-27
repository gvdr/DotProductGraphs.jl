"""
    nearestNeighbours(u, t, TempNet, k; side::Symbol=:AL)

Returns the indices of the k nearest nodes to point u in the embedding at time t.

# Arguments
* `u`: Query point (vector)
* `t`: Time index
* `TempNet`: TemporalNetworkEmbedding
* `k`: Number of nearest neighbors to return
* `side`: Which embedding to use (:AL or :AR)

# Returns
Vector of k indices of nearest nodes
"""
function nearestNeighbours(u, t, TempNet, k; side::Symbol=:AL)
  M = TempNet[t][side]
  n = size(M, 1)

  # Compute squared distances without intermediate allocations
  distances = Vector{Float64}(undef, n)
  @inbounds for i in 1:n
    dist_sq = zero(Float64)
    for j in axes(M, 2)
      dist_sq += abs2(M[i, j] - u[j])
    end
    distances[i] = dist_sq
  end

  # Get indices of k smallest distances
  return partialsortperm(distances, 1:k)
end
