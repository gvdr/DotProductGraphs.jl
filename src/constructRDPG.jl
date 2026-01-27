"""
    constructRDPG(TempNet::TemporalNetworkEmbedding; tsteps=false)

Reconstructs adjacency matrices from a TemporalNetworkEmbedding.

# Arguments
* `TempNet`: The temporal network embedding
* `tsteps`: Time steps to reconstruct (default: all time steps)

# Returns
Vector of reconstructed adjacency matrices (one per time step)
"""
function constructRDPG(TempNet::TemporalNetworkEmbedding; tsteps=false)
    if tsteps === false
        tsteps = 1:length(TempNet)
    end

    # Matrix multiplication L * R' for each time step
    return [TempNet[t][:AL] * TempNet[t][:AR]' for t in tsteps]
end
