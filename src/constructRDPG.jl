"""
Constructs RDPG from TemporalNetworkEmbedding; if tsteps is false, the whole time series reconstruction will be returned
"""

function constructRDPG(TempNet::TemporalNetworkEmbedding; tsteps=false)
    # p ∈ s, n, t
    # pR' = a
    # round(a)
    if typeof(tsteps)==Bool
        tsteps = 1:length(TempNet)
    end


    Ls = [TempNet[t][:AL] for t in tsteps]
    Rs = [TempNet[t][:AR]' for t in tsteps]

    rdpg = Ls.*Rs
end
