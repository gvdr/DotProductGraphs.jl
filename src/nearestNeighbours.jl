"""
Returns a tuple of the indeces of the neares nodes to u
"""

function nearestNeighbours(u,t,TempNet, k;side::Symbol=:AL) 
  M = TempNet[t][side]
  subtract_func(m) = m-u
  direction_vecs = [subtract_func(m) for m in eachrow(M)]
  
  v = collect(enumerate(direction_vecs))

  
  û = [v[1] for v in partialsort(v,1:k, by=x->sum(abs2, x[2]))]
end