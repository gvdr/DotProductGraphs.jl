using LinearAlgebra

function truncated_svd(Mat, dim)

    L,Σ,Rt = svd(Mat)
    L = L[:,1:dim]
    Σ = Σ[1:dim]
    Rt = Rt[:,1:dim]
    return L, Σ, Rt
  
end

function svd_embedding(A,d; svd_engine = nothing)

  # by default, we use LinearAlgebra.svd
  # eventually, we want to move to KrylovKit.jl
  if isnothing(svd_engine)
    svd_engine = truncated_svd
  end


    L,Σ,R = svd_engine(A,d)
    L̂ = L * diagm(.√Σ)
    R̂ = R * diagm(.√Σ)
    return (L̂ = L̂, R̂ = R̂)
end

clamp_to_prob(x) = x > one(x) ? one(x) :
                   x < zero(x) ? zero(x) : x

function dot_product(L,R; to_prob = true)

  (size(L)[2] == size(R)[2]) || error("Embeddings must be of compatible size")

  Mat = to_prob ? clamp_to_prob.(L * R') : L * R'

  return Mat

end
