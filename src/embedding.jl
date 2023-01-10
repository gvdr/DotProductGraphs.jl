using LinearAlgebra

# some rice over LinearAlgebra.svd to return a truncated Singular Value Decomposition
function truncated_svd(Mat, dim)

    L,Σ,Rt = svd(Mat)
    L = L[:,1:dim]
    Σ = Σ[1:dim]
    Rt = Rt[:,1:dim]
    return L, Σ, Rt
  
end

# computes an svd embedding of a matrix A of size d
# using the svd engine defined by the user
# (or the defaulut one from LinearAlgebra)
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

# clamps a value x to be in the intervall [0,1]
clamp_to_prob(x) = x > one(x) ? one(x) :
                   x < zero(x) ? zero(x) : x


# computes the dot product between two embeddings
# and clamps values to be probablities
function dot_product(L,R; to_prob = true)

  (size(L)[2] == size(R)[2]) || error("Embeddings must be of compatible size")

  Mat = to_prob ? clamp_to_prob.(L * R') : L * R'

  return Mat

end
