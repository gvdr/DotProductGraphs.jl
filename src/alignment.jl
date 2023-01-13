#function procrustes(A,B;Rotation_Matrix::Bool=false,seedless::Bool=false)
#end

"""
    procrustes alignments:

    returns the rotation matrix Ω minimising the distances between Ω*A and B
"""
function ortho_procrustes_RM(A,B)
    U,_,V = svd(A*B')
    Ω = V * U'
    return Ω
end

"""
    modifies in place a matrix A, rotating it to minimise the distances to B
"""
function rotated_ortho_procrustes!(A,B)
    A .= ortho_procrustes_RM(A,B) * A
end

"""
    Returns a matrix Aᵣ, rotating it to minimise the distances to B
"""
function rotated_ortho_procrustes(A,B)
    Aᵣ = ortho_procrustes_RM(A,B) * A
    return Aᵣ
end

"""
    compute the minimum distance between any rotated Ω*A and B
"""
function distance_ortho_procrustes(A,B)
    Ω = ortho_procrustes_RM(A,B)
    D = sum(abs2,B - Ω * A)
    return D
end

#function seedless_procrustes()
#end