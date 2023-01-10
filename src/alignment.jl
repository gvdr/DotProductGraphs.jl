#function procrustes(A,B;Rotation_Matrix::Bool=false,seedless::Bool=false)
#end

function ortho_procrustes_RM(A,B)
    U,_,V = svd(A*B')
    Ω = V * U'
    return Ω
end

function rotated_ortho_procrustes!(A,B)
    A .= orthogonal_procrustes_RM(A,B) * A
end

function distance_ortho_procrustes(A,B)
    Ω = orthogonal_procrustes_RM(A,B)
    D = sum(abs2,B - Ω * A)
    return D
end

#function seedless_procrustes()
#end