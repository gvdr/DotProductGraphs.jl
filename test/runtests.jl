using DotProductGraphs
using Test

@testset "rotated_ortho_procrustes" begin
    # set up starting matrices and rotation matrices
    M2d = [0.5 0.5 0.5 0.5 0.0 0.0;
           1.0 1.0 1.0 1.0 0.5 0.5];
    M3d = [0.5 0.5 0.5 0.5 0.0 0.0;
           1.0 1.0 1.0 1.0 0.5 0.5;
           0.5 0.5 0.5 0.5 0.0 0.0];
    θ = π / 6;
    rot2d = [cos(θ) -sin(θ);
             sin(θ)  cos(θ)];
    rot3d = [1      0       0;
             0 cos(θ) -sin(θ);
             0 sin(θ)  cos(θ)];

    # obtain rotated matrices
    R2d = rot2d * M2d
    R3d = rot3d * M3d

    # tests that we can get back to original matrices
    @test M2d ≈ rotated_ortho_procrustes(R2d,M2d)
    @test M3d ≈ rotated_ortho_procrustes(R3d,M3d)
end

@testset "ortho_procrustes_RM" begin
    # set up starting matrix and rotation matrix
    M2d = [0.5 0.5 0.5 0.5 0.0 0.0;
           1.0 1.0 1.0 1.0 0.5 0.5];
    θ = π / 6;
    rot2d = [cos(θ) -sin(θ);
             sin(θ)  cos(θ)];
    or_rot2d = rot2d .* [1 -1;-1 1]


    # obtain rotated matrix
    R2d = rot2d * M2d

    # tests that we can get back the rotation matrix or its conjugate
    @test rot2d ≈ ortho_procrustes_RM(R2d,M2d) || or_rot2d ≈ ortho_procrustes_RM(R2d,M2d)
end

@testset "svd_embedding" begin
    # we build a 2 blocks SBM
    blocks2 = reshape([ones(5,5) zeros(5,5); zeros(5,5) ones(5,5)],10,10)
    L,R = svd_embedding(blocks2,2)
    target_L = target_R = [
    -1.0   0.0
    -1.0   0.0
    -1.0   0.0
    -1.0   0.0
    -1.0   0.0
     0.0  -1.0
     0.0  -1.0
     0.0  -1.0
     0.0  -1.0
     0.0  -1.0]

     # and test whether we can get the embeddings right
     # TODO we should think about what to do for non uniqueness
     # (maybe using rotated_ortho_procrustes?)
     @test L ≈ target_L || L ≈ target_L[:,[2,1]]
     @test R ≈ target_R || R ≈ target_R[:,[2,1]]
end

@testset "dot_product" begin
    # we build a 2 blocks SBM
    blocks2 = reshape([ones(5,5) zeros(5,5); zeros(5,5) ones(5,5)],10,10)
    L,R = svd_embedding(blocks2,2)

    @test dot_product(L,R) ≈ blocks2
end