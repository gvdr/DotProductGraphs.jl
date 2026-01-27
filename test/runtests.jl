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

@testset "TemporalNetworkEmbedding" begin
    TemporalNetworkEmbedding([], [], 0, 0)
end


@testset "constructRDPG" begin
    A = [[1 1 1;
          1 1 1;
          1 1 1],
         [1 1 0;
          1 1 0;
          1 1 0]]
    T = TemporalNetworkEmbedding(A,2)
    @test A ≈ constructRDPG(T)
end

@testset "constructRDPG" begin
    A = [[1 1 1;
          1 1 1;
          1 1 1],
         [1 1 0;
          1 1 0;
          1 0 0]]
    T = TemporalNetworkEmbedding(A,2)
    u = [0 1]'
    @test [3] == nearestNeighbours(u, 2, T, 1)
end


@testset "Base.getindex" begin
    A = [[1 1 1;
          1 1 1;
          1 1 1],
         [1 1 0;
          1 1 0;
          1 0 0]]
    T = TemporalNetworkEmbedding(A,2)

    @test T[1.0] == T[1,:AL]
    @test T[1.9999] ≈ T[2,:AL] atol=1e-3  # Allow small tolerance for interpolation near boundary
    @test T[1.4] ≈ 0.6*T[1,:AL]+0.4*T[2,:AL]
end

@testset "omni_embedding" begin
    # Create simple symmetric test matrices
    A = [Float64[1 1 0; 1 1 0; 0 0 1] for _ in 1:3]

    result = omni_embedding(A, 2)

    # Check output structure
    @test haskey(result, :L)
    @test haskey(result, :R)
    @test haskey(result, :Sigma)

    # Check dimensions: n=3, d=2, T=3
    @test size(result.L) == (3, 2, 3)
    @test size(result.R) == (3, 2, 3)
    @test length(result.Sigma) == 2

    # L and R should be equal for symmetric matrices
    @test result.L ≈ result.R
end

@testset "uase_embedding" begin
    # Create simple symmetric test matrices
    A = [Float64[1 1 0; 1 1 0; 0 0 1] for _ in 1:3]

    result = uase_embedding(A, 2)

    # Check output structure
    @test haskey(result, :L)
    @test haskey(result, :R)
    @test haskey(result, :Sigma)

    # Check dimensions: n=3, d=2, T=3
    @test size(result.L) == (3, 2, 3)
    @test size(result.R) == (3, 2, 3)
    @test length(result.Sigma) == 2
end

@testset "duase_embedding" begin
    # Create simple symmetric test matrices
    A = [Float64[1 1 0; 1 1 0; 0 0 1] for _ in 1:3]

    result = duase_embedding(A, 2)

    # Check output structure
    @test haskey(result, :X_shared)
    @test haskey(result, :Y)
    @test haskey(result, :Sigma)

    # Check dimensions: n=3, d=2, T=3
    @test size(result.X_shared) == (3, 2)
    @test size(result.Y) == (3, 2, 3)
    @test length(result.Sigma) == 2
end

@testset "mase_embedding" begin
    # Create simple symmetric test matrices
    A = [Float64[1 1 0; 1 1 0; 0 0 1] for _ in 1:3]

    result = mase_embedding(A, 2)

    # Check output structure
    @test haskey(result, :V)
    @test haskey(result, :Sigma)

    # Check dimensions: n=3, d=2
    @test size(result.V) == (3, 2)
    @test length(result.Sigma) == 2

    # Test with scores
    result_with_scores = mase_embedding(A, 2; return_scores=true)
    @test haskey(result_with_scores, :scores)
    @test length(result_with_scores.scores) == 3
    # Score matrices should be d × d
    @test all(size(s) == (2, 2) for s in result_with_scores.scores)
end

@testset "TemporalNetworkEmbedding with method" begin
    # Create test matrices - need to be larger for Arpack
    n = 10
    A = [rand(n, n) for _ in 1:3]
    A = [a + a' for a in A]  # symmetrize

    # Test :omni method
    T_omni = TemporalNetworkEmbedding(A, 3, :omni)
    @test T_omni.n == n
    @test T_omni.d == 3
    @test size(T_omni.AL) == (n, 3, 3)

    # Test :uase method
    T_uase = TemporalNetworkEmbedding(A, 3, :uase)
    @test T_uase.n == n
    @test T_uase.d == 3
    @test size(T_uase.AL) == (n, 3, 3)

    # Test :mase method
    T_mase = TemporalNetworkEmbedding(A, 3, :mase)
    @test T_mase.n == n
    @test T_mase.d == 3
    @test size(T_mase.AL) == (n, 3, 3)
    # MASE should have same embedding for all time points
    @test T_mase.AL[:,:,1] ≈ T_mase.AL[:,:,2]
    @test T_mase.AL[:,:,2] ≈ T_mase.AL[:,:,3]

    # Test invalid method
    @test_throws ErrorException TemporalNetworkEmbedding(A, 3, :invalid_method)
end

@testset "Multi-graph embedding reconstruction" begin
    # Test that embeddings can approximately reconstruct the original matrices
    n = 10
    d = 3

    # Create a simple low-rank matrix and add small perturbations
    L_true = randn(n, d)
    P_base = clamp.(L_true * L_true', 0, 1)
    A = [P_base + 0.1 * randn(n, n) for _ in 1:5]
    A = [clamp.(a + a', 0, 1) / 2 for a in A]  # symmetrize and normalize

    # Test omni embedding reconstruction
    result_omni = omni_embedding(A, d)
    for t in 1:5
        L_t = result_omni.L[:, :, t]
        P_reconstructed = L_t * L_t'
        # Reconstruction should be reasonable (not exact due to noise)
        @test size(P_reconstructed) == (n, n)
    end

    # Test uase embedding reconstruction
    result_uase = uase_embedding(A, d)
    for t in 1:5
        L_t = result_uase.L[:, :, t]
        P_reconstructed = L_t * L_t'
        @test size(P_reconstructed) == (n, n)
    end

    # Test duase embedding reconstruction
    result_duase = duase_embedding(A, d)
    for t in 1:5
        Y_t = result_duase.Y[:, :, t]
        P_reconstructed = result_duase.X_shared * Y_t'
        @test size(P_reconstructed) == (n, n)
    end
end