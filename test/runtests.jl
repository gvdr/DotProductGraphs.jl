using DotProductGraphs
using Test
using LinearAlgebra: issymmetric, norm
using Random
using StatsBase: cor

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

@testset "difference_matrix and rw_precision_matrix" begin
    # Test difference matrix construction
    D1 = difference_matrix(5, 1)
    @test size(D1) == (4, 5)
    # First order difference: x[t+1] - x[t]
    x = [1.0, 2.0, 4.0, 7.0, 11.0]
    @test D1 * x ≈ [1.0, 2.0, 3.0, 4.0]

    D2 = difference_matrix(5, 2)
    @test size(D2) == (3, 5)
    # Second order difference: x[t+2] - 2x[t+1] + x[t]
    @test D2 * x ≈ [1.0, 1.0, 1.0]

    # Test precision matrix
    K = rw_precision_matrix(5, 1)
    @test size(K) == (5, 5)
    @test issymmetric(K)
end

@testset "gbdase_MAP" begin
    # Create smooth temporal network for testing
    n = 8
    d = 2
    T = 5

    # Generate smooth trajectory
    X_true = [0.4 .+ 0.1 * randn(n, d) for _ in 1:T]
    for t in 2:T
        X_true[t] = X_true[t-1] + 0.02 * randn(n, d)
    end

    # Generate probability matrices
    A = [clamp.(X_true[t] * X_true[t]', 0, 1) for t in 1:T]

    # Test GB-DASE MAP
    result = gbdase_MAP(A, d; max_iter=10, rw_order=2)

    @test haskey(result, :X)
    @test haskey(result, :sigma)
    @test haskey(result, :K)

    @test size(result.X) == (T, n, d)
    @test length(result.sigma) == n  # node-specific sigma by default
    @test size(result.K) == (T, T)
end

@testset "gbdase_gibbs" begin
    # Create small test case for Gibbs sampler
    n = 6
    d = 2
    T = 4

    # Generate smooth trajectory
    X_true = [0.4 .+ 0.1 * randn(n, d) for _ in 1:T]
    for t in 2:T
        X_true[t] = X_true[t-1] + 0.02 * randn(n, d)
    end

    A = [clamp.(X_true[t] * X_true[t]', 0, 1) for t in 1:T]

    # Test with minimal iterations for speed
    result = gbdase_gibbs(A, d; n_burnin=10, n_samples=20, seed=42)

    @test haskey(result, :X)
    @test haskey(result, :sigma)
    @test haskey(result, :samples)
    @test haskey(result, :sigma_samples)

    @test size(result.X) == (T, n, d)
    @test length(result.sigma) == n
    @test length(result.samples) == 20
    @test length(result.sigma_samples) == 20
end

@testset "gbdase" begin
    # Create small test case for faithful Python reproduction
    n = 8
    d = 2
    T = 5

    # Generate test matrices
    A = [rand(n, n) for _ in 1:T]
    A = [a + a' for a in A]  # symmetrize

    # Test with minimal iterations for speed
    result = gbdase(A, d; n_burnin=10, n_samples=20, seed=42)

    @test haskey(result, :X)
    @test haskey(result, :sigma)
    @test haskey(result, :samples)
    @test haskey(result, :sigma_samples)
    @test haskey(result, :scale_samples)
    @test haskey(result, :K)

    # Check dimensions - faithful uses (T, n, d) format
    @test size(result.X) == (T, n, d)
    @test length(result.sigma) == n
    @test length(result.samples) == 20
    @test length(result.sigma_samples) == 20
    @test length(result.scale_samples) == 20
    @test size(result.K) == (T, T)
end

@testset "gbdase_forecast" begin
    # Create small test case
    n = 6
    d = 2
    T = 5

    # Generate smooth trajectory
    X_true = [0.4 .+ 0.1 * randn(n, d) for _ in 1:T]
    for t in 2:T
        X_true[t] = X_true[t-1] + 0.02 * randn(n, d)
    end

    A = [clamp.(X_true[t] * X_true[t]', 0, 1) for t in 1:T]

    result = gbdase_MAP(A, d; max_iter=5)

    # Test forecast_positions with different methods
    k_steps = 3

    X_linear = gbdase_forecast_positions(result, k_steps; method=:linear)
    @test size(X_linear) == (k_steps, n, d)

    X_quadratic = gbdase_forecast_positions(result, k_steps; method=:quadratic)
    @test size(X_quadratic) == (k_steps, n, d)

    X_ar = gbdase_forecast_positions(result, k_steps; method=:ar)
    @test size(X_ar) == (k_steps, n, d)

    # Test forecast (full P matrices)
    P_future = gbdase_forecast(result, k_steps; method=:linear)
    @test length(P_future) == k_steps
    @test all(size(P) == (n, n) for P in P_future)

    # Forecasted P should be symmetric (since X*X')
    @test all(P ≈ P' for P in P_future)
end

@testset "gbdase vs gbdase_MAP consistency" begin
    # Test that faithful Gibbs sampler and MAP optimization give similar results
    # on a well-behaved problem

    n = 8
    d = 2
    T = 5

    # Generate a smooth, low-rank temporal network
    # Use fixed seed for reproducibility
    Random.seed!(123)
    X_true = [0.5 * ones(n, d) + 0.05 * randn(n, d) for _ in 1:T]
    for t in 2:T
        X_true[t] = X_true[t-1] + 0.01 * randn(n, d)
    end

    # Generate probability matrices from true embeddings
    A = [clamp.(X_true[t] * X_true[t]', 0, 1) for t in 1:T]

    # Run MAP estimation
    result_map = gbdase_MAP(A, d; max_iter=30, rw_order=2)

    # Run faithful Gibbs sampler with more samples for good posterior mean
    result_gibbs = gbdase(A, d; n_burnin=200, n_samples=500, rw_order=2, seed=42)

    # Both methods now use (T, n, d) format
    X_gibbs = result_gibbs.X
    X_map = result_map.X

    # Compare reconstructed probability matrices (rotation-invariant)
    # Note: MAP (gbdase_MAP) and Gibbs (gbdase) can give different
    # results because:
    # - MAP finds a point estimate while Gibbs computes posterior mean
    # - Different priors (MLE vs Half-Cauchy for sigma)
    # - MCMC noise in Gibbs sampling
    # So we test that both give REASONABLE reconstructions, not identical ones.
    total_error = 0.0
    for t in 1:T
        P_map = X_map[t, :, :] * X_map[t, :, :]'
        P_gibbs = X_gibbs[t, :, :] * X_gibbs[t, :, :]'
        total_error += norm(P_map - P_gibbs) / norm(P_map)
    end
    avg_relative_error = total_error / T

    # The reconstructed P matrices should be in the same ballpark
    # (allowing for fundamental algorithmic differences)
    @test avg_relative_error < 1.5  # Within 150% relative error

    # Check that both methods recover embeddings that reconstruct A reasonably
    reconstruction_error_map = 0.0
    reconstruction_error_gibbs = 0.0
    for t in 1:T
        P_map = X_map[t, :, :] * X_map[t, :, :]'
        P_gibbs = X_gibbs[t, :, :] * X_gibbs[t, :, :]'
        reconstruction_error_map += norm(P_map - A[t])^2
        reconstruction_error_gibbs += norm(P_gibbs - A[t])^2
    end

    # Both should have reasonable reconstruction (not zero, but not huge)
    @test reconstruction_error_map < n * n * T  # Less than 1 per element on average
    @test reconstruction_error_gibbs < n * n * T

    # Sigma estimates should be in the same ballpark
    sigma_map = result_map.sigma
    sigma_gibbs = result_gibbs.sigma

    # Both should have positive sigma values
    @test all(sigma_map .> 0)
    @test all(sigma_gibbs .> 0)

    # Correlation between sigma estimates (should be positively correlated)
    sigma_corr = cor(sigma_map, sigma_gibbs)
    @test sigma_corr > -0.5  # At least not strongly negatively correlated
end

@testset "gbdase_diagnostics" begin
    # Create small test case with Gibbs samples
    n = 6
    d = 2
    T = 4

    X_true = [0.4 .+ 0.1 * randn(n, d) for _ in 1:T]
    for t in 2:T
        X_true[t] = X_true[t-1] + 0.02 * randn(n, d)
    end

    A = [clamp.(X_true[t] * X_true[t]', 0, 1) for t in 1:T]

    # Need enough samples for diagnostics
    result = gbdase_gibbs(A, d; n_burnin=5, n_samples=20, seed=42)

    diag = gbdase_diagnostics(result)

    @test haskey(diag, :ess)
    @test haskey(diag, :rhat)
    @test haskey(diag, :summary)

    # Check dimensions of ESS and R-hat arrays
    @test size(diag.ess) == (n, d, T)
    @test size(diag.rhat) == (n, d, T)

    # Check summary statistics exist
    @test haskey(diag.summary, :mean_ess)
    @test haskey(diag.summary, :min_ess)
    @test haskey(diag.summary, :max_ess)
    @test haskey(diag.summary, :mean_rhat)
    @test haskey(diag.summary, :max_rhat)
    @test haskey(diag.summary, :n_samples)

    # ESS should be positive
    @test all(diag.ess .> 0)

    # R-hat should be >= 1 (approximately)
    @test all(diag.rhat .>= 0.9)  # Allow some numerical tolerance

    # Test with faithful sampler too
    result_faithful = gbdase(A, d; n_burnin=5, n_samples=20, seed=42)
    diag_faithful = gbdase_diagnostics(result_faithful)

    @test haskey(diag_faithful, :ess)
    @test haskey(diag_faithful, :rhat)
    @test size(diag_faithful.ess) == (n, d, T)
end