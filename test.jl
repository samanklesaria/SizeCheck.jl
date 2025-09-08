using SizeCheck
using Test

@testset "Basic Functionality" begin
    @sizecheck function matrix_multiply(a_NK, b_KM)
        result_NM = a_NK * b_KM
        return result_NM
    end

    a_NK = randn(3, 4)  # N=3, K=4
    b_KM = randn(4, 5)  # K=4, M=5
    @test size(matrix_multiply(a_NK, b_KM)) == (3, 5)

    a_NK = randn(3, 4)  # N=3, K=4
    b_KM = randn(5, 6)  # Wrong! K=5 doesn't match K=4
    @test_throws "Dimension" matrix_multiply(a_NK, b_KM)

    @sizecheck function test_augmented_assignment(data_NF)
        weights_FK = randn(size(data_NF, 2), 10)  # F matches, K=10
        data_NF += randn(size(data_NF))  # Should work - same dimensions
        result_NK = data_NF * weights_FK
        return result_NK
    end

    data_NF = randn(5, 8)  # N=5, F=8
    @test size(test_augmented_assignment(data_NF)) == (5, 10)
end

@testset "Macro Expansion Tests" begin
    # Test that dimension variables are created without runtime dictionary
    expansion = @macroexpand @sizecheck function test_expansion(a_NK)
        return a_NK, N, K
    end

    # Convert to string for easier testing
    expansion_str = string(expansion)

    # Should contain dimension variable assignments
    @test contains(expansion_str, "N = size(a_NK, 1)")
    @test contains(expansion_str, "K = size(a_NK, 2)")

    # Test expansion with mixed annotated and regular variables
    expansion2 = @macroexpand @sizecheck function test_mixed(a_NK, regular_var)
        result_NM = a_NK * regular_var
        return result_NM
    end

    expansion2_str = string(expansion2)
    @test contains(expansion2_str, "N = size(a_NK, 1)")
    @test contains(expansion2_str, "K = size(a_NK, 2)")
end

@testset "Dimension Variable Access" begin
    @sizecheck function access_test(a_NK, b_KM)
        c_NM = a_NK * b_KM
        return c_NM, N, K, M
    end

    a = rand(3, 4)
    b = rand(4, 5)
    result, n, k, m = access_test(a, b)

    @test n == 3
    @test k == 4
    @test m == 5
    @test size(result) == (3, 5)
end

@testset "Edge Cases and Error Conditions" begin
    # Test dimension variables are properly scoped (don't leak outside functions)
    @sizecheck function scoped_test(a_XY)
        return a_XY, X, Y
    end

    a = rand(2, 3)
    result, x, y = scoped_test(a)
    @test x == 2
    @test y == 3

    # X and Y should not be defined in global scope
    @test_throws UndefVarError X
    @test_throws UndefVarError Y

    # Test variables without size annotations are handled correctly
    @sizecheck function mixed_vars(annotated_NK, regular_var, also_NK)
        sum_result = annotated_NK + also_NK  # Should work - same dimensions
        other = regular_var * 2
        return sum_result, other, N, K
    end

    a1 = rand(3, 4)
    a2 = rand(3, 4)  # Same dimensions as a1
    result, other, n, k = mixed_vars(a1, 10, a2)
    @test n == 3
    @test k == 4
    @test other == 20

    # Test complex dimension relationships
    @sizecheck function complex_dims(a_NK, b_KM, c_NP)
        temp_NM = a_NK * b_KM
        result = hcat(temp_NM, c_NP)  # Should be N × (M+P)
        return result, N, K, M, P
    end

    a = rand(4, 3)  # N=4, K=3
    b = rand(3, 5)  # K=3, M=5
    c = rand(4, 2)  # N=4, P=2
    result, n, k, m, p = complex_dims(a, b, c)
    @test n == 4
    @test k == 3
    @test m == 5
    @test p == 2
    @test size(result) == (4, 7)  # N × (M+P) = 4 × 7
end

@testset "Numerical Constant Dimensions" begin
    # Test single numerical constants
    @sizecheck function test_constants(a_N3, b_2K)
        result_NK = zeros(N, K)
        return result_NK, N, K
    end

    a = rand(4, 3)  # N=4, second dim=3
    b = rand(2, 5)  # first dim=2, K=5
    result, n, k = test_constants(a, b)
    @test n == 4
    @test k == 5
    @test size(result) == (4, 5)

    # Test error case for wrong constant
    @sizecheck function test_constant_error(a_N3)
        return a_N3, N
    end

    a_wrong = rand(2, 4)  # Second dimension is 4, but expected 3
    @test_throws "Dimension 3 mismatch" test_constant_error(a_wrong)

    # Test complex mix of constants and variables
    @sizecheck function test_complex_constants(a_N32, b_4K2, c_N4)
        return a_N32, b_4K2, c_N4, N, K
    end

    a = rand(3, 3, 2)  # N=3, second=3, third=2
    b = rand(4, 5, 2)  # first=4, K=5, third=2
    c = rand(3, 4)     # N=3, second=4
    result_a, result_b, result_c, n, k = test_complex_constants(a, b, c)
    @test n == 3
    @test k == 5

    # Test error in complex case
    b_wrong = rand(3, 5, 2)  # first=3, but expected 4
    @test_throws "Dimension 4 mismatch" test_complex_constants(a, b_wrong, c)

    # Test macro expansion with constants
    expansion = @macroexpand @sizecheck function test_expansion_constants(a_N3, b_2K)
        return a_N3, b_2K, N, K
    end

    expansion_str = string(expansion)
    @test contains(expansion_str, "N = size(a_N3, 1)")  # Variable dimension
    @test contains(expansion_str, "K = size(b_2K, 2)")  # Variable dimension
    @test contains(expansion_str, "current_size != 3")  # Constant check
    @test contains(expansion_str, "current_size != 2")  # Constant check
end
