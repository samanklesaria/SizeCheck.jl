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

@testset "Destructuring Assignment" begin
    # Helper function that returns multiple arrays
    function get_arrays(n, k, m)
        return rand(n, k), rand(k, m), rand(n, m)
    end

    # Test simple destructuring with one size-annotated variable
    @sizecheck function test_simple_destructuring()
        arrays = get_arrays(3, 4, 5)
        result_NK, status = arrays[1], "success"
        return result_NK, status, N, K
    end

    result, status, n, k = test_simple_destructuring()
    @test n == 3
    @test k == 4
    @test status == "success"
    @test size(result) == (3, 4)

    # Test multiple size-annotated variables in destructuring
    @sizecheck function test_multiple_destructuring()
        a_NK, b_KM, c_NM = get_arrays(4, 3, 5)
        return a_NK, b_KM, c_NM, N, K, M
    end

    a, b, c, n, k, m = test_multiple_destructuring()
    @test n == 4
    @test k == 3
    @test m == 5
    @test size(a) == (4, 3)
    @test size(b) == (3, 5)
    @test size(c) == (4, 5)

    # Test mixed size-annotated and regular variables in destructuring
    @sizecheck function test_mixed_destructuring()
        matrix_NK, scalar, vector_N = rand(2, 3), 42, rand(2)
        return matrix_NK, scalar, vector_N, N, K
    end

    matrix, scalar, vector, n, k = test_mixed_destructuring()
    @test n == 2
    @test k == 3
    @test scalar == 42
    @test size(matrix) == (2, 3)
    @test size(vector) == (2,)

    # Test destructuring with numerical constants
    @sizecheck function test_destructuring_constants()
        a_N3, b_3K = rand(4, 3), rand(3, 5)
        return a_N3, b_3K, N, K
    end

    a, b, n, k = test_destructuring_constants()
    @test n == 4
    @test k == 5
    @test size(a) == (4, 3)
    @test size(b) == (3, 5)

    # Test error case: dimension mismatch in destructuring
    @sizecheck function test_destructuring_error()
        # This should fail because both variables claim K but have different sizes
        a_NK, b_KM = rand(3, 4), rand(5, 6)  # K=4 vs K=5 mismatch
        return a_NK, b_KM, N, K, M
    end

    @test_throws "Dimension K mismatch" test_destructuring_error()

    # Test destructuring with shared dimensions validates correctly
    @sizecheck function test_destructuring_shared_dims()
        # These should work - N and K are consistent
        a_NK, b_NM, c_KM = rand(3, 4), rand(3, 5), rand(4, 5)
        temp = a_NK * c_KM  # Should be 3×5
        return temp, N, K, M
    end

    result, n, k, m = test_destructuring_shared_dims()
    @test n == 3
    @test k == 4
    @test m == 5
    @test size(result) == (3, 5)

    # Test macro expansion includes destructuring checks
    expansion = @macroexpand @sizecheck function test_expansion_destructuring()
        a_NK, b_KM = rand(2, 3), rand(3, 4)
        return a_NK, b_KM, N, K, M
    end

    expansion_str = string(expansion)
    @test contains(expansion_str, "N = size(a_NK, 1)")
    @test contains(expansion_str, "K = size(a_NK, 2)")
    @test contains(expansion_str, "M = size(b_KM, 2)")
    @test !contains(expansion_str, "_dims_")
end

@testset "Explicit Dimension Assignment" begin

    # Test 1: Normal dimension tracking (baseline)
    @sizecheck function normal_dim_test(x_N, y_N)
        z_N = x_N + y_N
        return z_N
    end

    @test_throws "Dimension N mismatch" normal_dim_test(randn(3), randn(4))

    # Test 2: Explicit assignment before use - should work with "explicitly provided" message
    @sizecheck function explicit_before_test()
        N = 5
        x_N = randn(4)  # Size mismatch with N=5
        return x_N
    end

    exception = nothing
    try
        explicit_before_test()
    catch e
        exception = e
    end
    @test exception !== nothing
    @test occursin("explicitly provided", string(exception))
    @test occursin("Dimension N mismatch", string(exception))

    # Test 3: Explicit assignment after use - should fail during macro expansion
    macro_error = nothing
    try
        @eval @sizecheck function explicit_after_test()
            x_N = randn(3)
            N = 5  # This should fail
            return x_N
        end
    catch e
        macro_error = e
    end
    @test macro_error !== nothing
    @test occursin("Cannot assign to dimension variable N after it has been used", string(macro_error))

    # Test 4: Multiple explicit assignments before use
    @sizecheck function multiple_explicit_test()
        N = 3
        M = 4
        x_N = randn(N)
        y_M = randn(M)
        z_NM = randn(2, 4)  # Mismatch with N=3
        return z_NM
    end
    @test_throws "explicitly provided" multiple_explicit_test()

    # Test 5: Mixed explicit and inferred dimensions
    @sizecheck function mixed_explicit_test(x_N)
        M = 4  # Explicit
        y_M = randn(M)
        z_NM = randn(N, 3)  # Should fail - M=4 but z_NM has M=3
        return z_NM
    end
    @test_throws "explicitly provided" mixed_explicit_test(randn(2))


    # Test 6: Explicit assignment works when sizes match
    @sizecheck function explicit_match_test()
        N = 3
        x_N = randn(3)  # Sizes match
        return x_N, N
    end

    result, n = explicit_match_test()
    @test n == 3
    @test size(result) == (3,)

    # Test 7: Multiple dimension variables, some explicit, some inferred
    @sizecheck function mixed_assignment_test(a_NK)
        M = 5  # Explicit
        b_KM = randn(K, M)  # K from a_NK, M explicit
        c_NM = randn(N, 4)  # Should fail - M=5 but c_NM has M=4
        return c_NM
    end
    @test_throws "explicitly provided" mixed_assignment_test(randn(3, 4))

    # Test 8: Ensure normal dimension tracking still works after explicit assignment
    @sizecheck function normal_after_explicit_test(x_N)
        M = 4
        y_M = randn(M)
        z_N = randn(5)  # Should fail - N from x_N vs N=5
        return z_N
    end

    @test_throws (s -> occursin("variable x_N", s) && !occursin("explicitly provided", s)) normal_after_explicit_test(randn(3))

end
