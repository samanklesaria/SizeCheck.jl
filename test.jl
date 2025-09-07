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

    # Should NOT contain any _dims_ dictionary
    @test !contains(expansion_str, "_dims_")
    @test !contains(expansion_str, "Dict")

    # Test expansion with mixed annotated and regular variables
    expansion2 = @macroexpand @sizecheck function test_mixed(a_NK, regular_var)
        result_NM = a_NK * regular_var
        return result_NM
    end

    expansion2_str = string(expansion2)
    @test contains(expansion2_str, "N = size(a_NK, 1)")
    @test contains(expansion2_str, "K = size(a_NK, 2)")
    @test !contains(expansion2_str, "_dims_")
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

@testset "No Runtime Dictionary Overhead" begin
    # Generate a function and check its bytecode doesn't contain dictionary operations
    @sizecheck function no_dict_test(a_NK, b_KM)
        c_NM = a_NK * b_KM
        return c_NM, N, K, M
    end

    # Get the method and its code info
    method = methods(no_dict_test).ms[1]
    code_info = Base.uncompressed_ast(method)
    code_str = string(code_info)

    # Should not contain any dictionary-related operations
    @test !contains(code_str, "Dict")
    @test !contains(code_str, "haskey")
    @test !contains(code_str, "getindex")
    @test !contains(code_str, "setindex!")

    # Should contain direct size calls for dimension variables
    @test contains(code_str, "size")

    # Verify the function actually works
    a = rand(3, 4)
    b = rand(4, 5)
    result, n, k, m = no_dict_test(a, b)
    @test n == 3
    @test k == 4
    @test m == 5
end
