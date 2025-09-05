using SizeCheck
using Test

# Example from the README: Matrix multiplication with automatic shape checking
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
