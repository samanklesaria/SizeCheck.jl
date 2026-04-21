module SizeCheck

using MLStyle

export @sizecheck

"""
Automatically adds runtime shape checking to Julia functions based on
size-annotated variable names. Variables with underscores followed by dimension
letters (e.g., `x_NK`) are validated to ensure consistent shapes.

Dimension annotations can contain:
- Variable dimensions (uppercase letters): `N`, `K`, `M` - stored in variables of the same name
- Constant dimensions (single digits): `3`, `4`, `2` - checked for exact size

The macro automatically adds shape validation for:
- **Function arguments** with underscores in their names
- **Variable assignments** to names containing underscores, including destructuring assignments

The dimensions are scoped to the function they are defined in.
For example, if you define a function `foo` with a parameter `x_NK`, the dimension `N` is only valid within the scope of `foo`.
If you define another function `bar` with a parameter `y_NL`, this dimension `N` can differ from the one in `foo`,
but it is only valid within the scope of `bar`.

Examples:
```julia
@sizecheck function matrix_multiply(a_NK, b_KM)
    result_NM = a_NK * b_KM
    return result_NM, N, K, M  # Dimension variables accessible
end

# This works fine
a_NK = randn(3, 4)  # N=3, K=4
b_KM = randn(4, 5)  # K=4, M=5
result = matrix_multiply(a_NK, b_KM)  # size: (3, 5)

# This raises an error
a_NK = randn(3, 4)
b_KM = randn(5, 6)  # Wrong! K dimensions don't match
result = matrix_multiply(a_NK, b_KM)  # Error!

@sizecheck function with_constants(data_N3, weights_3K)
    # data_N3: first dim variable N, second dim exactly 3
    # weights_3K: first dim exactly 3, second dim variable K
    result_NK = data_N3 * weights_3K
    return result_NK, N, K
end
```
"""
macro sizecheck(expr)
    @match expr begin
        Expr(:function, signature, body) => begin
            new_body = transform_function_body(signature, body)
            esc(Expr(:function, signature, new_body))
        end
        _ => error("@sizecheck can only be applied to function definitions")
    end
end

function transform_function_body(signature, body)
    # Track dimensions at compile time for error messages
    dim_tracking = Dict{Char,Union{Symbol,Nothing}}()

    # Extract function arguments and generate checks for them
    arg_checks = generate_argument_checks(signature, dim_tracking)

    # Transform the function body
    transformed_body = transform_ast(body, dim_tracking)

    # Insert argument checks at the beginning
    @match transformed_body begin
        Expr(:block, stmts...) => Expr(:block, arg_checks..., stmts...)
        _ => Expr(:block, arg_checks..., transformed_body)
    end
end

function transform_ast(expr, dim_tracking)
    @match expr begin
        # Handle regular assignment expressions
        Expr(:(=), lhs, rhs) => transform_assignment(lhs, rhs, dim_tracking)

        # Handle block expressions recursively
        Expr(:block, stmts...) => Expr(:block, map(e -> transform_ast(e, dim_tracking), stmts)...)

        # Handle other expressions recursively
        Expr(head, args...) => Expr(head, map(e -> transform_ast(e, dim_tracking), args)...)

        # Leave literals and symbols unchanged
        _ => expr
    end
end

function transform_assignment(lhs, rhs, dim_tracking)
    # Check for explicit dimension variable assignments
    if isa(lhs, Symbol)
        lhs_str = string(lhs)
        if length(lhs_str) == 1 && isuppercase(lhs_str[1])
            dim_char = lhs_str[1]
            if haskey(dim_tracking, dim_char) && dim_tracking[dim_char] !== nothing
                # Dimension variable already used, throw error
                error("Cannot assign to dimension variable $lhs after it has been used in a size annotation")
            elseif !haskey(dim_tracking, dim_char)
                # Mark this dimension as explicitly assigned
                dim_tracking[dim_char] = nothing
            end
        end
    end

    # Handle destructuring assignment (tuple on left side)
    if isa(lhs, Expr) && lhs.head == :tuple
        # Extract size-annotated variables from tuple
        annotated_vars = []
        for var in lhs.args
            annotation = parse_size_annotation(var)
            if annotation !== nothing
                push!(annotated_vars, annotation)
            end
        end

        if !isempty(annotated_vars)
            # Generate the assignment and checks
            assignment = Expr(:(=), lhs, transform_ast(rhs, dim_tracking))
            checks = []
            for (var_name, dims) in annotated_vars
                check_expr = generate_size_check(var_name, dims, dim_tracking)
                push!(checks, check_expr)
            end
            return Expr(:block, assignment, checks...)
        else
            return Expr(:(=), lhs, transform_ast(rhs, dim_tracking))
        end
    else
        # Handle single variable assignment
        annotation = parse_size_annotation(lhs)
        if annotation !== nothing
            var_name, dims = annotation
            check_expr = generate_size_check(var_name, dims, dim_tracking)
            return Expr(:block,
                Expr(:(=), lhs, transform_ast(rhs, dim_tracking)),
                check_expr
            )
        else
            return Expr(:(=), lhs, transform_ast(rhs, dim_tracking))
        end
    end
end

function parse_size_annotation(expr)
    @match expr begin
        s::Symbol => begin
            str_s = string(s)
            parts = split(str_s, '_')
            if length(parts) >= 2
                dims_part = last(parts)
                if all(c -> isuppercase(c) || isdigit(c), dims_part) && !isempty(dims_part)
                    var_name = s
                    dims = collect(dims_part)
                    return (var_name, dims)
                end
            end
            return nothing
        end
        _ => nothing
    end
end

function generate_size_check(var_name, dims, dim_tracking)
    checks = []

    # Check that the number of dimensions matches the number of annotations
    n_annotations = length(dims)
    first_dim = string(dims[1])
    ndims_check = quote
        let actual_ndims = ndims($var_name)
            if actual_ndims != $n_annotations
                error("Dimension $($first_dim) mismatch: variable $($(QuoteNode(var_name))) has $actual_ndims dimensions but expected $($n_annotations)")
            end
        end
    end
    push!(checks, ndims_check)

    for (i, dim) in enumerate(dims)
        if isdigit(dim)
            # Numerical constant - generate size check
            expected_size = parse(Int, string(dim))
            dim_check = quote
                let current_size = size($var_name, $i)
                    if current_size != $expected_size
                        error("Dimension $($dim) mismatch: variable $($(QuoteNode(var_name))) has size $current_size but expected size $($expected_size)")
                    end
                end
            end
        else
            # Variable dimension
            dim_var = Symbol(dim)

            if haskey(dim_tracking, dim)
                # Dimension already seen, generate comparison check using existing variable
                first_var = dim_tracking[dim]
                if first_var === nothing
                    # Explicitly provided dimension
                    dim_check = quote
                        let current_size = size($var_name, $i)
                            if $dim_var != current_size
                                error("Dimension $($dim) mismatch: variable $($(QuoteNode(var_name))) has size $current_size but explicitly provided has size $($dim_var)")
                            end
                        end
                    end
                else
                    # Regular dimension tracking
                    dim_check = quote
                        let current_size = size($var_name, $i)
                            if $dim_var != current_size
                                error("Dimension $($dim) mismatch: variable $($(QuoteNode(var_name))) has size $current_size but variable $($(QuoteNode(first_var))) has size $($dim_var)")
                            end
                        end
                    end
                end
            else
                # First time seeing this dimension, create the variable
                dim_tracking[dim] = var_name
                dim_check = :($dim_var = size($var_name, $i))
            end
        end

        push!(checks, dim_check)
    end

    Expr(:block, checks...)
end

function generate_argument_checks(signature, dim_tracking)
    args = extract_function_args(signature)
    checks = []

    for arg in args
        annotation = parse_size_annotation(arg)
        if annotation !== nothing
            var_name, dims = annotation
            check_expr = generate_size_check(var_name, dims, dim_tracking)
            push!(checks, check_expr)
        end
    end

    return checks
end

function extract_function_args(signature)
    @match signature begin
        # Function name with arguments
        Expr(:call, name, args...) => args
        # Just function name (no arguments)
        name::Symbol => []
        _ => []
    end
end

end # module SizeCheck
