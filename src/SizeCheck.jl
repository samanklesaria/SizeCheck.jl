module SizeCheck

using MLStyle

export @sizecheck

"""
    @sizecheck function_definition

Automatically adds runtime shape checking to Julia functions based on
size-annotated variable names. Variables with underscores followed by dimension
letters (e.g., `x_NK`) are validated to ensure consistent shapes.

Example:
```julia
@sizecheck function matrix_multiply(a_NK, b_KM)
    result_NM = a_NK * b_KM
    return result_NM
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
    dim_tracking = Dict{Char,Symbol}()

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

        # Handle augmented assignments
        Expr(op, lhs, rhs) => begin
            if op in [:+=, :-=, :*=, :/=, :%=, :^=]
                transform_augmented_assignment(op, lhs, rhs, dim_tracking)
            else
                Expr(op, map(e -> transform_ast(e, dim_tracking), [lhs, rhs])...)
            end
        end

        # Handle block expressions recursively
        Expr(:block, stmts...) => Expr(:block, map(e -> transform_ast(e, dim_tracking), stmts)...)

        # Handle other expressions recursively
        Expr(head, args...) => Expr(head, map(e -> transform_ast(e, dim_tracking), args)...)

        # Leave literals and symbols unchanged
        _ => expr
    end
end

function transform_assignment(lhs, rhs, dim_tracking)
    # Check if lhs has size annotation
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

function transform_augmented_assignment(op, lhs, rhs, dim_tracking)
    annotation = parse_size_annotation(lhs)
    if annotation !== nothing
        var_name, dims = annotation
        check_expr = generate_size_check(var_name, dims, dim_tracking)
        return Expr(:block,
            Expr(op, lhs, transform_ast(rhs, dim_tracking)),
            check_expr
        )
    else
        return Expr(op, lhs, transform_ast(rhs, dim_tracking))
    end
end

function parse_size_annotation(expr)
    @match expr begin
        s::Symbol => begin
            str_s = string(s)
            parts = split(str_s, '_')
            if length(parts) >= 2
                dims_part = last(parts)
                if all(c -> isuppercase(c), dims_part) && !isempty(dims_part)
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

    for (i, dim) in enumerate(dims)
        dim_var = Symbol(dim)

        if haskey(dim_tracking, dim)
            # Dimension already seen, generate comparison check using existing variable
            first_var = dim_tracking[dim]
            dim_check = quote
                let current_size = size($var_name, $i)
                    if $dim_var != current_size
                        error("Dimension $($dim) mismatch: variable $($(QuoteNode(var_name))) has size $current_size but variable $($(QuoteNode(first_var))) has size $($dim_var)")
                    end
                end
            end
        else
            # First time seeing this dimension, create the variable
            dim_tracking[dim] = var_name
            dim_check = :($dim_var = size($var_name, $i))
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
