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
    # Create the _dims_ dictionary declaration
    dims_decl = :(local _dims_ = Dict{Char, Tuple{Int, Symbol}}())

    # Extract function arguments and generate checks for them
    arg_checks = generate_argument_checks(signature)

    # Transform the function body
    transformed_body = transform_ast(body)

    # Insert the _dims_ declaration and argument checks at the beginning
    @match transformed_body begin
        Expr(:block, stmts...) => Expr(:block, dims_decl, arg_checks..., stmts...)
        _ => Expr(:block, dims_decl, arg_checks..., transformed_body)
    end
end

function transform_ast(expr)
    @match expr begin
        # Handle regular assignment expressions
        Expr(:(=), lhs, rhs) => transform_assignment(lhs, rhs)

        # Handle augmented assignments
        Expr(op, lhs, rhs) => begin
            if op in [:+=, :-=, :*=, :/=, :%=, :^=]
                transform_augmented_assignment(op, lhs, rhs)
            else
                Expr(op, map(transform_ast, [lhs, rhs])...)
            end
        end

        # Handle block expressions recursively
        Expr(:block, stmts...) => Expr(:block, map(transform_ast, stmts)...)

        # Handle other expressions recursively
        Expr(head, args...) => Expr(head, map(transform_ast, args)...)

        # Leave literals and symbols unchanged
        _ => expr
    end
end

function transform_assignment(lhs, rhs)
    # Check if lhs has size annotation
    if has_size_annotation(lhs)
        var_name, dims = parse_size_annotation(lhs)
        check_expr = generate_size_check(var_name, dims)
        return Expr(:block,
            Expr(:(=), lhs, transform_ast(rhs)),
            check_expr
        )
    else
        return Expr(:(=), lhs, transform_ast(rhs))
    end
end

function transform_augmented_assignment(op, lhs, rhs)
    if has_size_annotation(lhs)
        var_name, dims = parse_size_annotation(lhs)
        check_expr = generate_size_check(var_name, dims)
        return Expr(:block,
            Expr(op, lhs, transform_ast(rhs)),
            check_expr
        )
    else
        return Expr(op, lhs, transform_ast(rhs))
    end
end

function has_size_annotation(expr)
    @match expr begin
        s::Symbol => begin
            str_s = string(s)
            parts = split(str_s, '_')
            if length(parts) >= 2
                dims_part = last(parts)
                all(c -> isuppercase(c), dims_part) && !isempty(dims_part)
            else
                false
            end
        end
        _ => false
    end
end

function parse_size_annotation(expr)
    @match expr begin
        s::Symbol => begin
            str_s = string(s)
            parts = split(str_s, '_')
            var_name = s
            dims = collect(last(parts))
            (var_name, dims)
        end
        _ => error("Invalid size annotation")
    end
end

function generate_size_check(var_name, dims)
    checks = []

    for (i, dim) in enumerate(dims)
        dim_check = quote
            let current_size = size($var_name, $i)
                if haskey(_dims_, $(QuoteNode(dim)))
                    stored_size, stored_var = _dims_[$(QuoteNode(dim))]
                    if stored_size != current_size
                        error("Dimension $($dim) mismatch: variable $($(QuoteNode(var_name))) has size $current_size but variable $stored_var has size $stored_size")
                    end
                else
                    _dims_[$(QuoteNode(dim))] = (current_size, $(QuoteNode(var_name)))
                end
            end
        end
        push!(checks, dim_check)
    end

    Expr(:block, checks...)
end

function generate_argument_checks(signature)
    args = extract_function_args(signature)
    checks = []

    for arg in args
        if has_size_annotation(arg)
            var_name, dims = parse_size_annotation(arg)
            check_expr = generate_size_check(var_name, dims)
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
