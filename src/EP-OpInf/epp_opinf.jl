"""
$(SIGNATURES)

Energy preserving penalty operator inference optimization (EPP)

## Arguments
- `D`: data matrix
- `Rt`: transpose of the derivative matrix (or residual matrix)
- `dims`: dimensions of the operators
- `operators_symbols`: symbols of the operators
- `options`: options for the operator inference set by the user
- `IG`: Initial Guesses

## Returns
- Inferred operators

## Note
- This is currently implemented for linear + quadratic operators only
"""
function epp_opinf(D::Matrix, Rt::Union{Matrix,Transpose},
                   dims::AbstractArray, operators_symbols::AbstractArray,
                   options::AbstractOption, IG::Operators)
    @info "Initialize optimization model."
    model = Model(Ipopt.Optimizer; add_bridges = false)
    if options.optim.linear_solver != "none"
        set_attribute(model, "hsllib", options.optim.HSL_lib_path)
        set_attribute(model, "linear_solver", options.optim.linear_solver)
    end
    set_optimizer_attribute(model, "max_iter", options.optim.max_iter)
    set_string_names_on_creation(model, false)
    if !options.optim.verbose
        set_silent(model)
    end

    # Define the dimensions of the operators
    idx = findfirst(x -> x == :A, operators_symbols)
    n = dims[idx]  # dimension of the state space
    v = Int(n^2)
    s = Int(n*(n+1)÷2)

    # Linear operator
    if 1 in options.system.state
        @variable(model, Ahat[1:n, 1:n])
        if options.optim.initial_guess 
            if options.optim.SIGE
                ni = size(IG.A, 1)
                set_start_value.(Ahat[1:ni, 1:ni], IG.A)
            else
                set_start_value.(Ahat, IG.A)
            end
        end

        if options.optim.with_bnds
            set_lower_bound.(Ahat, options.linear_operator_bounds[1])
            set_upper_bound.(Ahat, options.linear_operator_bounds[2])
        end
        tmp = Ahat
    end

    # Control operator
    if 1 in options.system.control
        idx = findfirst(x -> x == :B, operators_symbols)
        m = dims[idx]  # dimension of the control space
        @variable(model, Bhat[1:n, 1:m])
        if options.optim.initial_guess
            set_start_value.(Bhat, IG.B)
        end
        tmp = (@isdefined tmp) ? hcat(tmp, Bhat) : Bhat
    end

    # Quadratic operator
    if 2 in options.system.state
        if !options.optim.nonredundant_operators
            @variable(model, Hhat[1:n, 1:v])
            if options.optim.initial_guess
                set_start_value.(Hhat, IG.H)
            end
            tmp = (@isdefined tmp) ? hcat(tmp, Hhat) : Hhat
        else
            @variable(model, Fhat[1:n, 1:s])

            if options.optim.initial_guess
                if options.optim.SIGE
                    # Insert the lower dimension initial guess into optimization variable F
                    ni = size(IG.F, 1)
                    xsq_idx = [1 + (n + 1) * (k - 1) - k * (k - 1) / 2 for k in 1:n]
                    insert_idx = [collect(x:x+(ni-i)) for (i, x) in enumerate(xsq_idx[1:ni])]
                    idx = Int.(reduce(vcat, insert_idx))
                    set_start_value.(Fhat[1:ni, idx], IG.F)
                else
                    set_start_value.(Fhat, IG.F)
                end
            end

            if options.optim.with_bnds
                set_lower_bound.(Fhat, options.quad_operator_bounds[1])
                set_upper_bound.(Fhat, options.quad_operator_bounds[2])
            end

            tmp = (@isdefined tmp) ? hcat(tmp, Fhat) : Fhat
        end
    end

    # TODO: Implement the bliinear operator
    # if options.system.is_bilin
    #     @variable(model, Nhat[1:n, 1:w])
    #     if options.optim.initial_guess
    #         set_start_value.(Nhat, IG.N)
    #     end
    #     tmp = (@isdefined tmp) ? hcat(tmp, Nhat) : Nhat
    # end

    # Constant operator
    if !iszero(options.system.constant)
        @variable(model, Khat[1:n, 1])
        if options.optim.initial_guess
            set_start_value.(Khat, IG.K)
        end
        tmp = (@isdefined tmp) ? hcat(tmp, Khat) : Khat
    end

    @info "Set penalty"
    if !options.optim.nonredundant_operators
        EP = @expression(
            model, 
            sum(sum((Hhat[i, n*(k-1)+j] + Hhat[k, n*(k-1)+i] + Hhat[k, n*(i-1)+j]).^2) for i=1:n, j=1:i, k=1:j)
        )
    else
        EP = @expression(
            model,
            sum(sum((delta(j,k)*Fhat[i,fidx(n,j,k)] + delta(i,k)*Fhat[j,fidx(n,i,k)] + delta(j,i)*Fhat[k,fidx(n,j,i)]).^2) for i=1:n, j=1:i, k=1:j)
        )
    end

    @info "Set objective"
    Ot = @expression(model, tmp')

    n_tmp, m_tmp = size(Rt)
    @variable(model, X[1:n_tmp, 1:m_tmp])
    @constraint(model, X .== D * Ot .- Rt)  # this method answer on: https://discourse.julialang.org/t/write-large-least-square-like-problems-in-jump/35931

    if (options.λ_lin == 0 && options.λ_quad == 0)
        @objective(model, Min, sum(X.^2) + options.α * EP)
    else
        if !options.optim.nonredundant_operators
            REG = @expression(model, options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Hhat.^2))
            @objective(model, Min, sum(X.^2) + REG + options.α * EP)
        else
            REG = @expression(model, options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Fhat.^2))
            @objective(model, Min, sum(X.^2) + REG + options.α * EP)
        end
    end

    @info "Setup Done."
    @info "Optimize model."
    JuMP.optimize!(model)
    @info """[EP-OpInf Results]
    Constraint           = Energy-Preserving Penalty
    EP Weight            = $(options.α)
    Linear Regulation    = $(options.λ_lin)
    Quadratic Regulation = $(options.λ_quad)
    Warm Start           = $(options.optim.initial_guess)
    order                = $(n)
    solve time           = $(solve_time(model))
    termination_status   = $(termination_status(model))
    primal_status        = $(primal_status(model))
    dual_state           = $(dual_status(model))
    dual objective value = $(dual_objective_value(model))
    objective_value      = $(objective_value(model))
    """

    operators = Operators()
    unpack_operators!(operators, value.(Ot)', dims, operators_symbols)
    return operators
end
