export NC_Optimize, NC_Optimize_output

"""
    NC_Optimize(D::Matrix, Rt::Union{Matrix, Transpose}, 
        options::AbstractOption, IG::Operators) → Ahat, Bhat, Fhat, Hhat, Nhat, Khat

Optimization version of Standard Operator Inference (NC)

## Arguments
- `D`: data matrix
- `Rt`: transpose of the derivative matrix
- `options`: options for the operator inference set by the user
- `IG`: Initial Guesses

## Returns
- Inferred operators

"""
function NC_Optimize(D::Matrix, Rt::Union{Matrix, Transpose}, 
                     dims::AbstractArray, operators_symbols::AbstractArray,
                     options::AbstractOption, IG::Operators)
    # # Some dimensions to unpack for convenience
    # n = options.system.dims[:n]
    # m = options.system.dims[:m]
    # s = options.system.dims[:s2]
    # v = options.system.dims[:v2]
    # w = options.system.dims[:w1]

    @info "Initialize optimization model."

    # Model with options
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

    for (i, symbol) in zip(dims, operators_symbols)
        if options.optim.initial_guess
            @variable(model, [1:i, 1:i], base_name=string(symbol), start=getproperty(IG, symbol))
        else
            @variable(model, [1:i, 1:i], base_name=string(symbol))
        end
    end

    for (idx, symbol) in enumerate(operators_symbols)
        if idx == 1
            tmp = model[symbol]
        else
            tmp = hcat(tmp, model[symbol])
        end
    end

    # # Construct large initial guess matrix
    # IG_all = [getproperty(IG, symbol) for symbol in operators_symbols]
    # IG_all = reduce(hcat, IG_all)'

    # # Construct the objective matrix
    # K, n = size(Rt)
    # @variable(model, Ot[1:(Int ∘ sum)(dims), 1:n])
    # if options.optim.initial_guess
    #     set_start_value.(Ot, IG_all)
    # end

    # # Linear A matrix
    # if options.system.is_lin
    #     @variable(model, Ahat[1:n, 1:n])
    #     if options.optim.initial_guess
    #         set_start_value.(Ahat, IG.A)  # set initial value of Ahat
    #     end
    #     tmp = Ahat  # temporary variable: which is the objective matrix holding linear matrix, Ahat
    # end

    # # Input B matrix
    # if options.system.has_control
    #     @variable(model, Bhat[1:n, 1:m])
    #     if options.optim.initial_guess
    #         set_start_value.(Bhat, IG.B) # set initial value of Bhat
    #     end
    #     tmp = (@isdefined tmp) ? hcat(tmp, Bhat) : Bhat  # concatenate Bhat to objective matrix
    # end

    # # Quadratic F or H matrix
    # if options.system.is_quad
    #     if options.optim.which_quad_term == "F"
    #         @variable(model, Fhat[1:n, 1:s])
    #         if options.optim.initial_guess
    #             set_start_value.(Fhat, IG.F)  # set initial value of Fhat
    #         end
    #         tmp = (@isdefined tmp) ? hcat(tmp, Fhat) : Fhat  # concatenate Fhat to objective matrix
    #     else
    #         @variable(model, Hhat[1:n, 1:v])
    #         if options.optim.initial_guess
    #             set_start_value.(Hhat, IG.H)  # set initial value of Hhat
    #         end
    #         tmp = (@isdefined tmp) ? hcat(tmp, Hhat) : Hhat # concatenate Hhat to objective matrix
    #     end
    # end

    # TODO: Cubic G/E matrices

    # # Bilinear N matrix
    # if options.system.is_bilin
    #     @variable(model, Nhat[1:n, 1:w])
    #     if options.optim.initial_guess
    #         set_start_value.(Nhat, IG.N)  # set initial value of Nhat
    #     end
    #     tmp = (@isdefined tmp) ? hcat(tmp, Nhat) : Nhat # concatenate Nhat to objective matrix
    # end

    # # Constant K matrix
    # if options.system.has_const
    #     @variable(model, Khat[1:n, 1])
    #     if options.optim.initial_guess
    #         set_start_value.(Khat, IG.K)  # set initial value of Khat
    #     end
    #     tmp = (@isdefined tmp) ? hcat(tmp, Khat) : Khat  # concatenate Khat to objective matrix
    # end

    # Create objective matrix with JuMP expression
    Ot = @expression(model, tmp')
    # n_tmp, m_tmp = size(Rt)
    # @variable(model, X[1:n_tmp, 1:m_tmp])
    @variable(model, X[1:K, 1:n])
    @constraint(model, X .== D * Ot .- Rt)  # this method answer on: https://discourse.julialang.org/t/write-large-least-square-like-problems-in-jump/35931
    if options.λ_lin == 0 && options.λ_quad == 0 
        REG = @expression(model, sum(X.^2))
    else
        if !options.optim.nonredundant_operators
        # if options.optim.which_quad_term == "H"
            # REG = @expression(model, sum(X.^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Hhat.^2))
            REG = @expression(model, sum(X.^2) + options.λ_lin*sum(model[:A].^2) + options.λ_quad*sum(model[:A2].^2))
        else
            # REG = @expression(model, sum(X.^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Fhat.^2))
            REG = @expression(model, sum(X.^2) + options.λ_lin*sum(model[:A].^2) + options.λ_quad*sum(model[:A2u].^2))
        end
    end
    # foo = sum(X.^2)
    # for symbol in operators_symbols
    #     foo += options.λ[symbol] * sum(model[symbol].^2)
    # end
    # REG = @expression(model, foo)

    # Define the objective of the optimization problem
    @objective(model, Min, REG)

    @info "Done."

    # Logging from the optimizer
    @info "Optimize model."
    JuMP.optimize!(model)
    @info """\n
    Constraint           = Non-Constrained
    Warm Start           = $(options.optim.initial_guess)
    order                = $(n)
    solve time           = $(solve_time(model))
    termination_status   = $(termination_status(model))
    primal_status        = $(primal_status(model))
    dual_state           = $(dual_status(model))
    dual objective value = $(dual_objective_value(model))
    objective_value      = $(objective_value(model))
    """

    # Assign the inferred matrices
    # Ahat = options.system.is_lin ? value.(Ahat) : 0
    # Bhat = options.system.has_control ? value.(Bhat)[:, :] : 0
    # Fhat = options.system.is_quad ? options.optim.which_quad_term=="F" ? value.(Fhat) : H2F(value.(Hhat)) : 0
    # Hhat = options.system.is_quad ? options.optim.which_quad_term=="H" ? value.(Hhat) : F2Hs(value.(Fhat)) : 0
    # Nhat = options.system.is_bilin ? value.(Nhat) : 0
    # Khat = options.system.has_const ? value.(Khat) : 0
    @info "Done."
    # return Ahat, Bhat, Fhat, Hhat, Nhat, Khat
    operators = Operators()
    unpack_operators!(operators, value.(Ot)', dims, operators_symbols, options)
    return operators
end


"""
    NC_Optimize_output(Y::Matrix, Xt_hat::Union{Matrix, Transpose}, options::AbstractOption) → C

Output optimization for the standard operator inference (for operator `C`)

## Arguments
- `Y`: the output matrix 
- `Xt_hat`: the state matrix

## Return
- the output state matrix `C`
"""
function NC_Optimize_output(Y::Matrix, Xhat_t::Union{Matrix, Transpose}, 
        options::AbstractOption)
    # Some dimensions to unpack for convenience
    # n = options.system.dims[:n]
    # l = options.system.dims[:l]

    K, n = size(Xhat_t)
    l = size(Y, 2)

    Yt = transpose(Y)

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

    @variable(model, Chat[1:l, 1:n])
    Chat_t = @expression(model, Chat')
    
    n_tmp, m_tmp = size(Yt)
    @variable(model, X[1:n_tmp, 1:m_tmp])
    @constraint(model, X .== Matrix(Xhat_t) * Chat_t .- Yt) 

    @objective(model, Min, sum(X.^2))
    @info "Done."

    @info "Optimize model."
    JuMP.optimize!(model)
    @info """[Output Optimization Results]
    Warm Start           = False
    order                = $(n)
    solve time           = $(solve_time(model))
    termination_status   = $(termination_status(model))
    primal_status        = $(primal_status(model))
    dual_state           = $(dual_status(model))
    dual objective value = $(dual_objective_value(model))
    objective_value      = $(objective_value(model))
    """

    Chat = value.(Chat)
    @info "Done."
    return Chat[:,:]
end
