export DoA, skp_stability_rad

"""
    DoA(P::AbstractArray{T}) where T

Compute the domain of attraction using inferred Lyapunov function P.

## Arguments
* `P::AbstractArray{T}`: Lyapunov function matrix

## Returns
* `rmin::Float64`: minimum radius of the domain of attraction
* `rmax::Float64`: maximum radius of the domain of attraction
"""
function DoA(P::AbstractArray{T}) where T
    λmin = (abs ∘ eigmin)(P) 
    λmax = (abs ∘ eigmax)(P)
    rmax = 1 / sqrt(λmin)
    rmin = 1 / sqrt(λmax)
    return [rmin, rmax]
end


"""
    skp_stability_rad(Ahat::AbstractArray{T}, Hhat::AbstractArray{T}, P::AbstractArray{T}; 
        div_by_2::Bool=true) where T

Estimate the stability radius of the system using the inferred Lyapunov function P by
Sawant, Kramer, and Peherstorfer (SKP).

## Arguments
* `Ahat::AbstractArray{T}`: inferred system matrix
* `Hhat::AbstractArray{T}`: inferred quadratic term
* `Q::AbstractArray{T}`: quadratic matrix for auxiliary function (not P for the Lyapunov function)
* `div_by_2::Bool=true`: whether to divide the stability radius by 2

## Returns
* `ρhat::Float64`: estimated stability radius

## References
N. Sawant, B. Kramer, and B. Peherstorfer, “Physics-informed regularization and structure preservation
for learning stable reduced models from data with operator inference,” Computer Methods in Applied 
Mechanics and Engineering, vol. 404, p. 115836, Feb. 2023, doi: 10.1016/j.cma.2022.115836.
"""
# function skp_stability_rad(Ahat::AbstractArray{T}, Hhat::AbstractArray{T}, Q::AbstractArray{T}; 
#         div_by_2::Bool=false) where T

#     if div_by_2
#         P = lyapc(Ahat', 0.5*Q)
#     else
#         P = lyapc(Ahat', Q)
#     end
#     L = cholesky(Q).L
#     σmin = minimum(svd(L).S)
#     ρhat = σmin^2 / sqrt(norm(P,2)) / norm(Hhat,2) / 2
#     return ρhat
# end
function skp_stability_rad(P::AbstractArray{T}, Ahat::AbstractArray{T}, Hhat::Union{AbstractArray{T},Nothing}=nothing,
                            Ghat::Union{AbstractArray{T},Nothing}=nothing; dims::Tuple=(1,2)) where T<:Real
    if (2 in dims) && (3 in dims)
        @assert !isnothing(Hhat) && !isnothing(Ghat) "Hhat and Ghat must be provided for quadratic-cubic systems."
        Q = -(Ahat' * P + P * Ahat)
        L = cholesky(Q).L
        σmin = minimum(svd(L).S)

        part1 = sqrt(norm(P,2) * norm(Hhat,2) + 2*σmin^2 * norm(Ghat,2)) / 2 / norm(Ghat,2)
        part2 = sqrt(norm(P,2)) * norm(Hhat,2) / 2 / norm(Ghat,2)
        c_skp = part1 - part2 

    elseif (2 in dims) && !(3 in dims)
        @assert !isnothing(Hhat) "Hhat must be provided for quadratic systems."
        Q = -(Ahat' * P + P * Ahat)
        L = cholesky(Q).L
        σmin = minimum(svd(L).S)
        c_skp = σmin^2 / sqrt(norm(P,2)) / norm(Hhat,2) / 2

    elseif !(2 in dims) && (3 in dims)
        @assert !isnothing(Ghat) "Ghat must be provided for cubic systems."
        Q = -(Ahat' * P + P * Ahat)
        L = cholesky(Q).L
        σmin = minimum(svd(L).S)
        c_skp = σmin / sqrt(2 * norm(Ghat,2))

    else
        error("Invalid dimensions. Support only for quadratic and cubic systems.")
    end
    λmax_P = maximum(eigvals(P))
    ρ_skp = c_skp / sqrt(λmax_P)
    return ρ_skp
end


function sampling_memoryless(V::Function, Vdot::Function, ns::Real, N::Int, 
        state_space::Union{Array,Tuple}; uniform_state_space::Bool=true, history=false)
    c_hat_star = Inf
    xi = zeros(N)
    if history
        chistory = zeros(Int(ns))
        xi_all = zeros(N,Int(ns))
    end
    is_array = typeof(state_space) <: Array
    for j = 1:Int(ns)
        if uniform_state_space && !is_array
            rand!(Uniform(state_space[1], state_space[2]), xi)
        else
            @inbounds for i = 1:N
                xi[i] = rand(Uniform(state_space[i][1], state_space[i][2]))
            end
        end

        Vdot_xi = Vdot(xi)
        V_xi = V(xi)

        if Vdot_xi >= 0 && V_xi < c_hat_star
            c_hat_star = V_xi
        end

        if history
            xi_all[:,j] = xi
            chistory[j] += c_hat_star
        end
    end
    if history
        return c_hat_star, chistory, xi_all
    else
        return c_hat_star
    end
end


function sampling_memoryless(V::Function, Vdot::Function, ns::Real, N::Int, Nl::Int, gp::Int, 
        state_space::Union{Array,Tuple}, lifter::lifting; uniform_state_space::Bool=true, history=false)
    c_hat_star = Inf

    xi = zeros(N)
    xi_lift = zeros(Nl)
    if history
        chistory = zeros(Int(ns))
        xi_all = zeros(N,Int(ns))
    end
    is_array = typeof(state_space) <: Array
    for j = 1:Int(ns)
        if uniform_state_space && !is_array
            rand!(Uniform(state_space[1], state_space[2]), xi)
            xi_lift = vec(lifter.map(xi, gp))
        else
            @inbounds for i = 1:N
                xi[i] = rand(Uniform(state_space[i][1], state_space[i][2]))
            end
            xi_lift = vec(lifter.map(xi, gp))
        end

        Vdot_xi = Vdot(xi_lift)
        V_xi = V(xi_lift)

        if Vdot_xi >= 0 && V_xi < c_hat_star
            c_hat_star = V_xi
        end

        if history
            xi_all[:,j] = xi
            chistory[j] += c_hat_star
        end
    end
    if history
        return c_hat_star, chistory, xi_all
    else
        return c_hat_star
    end
end


function sampling_with_memory(V::Function, Vdot::Function, ns::Real, N::Int, 
        state_space::Union{Array,Tuple}; uniform_state_space::Bool=true, history=false)
    c_underbar_star = 0
    c_bar_star = Inf
    E = [0.0]
    sizehint!(E, Int(ns)+1)
    xi = zeros(N)
    if history
        chistory = zeros(Int(ns))
        xi_all = zeros(N,Int(ns))
    end
    is_array = typeof(state_space) <: Array
    for j = 1:Int(ns)
        if uniform_state_space && !is_array
            rand!(Uniform(state_space[1], state_space[2]), xi)
        else
            @inbounds for i = 1:N
                xi[i] = rand(Uniform(state_space[i][1], state_space[i][2]))
            end
        end

        Vdot_xi = Vdot(xi)
        V_xi = V(xi)

        if Vdot_xi < 0 && V_xi < c_bar_star
            push!(E, V_xi)
            if V_xi > c_underbar_star
                c_underbar_star = V_xi
            end
        elseif Vdot_xi >= 0 && V_xi < c_bar_star
            c_bar_star = V_xi
            if c_underbar_star >= c_bar_star
                c_underbar_star = maximum(filter(c -> c < c_bar_star, E))
            end
        end

        if history
            xi_all[:,j] = xi
            chistory[j] += c_underbar_star
        end
    end
    if history
        return c_underbar_star, chistory, xi_all
    else
        return c_underbar_star
    end
end


function sampling_with_memory(V::Function, Vdot::Function, ns::Real, N::Int, Nl::Int, gp::Int,
        state_space::Union{Array,Tuple}, lifter::lifting; uniform_state_space::Bool=true, history=false)
    c_underbar_star = 0
    c_bar_star = Inf
    E = [0.0]
    sizehint!(E, Int(ns)+1)
    xi = zeros(N)
    xi_lift = zeros(Nl)
    if history
        chistory = zeros(Int(ns))
        xi_all = zeros(N,Int(ns))
    end
    is_array = typeof(state_space) <: Array
    for j = 1:Int(ns)
        if uniform_state_space && !is_array
            rand!(Uniform(state_space[1], state_space[2]), xi)
            xi_lift = vec(lifter.map(xi, gp))
        else
            @inbounds for i = 1:N
                xi[i] = rand(Uniform(state_space[i][1], state_space[i][2]))
            end
            xi_lift = vec(lifter.map(xi, gp))
        end

        Vdot_xi = Vdot(xi_lift)
        V_xi = V(xi_lift)

        if Vdot_xi < 0 && V_xi < c_bar_star
            push!(E, V_xi)
            if V_xi > c_underbar_star
                c_underbar_star = V_xi
            end
        elseif Vdot_xi >= 0 && V_xi < c_bar_star
            c_bar_star = V_xi
            if c_underbar_star >= c_bar_star
                c_underbar_star = maximum(filter(c -> c < c_bar_star, E))
            end
        end

        if history
            xi_all[:,j] = xi
            chistory[j] += c_underbar_star
        end
    end

    if history
        return c_underbar_star, chistory, xi_all
    else
        return c_underbar_star
    end
end




"""
use
- Stratified Sampling: Divide the state space into strata and sample more systematically from each stratum.
- Quasi-Monte Carlo Methods: These methods use low-discrepancy sequences instead of random sampling to
  ensure a more uniform coverage of the space, which can lead to faster convergence in higher dimensions.
"""
function enhanced_sampling_with_memory(V::Function, V_dot::Function, ns::Real, N::Int, 
        state_space::Array, n_strata::Int; history=false, sampler="sobol")
    
    @assert (sampler in ["sobol", "faure", "halton", "golden"]) "Invalid sampler."
    sampler_dict = Dict(
        "sobol" => QuasiMonteCarlo.SobolSample(),
        "faure" => QuasiMonteCarlo.FaureSample(),
        "halton" => QuasiMonteCarlo.HaltonSample(),
        "golden" => QuasiMonteCarlo.GoldenSample()
    )

    function stratify_state_space(state_space, n_strata)
        num_dims = length(state_space)
        n_strata_per_dim = round(Int, n_strata ^ (1 / num_dims))

        # Initialize ranges for each dimension
        ranges = [range(state_space[dim][1], stop=state_space[dim][2], length=n_strata_per_dim+1) for dim in 1:num_dims]

        # Preallocate lower and upper bounds
        lb = Array{Float64,2}(undef, n_strata_per_dim ^ num_dims, num_dims)
        ub = Array{Float64,2}(undef, n_strata_per_dim ^ num_dims, num_dims)

        # Non-recursive function to generate strata
        indices = ones(Int, num_dims)
        for i in 1:size(lb, 1)
            for dim in 1:num_dims
                lb[i, dim], ub[i, dim] = ranges[dim][indices[dim]], ranges[dim][indices[dim] + 1]
            end

            # Update indices
            dim = 1
            while dim <= num_dims
                indices[dim] += 1
                if indices[dim] <= n_strata_per_dim
                    break
                else
                    indices[dim] = 1
                    dim += 1
                end
            end
        end

        return lb, ub
    end

    c_underbar_star = 0
    c_bar_star = Inf
    E = [0.0]
    sizehint!(E, Int(ns)+1)

    if history
        chistory = zeros(Int(ns))
        xi_all = zeros(N,Int(ns))
    end

    # Stratify the state space into n_strata
    strata_lb, strata_ub = stratify_state_space(state_space, n_strata)
    n_strata = size(strata_lb, 1)

    xi = Vector{Float64}(undef, N)
    sample_per_stratum = div(Int(ns), n_strata)
    ct = 1

    # Shuffle the strata (experimental)
    strata_indices = shuffle(1:n_strata)
    for k in 1:n_strata
        # lb = strata_lb[k,:]
        # ub = strata_ub[k,:]

        lb = strata_lb[strata_indices[k],:]
        ub = strata_ub[strata_indices[k],:]
        # sobol = Sobol.SobolSeq(lb, ub)
        samples = QuasiMonteCarlo.sample(sample_per_stratum, lb, ub, sampler_dict[sampler])
        for i = 1:sample_per_stratum
            # Quasi-Monte Carlo sampling within the current stratum
            # Sobol.next!(sobol, xi)
            xi = samples[:,i]
            
            if V_dot(xi) < 0 && V(xi) < c_bar_star
                push!(E, V(xi))
                if V(xi) > c_underbar_star
                    c_underbar_star = V(xi)
                end
            elseif V_dot(xi) >= 0 && V(xi) < c_bar_star
                c_bar_star = V(xi)
                if c_underbar_star >= c_bar_star
                    c_underbar_star = try
                        maximum(filter(c -> c < c_bar_star, E))
                    catch e 
                        if isa(e, MethodError) 
                            continue
                        else
                            @error e 
                        end
                    end
                end
            end

            if history
                xi_all[:,ct] = xi
                chistory[ct] += c_underbar_star
            end

            ct += 1  # increment counter
        end
    end

    if history
        return c_underbar_star, chistory, xi_all
    else
        return c_underbar_star
    end
end


function enhanced_sampling_with_memory(V::Function, V_dot::Function, ns::Real, N::Int, Nl::Int,
        gp::Int, state_space::Array, n_strata::Int, lifter::lifting; history=false, sampler="sobol")

    @assert (sampler in ["sobol", "faure", "halton", "golden"]) "Invalid sampler."
    sampler_dict = Dict(
        "sobol" => QuasiMonteCarlo.SobolSample(),
        "faure" => QuasiMonteCarlo.FaureSample(),
        "halton" => QuasiMonteCarlo.HaltonSample(),
        "golden" => QuasiMonteCarlo.GoldenSample(),
    )

    function stratify_state_space(state_space, n_strata)
        num_dims = length(state_space)
        n_strata_per_dim = round(Int, n_strata ^ (1 / num_dims))

        # Initialize ranges for each dimension
        ranges = [range(state_space[dim][1], stop=state_space[dim][2], length=n_strata_per_dim+1) for dim in 1:num_dims]

        # Preallocate lower and upper bounds
        lb = Array{Float64,2}(undef, n_strata_per_dim ^ num_dims, num_dims)
        ub = Array{Float64,2}(undef, n_strata_per_dim ^ num_dims, num_dims)

        # Non-recursive function to generate strata
        indices = ones(Int, num_dims)
        for i in 1:size(lb, 1)
            for dim in 1:num_dims
                lb[i, dim], ub[i, dim] = ranges[dim][indices[dim]], ranges[dim][indices[dim] + 1]
            end

            # Update indices
            dim = 1
            while dim <= num_dims
                indices[dim] += 1
                if indices[dim] <= n_strata_per_dim
                    break
                else
                    indices[dim] = 1
                    dim += 1
                end
            end
        end

        return lb, ub
    end

    c_underbar_star = 0
    c_bar_star = Inf
    E = [0.0]
    sizehint!(E, Int(ns)+1)

    if history
        chistory = zeros(Int(ns))
        xi_all = zeros(N,Int(ns))
    end

    # Stratify the state space into n_strata
    strata_lb, strata_ub = stratify_state_space(state_space, n_strata)
    n_strata = size(strata_lb, 1)

    xi = Vector{Float64}(undef, N)
    xi_lift = Vector{Float64}(undef, Nl)
    sample_per_stratum = div(Int(ns), n_strata)
    ct = 1
    for k in 1:n_strata
        lb = strata_lb[k,:]
        ub = strata_ub[k,:]

        # sobol = Sobol.SobolSeq(lb, ub)
        samples = QuasiMonteCarlo.sample(sample_per_stratum, lb, ub, sampler_dict[sampler])
        for i = 1:sample_per_stratum
            # Quasi-Monte Carlo sampling within the current stratum
            # Sobol.next!(sobol, xi)
            xi = samples[:,i]
            xi_lift = vec(lifter.map(xi, gp))

            if V_dot(xi_lift) < 0 && V(xi_lift) < c_bar_star
                push!(E, V(xi_lift))
                if V(xi_lift) > c_underbar_star
                    c_underbar_star = V(xi_lift)
                end
            elseif V_dot(xi_lift) >= 0 && V(xi_lift) < c_bar_star
                c_bar_star = V(xi_lift)
                if c_underbar_star >= c_bar_star
                    c_underbar_star = try
                        maximum(filter(c -> c < c_bar_star, E))
                    catch e 
                        if isa(e, MethodError) 
                            continue
                        else
                            @error e 
                        end
                    end
                end
            end

            if history
                xi_all[:,ct] = xi
                chistory[ct] += c_underbar_star
            end

            ct += 1  # increment counter
        end
    end

    if history
        return c_underbar_star, chistory, xi_all
    else
        return c_underbar_star
    end
end



function doa_sampling(V, V_dot, ns, N, state_space; Nl=0, gp=1,
        n_strata=Int(2^N), method="memoryless", lifter=nothing, uniform_state_space=true, history=false, sampler="sobol")
    if method == "memoryless"
        if isnothing(lifter)
            return sampling_memoryless(V, V_dot, ns, N, state_space; uniform_state_space=uniform_state_space, history=history)
        else
            return sampling_memoryless(V, V_dot, ns, N, Nl, gp, state_space, lifter; uniform_state_space=uniform_state_space, history=history)
        end
    elseif method == "memory"
        if isnothing(lifter)
            return sampling_with_memory(V, V_dot, ns, N, state_space; uniform_state_space=uniform_state_space, history=history)
        else
            return sampling_with_memory(V, V_dot, ns, N, Nl, gp, state_space, lifter; uniform_state_space=uniform_state_space, history=history)
        end
    elseif method == "enhanced"
        if isnothing(lifter)
            return enhanced_sampling_with_memory(V, V_dot, ns, N, state_space, n_strata; history=history, sampler=sampler)
        else
            return enhanced_sampling_with_memory(V, V_dot, ns, N, Nl, gp, state_space, n_strata, lifter; history=history, sampler=sampler)
        end
    else
        error("Invalid method. Options are memoryless, memory, and enhanced.")
    end
end

"""
LEDOA: Largest Estimated Domain of Attraction
"""
function LEDOA(V::Function, V_dot::Function, N::Int; linear_solver::Union{String,Nothing}="ma57", 
                HSL_lib_path::Union{String,Nothing}=nothing, verbose::Bool=true, 
                ci::Real=1e2, xi::Union{AbstractArray,Nothing}=nothing, δ::Real=1.0)

    # ipopt = optimizer_with_attributes(
    #     Ipopt.Optimizer, 
    #     "print_level" => verbose ? 5 : 0, 
    #     "linear_solver" => isnothing(linear_solver) ? "mumps" : linear_solver, 
    #     "hsllib" => HSL_lib_path
    # )
    # model = Model(
    #     optimizer_with_attributes(
    #         Alpine.Optimizer,
    #         "nlp_solver" => ipopt,
    #     ),
    # )

    # model = Model(
    #     optimizer_with_attributes(
    #         Ipopt.Optimizer, 
    #         "print_level" => verbose ? 5 : 0, 
    #         "linear_solver" => isnothing(linear_solver) ? "mumps" : linear_solver, 
    #         "hsllib" => HSL_lib_path
    #     ),
    # )

    # if !isnothing(linear_solver)
    #     if !isnothing(HSL_lib_path)
    #         set_optimizer_attribute(model, "hsllib", HSL_lib_path)
    #     end
    #     set_optimizer_attribute(model, "linear_solver", linear_solver)
    # end

    # Set up verbose or silent
    # if verbose
    #     unset_silent(model)
    # else
    #     set_silent(model)
    # end 
    # set_optimizer_attribute(model, "print_level", verbose ? 5 : 0)  # Adjusting print level based on verbose flag

    model = Model(SCS.Optimizer)
    # Set up verbose or silent

    set_string_names_on_creation(model, false)

    # register(model, :V, 1, V; autodiff=true)
    # register(model, :V_dot, 1, V_dot; autodiff=true)

    @variable(model, c >= eps())
    @variable(model, x[1:N])

    # Set initial values
    # set_start_value(c, ci)
    # if !isnothing(xi)
    #     set_start_value.(x, xi)
    # end


    # @expression(model, vx, V(x))
    # @expression(model, vxdot, V_dot(x))

    @operator(model, vx, N, (x...) -> V(collect(x)))
    @operator(model, vxdot, N, (x...) -> V_dot(collect(x)))
    @constraint(model, vx(x...) == c)
    @constraint(model, vxdot(x...) >= 0)
    @constraint(model, δ^2 <= sum(x[i]^2 for i = 1:N))

    # @NLconstraint(model, V(x...) == c)
    # @NLconstraint(model, V_dot(x...) >= 0)

    # @constraint(model, [i = 1:N], xtol <= x[i])
    # @objective(model, Min, vx(x...))
    @objective(model, Min, c)
    optimize!(model)
    # status = termination_status(model)
    # if status == MOI.OPTIMAL
    #     return value(c), value.(x)
    # else
    #     @info """Optimization failed: 
    #     $(status)
    #     """
    #     return nothing, nothing
    # end
    # x_star = value.(x)

    # @assert is_solved_and_feasible(model)

    return value(c), value.(x)
end

