export DoA, est_stability_rad

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
    est_stability_rad(Ahat::AbstractArray{T}, Hhat::AbstractArray{T}, P::AbstractArray{T}; 
        div_by_2::Bool=true) where T

Estimate the stability radius of the system using the inferred Lyapunov function P.

## Arguments
* `Ahat::AbstractArray{T}`: inferred system matrix
* `Hhat::AbstractArray{T}`: inferred quadratic term
* `P::AbstractArray{T}`: inferred Lyapunov function matrix
* `div_by_2::Bool=true`: whether to divide the stability radius by 2

## Returns
* `ρhat::Float64`: estimated stability radius

## References
N. Sawant, B. Kramer, and B. Peherstorfer, “Physics-informed regularization and structure 
preservation for learning stable reduced models from data with operator inference.” arXiv, Jul. 
06, 2021. Accessed: Jan. 29, 2023. [Online]. Available: http://arxiv.org/abs/2107.02597
"""
function est_stability_rad(Ahat::AbstractArray{T}, Hhat::AbstractArray{T}, P::AbstractArray{T}; 
        div_by_2::Bool=true) where T

    if div_by_2
        LLt = lyapc(Ahat', 0.5*P)
    else
        LLt = lyapc(Ahat', P)
    end
    L = cholesky(LLt).L
    σmin = minimum(svd(L).S)
    ρhat = σmin / sqrt(norm(P,2)) / norm(Hhat,2) / 2
    return ρhat
end


function sampling_memoryless(V::Function, Vdot::Function, ns::Int, N::Int, 
        state_space::Union{Array{Tuple,1},Tuple}; uniform_state_space::Bool=true, history=false)
    c_hat_star = Inf

    xi = zeros(N)
    if history
        chistory = zeros(ns)
        xi_all = zeros(N,ns)
    end
    for j = 1:ns
        if uniform_state_space
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


function sampling_memoryless(V::Function, Vdot::Function, ns::Int, N::Int, Nl::Int, gp::Int, 
        state_space::Union{Array{Tuple,1},Tuple}, lifter::lifting; uniform_state_space::Bool=true, history=false)
    c_hat_star = Inf

    xi = zeros(N)
    xi_lift = zeros(Nl)
    if history
        chistory = zeros(ns)
        xi_all = zeros(N,ns)
    end
    for j = 1:ns
        if uniform_state_space
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


function sampling_with_memory(V::Function, Vdot::Function, ns::Int, N::Int, 
        state_space::Union{Array{Tuple,1},Tuple}; uniform_state_space::Bool=true, history=false)
    c_underbar_star = 0
    c_bar_star = Inf
    E = [0.0]
    sizehint!(E, ns+1)
    xi = zeros(N)
    if history
        chistory = zeros(ns)
        xi_all = zeros(N,ns)
    end
    for j = 1:ns
        if uniform_state_space
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


function sampling_with_memory(V::Function, Vdot::Function, ns::Int, N::Int, Nl::Int, gp::Int,
        state_space::Union{Array{Tuple,1},Tuple}, lifter::lifting; uniform_state_space::Bool=true, history=false)
    c_underbar_star = 0
    c_bar_star = Inf
    E = [0.0]
    sizehint!(E, ns+1)
    xi = zeros(N)
    xi_lift = zeros(Nl)
    if history
        chistory = zeros(ns)
        xi_all = zeros(N,ns)
    end
    for j = 1:ns
        if uniform_state_space
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
function enhanced_sampling_with_memory(V::Function, V_dot::Function, ns::Int, N::Int, 
        state_space::Array{Tuple}, n_strata::Int)

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

    function low_discrepancy_sequence(dim, n)
        sobol = Sobol.SobolSeq(dim)
        points = [Sobol.next!(sobol) for _ in 1:n]
        return points
    end

    c_underbar_star = 0
    c_bar_star = Inf
    E = [0]
    sizehint!(E, ns+1)

    # Stratify the state space into n_strata
    strata_lb, strata_ub = stratify_state_space(state_space, n_strata)
    n_strata = size(strata_lb, 1)

    xi = Vector{Float64}(undef, N)
    sample_per_stratum = div(ns, n_strata)
    for k in n_strata
        lb = strata_lb[k,:]
        ub = strata_ub[k,:]

        sobol = Sobol.SobolSeq(lb, ub)
        for i = 1:sample_per_stratum
            # Quasi-Monte Carlo sampling within the current stratum
            Sobol.next!(sobol, xi)

            if V_dot(xi) < 0 && V(xi) < c_bar_star
                push!(E, V(xi))
                if V(xi) > c_underbar_star
                    c_underbar_star = V(xi)
                end
            elseif V_dot(xi) >= 0 && V(xi) < c_bar_star
                c_bar_star = V(xi)
                if c_underbar_star >= c_bar_star
                    c_underbar_star = maximum(filter(c -> c < c_bar_star, E))
                end
            end
        end
    end

    return c_underbar_star
end


function enhanced_sampling_with_memory(V::Function, V_dot::Function, ns::Int, N::Int, Nl::Int,
        gp::Int, state_space::Array{Tuple}, n_strata::Int, lifter::lifting)

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

    function low_discrepancy_sequence(dim, n)
        sobol = Sobol.SobolSeq(dim)
        points = [Sobol.next!(sobol) for _ in 1:n]
        return points
    end

    c_underbar_star = 0
    c_bar_star = Inf
    E = [0]
    sizehint!(E, ns+1)

    # Stratify the state space into n_strata
    strata_lb, strata_ub = stratify_state_space(state_space, n_strata)
    n_strata = size(strata_lb, 1)

    xi = Vector{Float64}(undef, N)
    xi_lift = Vector{Float64}(undef, Nl)
    sample_per_stratum = div(ns, n_strata)
    for k in n_strata
        lb = strata_lb[k,:]
        ub = strata_ub[k,:]

        sobol = Sobol.SobolSeq(lb, ub)
        for i = 1:sample_per_stratum
            # Quasi-Monte Carlo sampling within the current stratum
            Sobol.next!(sobol, xi)
            xi_lift = vec(lifter.map(xi, gp))

            if V_dot(xi_lift) < 0 && V(xi_lift) < c_bar_star
                push!(E, V(xi_lift))
                if V(xi_lift) > c_underbar_star
                    c_underbar_star = V(xi_lift)
                end
            elseif V_dot(xi_lift) >= 0 && V(xi_lift) < c_bar_star
                c_bar_star = V(xi_lift)
                if c_underbar_star >= c_bar_star
                    c_underbar_star = maximum(filter(c -> c < c_bar_star, E))
                end
            end
        end
    end

    return c_underbar_star
end



function doa_sampling(V, V_dot, ns, N, state_space; Nl=0, gp=1,
        n_strata=Int(2^N), method="memoryless", lifter=nothing, uniform_state_space=true, history=false)
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
            return enhanced_sampling_with_memory(V, V_dot, ns, N, state_space, n_strata)
        else
            return enhanced_sampling_with_memory(V, V_dot, ns, N, Nl, gp, state_space, n_strata, lifter)
        end
    else
        error("Invalid method. Options are memoryless, memory, and enhanced.")
    end
end

# function zubov_error(X, A, F, P, Q)
#     n, m = size(X)
#     # Construct some values used in the optimization
#     X = n < m ? X : X'  # here we want the row to be the states and columns to be time
#     X2 = squareMatStates(X)'
#     X = X' # now we want the columns to be the states and rows to be time
#     return norm(X*A'*P*X' + X2*F'*P*X' - 0.25*X*P*X'*X*Q'*X' + 0.5*X*Q'*X', 2)
# end


