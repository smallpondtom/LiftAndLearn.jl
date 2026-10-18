export lifting, lifted_basis

"""
$(TYPEDEF)

Lifting map for Lift & Learn (LnL).

Constructed as `lifting(N, Nl, lift_funcs)`, where `lift_funcs` contains the `Nl - N`
functions that compute the additional lifted variables. Each lifting function receives the
original variables as a vector `x` of length `N` (one entry per variable, each a vector or
matrix of grid values) and returns the new lifted variable, e.g. `x -> x[1] .^ 2`.

## Fields
- `N::Int64`: number of variables of the original nonlinear dynamics.
- `Nl::Int64`: number of variables of the lifted system.
- `lift_funcs::AbstractArray{Function}`: the `Nl - N` lifting functions.
- `map::Function`: map the data to all lifted variables (original variables followed by the
  additional lifted ones). Accepts a vector of per-variable arrays, or a stacked array
  `map(x, gp)` whose rows are split into `N` blocks of `gp` grid points.
- `mapNL::Function`: map the data to only the additional lifted variables; same call
  signatures as `map`.

## Notes
- Throws an error if `Nl - N != length(lift_funcs)`.

## Examples
```julia
lifter = lifting(2, 3, [x -> x[1] .^ 2])
W = lifter.map(X, gp)   # X has 2*gp rows, W has 3*gp rows
```
"""
struct lifting
    N::Int64
    Nl::Int64
    lift_funcs::AbstractArray{Function}
    map::Function
    mapNL::Function

    function lifting(N, Nl, lift_funcs)
        if (Nl - N) != length(lift_funcs)
            error("Number of lifting functions does not match given dimension of lifted states")
        end

        lift_map = Vector{Function}(undef, Nl)
        for i in 1:N
            lift_map[i] = x -> x[i]
        end

        i = 0
        for j in (N+1):Nl
            lift_map[j] = lift_funcs[i+=1]
        end

        # Map all the original and the lifted variables
        function map(x::Union{Vector{Vector{T}},Vector{Matrix{T}},
            Vector{SparseVector{T,Int64}},Vector{SparseMatrixCSC{T,Int64}}}) where {T<:Real}
            splat(idx) = lift_map[idx](x)
            return reduce(vcat, splat.(1:Nl))
        end

        function map(x::AbstractArray{T}, gp::Int=1) where {T<:Real}
            xsep = Vector{Matrix{T}}(undef, N) 
            for k in 1:N-1
                xsep[k] = x[Int((k-1)*gp+1):Int(k*gp), :]
            end
            xsep[N] = x[Int((N-1)*gp+1):end, :]
            return map(xsep)
        end

        # Map only the nonlinear lifted variables
        function mapNL(x::Union{Vector{Vector{T}},Vector{Matrix{T}},
            Vector{SparseVector{T,Int64}},Vector{SparseMatrixCSC{T,Int64}}}) where {T<:Real}
            splat(idx) = lift_map[idx](x)
            return reduce(vcat, splat.(N+1:Nl))
        end

        function mapNL(x::AbstractArray{T}, gp::Int=1) where {T<:Real}
            xsep = Vector{Matrix{T}}(undef, N) 
            for p in 1:N-1
                xsep[p] = x[Int((p-1)*gp+1):Int(p*gp), :]
            end
            xsep[N] = x[Int((N-1)*gp+1):end, :]
            return mapNL(xsep)
        end

        new(N, Nl, lift_funcs, map, mapNL)
    end
end


"""
$(SIGNATURES)

Create the block-diagonal Proper Orthogonal Decomposition (POD) basis of the lifted
snapshot data, with one POD basis per lifted variable.

The rows of `W` are split into `Nl` blocks of `gp` rows; the leading `ro[i]` left singular
vectors of block `i` form the `i`-th diagonal block of the basis.

## Arguments
- `W::Matrix`: lifted snapshot matrix of size ``(N_l \\cdot gp) \\times K`` (columns = snapshots).
- `Nl::Real`: number of lifted variables.
- `gp::Integer`: number of grid points of each variable.
- `ro::Vector`: reduced dimension of each variable's basis (each `ro[i] <= gp`).

## Returns
- `Vr::BlockDiagonal`: block-diagonal basis of size ``(N_l \\cdot gp) \\times \\sum_i ro_i``.
"""
function lifted_basis(W::Matrix, Nl::Real, gp::Integer, ro::Vector)
    @assert size(W, 1) == Nl*gp "Number of rows of W must be equal to Nl*gp"
    @assert all(ro .<= gp) "Reduced order must be less than or equal to the number of grid points"

    V = Vector{Matrix{Float64}}()
    for i in 1:Nl
        w = svd(W[(i-1)*gp+1:i*gp, :])
        push!(V, w.U[:, 1:ro[i]])
    end
    Vr = BlockDiagonal(V)
    return Vr
end
