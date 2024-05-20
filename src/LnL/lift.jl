export lifting, liftedBasis

"""
$(TYPEDEF)

Lifting map structure.

## Fields
- `N`: number of variables of the original nonlinear dynamics
- `Nl`: number of variables of the lifted system
- `lift_funcs`: array of lifting transformation functions 
- `map`: function to map the data to the new mapped states including original states
- `mapNL`: function to map the data to only the additional lifted states 

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
    liftedBasis(W, Nl, gp, ro) → Vr

Create the block-diagonal POD basis for the new lifted system data

## Arguments
- `w`: lifted data matrix
- `Nl`: number of variables of the lifted state dynamics
- `gp`: number of grid points for each variable
- `ro`: vector of the reduced orders for each basis

## Return
- `Vr`: block diagonal POD basis

"""
function liftedBasis(W::Matrix, Nl::Real, gp::Integer, ro::Vector)
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
