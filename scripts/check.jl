using BenchmarkTools
using Random
using Kronecker
using LinearAlgebra
using LiftAndLearn
const LnL = LiftAndLearn

##
@benchmark LnL.vech(x * x') setup=(x=rand(10000))

##
@inline function ⊘(x::AbstractVector{T}) where T
    n = length(x)
    result = Vector{T}(undef, n*(n+1) ÷ 2)
    k = 1
    @inbounds for i in 1:n
        for j in i:n
            result[k] = x[i] * x[j]
            k += 1
        end
    end
    return result
end
@benchmark ⊘(x) setup=(x=rand(10000))

##
function stratify_state_space(state_space, n_strata)
    num_dims = length(state_space)
    n_strata_per_dim = round(Int, n_strata ^ (1 / num_dims))

    # Initialize ranges for each dimension
    ranges = [range(state_space[dim][1], stop=state_space[dim][2], length=n_strata_per_dim+1) for dim in 1:num_dims]

    # Recursive function to generate strata
    function generate_strata(dim, current_stratum)
        if dim > num_dims
            push!(strata, copy(current_stratum))
        else
            for val in ranges[dim][1:end-1]
                current_stratum[dim] = (val, val + step(ranges[dim]))
                generate_strata(dim + 1, current_stratum)
            end
        end
    end

    strata = []
    generate_strata(1, Array{Tuple{Float64, Float64}, 1}(undef, num_dims))

    return strata
end

# Example usage
state_space = ((-10.0, 10.0), (-5.0, 5.0), (0.0, 20.0), (-1,1), (-2,2)) # 3-dimensional space
n_strata = Int(5^5) # Example: divide into 27 strata
@benchmark stratify_state_space(state_space, n_strata)

##
function stratify_state_space(state_space, n_strata)
    num_dims = length(state_space)
    n_strata_per_dim = round(Int, n_strata ^ (1 / num_dims))

    # Initialize ranges for each dimension
    ranges = [range(state_space[dim][1], stop=state_space[dim][2], length=n_strata_per_dim+1) for dim in 1:num_dims]

    # Preallocate strata
    strata = Array{Array{Tuple{Float64, Float64},1},1}(undef, n_strata_per_dim ^ num_dims)
    current_stratum = Array{Tuple{Float64, Float64}, 1}(undef, num_dims)

    # Non-recursive function to generate strata
    indices = ones(Int, num_dims)
    for i in 1:size(strata, 1)
        for dim in 1:num_dims
            current_stratum[dim] = (ranges[dim][indices[dim]], ranges[dim][indices[dim] + 1])
        end
        strata[i] = copy(current_stratum)

        # Update indices
        dim = 1
        for dim in 1:num_dims
            indices[dim] += 1
            if indices[dim] <= n_strata_per_dim
                break
            else
                indices[dim] = 1
                dim += 1
            end
        end
    end

    return strata
end
state_space = ((-10.0, 10.0), (-5.0, 5.0), (0.0, 20.0)) # 3-dimensional space
n_strata = Int(2^3) # Example: divide into 27 strata
strate = stratify_state_space(state_space, n_strata)
# state_space = ((-10.0, 10.0), (-5.0, 5.0), (0.0, 20.0), (-1,1), (-2,2)) # 3-dimensional space
# n_strata = Int(5^5) # Example: divide into 27 strata
# @benchmark stratify_state_space(state_space, n_strata)


##
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

# state_space = ((-10.0, 10.0), (-5.0, 5.0), (0.0, 20.0), (-1,1), (-2,2)) # 3-dimensional space
# n_strata = Int(5^5) # Example: divide into 27 strata
# lb, ub = stratify_state_space(state_space, n_strata)
state_space = ((-10.0, 10.0), (-5.0, 5.0), (0.0, 20.0)) # 3-dimensional space
n_strata = Int(2^3) # Example: divide into 27 strata
lb, ub = stratify_state_space(state_space, n_strata)

##
@inline function Unique_Kronecker(x::AbstractArray{T}, y::AbstractArray{T}) where {T<:Number}
    n = length(x)
    m = length(y)
    result = Array{T}(undef, n*(n+1) ÷ 2)
    k = 1
    @inbounds for i in 1:n
        for j in i:m
            result[k] = x[i] * y[j]
            k += 1
        end
    end
    return result
end


@inline function Unique_Kronecker(x::AbstractArray{T}) where {T<:Number}
    n = length(x)
    result = Array{T}(undef, n*(n+1) ÷ 2)
    k = 1
    @inbounds for i in 1:n
        for j in i:n
            result[k] = x[i] * y[j]
            k += 1
        end
    end
    return result
end


@inline function Multi_Unique_Kronecker(x::AbstractArray{T}, times::Integer) where {T<:Number}
    result = x
    for _ in 1:(times-1)
        result = Unique_Kronecker(result, x)
    end
    return result
end

Unique_Kronecker(x::AbstractArray{T}) where {T<:Number} = Unique_Kronecker(x, x)
⊘(x::AbstractArray{T}, y::AbstractArray{T}) where {T<:Number} = Unique_Kronecker(x, y)

# ⊘(x::AbstractArray{T}) where {T<:Number} = ⊘(x, x)
# ⊘(x::AbstractArray{T}, y::AbstractArray{T}...) where {T<:Number} = ⊘(x, ⊘(y...))
# ⊘(x::AbstractArray{T}) where T = x
# ⊘(x::AbstractArray{T}...) where {T<:Number} = ⊘(x...)
# ⊘(x::AbstractArray{T}) where T = x ⊘ x


##

x = [1,2,3,4,5,6]
x ⊘ x
# Unique_Kronecker(x)
# Multi_Unique_Kronecker(x, 3)

##
x = [1,2,3,4]
a = kron(x,x)
b = x ⊗ x

function invec(r::AbstractArray, m::Int, n::Int)::VecOrMat
    tmp = vec(1.0I(n))'
    foo = (tmp ⊗ 1.0I(m))
    bar = (1.0I(n) ⊗ r)
    return  foo * bar
end

invec(x, 2, 2)

# @benchmark invec(x, 100, 100) setup=(x=rand(10000))

##
function invec2(r::AbstractArray, m::Int, n::Int)::VecOrMat
    tmp = vec(1.0I(n))'
    return kron(tmp, 1.0I(m)) * kron(1.0I(n), r)
end

invec(x, 2, 2)

@benchmark invec2(x, 100, 100) setup=(x=rand(10000))


##
# POD test 1
n = 10
m = 3
l = 2
p = 1
r = 5

A = round.(rand(n, n), digits=2)
B = round.(rand(n, m), digits=2)
C = round.(rand(l, n), digits=2)
K = round.(rand(n, 1), digits=2)
H = round.(rand(n, n^2), digits=2)
F = LnL.H2F(H)
N = round.(rand(n, n), digits=2)
op = LnL.operators(A=A, B=B, C=C, K=K, F=F, H=H, N=N)

Vr = round.(rand(n, r), digits=2)
Ahat = Vr' * A * Vr
Bhat = Vr' * B
Chat = C * Vr
Khat = Vr' * K
Hhat = Vr' * H * kron(Vr, Vr)
Fhat = Vr' * F * LnL.elimat(n) * kron(Vr, Vr) * LnL.dupmat(r)
Nhat = Vr' * N * Vr
op_naive = LnL.operators(A=Ahat, B=Bhat, C=Chat, K=Khat, F=Fhat, H=Hhat, N=Nhat)

system = LnL.sys_struct(
    is_lin=true, 
    is_quad=true, 
    is_bilin=true, 
    has_control=true, 
    has_output=true, 
    has_const=true,
)
options = LnL.LS_options(system=system)
op_rom = LnL.intrusiveMR(op, Vr, options)

##
using Test
for field in fieldnames(typeof(op_rom))
    if field !== :f && field !== :Q
        mat_rom = getfield(op_rom, field)
        mat_naive = getfield(op_naive, field)
        println(field)
        @test (mat_rom .== mat_naive)
    end
end


##
x = rand(10000)

@benchmark x ⊘ x

##
@benchmark vech(x * x')

##
@test all(x ⊘ x .== vech(x * x'))


## 
x = rand(10000)

@benchmark kron(x,x)

## 
@benchmark x ⊗ x

##
function squareMatStates1(Xmat)
    function vech_col(X)
        return X ⊘ X'
    end
    tmp = vech_col.(eachcol(Xmat))
    return reduce(hcat, tmp)
end


function squareMatStates2(Xmat)
    function vech_col(X)
        return vech(X * X')
    end
    tmp = vech_col.(eachcol(Xmat))
    return reduce(hcat, tmp)
end

@benchmark squareMatStates2(Xmat) setup=(Xmat=rand(512, 10000))

## 
@benchmark squareMatStates1(Xmat) setup=(Xmat=rand(512, 10000))

##

function kronMatStates(Xmat)
    function vec_col(X)
        return X ⊗ X
    end
    tmp = vec_col.(eachcol(Xmat))
    return reduce(hcat, tmp)
end

Xmat = [1 2 3; 4 5 6; 7 8 9]
kronMatStates(Xmat)

##
using SparseArrays
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
            println(splat.(1:Nl))
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

lifter = lifting(2, 4, [x -> sin.(x[1]), x -> cos.(x[1])])
x = [π π/2 0; 1.0 0 -1.0]
xsep = [x[1:1,:], x[2:2,:]]
lift_data = lifter.map(x)

##

function test1(n,N)
    z = zeros(n)
    for i in 1:N
        z = rand(n)
    end
end

function test2(n,N)
    z = Vector{Float64}(undef, n)
    for i in 1:N
        z = rand(n)
    end
end
##
@benchmark test1(100,10000)
##
@benchmark test2(100,10000)