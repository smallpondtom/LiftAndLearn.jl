export Operators

"""
$(TYPEDEF)

Organize the operators of the system in a structure. The operators currently 
supported are up to second order.

# Fields
- `A`: linear state operator
- `B`: linear input operator
- `C`: linear output operator
- `D`: linear input-output coupling operator
- `A2`: quadratic state operator with redundancy
- `A3`: cubic state operator with redundancy
- `A4`: quartic state operator with redundancy
- `A2u`: quadratic state operator with no redundancy
- `A3u`: cubic state operator with no redundancy
- `A4u`: quartic state operator with no redundancy
- `A2t`: quadratic state operator with redundancy in 3-dim tensor form
- `N`: bilinear (state-input) operator
- `K`: constant operator
- `f`: nonlinear function operator f(x,u)
- `dims`: dimensions of the operators

# Note 
- Currently only supports
    - state: up to 4th order
    - input: only B matrix
    - output: only C and D matrices
    - state-input-coupling: bilinear 
    - constant term: K matrix
    - nonlinearity: f(x,u)
"""
Base.@kwdef mutable struct Operators
    # Linear IO sytem operators
    A::Union{AbstractArray{<:Number},Real} = 0                                             # linear
    B::Union{AbstractArray{<:Number},Real} = 0                                             # control
    C::Union{AbstractArray{<:Number},Real} = 0                                             # output
    D::Union{AbstractArray{<:Number},Real} = 0                                             # input-output coupling

    # Polynomial state operators
    A2::Union{AbstractArray{<:Number},Real} = 0                                            # quadratic redundant
    A3::Union{AbstractArray{<:Number},Real} = 0                                            # cubic redundant
    A4::Union{AbstractArray{<:Number},Real} = 0                                            # quartic redundant

    # Polynomial state operators (unique/non-redundant)
    A2u::Union{AbstractArray{<:Number},Real} = 0                                           # quadratic non-redundant
    A3u::Union{AbstractArray{<:Number},Real} = 0                                           # cubic non-redundant
    A4u::Union{AbstractArray{<:Number},Real} = 0                                           # quartic non-redundant

    # Polynomial state operators (tensor/unflattened)
    A2t::Union{AbstractArray{<:Number},Real} = 0                                           # quadratic tensor

    # State-input coupled operators
    N::Union{AbstractArray{<:Number},AbstractArray{<:AbstractArray{<:Number}},Real} = 0    # bilinear

    # Constant operators
    K::Union{AbstractArray{<:Number},Real} = 0                                             # constant

    # Nonlinear function operator (default using defined operators)
    f::Function =
        begin
            if size(B,2) == 1
                (x,u) -> A2u*⊘(x, iszero(A2u) ? 1 : 2) + A3u*⊘(x, iszero(A3u) ? 1 : 3) + A4u*⊘(x, iszero(A4u) ? 1 : 4) + (N*x)*u[1]
            else
                (x,u) -> A2u*⊘(x, iszero(A2u) ? 1 : 2) + A3u*⊘(x, iszero(A3u) ? 1 : 3) + A4u*⊘(x, iszero(A4u) ? 1 : 4) + sum([(N[i] * x) * u[i] for i in 1:size(B,2)]) 
            end
        end 

    # Dimensions
    dims::Dict{Symbol, Int64} = Dict(
        :A => iszero(A) ? 0 : size(A,1), 
        :B => iszero(B) ? 0 : size(B,2), 
        :C => iszero(C) ? 0 : size(C,1), 
        :N => iszero(N) ? 0 : size(A,1) * size(B,2),
        :A2 => iszero(A2) ?  0 : size(A2,2), 
        :A3 => iszero(A3) ?  0 : size(A3,2),
        :A4 => iszero(A4) ?  0 : size(A4,2),
        :A2u => iszero(A2u) ? 0 : size(A2u,2), 
        :A3u => iszero(A3u) ? 0 : size(A3u,2), 
        :A4u => iszero(A4u) ? 0 : size(A4u,2), 
    ) 
end