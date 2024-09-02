export Operators

include("polytools.jl")
include("legacy.jl")

"""
$(TYPEDEF)

Organize the operators of the system in a structure. The operators currently 
supported are up to second order.

## Fields
- `A`: linear state operator
- `B`: linear input operator
- `C`: linear output operator
- `F`: quadratic state operator with no redundancy
- `H`: quadratic state operator with redundancy
- `Q`: quadratic state operator with redundancy in 3-dim tensor form
- `G`: cubic state operator with redundancy
- `E`: cubic state operator with no redundancy
- `K`: constant operator
- `N`: bilinear (state-input) operator
- `f`: nonlinear function operator f(x,u)
"""
# Base.@kwdef mutable struct Operators
#     A::Union{AbstractArray{<:Number},Real} = 0                                           # linear
#     B::Union{AbstractArray{<:Number},Real} = 0                                           # control
#     C::Union{AbstractArray{<:Number},Real} = 0                                           # output
#     H::Union{AbstractArray{<:Number},Real} = 0                                           # quadratic redundant
#     F::Union{AbstractArray{<:Number},Real} = 0                                           # quadratic non-redundant
#     Q::Union{AbstractArray{<:Number},AbstractArray{<:AbstractArray{<:Number}},Real} = 0  # quadratic (array of 2D square matrices)
#     G::Union{AbstractArray{<:Number},Real} = 0                                           # cubic redundant
#     E::Union{AbstractArray{<:Number},Real} = 0                                           # cubic non-redundant
#     K::Union{AbstractArray{<:Number},Real} = 0                                           # constant
#     N::Union{AbstractArray{<:Number},AbstractArray{<:AbstractArray{<:Number}},Real} = 0  # bilinear
#     f::Function = x -> x                                                                 # nonlinear function
# end

# INFO: Make this into a dictionary
# Base.@kwdef mutable struct Operators
#     A::Union{AbstractArray{<:Number},Real} = 0                                             # linear
#     B::Union{AbstractArray{<:Number},Real} = 0                                             # control
#     C::Union{AbstractArray{<:Number},Real} = 0                                             # output
#     A2::Union{AbstractArray{<:Number},Real} = 0                                            # quadratic redundant
#     Ah2::Union{AbstractArray{<:Number},Real} = 0                                           # quadratic non-redundant
#     As2::Union{AbstractArray{<:Number},AbstractArray{<:AbstractArray{<:Number}},Real} = 0  # quadratic (array of 2D "s"quare matrices)
#     A3::Union{AbstractArray{<:Number},Real} = 0                                            # cubic redundant
#     Ah3::Union{AbstractArray{<:Number},Real} = 0                                           # cubic non-redundant
#     K::Union{AbstractArray{<:Number},Real} = 0                                             # constant
#     E::Union{AbstractArray{<:Number},AbstractArray{<:AbstractArray{<:Number}},Real} = 0    # bilinear
#     f::Function = x -> x                                                                   # nonlinear function
# end




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

    # Nonlinear function operator
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



# Base.@kwdef mutable struct Operators 
#     A::Union{AbstractArray{<:Number},Real} = 0  # linear state operator
#     Ak::Dict{<:Int,Union{AbstractArray{<:Number},Real}} = Dict(1=>0)  # polynomial state operators (quadratic, cubic, etc.)
#     Aku::Dict{<:Int,Union{AbstractArray{<:Number},Real}} = Dict(1=>0)  # "u"nique polynomial state operators (quadratic, cubic, etc.)
#     Akt::Dict{<:Int,Union{AbstractArray{<:Number},Real}} = Dict(1=>0)  # "t"ensor (unflattened) polynomial state operators (quadratic, cubic, etc.)
#     B::Union{AbstractArray{<:Number},Real} = 0  # control operator
#     Nk::Dict{<:Int,Union{AbstractArray{<:Number},AbstractArray{<:AbstractArray{<:Number}},Real}} = Dict(1=>0)  # polynomial state-input coupled operators (bilinaer etc.)
#     C::Union{AbstractArray{<:Number},Real} = 0  # output operator
#     K::Union{AbstractArray{<:Number},Real} = 0  # constant operator
#     f::Function = x -> x  # nonlinear function operator f(x,u)

#     # dimensions
#     dims::Dict{}
# end
