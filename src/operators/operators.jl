export operators

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
Base.@kwdef mutable struct operators
    A::Union{AbstractArray{<:Number},Real} = 0                                           # linear
    B::Union{AbstractArray{<:Number},Real} = 0                                           # control
    C::Union{AbstractArray{<:Number},Real} = 0                                           # output
    H::Union{AbstractArray{<:Number},Real} = 0                                           # quadratic redundant
    F::Union{AbstractArray{<:Number},Real} = 0                                           # quadratic non-redundant
    Q::Union{AbstractArray{<:Number},AbstractArray{<:AbstractArray{<:Number}},Real} = 0  # quadratic (array of 2D square matrices)
    G::Union{AbstractArray{<:Number},Real} = 0                                           # cubic redundant
    E::Union{AbstractArray{<:Number},Real} = 0                                           # cubic non-redundant
    K::Union{AbstractArray{<:Number},Real} = 0                                           # constant
    N::Union{AbstractArray{<:Number},AbstractArray{<:AbstractArray{<:Number}},Real} = 0  # bilinear
    f::Function = x -> x                                                                 # nonlinear function
end
