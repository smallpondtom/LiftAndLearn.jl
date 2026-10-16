export Operators

"""
$(TYPEDEF)

Container for the (full-order or reduced) operators of a polynomial system

```math
\\dot{\\mathbf{x}} = \\mathbf{A}\\mathbf{x} + \\mathbf{B}\\mathbf{u} + \\mathbf{A}_2(\\mathbf{x}\\otimes\\mathbf{x})
+ \\mathbf{A}_3(\\mathbf{x}\\otimes\\mathbf{x}\\otimes\\mathbf{x}) + \\mathbf{A}_4(\\mathbf{x}\\otimes\\mathbf{x}\\otimes\\mathbf{x}\\otimes\\mathbf{x})
+ \\mathbf{N}\\mathbf{x}u + \\mathbf{K}, \\qquad \\mathbf{y} = \\mathbf{C}\\mathbf{x} + \\mathbf{D}\\mathbf{u}.
```

Constructed with keyword arguments (`Base.@kwdef`); every operator defaults to the scalar
`0`, meaning "absent".

## Fields
- `A::Union{AbstractArray{<:Number},Real}=0`: linear state operator (``n \\times n``).
- `B::Union{AbstractArray{<:Number},Real}=0`: linear input operator (``n \\times m``).
- `C::Union{AbstractArray{<:Number},Real}=0`: linear output operator (``l \\times n``).
- `D::Union{AbstractArray{<:Number},Real}=0`: linear input-output (feedthrough) operator (``l \\times m``).
- `A2::Union{AbstractArray{<:Number},Real}=0`: quadratic state operator acting on ``\\mathbf{x}\\otimes\\mathbf{x}`` (redundant form, ``n \\times n^2``).
- `A3::Union{AbstractArray{<:Number},Real}=0`: cubic state operator (redundant form, ``n \\times n^3``).
- `A4::Union{AbstractArray{<:Number},Real}=0`: quartic state operator (redundant form, ``n \\times n^4``).
- `A2u::Union{AbstractArray{<:Number},Real}=0`: quadratic state operator acting on the unique Kronecker product ``\\mathbf{x}\\oslash\\mathbf{x}`` (non-redundant form, ``n \\times n(n+1)/2``).
- `A3u::Union{AbstractArray{<:Number},Real}=0`: cubic state operator (non-redundant form).
- `A4u::Union{AbstractArray{<:Number},Real}=0`: quartic state operator (non-redundant form).
- `A2t::Union{AbstractArray{<:Number},Real}=0`: quadratic state operator as a 3-dimensional tensor (``n \\times n \\times n``).
- `N::Union{AbstractArray{<:Number},AbstractArray{<:AbstractArray{<:Number}},Real}=0`: bilinear (state-input) operator; an ``n \\times n`` matrix for one input, or an ``n \\times n \\times m`` array / vector of ``n \\times n`` matrices for ``m`` inputs.
- `K::Union{AbstractArray{<:Number},Real}=0`: constant operator (``n \\times 1``).
- `f::Function=_default_nonlinear_function`: nonlinear right-hand side ``\\mathbf{f}(\\mathbf{x}, \\mathbf{u})``. Supply your own `(x, u) -> ...` closure for non-polynomial terms (e.g. `f=(x, u) -> -x.^3`). The default is a placeholder: `reproject` then uses the polynomial and bilinear operators (`A2u`, `A3u`, `A4u`, `N`) instead, and `_reconstruct_function!` turns it into an explicit closure.
- `O::Union{AbstractArray{<:Number},Real}=0`: concatenated operator matrix ``\\mathbf{O}`` returned by the OpInf regression.
- `dims::Dict{Symbol,Int64}`: dimensions of the operators, computed at construction from the operators given (`:A`, `:B`, `:C`, `:N`, `:A2`, `:A3`, `:A4`, `:A2u`, `:A3u`, `:A4u`; `0` if absent). It is not updated when fields are modified later.

## Notes
- Currently supports
    - state: up to 4th order,
    - input: only the ``\\mathbf{B}`` matrix,
    - output: only the ``\\mathbf{C}`` and ``\\mathbf{D}`` matrices,
    - state-input coupling: bilinear,
    - constant term: ``\\mathbf{K}``,
    - general nonlinearity: ``\\mathbf{f}(\\mathbf{x}, \\mathbf{u})``.

## Examples
```julia
op = Operators(A=[-2.0 0.0; 0.0 -1.0], A2u=[0.0 1.0 0.0; 0.0 1.0 0.0], B=reshape([0.0, -0.01], 2, 1))
```
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

    # # Nonlinear function operator (default using defined operators)
    # f::Function =
    #     begin
    #         if size(B,2) == 1
    #             (x,u) -> A2u*⊘(x, iszero(A2u) ? 1 : 2) + A3u*⊘(x, iszero(A3u) ? 1 : 3) + A4u*⊘(x, iszero(A4u) ? 1 : 4) + (N*x)*u[1]
    #         else
    #             (x,u) -> A2u*⊘(x, iszero(A2u) ? 1 : 2) + A3u*⊘(x, iszero(A3u) ? 1 : 3) + A4u*⊘(x, iszero(A4u) ? 1 : 4) + sum([(N[i] * x) * u[i] for i in 1:size(B,2)]) 
    #         end
    #     end 

    # Nonlinear function operator (default using defined operators) - not serialized
    f::Function = _default_nonlinear_function

    # Concatenated operators from the solution of OpInf
    O::Union{AbstractArray{<:Number},Real} = 0

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

# Define the default function separately
"""
$(SIGNATURES)

Build the default nonlinear right-hand side ``\\mathbf{f}(\\mathbf{x}, \\mathbf{u})`` of `ops`
from its non-redundant polynomial operators and its bilinear operator. Internal helper.

The returned closure evaluates
``\\mathbf{A}_{2u}(\\mathbf{x}\\oslash\\mathbf{x}) + \\mathbf{A}_{3u}(\\mathbf{x}\\oslash\\mathbf{x}\\oslash\\mathbf{x}) + \\mathbf{A}_{4u}(\\cdots) + \\sum_i (\\mathbf{N}_i\\mathbf{x})u_i``,
where absent (zero) operators contribute zero.

## Arguments
- `ops::Operators`: operators of the system.

## Returns
- `f::Function`: closure `(x, u) -> ...` (it reads the fields of `ops` at call time).

## Notes
- For more than one input (`size(ops.B, 2) > 1`) the bilinear term uses `ops.N[i]`, i.e.
  `N` must then be a vector of ``n \\times n`` matrices.
"""
function _default_nonlinear_function(ops::Operators)
    if size(ops.B,2) == 1
        return (x,u) -> ops.A2u*⊘(x, iszero(ops.A2u) ? 1 : 2) + ops.A3u*⊘(x, iszero(ops.A3u) ? 1 : 3) + ops.A4u*⊘(x, iszero(ops.A4u) ? 1 : 4) + (ops.N*x)*u[1]
    else
        return (x,u) -> ops.A2u*⊘(x, iszero(ops.A2u) ? 1 : 2) + ops.A3u*⊘(x, iszero(ops.A3u) ? 1 : 3) + ops.A4u*⊘(x, iszero(ops.A4u) ? 1 : 4) + sum([(ops.N[i] * x) * u[i] for i in 1:size(ops.B,2)]) 
    end
end

# Reconstruction after loading
"""
$(SIGNATURES)

Set `ops.f` to the default nonlinear function built by `_default_nonlinear_function`
(e.g. after construction or after loading an `Operators` from disk). Internal helper.

## Arguments
- `ops::Operators`: operators of the system (mutated).

## Returns
- `f::Function`: the new `ops.f`; mutates `ops` in place.
"""
function _reconstruct_function!(ops::Operators)
    ops.f = _default_nonlinear_function(ops)
end

"""
$(SIGNATURES)

Return the nonlinear right-hand side ``\\mathbf{f}(\\mathbf{x}, \\mathbf{u})`` of `ops` without
modifying it: the user-supplied `ops.f` if one was given, otherwise the default built from the
polynomial and bilinear operators by `_default_nonlinear_function`. Internal helper.

## Arguments
- `ops::Operators`: operators of the system.

## Returns
- `f::Function`: closure `(x, u) -> ...`.
"""
function _nonlinear_function(ops::Operators)
    return ops.f === _default_nonlinear_function ? _default_nonlinear_function(ops) : ops.f
end