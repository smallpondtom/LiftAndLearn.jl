"""
    proj_error(Xf, Vr) → PE

Compute the projection error

# Arguments
- `Xf`: reference state data
- `Vr`: POD basis

# Return
- `PE`: projection error
"""
function proj_error(Xf, Vr)
    # Projection error
    Xf_norm = norm(Xf, 2)
    PE = norm(Xf - Vr * Vr' * Xf, 2) / Xf_norm
    return PE
end


"""
    rel_state_error(Xf, X, Vr) → SE

Compute the relative state error

# Arguments
- `Xf`: reference state data
- `X`: testing state data
- `Vr`: POD basis

# Return
- `SE`: state error
"""
function rel_state_error(Xf, X, Vr)
    n = size(Xf, 1)
    Xf_norm = norm(Xf, 2)
    SE = norm(Xf - Vr[1:n, :] * X, 2) / Xf_norm
    return SE
end


"""
    rel_output_error(Yf, Y) → OE

Compute relative output error

# Arguments
- `Yf`: reference output data
- `Y`: testing output data

# Return
- `OE`: output error
"""
function rel_output_error(Yf, Y)
    n = size(Yf, 1)
    Yf_norm = norm(Yf, 2)
    OE = norm(Yf - Y[1:n, :], 2) / Yf_norm
    return OE
end


"""
    compute_all_errors(Xf, Yf, Xint, Yint, Xinf, Yinf, Vr) → PE, ISE, IOE, OSE, OOE

Compute all projection, state, and output errors

# Arguments
- `Xf`: reference state data
- `Yf`: reference output data
- `Xint`: intrusive model state data
- `Yint`: intrusive model output data
- `Xinf`: inferred model state data
- `Xint`: inferrred model output data
- `Vr`: POD basis

# Return
- `PE`: projection error
- `ISE`: intrusive state error
- `IOE`: intrusive output error
- `OSE`: operator inference state error
- `OOE`: operator inference output error
"""
function compute_all_errors(Xf, Yf, Xint, Yint, Xinf, Yinf, Vr)
    # Projection error
    Xf_norm = norm(Xf, 2)
    Yf_norm = norm(Yf, 2)
    PE = norm(Xf - Vr * Vr' * Xf, 2) / Xf_norm

    # State and output error of the intrusive model 
    ISE = norm(Xf - Vr * Xint, 2) / Xf_norm
    IOE = norm(Yf - Yint, 2) / Yf_norm

    # State errors of the operator inference model
    OSE = norm(Xf - Vr * Xinf, 2) / Xf_norm
    OOE = norm(Yf - Yinf, 2) / Yf_norm
    return PE, ISE, IOE, OSE, OOE
end

#

"""
$(SIGNATURES)

Compute the constraint residual which is the residual of the energy-preserving constraint 
```math
\\sum \\left| \\hat{h}_{ijk} + \\hat{h}_{jik} + \\hat{h}_{kji} \\right| \\quad 1 \\leq i,j,k \\leq r
```

## Arguments
- `X::AbstractArray`: the matrix to compute the constraint residual
- `r::Real`: the dimension of the system
- `redundant::Bool`: redundant or nonredundant operator
- `with_moment::Bool`: whether to compute the moment of the constraint residual

## Returns
- `ϵX`: the constraint residual
- `mmt`: the moment which is the sum of the constraint residual without absolute value
"""
function ep_constraint_residual(X::AbstractArray, r::Real, redundant::Bool=false; with_moment::Bool=false)
    ϵX = 0
    mmt = 0

    if with_moment
        if redundant
            for i in 1:r, j in 1:r, k in 1:r
                foo = X[i, r*(k-1)+j] + X[j, r*(k-1)+i] + X[k, r*(i-1)+j]
                ϵX += abs(foo)
                mmt += foo
            end
        else
            for i in 1:r, j in 1:r, k in 1:r
                foo = delta(j,k)*X[i, fidx(r,j,k)] + delta(i,k)*X[j, fidx(r,i,k)] + delta(j,i)*X[k, fidx(r,j,i)]
                ϵX += abs(foo)
                mmt += foo
            end
        end
        return ϵX, mmt
    else
        if redundant
            for i in 1:r, j in 1:r, k in 1:r
                foo = X[i, r*(k-1)+j] + X[j, r*(k-1)+i] + X[k, r*(i-1)+j]
                ϵX += abs(foo)
            end
        else
            for i in 1:r, j in 1:r, k in 1:r
                foo = delta(j,k)*X[i, fidx(r,j,k)] + delta(i,k)*X[j, fidx(r,i,k)] + delta(j,i)*X[k, fidx(r,j,i)]
                ϵX += abs(foo)
            end
        end
        return ϵX
    end
end


"""
$(SIGNATURES)

Compute the constraint violation which is the violation of the energy-preserving constraint
```math
\\sum \\langle \\mathbf{x}, \\mathbf{H}(\\mathbf{x}\\otimes\\mathbf{x})\\rangle \\quad \\forall \\mathbf{x} \\in \\mathcal{D}
```

## Arguments
- `Data::AbstractArray`: the data
- `X::AbstractArray`: the matrix to compute the constraint violation
- `redundant::Bool`: redundant or nonredundant operator

## Returns
- `viol`: the constraint violation
"""
function ep_constraint_violation(Data::AbstractArray, X::AbstractArray, redundant::Bool=false)
    _, m = size(Data)
    viol = zeros(m,1)
    if redundant
        for i in 1:m
            viol[i] = Data[:,i]' * X * kron(Data[:,i], Data[:,i])
        end
    else
        for i in 1:m
            viol[i] = Data[:,i]' * X * UniqueKronecker.vech(Data[:,i] * Data[:,i]')
        end
    end
    return viol
end


"""
$(SIGNATURES)

Check if the matrix is energy-preserving.

## Arguments
- `X::AbstractArray`: the matrix to check if it is energy-preserving
- `redundant::Bool`: redundant or nonredundant operator
- `tol::Real`: the tolerance

## Returns
- `Bool`: whether the matrix is energy-preserving
"""
function isenergypreserving(X::AbstractArray, redundant::Bool=false; tol=1e-8)
    r = size(X, 1)
    ϵX = ep_constraint_residual(X, r, redundant)
    return ϵX < tol
end