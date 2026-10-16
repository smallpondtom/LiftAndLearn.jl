"""
$(SIGNATURES)

Compute the relative projection error of the snapshot data onto a basis.

The error is measured as
```math
\\frac{\\| \\mathbf{X}_f - \\mathbf{V}_r \\mathbf{V}_r^\\top \\mathbf{X}_f \\|}{\\| \\mathbf{X}_f \\|}
```
where ``\\|\\cdot\\|`` is `norm(·, 2)`, i.e., the Frobenius norm for matrices (the entries are
treated as one long vector).

## Arguments
- `Xf`: reference state snapshot matrix of size ``n \\times K`` (columns are snapshots), or a single state vector.
- `Vr`: basis matrix of size ``n \\times r`` (e.g., the POD basis), assumed to have orthonormal columns.

## Returns
- `PE::Real`: the relative projection error.
"""
function proj_error(Xf, Vr)
    # Projection error
    Xf_norm = norm(Xf, 2)
    PE = norm(Xf - Vr * Vr' * Xf, 2) / Xf_norm
    return PE
end


"""
$(SIGNATURES)

Compute the relative state error of a reduced-order solution lifted back to the full space.

The error is ``\\| \\mathbf{X}_f - \\mathbf{V}_r[1:n,:] \\, \\mathbf{X} \\| / \\| \\mathbf{X}_f \\|`` in the
Frobenius norm, where ``n`` is the number of rows of `Xf`. Only the first ``n`` rows of `Vr` are
used, so a basis of a lifted (larger) state can be compared against the original state data.

## Arguments
- `Xf`: reference (full-order) state snapshot matrix of size ``n \\times K`` (columns are snapshots).
- `X`: reduced-order state trajectory of size ``r \\times K``.
- `Vr`: basis matrix with at least ``n`` rows and ``r`` columns (e.g., the POD basis).

## Returns
- `SE::Real`: the relative state error.
"""
function rel_state_error(Xf, X, Vr)
    n = size(Xf, 1)
    Xf_norm = norm(Xf, 2)
    SE = norm(Xf - Vr[1:n, :] * X, 2) / Xf_norm
    return SE
end


"""
$(SIGNATURES)

Compute the relative output error ``\\| \\mathbf{Y}_f - \\mathbf{Y}[1:n,:] \\| / \\| \\mathbf{Y}_f \\|``
in the Frobenius norm.

Only the first ``n`` rows of `Y` are compared, where ``n`` is the number of rows of `Yf`.

## Arguments
- `Yf`: reference output data of size ``n \\times K`` (columns are time instances).
- `Y`: output data to test, with at least ``n`` rows and ``K`` columns.

## Returns
- `OE::Real`: the relative output error.
"""
function rel_output_error(Yf, Y)
    n = size(Yf, 1)
    Yf_norm = norm(Yf, 2)
    OE = norm(Yf - Y[1:n, :], 2) / Yf_norm
    return OE
end


"""
$(SIGNATURES)

Compute the projection error and the relative state and output errors of an intrusive model
and an Operator Inference (OpInf) model in one call.

All errors are relative errors in the Frobenius norm (see [`proj_error`](@ref),
[`rel_state_error`](@ref), and [`rel_output_error`](@ref)). Reduced states are lifted back to
the full space with `Vr` before comparison.

## Arguments
- `Xf`: reference state snapshot matrix of size ``n \\times K`` (columns are snapshots).
- `Yf`: reference output data of size ``q \\times K``.
- `Xint`: reduced state trajectory of the intrusive model, size ``r \\times K``.
- `Yint`: output data of the intrusive model, size ``q \\times K``.
- `Xinf`: reduced state trajectory of the inferred (OpInf) model, size ``r \\times K``.
- `Yinf`: output data of the inferred (OpInf) model, size ``q \\times K``.
- `Vr`: basis matrix of size ``n \\times r`` (e.g., the POD basis).

## Returns
- `PE::Real`: projection error of `Xf` onto `Vr`.
- `ISE::Real`: intrusive model state error.
- `IOE::Real`: intrusive model output error.
- `OSE::Real`: OpInf model state error.
- `OOE::Real`: OpInf model output error.
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

Compute the residual of the energy-preserving constraint of a quadratic operator.

The residual is
```math
\\sum_{i,j,k=1}^{r} \\left| \\hat{h}_{ijk} + \\hat{h}_{jik} + \\hat{h}_{kji} \\right|
```
where ``\\hat{h}_{ijk}`` are the entries of the quadratic operator viewed as a third-order tensor.
A quadratic operator is energy-preserving when this residual is zero.

## Arguments
- `X::AbstractArray`: quadratic operator matrix, of size ``r \\times r^2`` if `redundant=true` or
  ``r \\times r(r+1)/2`` (unique Kronecker / `vech` ordering) if `redundant=false`.
- `r::Real`: the reduced dimension (number of rows of `X`).
- `redundant::Bool=false`: whether `X` acts on the redundant Kronecker product
  ``\\mathbf{x}\\otimes\\mathbf{x}`` (`true`) or on the non-redundant (unique) product (`false`).

## Keyword Arguments
- `with_moment::Bool=false`: if `true`, also return the moment (the sum of the residual terms
  without the absolute value).

## Returns
- `ϵX::Real`: the constraint residual.
- `mmt::Real`: the moment of the constraint residual (only returned when `with_moment=true`).
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

Evaluate the energy-preserving constraint violation of a quadratic operator on each snapshot.

For every snapshot ``\\mathbf{x}`` (column of `Data`) this computes
```math
\\langle \\mathbf{x}, \\mathbf{H}(\\mathbf{x}\\otimes\\mathbf{x})\\rangle ,
```
which vanishes for an energy-preserving quadratic operator ``\\mathbf{H}``.

## Arguments
- `Data::AbstractArray`: state snapshot matrix of size ``r \\times m`` (columns are snapshots).
- `X::AbstractArray`: quadratic operator matrix, of size ``r \\times r^2`` if `redundant=true` or
  ``r \\times r(r+1)/2`` if `redundant=false`.
- `redundant::Bool=false`: whether `X` acts on ``\\mathbf{x}\\otimes\\mathbf{x}`` (`true`) or on
  the unique Kronecker product `UniqueKronecker.vech(x * x')` (`false`).

## Returns
- `viol::Matrix`: an ``m \\times 1`` matrix whose ``i``-th entry is the violation for snapshot ``i``.
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

Check whether a quadratic operator is energy-preserving, i.e., whether its
[`ep_constraint_residual`](@ref) is below a tolerance.

## Arguments
- `X::AbstractArray`: quadratic operator matrix; the reduced dimension is taken as `size(X, 1)`.
- `redundant::Bool=false`: whether `X` acts on the redundant (`true`) or the non-redundant
  (`false`) Kronecker product.

## Keyword Arguments
- `tol::Real=1e-8`: tolerance on the constraint residual.

## Returns
- `::Bool`: `true` if the constraint residual is strictly less than `tol`.
"""
function isenergypreserving(X::AbstractArray, redundant::Bool=false; tol=1e-8)
    r = size(X, 1)
    ϵX = ep_constraint_residual(X, r, redundant)
    return ϵX < tol
end