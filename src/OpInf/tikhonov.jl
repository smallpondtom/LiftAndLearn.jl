"""
    tikhonov(b::AbstractArray, A::AbstractArray, Γ::AbstractMatrix, tol::Real;
        flag::Bool=false)

Tikhonov regression

## Arguments
- `b::AbstractArray`: right hand side of the regression problem
- `A::AbstractArray`: left hand side of the regression problem
- `Γ::AbstractMatrix`: Tikhonov matrix
- `tol::Real`: tolerance for the singular values
- `flag::Bool`: flag for the tolerance

## Returns
- regression solution
"""
function tikhonov(b::AbstractArray, A::AbstractArray, Γ::AbstractMatrix, tol::Real; flag::Bool=false)
    if flag
        Ag = A' * A + Γ' * Γ
        Ag_svd = svd(Ag)
        sing_idx = findfirst(Ag_svd.S .< tol)

        # If singular values are nearly singular, truncate at a certain threshold
        # and fill in the rest with zeros
        if sing_idx !== nothing
            @warn "Rank difficient, rank = $(sing_idx), tol = $(Ag_svd.S[sing_idx]).\n"
            foo = [1 ./ Ag_svd.S[1:sing_idx-1]; zeros(length(Ag_svd.S[sing_idx:end]))]
            bar = Ag_svd.Vt' * Diagonal(foo) * Ag_svd.U'
            return bar * (A' * b)
        else
            return Ag \ (A' * b)
        end
    else
        return (A' * A + Γ' * Γ) \ (A' * b)
    end
end


"""
    tikhonovMatrix!(Γ::AbstractArray, dims::Dict, options::AbstractOption)

Construct the Tikhonov matrix

## Arguments
- `Γ::AbstractArray`: Tikhonov matrix (pass by reference)
- `options::AbstractOption`: options for the operator inference set by the user

## Returns
- `Γ`: Tikhonov matrix (pass by reference)
"""
function tikhonovMatrix!(Γ::AbstractArray, dims::AbstractArray, operator_symbols::AbstractArray, 
                         λ::TikhonovParameter)
    si = 1
    for (d, symbol) in zip(dims, operator_symbols)
        symbol_str  = string(symbol)
        if (length(symbol_str) >= 2) && ('A' in symbol_str)
            Γ[si:si+d-1] .= getproperty(λ, Symbol(symbol_str[1:2]))
        else
            Γ[si:si+d-1] .= getproperty(λ, symbol)
        end
        si += d
    end
end
