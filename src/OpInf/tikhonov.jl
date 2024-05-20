"""
    tikhonov(b::AbstractArray, A::AbstractArray, Γ::AbstractMatrix, tol::Real;
        flag::Bool=false) → x

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
function tikhonovMatrix!(Γ::AbstractArray, options::AbstractOption)
    n = options.system.dims[:n]
    p = options.system.dims[:p]
    s2 = options.system.dims[:s2]
    v2 = options.system.dims[:v2]
    s3 = options.system.dims[:s3]
    v3 = options.system.dims[:v3]
    w1 = options.system.dims[:w1]

    λ = options.λ
    si = 0  # start index
    if n != 0
        Γ[1:n] .= λ.lin
        si += n 
    end

    if p != 0
        Γ[si+1:si+p] .= λ.ctrl
        si += p
    end

    if options.optim.which_quad_term == "F"
        if s2 != 0
            Γ[si+1:si+s2] .= λ.quad
        end
        si += s2
    else
        if v2 != 0
            Γ[si+1:si+v2] .= λ.quad
        end
        si += v2
    end

    if options.system.is_cubic
        if options.optim.which_cubic_term == "E"
            if s3 != 0
                Γ[si+1:si+s3] .= λ.cubic
            end
            si += s3
        else
            if v3 != 0
                Γ[si+1:si+v3] .= λ.cubic
            end
            si += v3
        end
    end

    if w1 != 0
        Γ[si+1:si+w1] .= λ.bilin
    end
end
