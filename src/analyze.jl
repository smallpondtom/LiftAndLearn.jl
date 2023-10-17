"""
File containing functions that analyze the results for operator inference and
LnL.
"""


"""
Plotting the error bounds for the test results with median, 1st quartile, and 
3rd quartile.

# Arguments
- `p`: plot figure
- `X`: reference data
- `Y`: testing data
- `name`: plot label
"""
# function errBnds(p, X, Y, name="Intrusive")
#     X = vec(X)'
#     Ym = median(Y, dims=1)
#     n = length(X)
#     q1 = quantile.(eachcol(Y), fill(0.25, n))
#     q3 = quantile.(eachcol(Y), fill(0.75, n))
#     ϵ⁻ = Ym' .- q1
#     ϵ⁺ = q3 .- Ym'
#     plot!(p, X', Ym', ribbon=(ϵ⁻', ϵ⁺'), fillalpha=0.25, lc=:blue, fillcolor=:green, label=name)
#     scatter!(p, X, Ym, ms=10, yerror=(ϵ⁻, ϵ⁺), legend=false, mc=:blue, shape=:star5)
#     return nothing
# end


"""
Compute the projection error

# Arguments
- `Xf`: reference state data
- `Vr`: POD basis

# Return
- `PE`: projection error
"""
function compProjError(Xf, Vr)
    # Projection error
    Xf_norm = norm(Xf, 2)
    PE = norm(Xf - Vr * Vr' * Xf, 2) / Xf_norm
    return PE
end


"""
Compute the state error

# Arguments
- `Xf`: reference state data
- `X`: testing state data
- `Vr`: POD basis

# Return
- `SE`: state error
"""
function compStateError(Xf, X, Vr)
    n = size(Xf, 1)
    Xf_norm = norm(Xf, 2)
    SE = norm(Xf - Vr[1:n, :] * X, 2) / Xf_norm
    return SE
end


"""
Compute output error

# Arguments
- `Yf`: reference output data
- `Y`: testing output data

# Return
- `OE`: output error
"""
function compOutputError(Yf, Y)
    n = size(Yf, 1)
    Yf_norm = norm(Yf, 2)
    OE = norm(Yf - Y[1:n, :], 2) / Yf_norm
    return OE
end


"""
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
function compError(Xf, Yf, Xint, Yint, Xinf, Yinf, Vr)
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


# function constraintResidual(Qall::Matrix, Hall::Matrix,
#     rmin::Real, rmax::Real, m::Real)
#     ϵQ = zeros(rmax - (rmin - 1), m)
#     ϵH = zeros(rmax - (rmin - 1), m)
#     for r in rmin:rmax
#         for p in 1:m
#             Q = Qall[r-(rmin-1), p]
#             H = Hall[r-(rmin-1), p]
#             ϵQ_tmp = 0
#             ϵH_tmp = 0
#             for i in 1:r, j in 1:r, k in 1:r
#                 ϵQ_tmp += Q[i][j, k] + Q[j][i, k] + Q[k][j, i]
#                 ϵH_tmp += H[i, r*(j-1)+k] + H[j, r*(i-1)+k] + H[k, r*(j-1)+i]
#             end
#             ϵQ[r-(rmin-1), p] = ϵQ_tmp / (r^3)
#             ϵH[r-(rmin-1), p] = ϵH_tmp / (r^3)
#         end
#     end
#     return ϵQ, ϵH
# end


# function constraintResidual(Q::Array{Matrix{Float64}}, H::Union{Matrix, SparseMatrixCSC}, r::Real)
#     ϵQ = 0
#     ϵH = 0
#     for i in 1:r, j in 1:r, k in 1:r
#         ϵQ += Q[i][j, k] + Q[j][i, k] + Q[k][j, i]
#         ϵH += H[i, r*(j-1)+k] + H[j, r*(i-1)+k] + H[k, r*(j-1)+i]
#     end
#     ϵQ /= r^3
#     ϵH /= r^3
#     return ϵQ, ϵH
# end


# function constraintResidual(Hall::Matrix, rmin::Real, rmax::Real, m::Real)
#     ϵH = zeros(rmax - (rmin - 1), m)
#     for r in rmin:rmax
#         for p in 1:m
#             H = Hall[r-(rmin-1), p]
#             ϵH_tmp = 0
#             for i in 1:r, j in 1:i, k in 1:j
#                 ϵH_tmp += H[i, r*(j-1)+k] + H[j, r*(i-1)+k] + H[k, r*(j-1)+i]
#             end
#             ϵH[r-(rmin-1), p] = ϵH_tmp
#         end
#     end
#     return ϵH
# end


function constraintResidual(X::Union{Matrix, SparseMatrixCSC}, r::Real, which_quad::String="H"; with_mmt=false)
    ϵX = 0
    mmt = 0

    if with_mmt
        if which_quad == "H"
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
        if which_quad == "H"
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


function constraintViolation(Data::AbstractArray, X::Union{Matrix, SparseMatrixCSC}, which_quad::String="H")
    _, m = size(Data)
    viol = zeros(m,1)
    if which_quad == "H"
        for i in 1:m
            viol[i] = Data[:,i]' * X * kron(Data[:,i], Data[:,i])
        end
    else
        for i in 1:m
            viol[i] = Data[:,i]' * X * vech(Data[:,i] * Data[:,i]')
        end
    end
    return viol
end


"""
Choose reduced order (ro) that preserves an acceptable energy.

# Arguments
- `Σ`: Singular value vector from the SVD of some Hankel Matrix
"""
function choose_ro(Σ::Vector; en_low=-15)
    # Energy loss from truncation
    en = 1 .- sqrt.(cumsum(Σ .^ 2)) / norm(Σ)

    # loop through ROM sizes
    en_vals = map(x -> 10.0^x, -1.0:-1.0:en_low)
    r_all = Vector{Float64}()
    for rr = axes(en_vals, 1)
        # determine # basis functions to retain based on energy lost
        en_thresh = en_vals[rr]
        push!(r_all, findfirst(x -> x < en_thresh, en))
    end

    return Int.(r_all), en
end
