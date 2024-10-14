export epopinf

include("ephec_opinf.jl")
include("epsic_opinf.jl")
include("epp_opinf.jl")

"""
    fidx(n::Int, j::Int, k::Int) → Int

Auxiliary function for the `F` matrix indexing.

## Arguments 
- `n`: row dimension of the F matrix
- `j`: row index 
- `k`: col index

## Returns
- index corresponding to the `F` matrix
"""
function fidx(n,j,k)
    if j >= k
        return Int((n - k/2)*(k - 1) + j)
    else
        return Int((n - j/2)*(j - 1) + k)
    end
end


"""
    delta(v::Int, w::Int) → Float64

Another auxiliary function for the `F` matrix

## Arguments
- `v`: first index
- `w`: second index

## Returns
- coefficient of 1.0 or 0.5
"""
function delta(v,w)
    return v == w ? 1.0 : 0.5
end


"""
$(SIGNATURES)

Energy-preserving Operator Inference (EPOpInf) optimization problem.
"""
function epopinf(X::AbstractArray, Vn::AbstractArray, options::AbstractOption; 
                 U::AbstractArray=zeros(1,1), Xdot::AbstractArray=[], IG::Operators=Operators())

    Ut = fat2tall(U)  # make sure that the U-matrix is tall

    if isempty(Xdot)
        # Approximate the derivative data with finite difference
        Xdot, idx = time_derivative_approx(X, options)
        Xhat = Vn' * X[:, idx]  # fix the index of states
        Xhat_t = transpose(Xhat)
        Ut = iszero(Ut) ? 0 : Ut[idx, :]  # fix the index of inputs
        Rt = Xdot' * Vn
    else
        Xhat = Vn' * X
        Xhat_t = transpose(Xhat)
        Rt = transpose(Vn' * Xdot)
    end

    D, dims, operators_symbols = get_data_matrix(Xhat, Xhat_t, Ut, options; verbose=true)
    if options.method == :EPHEC
        return ephec_opinf(D, Rt, dims, operators_symbols, options, IG)
    elseif options.method == :EPSIC
        return epsic_opinf(D, Rt, dims, operators_symbols, options, IG)
    elseif options.method == :EPP
        return epp_opinf(D, Rt, dims, operators_symbols, options, IG)
    else
        error("Incorrect optimization options. Currently only EPHEC, EPSIC, and EPP are supported.")
    end
end