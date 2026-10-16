export epopinf

"""
$(SIGNATURES)

Return the dual objective value of a solved JuMP `model`, or `"n/a"` if the solver does not
provide it (e.g. Ipopt for quadratic objectives). Used in the solver log summaries. Internal helper.

## Arguments
- `model::JuMP.Model`: the solved optimization model.

## Returns
- `val::Union{Float64,String}`: the dual objective value, or `"n/a"`.
"""
function _dual_objective_or_na(model)
    try
        return dual_objective_value(model)
    catch
        return "n/a"
    end
end

include("ephec_opinf.jl")
include("epsic_opinf.jl")
include("epp_opinf.jl")

"""
$(SIGNATURES)

Return the column index of the monomial ``x_j x_k`` in the non-redundant quadratic operator
``\\mathbf{F}`` (`A2u`, of size ``n \\times n(n+1)/2``), whose columns are ordered as
``x_1x_1, x_1x_2, \\ldots, x_1x_n, x_2x_2, \\ldots, x_nx_n``. Internal helper.

## Arguments
- `n`: state dimension.
- `j`: index of the first state variable.
- `k`: index of the second state variable (the result is symmetric in `j` and `k`).

## Returns
- `idx::Int`: column index of ``x_j x_k`` in ``\\mathbf{F}``.
"""
function fidx(n,j,k)
    if j >= k
        return Int((n - k/2)*(k - 1) + j)
    else
        return Int((n - j/2)*(j - 1) + k)
    end
end


"""
$(SIGNATURES)

Return the weight of an entry of the non-redundant quadratic operator ``\\mathbf{F}`` in the
energy-preserving constraint: `1.0` for a squared monomial (``v = w``) and `0.5` for a
cross term (``v \\neq w``), which appears twice in the redundant operator. Internal helper.

## Arguments
- `v`: first state index.
- `w`: second state index.

## Returns
- `coeff::Float64`: `1.0` if `v == w`, otherwise `0.5`.
"""
function delta(v,w)
    return v == w ? 1.0 : 0.5
end


"""
$(SIGNATURES)

Infer the reduced operators by Energy-Preserving Operator Inference (EP-OpInf), i.e. a
least-squares OpInf problem whose quadratic operator is constrained (or penalized) to be
energy preserving, solved with JuMP and Ipopt.

The method is chosen by `options.method`: `:EPHEC` (hard equality constraints, see
`ephec_opinf`), `:EPSIC` (soft inequality constraints, see `epsic_opinf`) or `:EPP`
(penalty, see `epp_opinf`). If `Xdot` is empty, the time derivative is approximated from
`X` by finite differences with `time_derivative_approx`.

## Arguments
- `X::AbstractArray`: full state snapshot matrix of size ``n \\times K`` (rows = state, columns = time).
- `Vn::AbstractArray`: basis (e.g. POD basis) of size ``n \\times r``.
- `options::AbstractOption`: `EPHECOpInfOption`, `EPSICOpInfOption` or `EPPOpInfOption`.

## Keyword Arguments
- `U::AbstractArray=zeros(1,1)`: input data (fat matrices are transposed to tall).
- `Xdot::AbstractArray=[]`: full time-derivative data of size ``n \\times K``; if empty it is approximated by finite differences.
- `IG::Operators=Operators()`: initial guess of the operators (used if `options.optim.initial_guess=true`).

## Returns
- `op::Operators`: the inferred reduced operators.

## Notes
- Only linear and quadratic state operators, a linear input operator and a constant term are supported.
- Throws an error for an unsupported `options.method`.
"""
function epopinf(X::AbstractArray, Vn::AbstractArray, options::AbstractOption; 
                 U::AbstractArray=zeros(1,1), Xdot::AbstractArray=[], IG::Operators=Operators())

    Ut = fat2tall(U)  # make sure that the U-matrix is tall

    if isempty(Xdot)
        # Approximate the derivative data with finite difference
        Xdot, idx = time_derivative_approx(X, options)
        Xhat = Vn' * X[:, idx]  # fix the index of states
        Xhat_t = transpose(Xhat)
        Ut = iszero(Ut) ? [0.0] : Ut[idx, :]  # fix the index of inputs
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