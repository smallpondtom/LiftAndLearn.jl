export opinf

# Import necessary utility functions
include("time_derivative_approx.jl")
include("get_data_matrix.jl")
include("unpack_operators.jl")
include("leastsquares.jl")
include("tikhonov.jl")
include("reproject.jl")

"""
$(SIGNATURES)

Solve the Operator Inference (OpInf) regression ``\\mathbf{D}\\mathbf{O}^\\top \\approx \\mathbf{R}^\\top``
with or without Tikhonov regularization and unpack the result into an `Operators` structure.

If `options.with_reg` is `true`, a diagonal Tikhonov matrix ``\\boldsymbol{\\Gamma}`` is
built from `options.λ` with `tikhonov_matrix!` and the problem is solved with `tikhonov`;
otherwise `standard_least_squares` is used. The solver settings (`use_gpu`,
`use_normal_equations`, `use_backslash`, `chunk_size`, `tolerance`, ...) are taken from
`options`. If `options.system.output` is nonzero, the output operator ``\\mathbf{C}`` is
also fitted from `Yt` and `Xhat_t` (see `unpack_operators!`).

## Arguments
- `D::AbstractArray`: data matrix ``\\mathbf{D}`` of size ``K \\times d`` (rows = time).
- `Rt::AbstractArray`: transposed derivative data ``\\mathbf{R}^\\top`` of size ``K \\times r``.
- `Yt::AbstractArray`: transposed output data of size ``K \\times l``.
- `Xhat_t::AbstractArray`: transposed reduced state data of size ``K \\times r``.
- `dims::AbstractArray`: number of columns of ``\\mathbf{D}`` belonging to each operator.
- `operator_symbols::AbstractArray`: operator symbols (e.g. `:A`, `:B`, `:A2u`) in the same order as `dims`.
- `options::AbstractOption`: OpInf options (typically an `LSOpInfOption`).

## Returns
- `operators::Operators`: the inferred operators; the full operator matrix ``\\mathbf{O}`` is stored in `operators.O`.
"""
function leastsquares_solve(D::AbstractArray, Rt::AbstractArray, 
                            Yt::AbstractArray, Xhat_t::AbstractArray, 
                            dims::AbstractArray, operator_symbols::AbstractArray, 
                            options::AbstractOption)

    # compute least squares (pseudo inverse)
    if options.with_reg 
        # Preallocate the Tikhonov weight Matrix
        Γ = spzeros(sum(dims))

        # Construct the Tikhonov matrix
        tikhonov_matrix!(Γ, dims, operator_symbols, options.λ)
        Γ = spdiagm(0 => Γ)  # convert to sparse diagonal matrix

        Ot = tikhonov(Rt, D, Γ;
                      tol=options.tolerance,
                      use_gpu=options.use_gpu,
                      use_normal_form=options.use_normal_equations,
                      use_svd_truncation=options.use_svd_truncation,
                      use_backslash=options.use_backslash,
                      chunk_size=options.chunk_size,
                      max_iterations=options.max_iterations,
                      estimate_memory=options.estimate_memory,
                      preconditioning=options.preconditioning,)
    else
        # Ot = D \ Rt  # INFO: This is not optimal
        Ot = standard_least_squares(D, Rt; 
                                    use_gpu=options.use_gpu, 
                                    use_normal_equations=options.use_normal_equations,
                                    chunk_size=options.chunk_size,
                                    tolerance=options.tolerance,
                                    use_backslash=options.use_backslash,
                                    algorithm=options.algorithm,
                                    estimate_memory=options.estimate_memory,
                                    max_iterations=options.max_iterations,)
    end

    # Extract the operators from the operator matrix O
    O = transpose(Ot)

    # Extract the operators
    operators = Operators(O=O)

    # Unpack the operators
    unpack_operators!(operators, O, Yt, Xhat_t, dims, operator_symbols, options)

    return operators
end


"""
$(SIGNATURES)

Infer the reduced operators by Operator Inference (OpInf) from full-order state data.

The state data is projected onto the basis `Vn` (``\\hat{\\mathbf{X}} = \\mathbf{V}_n^\\top\\mathbf{X}``).
If `Xdot` is empty, the time-derivative data is approximated from `X` by finite
differences with `time_derivative_approx` using `options.data.deriv_type` and
`options.data.Δt`, and the state, input and output data are restricted to the time
indices returned by the scheme. The regression is then built with `get_data_matrix` and
solved with `leastsquares_solve`.

## Arguments
- `X::AbstractArray`: full state snapshot matrix of size ``n \\times K`` (rows = state, columns = time).
- `Vn::AbstractArray`: basis (e.g. POD basis) of size ``n \\times r``.
- `options::AbstractOption`: OpInf options (typically an `LSOpInfOption`).

## Keyword Arguments
- `U::AbstractArray=[0.0]`: input data (``K \\times m`` or ``m \\times K``; fat matrices are transposed to tall).
- `Y::AbstractArray=[0.0]`: output data (``K \\times l`` or ``l \\times K``; fat matrices are transposed to tall).
- `Xdot::AbstractArray=[]`: full time-derivative data of size ``n \\times K``; if empty it is approximated by finite differences.

## Returns
- `op::Operators`: the inferred reduced operators.
"""
function opinf(X::AbstractArray, Vn::AbstractArray, options::AbstractOption; 
               U::AbstractArray=[0.0], Y::AbstractArray=[0.0],
               Xdot::AbstractArray=[])::Operators
    Ut = fat2tall(U)  # make sure that the U-matrix is tall
    Yt = fat2tall(Y)  # make sure that the Y-matrix is tall

    if isempty(Xdot)
        # Approximate the derivative data with finite difference
        Xdot, idx = time_derivative_approx(X, options)
        Xhat = Vn' * X[:, idx]  # fix the index of states
        Xhat_t = Xhat'
        Ut = iszero(Ut) ? [0.0] : Ut[idx, :]  # fix the index of inputs
        Yt = iszero(Yt) ? [0.0] : Yt[idx, :]  # fix the index of outputs
        Rt = Xdot' * Vn
    else
        Xhat = Vn' * X
        Xhat_t = Xhat'
        Rt = Xdot' * Vn  
    end

    D, dims, op_symbols = get_data_matrix(Xhat, Xhat_t, Ut, options; verbose=true)
    op = leastsquares_solve(D, Rt, Yt, Xhat_t, dims, op_symbols, options)
    return op
end


"""
$(SIGNATURES)

Infer the operators by Operator Inference (OpInf) directly from reduced (already
projected) state and derivative data.

## Arguments
- `Xhat::AbstractArray`: reduced state snapshot matrix of size ``r \\times K`` (rows = reduced state, columns = time).
- `options::AbstractOption`: OpInf options (typically an `LSOpInfOption`).

## Keyword Arguments
- `U::AbstractArray=zeros(1,1)`: input data (fat matrices are transposed to tall).
- `Y::AbstractArray=zeros(1,1)`: output data (fat matrices are transposed to tall).
- `Xhatdot::AbstractArray=[]`: reduced time-derivative data of size ``r \\times K``; required.

## Returns
- `op::Operators`: the inferred operators.

## Notes
- `Xhatdot` must be provided; an error is thrown if it is empty.
"""
function opinf(Xhat::AbstractArray, options::AbstractOption; 
               U::AbstractArray=zeros(1,1), Y::AbstractArray=zeros(1,1),
               Xhatdot::AbstractArray=[])
    Ut = fat2tall(U)
    Yt = fat2tall(Y)

    if isempty(Xhatdot)
        error("Xhatdot is required for this function. Please provide the time derivative data.")
    else
        Xhat_t = Xhat'
        Rt = Xhatdot'
    end

    D, dims, op_symbols = get_data_matrix(Xhat, Xhat_t, Ut, options; verbose=true)
    op = leastsquares_solve(D, Rt, Yt, Xhat_t, dims, op_symbols, options)
    return op
end


"""
$(SIGNATURES)

Infer the reduced operators by Operator Inference (OpInf) with the reprojection scheme.

Instead of using finite differences, the derivative data is generated by evaluating the
full-order model `full_op` at the reconstructed reduced states and projecting back onto
`Vn` (see `reproject`).

## Arguments
- `X::AbstractArray`: full state snapshot matrix of size ``n \\times K`` (rows = state, columns = time).
- `Vn::AbstractArray`: basis (e.g. POD basis) of size ``n \\times r``.
- `full_op::Operators`: full-order model operators.
- `options::AbstractOption`: OpInf options (typically an `LSOpInfOption`).

## Keyword Arguments
- `U::AbstractArray=zeros(1,1)`: input data (fat matrices are transposed to tall).
- `Y::AbstractArray=zeros(1,1)`: output data (fat matrices are transposed to tall).
- `return_derivative::Bool=false`: also return the reprojected derivative data ``\\mathbf{R}^\\top``.

## Returns
- `op::Operators`: the inferred reduced operators.
- `Rt::AbstractArray`: reprojected derivative data of size ``K \\times r`` (only if `return_derivative=true`).

## Notes
- `reproject` reads one input row per snapshot, so `U` must have ``K`` time samples (pass
  zeros for an autonomous system); the default `zeros(1,1)` only works for ``K = 1``.
"""
function opinf(X::AbstractArray, Vn::AbstractArray, full_op::Operators, options::AbstractOption;
               U::AbstractArray=zeros(1,1), Y::AbstractArray=zeros(1,1), return_derivative::Bool=false)
    Ut = fat2tall(U)
    Yt = fat2tall(Y)

    Xhat = Vn' * X
    Xhat_t = transpose(Xhat)

    # Reproject
    Rt = reproject(Xhat, Vn, Ut, full_op, options)
    D, dims, op_symbols = get_data_matrix(Xhat, Xhat_t, Ut, options; verbose=true)
    op = leastsquares_solve(D, Rt, Yt, Xhat_t, dims, op_symbols, options)

    if return_derivative
        return op, Rt
    else
        return op
    end
end

