"""
$(SIGNATURES)

Generate the derivative data by reprojection, which removes the closure error caused by
the truncated POD modes.

For each snapshot ``\\hat{\\mathbf{x}}_i`` the full-order right-hand side
``\\mathbf{f}(\\mathbf{V}\\hat{\\mathbf{x}}_i, \\mathbf{u}_i) = \\mathbf{A}\\mathbf{x} + \\mathbf{B}\\mathbf{u} + \\mathbf{K} + \\mathbf{f}_{\\text{nl}}(\\mathbf{x}, \\mathbf{u})``
of `op` is evaluated at the reconstructed state and projected back onto `V`.

## Arguments
- `Xhat::AbstractArray`: reduced state snapshot matrix of size ``r \\times K`` (columns = snapshots).
- `V::AbstractArray`: basis (e.g. POD basis) of size ``n \\times r``.
- `Ut::AbstractArray`: tall input data matrix of size ``K \\times m``.
- `op::Operators`: full-order model operators. The nonlinear part is `op.f` if a custom function
  was supplied, otherwise the default built from the polynomial and bilinear operators of `op`.
- `options::AbstractOption`: OpInf options.

## Returns
- `Rt::AbstractArray`: reprojected derivative data ``\\mathbf{R}^\\top`` of size ``K \\times r``.
"""
function reproject(Xhat::AbstractArray, V::AbstractArray, Ut::AbstractArray,
    op::Operators, options::AbstractOption)::AbstractArray
    
    n, K = size(Xhat)
    Rt = zeros(K, n)  # Left hand side of the regression problem

    # Use the user-supplied nonlinear function if given, else the default built from the operators
    fnl = _nonlinear_function(op)
    f = (x, u) -> op.A * x .+ op.B * u .+ op.K .+ fnl(x,u)

    for i in 1:K  # loop thru all data
        x = Xhat[:, i]  # julia automatically makes into column vec after indexing (still (spatial row)-(time col))
        xrec = V * x
        states = f(xrec, Ut[i, :])
        Rt[i, :] = V' * states
    end
    return Rt
end

