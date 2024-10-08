"""
    reproject(Xhat::AbstractArray, V::AbstractArray, U::AbstractArray,
        op::Operators, options::AbstractOption) → Rhat::AbstractArray

Reprojecting the data to minimize the error affected by the missing orders of the POD basis

## Arguments
- `Xhat::AbstractArray`: state data matrix projected onto the basis
- `V::AbstractArray`: POD basis
- `U::AbstractArray`: input data matrix
- `op::Operators`: full order model operators
- `options::AbstractOption`: options for the operator inference defined by the user

## Return
- `Rhat::AbstractArray`: R matrix (transposed) for the regression problem
"""
function reproject(Xhat::AbstractArray, V::AbstractArray, U::AbstractArray,
    op::Operators, options::AbstractOption)::AbstractArray
    
    n, K = size(Xhat)
    Rt = zeros(K, n)  # Left hand side of the regression problem

    # Assuming the user gave the nonlinear functional or the Operator structure predefined the nonlinear functional
    f = (x, u) -> op.A * x .+ op.B * u .+ op.K .+ op.f(x,u)

    for i in 1:K  # loop thru all data
        x = Xhat[:, i]  # julia automatically makes into column vec after indexing (still (spatial row)-(time col))
        xrec = V * x
        states = f(xrec, U[i, :])
        Rt[i, :] = V' * states
    end
    return Rt
end

